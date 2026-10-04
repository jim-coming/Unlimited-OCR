#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TOC hierarchy + procedure-step aware chunking (v2) for the current
Unlimited-OCR -> Qwen RAG pipeline.

Main idea
---------
1. Read the original PDF TOC layout with PyMuPDF.
2. Infer TOC hierarchy from indentation instead of treating every title as
   the same level.
3. Exclude TOC pages from retrieval.
4. Match TOC titles back to OCR body pages.
5. Keep the full hierarchical section path, e.g.
      溫度過高 > 動力控制系統電動水泵無作用
6. Inside each section, split by procedure headings such as:
      1 檢查...
      2 檢查...
   while NOT treating substeps "1. xxx" / "2. xxx" as new chunks.
7. Pack complete procedure blocks into compact retrieval chunks.
8. If one procedure block is unusually long, split only inside that step.
9. Prefix every chunk with the section path.
10. Build Qwen3.5 GGUF embeddings compatible with Rev3.

Original files are NOT modified.

Default outputs
---------------
./flash/rag/all_chunks_toc_step.jsonl
./flash/rag/all_embeddings_toc_step.pt
"""

import argparse
import difflib
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from llama_cpp import Llama

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = str(ROOT / "flash/rag/all_chunks.jsonl")
DEFAULT_CACHE_ROOT = str(ROOT / "flash/cache")
DEFAULT_OUTPUT_JSONL = str(ROOT / "flash/rag/all_chunks_toc_step.jsonl")
DEFAULT_MODEL = str(ROOT / "models/Qwen3.5-2B-Q5_K_M.gguf")
DEFAULT_OUTPUT_EMBED = str(ROOT / "flash/rag/all_embeddings_toc_step.pt")

QWEN_DIM = 2048

# Retrieval chunk target after section/step segmentation.
DEFAULT_TARGET_CHARS = 320
DEFAULT_OVERLAP_CHARS = 40
DEFAULT_HARD_MAX_CHARS = 520

LEAK_PREFIXES = (
    "Do not use special characters",
    "Treat all tabular layout as plain text with spacing.",
    "Preserve the original document structure.",
    "Output section titles and headings using Markdown heading syntax",
    "Use Markdown headings only for visually distinct section titles",
    "Preserve tables, lists, and the original reading order.",
    "Do not summarize, rewrite, infer missing text",
)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_no}: {e}"
                ) from e
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_match(text):
    """
    Normalize OCR/PDF text only for title matching.
    The original content is still kept verbatim for RAG.
    """
    s = str(text or "").strip()
    s = re.sub(r"^#{1,6}\s*", "", s)
    s = s.replace(" ", "").replace("\u3000", "")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("：", ":")
    s = s.replace("＂", '"').replace("“", '"').replace("”", '"')
    s = re.sub(r"[.…·．。]+$", "", s)
    return s


def strip_prompt_leakage(text):
    kept = []
    removed = []

    for line in text.splitlines():
        s = line.strip()
        if any(s.startswith(prefix) for prefix in LEAK_PREFIXES):
            removed.append(line)
            continue
        kept.append(line)

    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out, removed


def page_number_from_name(page_dir):
    m = re.search(r"page_(\d+)$", page_dir.name)
    return int(m.group(1)) if m else None


def locate_clean_pages(cache_root, sample_record):
    meta = sample_record.get("metadata") or {}
    pdf_sha = str(meta.get("pdf_sha256") or "").strip()
    cfg_hash = str(meta.get("ocr_config_hash") or "").strip()

    if not pdf_sha or not cfg_hash:
        return []

    base = cache_root / pdf_sha / cfg_hash / "pages"
    if not base.is_dir():
        return []

    pages = []
    for page_dir in sorted(base.glob("page_*")):
        page_no = page_number_from_name(page_dir)
        clean = page_dir / "clean.md"
        if page_no is None or not clean.is_file():
            continue

        text = clean.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        text, _ = strip_prompt_leakage(text)
        pages.append((page_no, text))

    return pages


def repeated_short_lines(
    pages,
    min_pages=3,
    max_chars=32,
    edge_lines=4,
):
    """
    Detect recurring page headers/footers ONLY near page edges.

    IMPORTANT:
    Do not scan the whole page for repeated short lines. In service manuals,
    legitimate procedure lines such as:
        "1 檢查保險絲"
        "1. 將電源開關OFF。"
        "是 > 到2。"
    can repeat on many pages. Treating every repeated line as boilerplate
    would delete real troubleshooting content.

    We therefore only count short lines among the first/last few non-empty
    OCR lines on each page.
    """
    counter = Counter()

    for _, text in pages:
        nonempty = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not nonempty:
            continue

        edge = nonempty[:edge_lines] + nonempty[-edge_lines:]
        seen_this_page = set()

        for s in edge:
            if len(s) > max_chars:
                continue

            if s.lower().startswith("<table"):
                continue

            key = normalize_match(s)
            if not key:
                continue

            if key not in seen_this_page:
                counter[key] += 1
                seen_this_page.add(key)

    return {k for k, c in counter.items() if c >= min_pages}


def remove_noise_lines(text, recurring):
    kept = []

    for line in text.splitlines():
        s = line.strip()

        if not s:
            kept.append("")
            continue

        key = normalize_match(s)

        if key in recurring:
            continue

        # Page footer, e.g. 28-9.
        if re.fullmatch(r"\d{1,3}-\d{1,3}", s):
            continue

        # Vertical OCR numbering noise.
        if re.fullmatch(r"\d{1,3}", s):
            continue

        kept.append(line)

    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


# ======================================================================
# TOC layout parsing from original PDF
# ======================================================================

TOC_PAGE_RE = re.compile(r"^(.*?)(?:[.…·．]{2,}|\s+)\s*(\d{1,3})\s*$")


def _pdf_page_lines(page):
    """
    Return visual text lines with x/y coordinates from the PDF text layer.
    """
    d = page.get_text("dict")
    rows = []

    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            text = "".join(str(s.get("text", "")) for s in spans).strip()
            if not text:
                continue

            bbox = line.get("bbox") or [0, 0, 0, 0]
            rows.append({
                "text": text,
                "x0": float(bbox[0]),
                "y0": float(bbox[1]),
                "x1": float(bbox[2]),
                "y1": float(bbox[3]),
                "spans": spans,
            })

    return rows


def _strip_toc_page_number(text):
    m = TOC_PAGE_RE.match(text.strip())
    if not m:
        return None

    title = m.group(1).strip(" .…·．。\t")
    page_no = int(m.group(2))

    if not title:
        return None

    return title, page_no


def extract_toc_hierarchy_from_pdf(pdf_path):
    """
    Parse TOC entries using visual indentation from the PDF itself.

    Works especially well for this manual because the TOC is two-column and
    child entries are visually indented.

    Returns:
      {
        "toc_pages": [1],
        "entries": [
          {"title": "...", "page": 4, "level": 1, "path": [...]},
          ...
        ]
      }
    """
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF (fitz) is not available. "
            "Install/import PyMuPDF or use text fallback."
        )

    pdf = fitz.open(str(pdf_path))
    toc_pages = []
    all_entries = []

    try:
        for pno in range(len(pdf)):
            page = pdf[pno]
            plain = page.get_text("text")

            # Conservative TOC page detection.
            if "目錄" not in plain and "目录" not in plain:
                continue

            lines = _pdf_page_lines(page)
            candidates = []

            # First collect direct "title .... page" lines.
            # Also support one wrapped title line immediately before it.
            pending_by_column = {0: None, 1: None}
            width = float(page.rect.width)

            lines = sorted(lines, key=lambda x: (x["y0"], x["x0"]))

            for row in lines:
                text = row["text"].strip()

                # Ignore the page title / TOC label itself.
                if normalize_match(text) in {"目錄", "目录", "故障查修"}:
                    continue

                # Ignore pure vertical numbering / page artifacts.
                if re.fullmatch(r"\d{1,3}", text):
                    continue

                column = 0 if row["x0"] < width / 2 else 1

                parsed = _strip_toc_page_number(text)
                if parsed:
                    title, target_page = parsed

                    pending = pending_by_column.get(column)
                    if pending:
                        # Join wrapped TOC line only when visual indentation
                        # is approximately the same.
                        if abs(float(pending["x0"]) - float(row["x0"])) <= 4.0:
                            title = str(pending["text"]).strip() + title
                        pending_by_column[column] = None

                    candidates.append({
                        "title": title.strip(),
                        "page": target_page,
                        "x0": float(row["x0"]),
                        "y0": float(row["y0"]),
                        "column": column,
                    })
                    continue

                # Potential wrapped first half of a TOC title.
                if len(text) <= 40:
                    pending_by_column[column] = row

            if len(candidates) < 3:
                continue

            toc_pages.append(pno + 1)

            # Reading order: left column top-to-bottom, then right.
            candidates.sort(key=lambda r: (r["column"], r["y0"]))

            # Infer hierarchy level from indentation clusters per column.
            by_col = defaultdict(list)
            for c in candidates:
                by_col[c["column"]].append(c)

            col_x_clusters = {}
            for col, items in by_col.items():
                vals = sorted({round(float(x["x0"]), 1) for x in items})

                # Merge nearly-equal x positions.
                clusters = []
                for x in vals:
                    if not clusters or abs(x - clusters[-1]) > 4.0:
                        clusters.append(x)

                col_x_clusters[col] = clusters

            # Determine whether the first left-column indentation is a
            # document-root TOC entry. In this PDF it is "重要故障查修":
            # one visually less-indented entry, followed by normal headings.
            left_clusters = col_x_clusters.get(0, [])
            left_root_offset = 0

            if len(left_clusters) >= 3:
                left_root_offset = 1

            entries = []
            hierarchy = {}

            for c in candidates:
                clusters = col_x_clusters[c["column"]]
                x = float(c["x0"])
                nearest_idx = min(
                    range(len(clusters)),
                    key=lambda i: abs(clusters[i] - x),
                )

                if c["column"] == 0:
                    if left_root_offset and nearest_idx == 0:
                        # Root/title-like TOC entry. Keep it as level 0.
                        level = 0
                    else:
                        level = max(1, nearest_idx - left_root_offset + 1)
                else:
                    # Right column is a continuation. Its least-indented
                    # entry corresponds to level 1.
                    level = nearest_idx + 1

                title = c["title"].strip()

                # Update hierarchy stack.
                hierarchy[level] = title
                for k in list(hierarchy):
                    if k > level:
                        del hierarchy[k]

                if level == 0:
                    path = [title]
                else:
                    path = [
                        hierarchy[k]
                        for k in sorted(hierarchy)
                        if 1 <= k <= level
                    ]

                item = dict(c)
                item["level"] = level
                item["path"] = path
                entries.append(item)

            all_entries.extend(entries)

    finally:
        pdf.close()

    # De-duplicate by normalized title, preserving first occurrence.
    unique = []
    seen = set()

    for e in all_entries:
        key = normalize_match(e["title"])
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(e)

    return {
        "toc_pages": sorted(set(toc_pages)),
        "entries": unique,
    }


def extract_toc_text_fallback(pages):
    """
    Text-only fallback when the original PDF text layer is unavailable.
    Levels are less reliable than the visual-indent parser.
    """
    entries = []
    toc_pages = []

    for page_no, text in pages:
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        if not any(normalize_match(x) in {"目錄", "目录"} for x in lines[:12]):
            continue

        toc_pages.append(page_no)
        pending = []

        for line in lines:
            if normalize_match(line) in {"目錄", "目录", "故障查修"}:
                continue

            m = TOC_PAGE_RE.match(line)
            if m:
                title = m.group(1).strip(" .…·．。\t")
                if pending:
                    title = "".join(pending) + title
                    pending = []

                if title:
                    entries.append({
                        "title": title,
                        "page": int(m.group(2)),
                        "level": 1,
                        "path": [title],
                    })
                continue

            if len(line) <= 40:
                pending.append(line)

    return {
        "toc_pages": sorted(set(toc_pages)),
        "entries": entries,
    }


# ======================================================================
# Body segmentation using TOC hierarchy
# ======================================================================

def make_title_maps(toc_entries):
    exact = {}
    entries = []

    for e in toc_entries:
        key = normalize_match(e["title"])
        if not key:
            continue

        exact[key] = list(e.get("path") or [e["title"]])
        entries.append({
            "key": key,
            "title": e["title"],
            "path": list(e.get("path") or [e["title"]]),
        })

    return exact, entries


def match_title_line(line, exact_map, title_entries):
    key = normalize_match(line)

    if key in exact_map:
        return exact_map[key]

    # Fuzzy fallback for OCR errors in a short standalone heading line.
    # Keep the threshold high so normal body sentences are not misclassified.
    if not key or len(key) > 42:
        return None

    best = None
    best_ratio = 0.0

    for e in title_entries:
        tk = e["key"]

        # Avoid comparing wildly different lengths.
        if abs(len(tk) - len(key)) > max(3, int(len(tk) * 0.25)):
            continue

        ratio = difflib.SequenceMatcher(None, key, tk).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best = e

    if best is not None and best_ratio >= 0.92:
        return best["path"]

    return None


def segment_body_pages(pages, toc_pages, toc_entries, recurring):
    exact_map, title_entries = make_title_maps(toc_entries)

    segments = []
    current_path = []
    body = []
    start_page = None
    end_page = None

    def flush():
        nonlocal body, start_page, end_page

        text = "\n".join(body).strip()

        if text:
            segments.append({
                "section_path": list(current_path),
                "body": text,
                "page_start": start_page,
                "page_end": end_page,
            })

        body = []
        start_page = None
        end_page = None

    for page_no, raw_text in pages:
        if page_no in toc_pages:
            continue

        text = remove_noise_lines(raw_text, recurring)
        lines = [x.strip() for x in text.splitlines() if x.strip()]

        i = 0
        while i < len(lines):
            matched_path = match_title_line(
                lines[i],
                exact_map=exact_map,
                title_entries=title_entries,
            )

            if matched_path:
                flush()
                current_path = list(matched_path)
                i += 1
                continue

            if start_page is None:
                start_page = page_no

            end_page = page_no
            body.append(lines[i])
            i += 1

    flush()
    return segments


# ======================================================================
# Procedure-step aware chunking
# ======================================================================

# Main procedure heading: "1 檢查..." / "10 檢查..."
# Substeps like "1. 將..." do NOT match because of the period.
PROCEDURE_RE = re.compile(r"^\s*(\d{1,2})\s+(?![.)、])(\S.*)$")


def split_into_atoms(text):
    """
    Keep complete HTML tables together; otherwise use non-empty lines.
    """
    parts = re.split(r"(<table\b.*?</table>)", text, flags=re.I | re.S)
    atoms = []

    for part in parts:
        if not part or not part.strip():
            continue

        p = part.strip()

        if re.match(r"^<table\b", p, flags=re.I):
            atoms.append(p)
            continue

        for line in p.splitlines():
            line = line.strip()
            if line:
                atoms.append(line)

    return atoms


def split_procedure_blocks(body):
    """
    Split one semantic section into blocks:
      preamble (if any)
      step 1
      step 2
      ...

    Returns list of:
      {"step": 1 or None, "heading": "...", "text": "..."}
    """
    atoms = split_into_atoms(body)

    blocks = []
    current = []
    current_step = None
    current_heading = ""

    def flush():
        nonlocal current

        text = "\n".join(current).strip()
        if text:
            blocks.append({
                "step": current_step,
                "heading": current_heading,
                "text": text,
            })
        current = []

    for atom in atoms:
        m = PROCEDURE_RE.match(atom)

        if m:
            flush()
            current_step = int(m.group(1))
            current_heading = atom
            current = [atom]
            continue

        current.append(atom)

    flush()
    return blocks


def split_long_block(block_text, hard_max_chars, overlap_chars):
    """
    Only used if ONE procedure block itself is too long.
    Keep its first heading line in every subpiece.
    """
    atoms = split_into_atoms(block_text)

    if not atoms:
        return []

    heading = atoms[0] if PROCEDURE_RE.match(atoms[0]) else ""
    body_atoms = atoms[1:] if heading else atoms

    pieces = []
    current = []

    def render(items):
        content = "\n".join(items).strip()
        if heading and (not content.startswith(heading)):
            content = f"{heading}\n{content}".strip()
        return content

    def overlap_tail(items):
        if overlap_chars <= 0:
            return []

        tail = []
        total = 0

        for item in reversed(items):
            tail.append(item)
            total += len(item) + 1
            if total >= overlap_chars:
                break

        tail.reverse()
        return tail

    for atom in body_atoms:
        # Preserve complete tables even when they are large.
        if not current:
            candidate = render([atom])
        else:
            candidate = render(current + [atom])

        if len(candidate) <= hard_max_chars or not current:
            current.append(atom)
            continue

        finished = render(current)
        if finished:
            pieces.append(finished)

        current = overlap_tail(current) + [atom]

    tail = render(current)
    if tail:
        pieces.append(tail)

    return pieces


def pack_procedure_blocks(
    blocks,
    target_chars,
    hard_max_chars,
    overlap_chars,
):
    """
    Prefer whole procedure steps.

    - Small adjacent steps can be packed together.
    - A step larger than hard_max_chars is split internally.
    - We do not normally overlap whole independent steps; the repeated
      section-path prefix provides semantic continuity.
    """
    normalized = []

    for block in blocks:
        text = str(block["text"]).strip()

        if len(text) <= hard_max_chars:
            normalized.append({
                "steps": [] if block["step"] is None else [block["step"]],
                "text": text,
            })
            continue

        for piece in split_long_block(
            text,
            hard_max_chars=hard_max_chars,
            overlap_chars=overlap_chars,
        ):
            normalized.append({
                "steps": [] if block["step"] is None else [block["step"]],
                "text": piece,
            })

    chunks = []
    current_texts = []
    current_steps = []

    def flush():
        nonlocal current_texts, current_steps

        text = "\n".join(current_texts).strip()
        if text:
            chunks.append({
                "text": text,
                "steps": list(dict.fromkeys(current_steps)),
            })

        current_texts = []
        current_steps = []

    for item in normalized:
        text = item["text"]
        steps = item["steps"]

        if not current_texts:
            current_texts = [text]
            current_steps = list(steps)
            continue

        candidate = "\n".join(current_texts + [text]).strip()

        # Pack adjacent complete procedure blocks only while reasonably compact.
        if len(candidate) <= target_chars:
            current_texts.append(text)
            current_steps.extend(steps)
            continue

        flush()
        current_texts = [text]
        current_steps = list(steps)

    flush()
    return chunks


def build_chunks_from_segments(
    segments,
    sample_record,
    source_file,
    toc_pages,
    target_chars,
    hard_max_chars,
    overlap_chars,
):
    rows = []
    chunk_counter = 0

    for seg_idx, seg in enumerate(segments, 1):
        section_path = list(seg.get("section_path") or [])
        body = str(seg.get("body") or "").strip()

        blocks = split_procedure_blocks(body)

        packed = pack_procedure_blocks(
            blocks,
            target_chars=target_chars,
            hard_max_chars=hard_max_chars,
            overlap_chars=overlap_chars,
        )

        heading_prefix = ""
        if section_path:
            heading_prefix = "主題：" + " > ".join(section_path)

        for sub_idx, item in enumerate(packed, 1):
            chunk_counter += 1

            content = item["text"].strip()

            if heading_prefix:
                content = f"{heading_prefix}\n\n{content}".strip()

            document_id = str(sample_record.get("document_id") or "")
            page_start = int(seg.get("page_start") or 0)
            page_end = int(seg.get("page_end") or page_start)

            chunk_id = (
                f"{document_id[:16]}"
                f"-p{page_start:04d}-{page_end:04d}"
                f"-s{seg_idx:03d}-c{sub_idx:03d}"
            )

            meta = dict(sample_record.get("metadata") or {})
            meta.update({
                "section_path": section_path,
                "section_title": section_path[-1] if section_path else "",
                "procedure_steps": item["steps"],
                "segment_index": seg_idx,
                "sub_chunk_index": sub_idx,
                "toc_page_numbers": sorted(toc_pages),
                "chunking_strategy": "toc_hierarchy_plus_procedure_steps",
                "target_chunk_chars": target_chars,
                "hard_max_chunk_chars": hard_max_chars,
                "intra_step_overlap_chars": overlap_chars,
            })

            rows.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source_file": source_file,
                "source_path": str(sample_record.get("source_path") or ""),
                "page": page_start,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_index": chunk_counter,
                "page_chunk_index": sub_idx,
                "content": content,
                "content_sha256": sha256_text(content),
                "metadata": meta,
            })

    return rows


def fallback_small_chunks(source_rows, target_chars):
    """
    Minimal fallback if TOC hierarchy cannot be extracted.
    """
    out = []
    idx = 0

    for src in source_rows:
        text, removed = strip_prompt_leakage(
            str(src.get("content") or "")
        )

        atoms = split_into_atoms(text)
        current = []

        def flush():
            nonlocal current, idx
            content = "\n".join(current).strip()
            if not content:
                current = []
                return

            idx += 1
            rec = dict(src)
            rec["chunk_id"] = f"{src.get('chunk_id', 'chunk')}-fb{idx:04d}"
            rec["chunk_index"] = idx
            rec["content"] = content
            rec["content_sha256"] = sha256_text(content)

            meta = dict(src.get("metadata") or {})
            meta.update({
                "chunking_strategy": "fallback_fixed_lines",
                "target_chunk_chars": target_chars,
                "prompt_leak_lines_removed": len(removed),
            })
            rec["metadata"] = meta
            out.append(rec)
            current = []

        for atom in atoms:
            candidate = "\n".join(current + [atom]).strip()
            if current and len(candidate) > target_chars:
                flush()
            current.append(atom)

        flush()

    return out


# ======================================================================
# Qwen embeddings
# ======================================================================

def text_embedding(llm, text):
    result = llm.create_embedding(text)
    raw = result["data"][0]["embedding"]

    vec = torch.tensor(raw, dtype=torch.float32)

    if vec.ndim == 2:
        vec = vec.mean(dim=0)

    if vec.ndim != 1:
        raise RuntimeError(
            f"Unexpected embedding shape: {tuple(vec.shape)}"
        )

    if vec.shape[0] != QWEN_DIM:
        raise RuntimeError(
            f"Embedding dim mismatch: expected {QWEN_DIM}, "
            f"got {vec.shape[0]}"
        )

    return F.normalize(vec, p=2, dim=0)


def build_embeddings(rows, model_path, output_path, n_ctx):
    print()
    print("Loading Qwen GGUF...")
    t_load = time.perf_counter()

    llm = Llama(
        model_path=str(model_path),
        embedding=True,
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        verbose=False,
    )

    print(
        f"Qwen loaded in "
        f"{time.perf_counter() - t_load:.2f}s"
    )
    print("Generating embeddings...")

    vectors = []
    docs = []
    t0 = time.perf_counter()

    for i, rec in enumerate(rows, 1):
        content = str(rec.get("content") or "").strip()
        if not content:
            continue

        vec = text_embedding(llm, content)
        vectors.append(vec)

        doc = dict(rec)
        doc["text"] = content
        docs.append(doc)

        steps = (rec.get("metadata") or {}).get(
            "procedure_steps", []
        )

        print(
            f"\r[{i:4d}/{len(rows):4d}] "
            f"p={rec.get('page_start')}-{rec.get('page_end')} "
            f"steps={steps!s:<10} "
            f"chars={len(content):>4}",
            end="",
            flush=True,
        )

    print()

    if not vectors:
        raise RuntimeError("No embeddings generated.")

    embeddings = torch.stack(vectors, dim=0).float().cpu()
    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    db = {
        "embeddings": embeddings,
        "documents": docs,
        "metadata": {
            "embedding_model": str(model_path),
            "embedding_dim": int(embeddings.shape[1]),
            "pooling": "mean_if_token_level_then_l2",
            "chunk_count": len(docs),
            "chunking_strategy": "toc_hierarchy_plus_procedure_steps",
        },
    }

    torch.save(db, output_path)

    print()
    print("=" * 78)
    print("EMBEDDING FINISHED")
    print("=" * 78)
    print("Embedding shape:", embeddings.shape)
    print(
        "Norm min       :",
        float(embeddings.norm(dim=1).min()),
    )
    print(
        "Norm max       :",
        float(embeddings.norm(dim=1).max()),
    )
    print(
        "Embedding time :",
        f"{time.perf_counter() - t0:.2f}s",
    )
    print("Saved          :", output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-output", default=DEFAULT_OUTPUT_EMBED)

    parser.add_argument(
        "--pdf",
        default=None,
        help=(
            "Optional original PDF path. If omitted, use source_path "
            "from all_chunks.jsonl."
        ),
    )

    parser.add_argument(
        "--source-file",
        default=None,
        help='Optional: only process one PDF, e.g. "故障查修.pdf"',
    )

    parser.add_argument(
        "--target-chars",
        type=int,
        default=DEFAULT_TARGET_CHARS,
    )

    parser.add_argument(
        "--hard-max-chars",
        type=int,
        default=DEFAULT_HARD_MAX_CHARS,
    )

    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=DEFAULT_OVERLAP_CHARS,
    )

    parser.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Build chunks only; do not load Qwen.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    cache_root = Path(args.cache_root)
    output_jsonl = Path(args.output_jsonl)
    model_path = Path(args.model)
    output_embed = Path(args.embedding_output)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    source_rows = read_jsonl(input_path)

    grouped = defaultdict(list)
    for row in source_rows:
        name = str(row.get("source_file") or "unknown")

        if args.source_file and name != args.source_file:
            continue

        grouped[name].append(row)

    if not grouped:
        raise RuntimeError("No matching source documents.")

    all_out = []

    print("=" * 78)
    print("TOC HIERARCHY + PROCEDURE-STEP CHUNKER")
    print("=" * 78)

    for source_file, rows in grouped.items():
        sample = rows[0]
        pages = locate_clean_pages(
            cache_root,
            sample,
        )

        if not pages:
            print()
            print(source_file)
            print("  OCR clean.md cache not found -> fallback")
            all_out.extend(
                fallback_small_chunks(
                    rows,
                    target_chars=args.target_chars,
                )
            )
            continue

        if args.pdf:
            pdf_path = Path(args.pdf)
        else:
            pdf_path = Path(
                str(sample.get("source_path") or "")
            )

        toc_info = None
        toc_mode = "text-fallback"

        if pdf_path.is_file() and fitz is not None:
            try:
                toc_info = extract_toc_hierarchy_from_pdf(
                    pdf_path
                )
                if toc_info["entries"]:
                    toc_mode = "pdf-visual-indent"
            except Exception as e:
                print(
                    f"[WARN] PDF TOC layout parse failed: {e}"
                )

        if not toc_info or not toc_info["entries"]:
            toc_info = extract_toc_text_fallback(pages)

        toc_pages = set(toc_info.get("toc_pages") or [])
        toc_entries = list(toc_info.get("entries") or [])

        recurring = repeated_short_lines(pages)

        if toc_entries:
            segments = segment_body_pages(
                pages=pages,
                toc_pages=toc_pages,
                toc_entries=toc_entries,
                recurring=recurring,
            )

            doc_rows = build_chunks_from_segments(
                segments=segments,
                sample_record=sample,
                source_file=source_file,
                toc_pages=toc_pages,
                target_chars=args.target_chars,
                hard_max_chars=args.hard_max_chars,
                overlap_chars=args.overlap_chars,
            )
        else:
            segments = []
            doc_rows = fallback_small_chunks(
                rows,
                target_chars=args.target_chars,
            )
            toc_mode = "no-toc -> fallback"

        all_out.extend(doc_rows)

        print()
        print(source_file)
        print("  TOC mode    :", toc_mode)
        print("  PDF         :", pdf_path)
        print("  OCR pages   :", len(pages))
        print("  TOC pages   :", sorted(toc_pages))
        print("  TOC entries :", len(toc_entries))
        print("  segments    :", len(segments))
        print("  chunks      :", len(doc_rows))

        if toc_entries:
            print("  TOC hierarchy sample:")
            for e in toc_entries[:16]:
                print(
                    f"    L{e.get('level')} "
                    f"{' > '.join(e.get('path') or [e['title']])}"
                )

    if not all_out:
        raise RuntimeError("No chunks generated.")

    write_jsonl(
        output_jsonl,
        all_out,
    )

    lengths = [
        len(str(r.get("content") or ""))
        for r in all_out
    ]

    print()
    print("=" * 78)
    print("CHUNKING FINISHED")
    print("=" * 78)
    print("Total chunks :", len(all_out))
    print(
        "Length min/avg/max:",
        min(lengths),
        f"{sum(lengths) / len(lengths):.1f}",
        max(lengths),
    )
    print("Saved JSONL  :", output_jsonl)
    print("Original all_chunks.jsonl was NOT modified.")

    if args.no_embed:
        return

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    build_embeddings(
        rows=all_out,
        model_path=model_path,
        output_path=output_embed,
        n_ctx=args.n_ctx,
    )


if __name__ == "__main__":
    main()
