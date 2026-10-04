#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic RAG rechunker for Unlimited-OCR output.

Goal
----
Keep OCR and RAG chunking separate.

This script does NOT rerun OCR. It reads each page's existing clean.md, then asks a
local Qwen GGUF model to identify semantic section boundaries by ORIGINAL LINE
NUMBER only. Qwen is not allowed to rewrite the OCR content. The original OCR text
is then sliced deterministically into small retrieval chunks.

Pipeline
--------
Unlimited-OCR clean.md
  -> line-numbered page text
  -> Qwen section-boundary detector (JSON: start_line / level / exact title)
  -> deterministic section slicer using ORIGINAL OCR lines
  -> small 500-char retrieval chunks (default)
  -> flash/rag/all_chunks_semantic.jsonl

The original flash/rag/all_chunks.jsonl is left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from llama_cpp import Llama


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_FLASH_DIR = DEFAULT_ROOT / "flash"
DEFAULT_DOCUMENTS_FILE = DEFAULT_FLASH_DIR / "rag" / "documents.jsonl"
DEFAULT_OUTPUT_FILE = DEFAULT_FLASH_DIR / "rag" / "all_chunks_semantic.jsonl"
DEFAULT_MODEL = DEFAULT_ROOT / "models" / "Qwen3.5-2B-Q5_K_M.gguf"

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 60
DEFAULT_N_CTX = 4096
DEFAULT_N_BATCH = 512
DEFAULT_N_GPU_LAYERS = -1
DETECTOR_VERSION = 1


SYSTEM_PROMPT = r"""
你是一個「文件結構分析器」，不是摘要器，也不是問答助手。

你的唯一任務：從 OCR 頁面中找出真正的章節/主題/子主題起始行。

規則：
1. 只能選擇輸入中真實存在的整行文字作為標題；不可改寫、補字、摘要或創造標題。
2. 每個標題必須回傳它在輸入中的原始行號。
3. level 只使用 1、2、3：
   - 1 = 新的主要主題/章節
   - 2 = 該主題下的子主題
   - 3 = 更細的子主題
4. 不要把下列內容當標題：
   - 程序步驟或編號步驟（例如「1 檢查...」、「2. 將...」）
   - 一般清單項目
   - 表格內容、欄名、OK/NG、是/否
   - 頁碼、頁首頁尾、重複頁面標頭
   - 純數字或只有編號的行
5. 如果連續兩行分別是主題與子主題，可以分別回傳 level 1 與 level 2。
6. 判斷依據是文件語意結構與上下文，不要依賴特定領域關鍵字。
7. 不確定時寧可少切，不要亂切。
8. 只輸出 JSON array，不要輸出任何解釋或 Markdown code fence。

輸出格式：
[
  {"start_line": 9, "level": 1, "title": "原始標題文字"},
  {"start_line": 10, "level": 2, "title": "原始子標題文字"}
]

如果沒有可靠的 section boundary，輸出 []。
""".strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {exc}") from exc
            if isinstance(value, dict):
                yield value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def make_numbered_page(text: str) -> tuple[list[str], str]:
    """Return original lines and a line-numbered view for the detector."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    numbered: list[str] = []
    for idx, line in enumerate(lines, start=1):
        if line.strip():
            numbered.append(f"{idx}| {line.rstrip()}")
    return lines, "\n".join(numbered)


def repeated_short_lines(page_texts: list[str], min_ratio: float = 0.50) -> set[str]:
    """Find likely recurring page headers/footers without domain keywords."""
    if len(page_texts) < 2:
        return set()

    counter: Counter[str] = Counter()
    for text in page_texts:
        seen: set[str] = set()
        for raw in text.splitlines():
            line = normalize_for_compare(raw)
            if not line or len(line) > 80:
                continue
            if re.fullmatch(r"\d+", line):
                continue
            seen.add(line)
        counter.update(seen)

    threshold = max(2, int(len(page_texts) * min_ratio + 0.999))
    return {line for line, count in counter.items() if count >= threshold}


def build_detector_user_prompt(
    numbered_text: str,
    recurring_lines: set[str],
) -> str:
    recurring = "\n".join(f"- {line}" for line in sorted(recurring_lines))
    if not recurring:
        recurring = "（無）"

    return (
        "以下是 OCR 頁面，每行格式為『原始行號| 文字』。\n"
        "請只找真正的 section / subsection 起始行。\n\n"
        "以下文字在多頁重複出現，通常是頁首頁尾或重複文件標頭，請不要把它們當作新的 section boundary：\n"
        f"{recurring}\n\n"
        "OCR PAGE:\n"
        f"{numbered_text}\n"
    )


def extract_json_array(text: str) -> list[Any]:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        value = json.loads(text)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []
    return []


def validate_headings(raw: list[Any], lines: list[str]) -> list[dict[str, Any]]:
    """Validate model output against exact original lines; never trust rewritten titles."""
    output: list[dict[str, Any]] = []
    seen_lines: set[int] = set()

    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            line_no = int(item.get("start_line"))
            level = int(item.get("level"))
        except (TypeError, ValueError):
            continue

        if not (1 <= line_no <= len(lines)):
            continue
        if line_no in seen_lines:
            continue
        if level not in (1, 2, 3):
            level = max(1, min(3, level))

        actual = lines[line_no - 1].strip()
        if not actual:
            continue

        # Reject obvious non-heading forms independently of domain.
        if re.fullmatch(r"\d+", actual):
            continue
        if re.match(r"^\s*\d+[.)、]\s+", actual):
            continue
        if actual.lower() in {"ok", "ng", "ok或ng", "是或否", "是", "否"}:
            continue
        if actual.startswith("<table"):
            continue

        seen_lines.add(line_no)
        output.append(
            {
                "start_line": line_no,
                "level": level,
                # Always store exact OCR line rather than model's rewritten title.
                "title": actual,
            }
        )

    output.sort(key=lambda x: x["start_line"])
    return output


class QwenSectionDetector:
    def __init__(
        self,
        model_path: Path,
        n_ctx: int,
        n_batch: int,
        n_gpu_layers: int,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        print(f"[SectionDetector] Loading Qwen: {model_path}", flush=True)
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print("[SectionDetector] Qwen loaded", flush=True)

    def detect(
        self,
        page_text: str,
        recurring_lines: set[str],
        max_tokens: int = 600,
    ) -> list[dict[str, Any]]:
        lines, numbered = make_numbered_page(page_text)
        if not numbered.strip():
            return []

        user_prompt = build_detector_user_prompt(numbered, recurring_lines)

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
        )
        content = response["choices"][0]["message"]["content"]
        raw = extract_json_array(content)
        headings = validate_headings(raw, lines)

        # One conservative retry if JSON parsing failed completely while model emitted text.
        if not headings and str(content or "").strip() not in {"", "[]"}:
            retry = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {
                        "role": "assistant",
                        "content": str(content),
                    },
                    {
                        "role": "user",
                        "content": "上一個輸出不是可解析的 JSON array。請重新輸出，只有 JSON array，不要解釋。",
                    },
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
            retry_content = retry["choices"][0]["message"]["content"]
            headings = validate_headings(
                extract_json_array(retry_content),
                lines,
            )

        return headings


def cache_path_for_page(page_dir: Path) -> Path:
    return page_dir / "semantic_sections_qwen.json"


def load_cached_headings(
    page_dir: Path,
    page_text: str,
    model_path: Path,
) -> list[dict[str, Any]] | None:
    path = cache_path_for_page(page_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    expected = {
        "detector_version": DETECTOR_VERSION,
        "page_sha256": sha256_text(page_text),
        "model_path": str(model_path.resolve()),
    }
    for key, value in expected.items():
        if data.get(key) != value:
            return None

    headings = data.get("headings")
    return headings if isinstance(headings, list) else None


def save_cached_headings(
    page_dir: Path,
    page_text: str,
    model_path: Path,
    headings: list[dict[str, Any]],
) -> None:
    data = {
        "detector_version": DETECTOR_VERSION,
        "page_sha256": sha256_text(page_text),
        "model_path": str(model_path.resolve()),
        "headings": headings,
    }
    atomic_write_text(
        cache_path_for_page(page_dir),
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    )


def segment_page(
    page_text: str,
    headings: list[dict[str, Any]],
    carry_path: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Slice ORIGINAL OCR lines using validated section boundaries."""
    lines = page_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    by_line = {int(h["start_line"]): h for h in headings}

    # The incoming path helps page-start continuation text retain previous topic.
    stack = list(carry_path[:3])
    current_path = list(stack)
    body: list[str] = []
    segments: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal body
        text = "\n".join(body).strip()
        if text:
            segments.append(
                {
                    "section_path": list(current_path),
                    "body": text,
                }
            )
        body = []

    for line_no, raw in enumerate(lines, start=1):
        heading = by_line.get(line_no)
        if heading is None:
            body.append(raw)
            continue

        flush()
        level = int(heading["level"])
        title = str(heading["title"]).strip()

        # Resize stack to the requested hierarchy level.
        if level == 1:
            stack = [title]
        else:
            if len(stack) >= level:
                stack = stack[: level - 1]
            while len(stack) < level - 1:
                # If the model jumps a level, do not invent a missing title.
                break
            stack = stack[: level - 1] + [title]

        current_path = list(stack)

    flush()
    return segments, list(stack)


def atomic_blocks(text: str) -> list[str]:
    """Create small blocks while preserving HTML tables and original lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines


def split_section_into_chunks(
    section_path: list[str],
    body: str,
    target_size: int,
    overlap_chars: int,
) -> list[str]:
    """
    Retrieval-sized chunking inside one semantic section.

    - Never crosses semantic section boundaries.
    - Keeps a complete OCR line / HTML table atomic whenever possible.
    - Repeats the section path in every chunk.
    """
    blocks = atomic_blocks(body)
    if not blocks:
        return []

    prefix = ""
    if section_path:
        prefix = "主題：" + " > ".join(section_path)

    chunks: list[str] = []
    current: list[str] = []

    def render(blocks_: list[str]) -> str:
        payload = "\n".join(blocks_).strip()
        return f"{prefix}\n{payload}".strip() if prefix else payload

    def overlap_tail(blocks_: list[str]) -> list[str]:
        if overlap_chars <= 0:
            return []
        selected: list[str] = []
        total = 0
        for block in reversed(blocks_):
            selected.append(block)
            total += len(block)
            if total >= overlap_chars:
                break
        return list(reversed(selected))

    for block in blocks:
        # Preserve tables/long atomic lines even if they exceed the target.
        if not current:
            current = [block]
            continue

        candidate = render(current + [block])
        if len(candidate) <= target_size:
            current.append(block)
            continue

        chunks.append(render(current))
        current = overlap_tail(current)

        # Avoid a pathological loop if overlap alone is already near the limit.
        while current and len(render(current + [block])) > target_size:
            if len(current) == 1:
                current = []
                break
            current = current[1:]
        current.append(block)

    if current:
        chunks.append(render(current))

    # Exact de-duplication only.
    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk and chunk not in seen:
            seen.add(chunk)
            out.append(chunk)
    return out


def read_document_pages(record: dict[str, Any]) -> list[tuple[int, Path, str]]:
    cache_dir = Path(str(record["cache_dir"]))
    page_count = int(record.get("page_count") or 0)
    pages: list[tuple[int, Path, str]] = []
    for page_no in range(1, page_count + 1):
        page_dir = cache_dir / "pages" / f"page_{page_no:04d}"
        clean = page_dir / "clean.md"
        if not clean.is_file():
            continue
        text = clean.read_text(encoding="utf-8", errors="replace").strip()
        pages.append((page_no, page_dir, text))
    return pages


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Use local Qwen to detect semantic section boundaries in existing Unlimited-OCR clean.md and rebuild small RAG chunks."
    )
    p.add_argument("--documents-file", type=Path, default=DEFAULT_DOCUMENTS_FILE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    p.add_argument("--n-ctx", type=int, default=DEFAULT_N_CTX)
    p.add_argument("--n-batch", type=int, default=DEFAULT_N_BATCH)
    p.add_argument("--n-gpu-layers", type=int, default=DEFAULT_N_GPU_LAYERS)
    p.add_argument("--force-analyze", action="store_true", help="Ignore semantic section cache and rerun Qwen boundary detection.")
    p.add_argument("--max-detector-tokens", type=int, default=600)
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.chunk_size < 200:
        raise SystemExit("--chunk-size 建議至少 200")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        raise SystemExit("--chunk-overlap 必須 >= 0 且小於 --chunk-size")
    if not args.documents_file.is_file():
        raise SystemExit(f"找不到 documents.jsonl: {args.documents_file}")

    docs = list(jsonl_records(args.documents_file))
    docs = [d for d in docs if d.get("cache_dir") and d.get("document_id")]
    if not docs:
        raise SystemExit("documents.jsonl 沒有可處理的文件")

    # Determine whether a model load is needed at all; cached analyses can be reused.
    detector: QwenSectionDetector | None = None
    all_records: list[dict[str, Any]] = []

    for doc_idx, doc in enumerate(docs, start=1):
        source_file = str(doc.get("source_file") or "未知文件")
        document_id = str(doc["document_id"])
        source_path = str(doc.get("source_path") or "")
        cache_dir = Path(str(doc["cache_dir"]))
        pages = read_document_pages(doc)
        if not pages:
            print(f"[WARN] {source_file}: 沒有 clean.md，略過", flush=True)
            continue

        recurring = repeated_short_lines([text for _, _, text in pages])
        print(
            f"\n[{doc_idx}/{len(docs)}] {source_file}: pages={len(pages)}, recurring_lines={len(recurring)}",
            flush=True,
        )

        carry_path: list[str] = []
        doc_records: list[dict[str, Any]] = []
        chunk_counter = 0

        for page_no, page_dir, page_text in pages:
            headings = None
            if not args.force_analyze:
                headings = load_cached_headings(page_dir, page_text, args.model)

            if headings is None:
                if detector is None:
                    detector = QwenSectionDetector(
                        model_path=args.model,
                        n_ctx=args.n_ctx,
                        n_batch=args.n_batch,
                        n_gpu_layers=args.n_gpu_layers,
                    )
                headings = detector.detect(
                    page_text=page_text,
                    recurring_lines=recurring,
                    max_tokens=args.max_detector_tokens,
                )
                save_cached_headings(page_dir, page_text, args.model, headings)
                cache_tag = "Qwen"
            else:
                cache_tag = "cache"

            print(
                f"  page {page_no:04d}: headings={len(headings)} ({cache_tag})",
                flush=True,
            )
            for h in headings:
                print(
                    f"    L{h['level']} line {h['start_line']}: {h['title']}",
                    flush=True,
                )

            segments, carry_path = segment_page(
                page_text=page_text,
                headings=headings,
                carry_path=carry_path,
            )

            page_chunk_index = 0
            for segment_index, segment in enumerate(segments, start=1):
                section_path = list(segment.get("section_path") or [])
                body = str(segment.get("body") or "").strip()
                chunks = split_section_into_chunks(
                    section_path=section_path,
                    body=body,
                    target_size=args.chunk_size,
                    overlap_chars=args.chunk_overlap,
                )

                for sub_index, content in enumerate(chunks, start=1):
                    page_chunk_index += 1
                    chunk_counter += 1
                    chunk_id = (
                        f"{document_id[:16]}-p{page_no:04d}-s{segment_index:03d}-c{sub_index:03d}"
                    )
                    record = {
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "source_file": source_file,
                        "source_path": source_path,
                        "page": page_no,
                        "page_start": page_no,
                        "page_end": page_no,
                        "chunk_index": chunk_counter,
                        "page_chunk_index": page_chunk_index,
                        "content": content,
                        "content_sha256": sha256_text(content),
                        "metadata": {
                            "title": Path(source_file).stem,
                            "section_path": section_path,
                            "section_title": section_path[-1] if section_path else "",
                            "semantic_section_index": segment_index,
                            "semantic_chunk_sub_index": sub_index,
                            "chunking_strategy": "qwen_section_boundary_then_small_chunks",
                            "detector_model": str(args.model),
                            "detector_version": DETECTOR_VERSION,
                            "target_chunk_chars": args.chunk_size,
                            "chunk_overlap_chars": args.chunk_overlap,
                        },
                    }
                    doc_records.append(record)
                    all_records.append(record)

        per_doc = cache_dir / "semantic_chunks_qwen.jsonl"
        atomic_write_text(
            per_doc,
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in doc_records),
        )
        print(
            f"  -> semantic chunks={len(doc_records)} | {per_doc}",
            flush=True,
        )

    atomic_write_text(
        args.output,
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_records),
    )

    print("\n" + "=" * 78)
    print(f"DONE: semantic chunks={len(all_records)}")
    print(f"Output: {args.output}")
    print("Original all_chunks.jsonl was NOT modified.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
