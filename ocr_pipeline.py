#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unlimited-OCR 正式批次管線（Jetson / Transformers）

預設資料夾：
  輸入 PDF：./input
  快取與 RAG：./flash

主要功能：
1. 自動掃描 input 裡所有 PDF，PDF 有幾頁就逐頁辨識幾頁。
2. 使用 PDF SHA-256 + OCR 設定作為快取鍵；相同內容不重跑。
3. 每頁獨立快取，可在中斷後續跑。
4. 產生 RAG 友善的 document.md、document.txt、chunks.jsonl。
5. 產生目前 input 目錄的總索引 flash/rag/all_chunks.jsonl。
6. 支援 --watch 持續監看 input。
"""

from __future__ import annotations

import os

# 完全離線：只使用已下載的 Unlimited-OCR 快取。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import fcntl
import gc
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterable

import fitz
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = DEFAULT_ROOT / "input"
DEFAULT_FLASH_DIR = DEFAULT_ROOT / "flash"

DEFAULT_MODEL = "baidu/Unlimited-OCR"
DEFAULT_REVISION = "main"

# 單頁辨識使用官方 gundam 設定，較適合 16 GB Jetson。
DEFAULT_DPI = 200
DEFAULT_MAX_LENGTH = 8192
DEFAULT_BASE_SIZE = 1024
DEFAULT_IMAGE_SIZE = 640
DEFAULT_CROP_MODE = True
DEFAULT_NO_REPEAT_NGRAM_SIZE = 35
DEFAULT_NGRAM_WINDOW = 128
DEFAULT_PROMPT = "<image>document parsing."

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 180

OCR_CACHE_FORMAT_VERSION = 1
RAG_FORMAT_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(data: Any) -> str:
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def setup_logging(flash_dir: Path, verbose: bool) -> logging.Logger:
    log_dir = flash_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("unlimited_ocr_pipeline")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / "ocr_pipeline.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def normalize_ocr_markdown(text: str) -> str:
    """只做安全清理，不擅自修改 OCR 內容或表格結構。"""
    text = text.replace("\ufeff", "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 移除模型可能殘留的控制符號。
    control_tokens = (
        "<｜end▁of▁sentence｜>",
        "<|end_of_sentence|>",
        "<|Assistant|>",
        "<|User|>",
        "<PAGE>",
    )
    for token in control_tokens:
        text = text.replace(token, "")

    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n[ \t]+\n", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def markdown_for_rag(text: str) -> str:
    """保留標題、清單與表格；把本機圖片連結轉成文字標記。"""
    text = normalize_ocr_markdown(text)

    def image_replacement(match: re.Match[str]) -> str:
        alt = match.group(1).strip()
        return f"[圖片：{alt}]" if alt else "[圖片]"

    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", image_replacement, text)
    text = re.sub(r"<[^>\n]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def markdown_to_plain_text(text: str) -> str:
    text = markdown_for_rag(text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.*?)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!_)_(?!_)(.*?)_(?!_)", r"\1", text)

    # Markdown 表格分隔線沒有語意，純文字版移除。
    text = re.sub(
        r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def rewrite_page_image_links(text: str, page_number: int) -> str:
    prefix = f"pages/page_{page_number:04d}/"

    def replace(match: re.Match[str]) -> str:
        alt = match.group(1)
        target = match.group(2)
        return f"![{alt}]({prefix}{target})"

    return re.sub(
        r"!\[([^\]]*)\]\((images/[^)]+)\)",
        replace,
        text,
    )


def split_long_text(text: str, size: int, overlap: int) -> list[str]:
    """對單一超長區塊做有重疊的切割，優先在換行或標點處切。"""
    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        tentative_end = min(start + size, text_length)
        end = tentative_end

        if tentative_end < text_length:
            search_start = start + int(size * 0.60)
            boundaries = [
                text.rfind("\n", search_start, tentative_end),
                text.rfind("。", search_start, tentative_end),
                text.rfind("；", search_start, tentative_end),
                text.rfind(".", search_start, tentative_end),
                text.rfind(" ", search_start, tentative_end),
            ]
            best = max(boundaries)
            if best > start:
                end = best + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_length:
            break

        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def chunk_markdown(text: str, size: int, overlap: int) -> list[str]:
    """
    先依 Markdown 段落組塊；超長段落再切割。
    表格與清單在未超過 size 時會盡量保持完整。
    """
    text = markdown_for_rag(text)
    if not text:
        return []

    blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    output: list[str] = []
    current = ""

    for block in blocks:
        if len(block) > size:
            if current:
                output.append(current.strip())
                current = ""
            output.extend(split_long_text(block, size, overlap))
            continue

        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) <= size:
            current = candidate
            continue

        if current:
            output.append(current.strip())

        # 從上一塊保留少量尾端上下文。
        if output and overlap > 0:
            tail = output[-1][-overlap:].strip()
            current = f"{tail}\n\n{block}" if tail else block
            if len(current) > size:
                output.extend(split_long_text(current, size, overlap))
                current = ""
        else:
            current = block

    if current.strip():
        output.append(current.strip())

    # 去除意外產生的完全重複塊。
    deduplicated: list[str] = []
    for chunk in output:
        if not deduplicated or chunk != deduplicated[-1]:
            deduplicated.append(chunk)
    return deduplicated


def list_pdf_files(input_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: str(path).lower(),
    )


class OCRPipeline:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.input_dir = args.input_dir.resolve()
        self.flash_dir = args.flash_dir.resolve()
        self.cache_root = self.flash_dir / "cache"
        self.rag_root = self.flash_dir / "rag"

        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.rag_root.mkdir(parents=True, exist_ok=True)

        self.ocr_config = {
            "cache_format_version": OCR_CACHE_FORMAT_VERSION,
            "model": args.model,
            "revision": args.revision,
            "prompt": args.prompt,
            "dpi": args.dpi,
            "max_length": args.max_length,
            "base_size": DEFAULT_BASE_SIZE,
            "image_size": DEFAULT_IMAGE_SIZE,
            "crop_mode": DEFAULT_CROP_MODE,
            "no_repeat_ngram_size": DEFAULT_NO_REPEAT_NGRAM_SIZE,
            "ngram_window": DEFAULT_NGRAM_WINDOW,
            "torch_dtype": "bfloat16",
            "inference_mode": "single_page_gundam",
        }
        self.ocr_config_hash = sha256_json(self.ocr_config)

        self.tokenizer = None
        self.model = None

    def cache_dir_for(self, pdf_sha256: str) -> Path:
        return self.cache_root / pdf_sha256 / self.ocr_config_hash

    def manifest_path_for(self, pdf_sha256: str) -> Path:
        return self.cache_dir_for(pdf_sha256) / "manifest.json"

    def page_dir_for(self, cache_dir: Path, page_number: int) -> Path:
        return cache_dir / "pages" / f"page_{page_number:04d}"

    def load_model_if_needed(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 無法使用，Unlimited-OCR 正式管線需要 NVIDIA GPU。")

        self.logger.info("開始載入 tokenizer：%s", self.args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model,
            revision=self.args.revision,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.logger.info("開始載入模型：%s", self.args.model)
        self.model = AutoModel.from_pretrained(
            self.args.model,
            revision=self.args.revision,
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.eval().cuda()

        self.logger.info(
            "模型載入完成：GPU=%s，allocated=%.2f GB，reserved=%.2f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
        )

    @staticmethod
    def _native_text_is_usable(
        text: str,
        word_count: int,
    ) -> bool:
        text = str(text or "").strip()
        if word_count < 8 or len(text) < 45:
            return False

        meaningful = re.findall(
            r"[A-Za-z0-9\u3400-\u9fff]",
            text,
        )
        if len(meaningful) < 30:
            return False

        replacement_ratio = text.count("\ufffd") / max(1, len(text))
        return replacement_ratio <= 0.01

    def extract_native_page_text(
        self,
        pdf_path: Path,
        page_number: int,
    ) -> str | None:
        """Extract native text while preserving multi-column reading order.

        A row-first reconstruction interleaves the left and right columns of
        service manuals.  PyMuPDF blocks retain the layout well enough to read
        the complete left column before the right column.  Full-width headers
        and footers remain before/after the columns.
        """
        with fitz.open(pdf_path) as document:
            page = document.load_page(page_number - 1)
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            blocks = page.get_text("blocks", sort=False)

        normalized_blocks: list[tuple[float, float, float, float, str]] = []
        word_count = 0
        for block in blocks:
            x0, y0, x1, y1, raw_text = block[:5]
            text = normalize_ocr_markdown(str(raw_text or ""))
            if not text:
                continue
            normalized_blocks.append(
                (float(x0), float(y0), float(x1), float(y1), text)
            )
            word_count += len(re.findall(r"\S+", text))

        if not normalized_blocks:
            return None

        midpoint = page_width / 2.0
        gutter = max(12.0, page_width * 0.035)
        header_limit = page_height * 0.10
        footer_limit = page_height * 0.92
        header: list[tuple[float, float, float, float, str]] = []
        left: list[tuple[float, float, float, float, str]] = []
        right: list[tuple[float, float, float, float, str]] = []
        footer: list[tuple[float, float, float, float, str]] = []

        for block in normalized_blocks:
            x0, y0, x1, y1, _ = block
            if y1 <= header_limit or x0 < midpoint - gutter < midpoint + gutter < x1:
                header.append(block)
            elif y0 >= footer_limit:
                footer.append(block)
            elif (x0 + x1) / 2.0 < midpoint:
                left.append(block)
            else:
                right.append(block)

        # Only use column ordering when both columns contain meaningful text.
        if left and right:
            ordered_groups = (header, left, right, footer)
            ordered = [
                block
                for group in ordered_groups
                for block in sorted(group, key=lambda item: (item[1], item[0]))
            ]
        else:
            ordered = sorted(normalized_blocks, key=lambda item: (item[1], item[0]))

        # Blank lines preserve block boundaries for the semantic chunker.
        text = "\n\n".join(block[4] for block in ordered).strip()
        if not self._native_text_is_usable(
            text,
            word_count,
        ):
            return None

        return text

    def inspect_pdf(self, pdf_path: Path) -> dict[str, Any]:
        with fitz.open(pdf_path) as document:
            if document.needs_pass:
                raise RuntimeError("PDF 已加密，需要密碼，無法自動辨識。")

            metadata = document.metadata or {}
            return {
                "page_count": document.page_count,
                "pdf_metadata": {
                    key: value
                    for key, value in metadata.items()
                    if value not in (None, "")
                },
            }

    def create_or_update_manifest(
        self,
        pdf_path: Path,
        pdf_sha256: str,
        file_size: int,
        page_count: int,
        pdf_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        manifest_path = self.manifest_path_for(pdf_sha256)
        cache_dir = manifest_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)

        manifest = read_json(manifest_path, default=None)
        if not isinstance(manifest, dict):
            manifest = {
                "manifest_version": 1,
                "document_id": pdf_sha256,
                "pdf_sha256": pdf_sha256,
                "file_size": file_size,
                "page_count": page_count,
                "pdf_metadata": pdf_metadata,
                "ocr_config": self.ocr_config,
                "ocr_config_hash": self.ocr_config_hash,
                "source_files": [],
                "status": "pending",
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "completed_at": None,
                "pages": {},
            }

        manifest["file_size"] = file_size
        manifest["page_count"] = page_count
        manifest["pdf_metadata"] = pdf_metadata
        manifest["ocr_config"] = self.ocr_config
        manifest["ocr_config_hash"] = self.ocr_config_hash

        source_path = str(pdf_path.resolve())
        source_files = manifest.setdefault("source_files", [])
        if source_path not in source_files:
            source_files.append(source_path)
            source_files.sort()

        pages = manifest.setdefault("pages", {})
        for page_number in range(1, page_count + 1):
            pages.setdefault(
                str(page_number),
                {
                    "status": "pending",
                    "attempts": 0,
                    "last_error": None,
                    "updated_at": None,
                },
            )

        # PDF 頁數若縮短，移除 manifest 中超出的頁碼。
        for page_key in list(pages):
            try:
                if int(page_key) > page_count:
                    del pages[page_key]
            except ValueError:
                del pages[page_key]

        manifest["updated_at"] = utc_now()
        atomic_write_json(manifest_path, manifest)
        return manifest

    def page_is_cached(
        self,
        manifest: dict[str, Any],
        cache_dir: Path,
        page_number: int,
    ) -> bool:
        page_info = manifest.get("pages", {}).get(str(page_number), {})
        page_dir = self.page_dir_for(cache_dir, page_number)
        return (
            page_info.get("status") == "done"
            and (page_dir / "result.md").is_file()
            and (page_dir / "clean.md").is_file()
        )

    def render_page(self, pdf_path: Path, page_number: int, output_image: Path) -> None:
        with fitz.open(pdf_path) as document:
            page = document.load_page(page_number - 1)
            matrix = fitz.Matrix(self.args.dpi / 72.0, self.args.dpi / 72.0)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(output_image)

    def run_page_ocr(
        self,
        pdf_path: Path,
        cache_dir: Path,
        manifest: dict[str, Any],
        page_number: int,
    ) -> bool:
        page_key = str(page_number)
        page_info = manifest["pages"][page_key]
        page_info["status"] = "processing"
        page_info["attempts"] = int(page_info.get("attempts", 0)) + 1
        page_info["last_error"] = None
        page_info["updated_at"] = utc_now()
        manifest["status"] = "processing"
        manifest["updated_at"] = utc_now()
        atomic_write_json(cache_dir / "manifest.json", manifest)

        page_dir = self.page_dir_for(cache_dir, page_number)
        if page_dir.exists():
            shutil.rmtree(page_dir)
        page_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "[%s] 開始辨識第 %d/%d 頁",
            pdf_path.name,
            page_number,
            manifest["page_count"],
        )

        try:
            native_text = self.extract_native_page_text(
                pdf_path,
                page_number,
            )
            if native_text is not None:
                atomic_write_text(
                    page_dir / "result.md",
                    native_text + "\n",
                )
                atomic_write_text(
                    page_dir / "clean.md",
                    native_text + "\n",
                )

                page_info.update(
                    {
                        "status": "done",
                        "characters": len(native_text),
                        "extraction_method": "pdf_text_layer",
                        "last_error": None,
                        "updated_at": utc_now(),
                    }
                )
                manifest["updated_at"] = utc_now()
                atomic_write_json(
                    cache_dir / "manifest.json",
                    manifest,
                )
                self.logger.info(
                    "[%s] 第 %d 頁使用 PDF 原生文字層，"
                    "不載入 OCR 模型，文字數=%d",
                    pdf_path.name,
                    page_number,
                    len(native_text),
                )
                return True

            self.load_model_if_needed()

            with tempfile.TemporaryDirectory(prefix="unlimited_ocr_page_") as temp_dir:
                page_image = Path(temp_dir) / f"page_{page_number:04d}.png"
                self.render_page(pdf_path, page_number, page_image)

                with torch.inference_mode():
                    self.model.infer(
                        self.tokenizer,
                        prompt=self.args.prompt,
                        image_file=str(page_image),
                        output_path=str(page_dir),
                        base_size=DEFAULT_BASE_SIZE,
                        image_size=DEFAULT_IMAGE_SIZE,
                        crop_mode=DEFAULT_CROP_MODE,
                        max_length=self.args.max_length,
                        no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM_SIZE,
                        ngram_window=DEFAULT_NGRAM_WINDOW,
                        save_results=True,
                    )

            raw_result_path = page_dir / "result.md"
            if not raw_result_path.is_file():
                raise RuntimeError(
                    f"模型沒有產生預期結果檔：{raw_result_path}"
                )

            raw_text = raw_result_path.read_text(encoding="utf-8", errors="replace")
            clean_text = normalize_ocr_markdown(raw_text)
            atomic_write_text(page_dir / "clean.md", clean_text + "\n")

            page_info.update(
                {
                    "status": "done",
                    "characters": len(clean_text),
                    "last_error": None,
                    "updated_at": utc_now(),
                }
            )
            manifest["updated_at"] = utc_now()
            atomic_write_json(cache_dir / "manifest.json", manifest)

            self.logger.info(
                "[%s] 第 %d 頁完成，文字數=%d",
                pdf_path.name,
                page_number,
                len(clean_text),
            )
            return True

        except torch.OutOfMemoryError as error:
            error_text = f"CUDA OOM: {error}"
            self.logger.error(
                "[%s] 第 %d 頁記憶體不足。可降低 --max-length 或 --dpi。",
                pdf_path.name,
                page_number,
            )
            page_info.update(
                {
                    "status": "error",
                    "last_error": error_text,
                    "updated_at": utc_now(),
                }
            )
            manifest["status"] = "partial"
            manifest["updated_at"] = utc_now()
            atomic_write_json(cache_dir / "manifest.json", manifest)
            torch.cuda.empty_cache()
            return False

        except Exception as error:
            error_text = "".join(
                traceback.format_exception_only(type(error), error)
            ).strip()
            self.logger.exception(
                "[%s] 第 %d 頁辨識失敗：%s",
                pdf_path.name,
                page_number,
                error_text,
            )
            page_info.update(
                {
                    "status": "error",
                    "last_error": error_text,
                    "updated_at": utc_now(),
                }
            )
            manifest["status"] = "partial"
            manifest["updated_at"] = utc_now()
            atomic_write_json(cache_dir / "manifest.json", manifest)
            return False

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def assemble_rag_files(
        self,
        pdf_path: Path,
        cache_dir: Path,
        manifest: dict[str, Any],
    ) -> None:
        page_count = int(manifest["page_count"])
        page_markdown: list[tuple[int, str]] = []

        for page_number in range(1, page_count + 1):
            clean_path = self.page_dir_for(cache_dir, page_number) / "clean.md"
            if not clean_path.is_file():
                raise RuntimeError(f"缺少第 {page_number} 頁快取：{clean_path}")
            clean_text = clean_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()

            # 舊 OCR 快取也會在組裝 RAG 時改用可靠的 PDF 文字層，
            # 因此不需要 --force 重新跑模型。
            native_text = self.extract_native_page_text(
                pdf_path,
                page_number,
            )
            if native_text is not None:
                clean_text = native_text
                self.logger.info(
                    "[%s] 第 %d 頁 RAG 使用 PDF 原生文字層",
                    pdf_path.name,
                    page_number,
                )

            page_markdown.append((page_number, clean_text))

        title = (
            manifest.get("pdf_metadata", {}).get("title")
            or pdf_path.stem
        )
        front_matter = [
            "---",
            f'document_id: "{manifest["document_id"]}"',
            f'source_file: "{pdf_path.name}"',
            f'page_count: {page_count}',
            f'ocr_model: "{self.args.model}"',
            f'ocr_config_hash: "{self.ocr_config_hash}"',
            f'generated_at: "{utc_now()}"',
            "---",
            "",
            f"# {title}",
            "",
        ]

        document_md_parts = ["\n".join(front_matter)]
        document_txt_parts = [str(title)]

        chunk_records: list[dict[str, Any]] = []
        chunk_counter = 0

        for page_number, clean_text in page_markdown:
            document_text = rewrite_page_image_links(clean_text, page_number)
            document_md_parts.append(
                f"## 第 {page_number} 頁\n\n{document_text}\n"
            )

            plain_text = markdown_to_plain_text(clean_text)
            document_txt_parts.append(
                f"\n===== 第 {page_number} 頁 =====\n\n{plain_text}\n"
            )

            rag_text = markdown_for_rag(clean_text)
            for page_chunk_index, content in enumerate(
                chunk_markdown(
                    rag_text,
                    size=self.args.chunk_size,
                    overlap=self.args.chunk_overlap,
                ),
                start=1,
            ):
                chunk_counter += 1
                chunk_id = (
                    f'{manifest["document_id"][:16]}'
                    f"-p{page_number:04d}"
                    f"-c{page_chunk_index:04d}"
                )
                chunk_records.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": manifest["document_id"],
                        "source_file": pdf_path.name,
                        "source_path": str(pdf_path.resolve()),
                        "page": page_number,
                        "page_start": page_number,
                        "page_end": page_number,
                        "chunk_index": chunk_counter,
                        "page_chunk_index": page_chunk_index,
                        "content": content,
                        "content_sha256": hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest(),
                        "metadata": {
                            "title": title,
                            "pdf_sha256": manifest["pdf_sha256"],
                            "page_count": page_count,
                            "ocr_model": self.args.model,
                            "ocr_config_hash": self.ocr_config_hash,
                            "rag_format_version": RAG_FORMAT_VERSION,
                        },
                    }
                )

        document_md = "\n".join(document_md_parts).strip() + "\n"
        document_txt = "\n".join(document_txt_parts).strip() + "\n"
        chunks_jsonl = "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in chunk_records
        )

        atomic_write_text(cache_dir / "document.md", document_md)
        atomic_write_text(cache_dir / "document.txt", document_txt)
        atomic_write_text(cache_dir / "chunks.jsonl", chunks_jsonl)

        manifest["rag"] = {
            "format_version": RAG_FORMAT_VERSION,
            "chunk_size": self.args.chunk_size,
            "chunk_overlap": self.args.chunk_overlap,
            "chunk_count": len(chunk_records),
            "document_markdown": "document.md",
            "document_text": "document.txt",
            "chunks_jsonl": "chunks.jsonl",
            "updated_at": utc_now(),
        }
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        atomic_write_json(cache_dir / "manifest.json", manifest)

    def process_pdf(self, pdf_path: Path) -> dict[str, Any]:
        try:
            stat = pdf_path.stat()
            pdf_sha256 = sha256_file(pdf_path)
            pdf_info = self.inspect_pdf(pdf_path)
            page_count = int(pdf_info["page_count"])

            if page_count <= 0:
                raise RuntimeError("PDF 頁數為 0。")

            cache_dir = self.cache_dir_for(pdf_sha256)
            manifest = self.create_or_update_manifest(
                pdf_path=pdf_path,
                pdf_sha256=pdf_sha256,
                file_size=stat.st_size,
                page_count=page_count,
                pdf_metadata=pdf_info["pdf_metadata"],
            )

            if self.args.force and cache_dir.exists():
                self.logger.warning("[%s] --force：清除目前 OCR 快取", pdf_path.name)
                shutil.rmtree(cache_dir)
                manifest = self.create_or_update_manifest(
                    pdf_path=pdf_path,
                    pdf_sha256=pdf_sha256,
                    file_size=stat.st_size,
                    page_count=page_count,
                    pdf_metadata=pdf_info["pdf_metadata"],
                )

            missing_pages = [
                page_number
                for page_number in range(1, page_count + 1)
                if not self.page_is_cached(
                    manifest,
                    cache_dir,
                    page_number,
                )
            ]

            if not missing_pages:
                self.logger.info(
                    "[快取命中] %s，共 %d 頁，不重新 OCR",
                    pdf_path.name,
                    page_count,
                )
                # 即使 OCR 快取命中，也重新整理 RAG，讓 chunk 參數可獨立變更。
                self.assemble_rag_files(
                    pdf_path=pdf_path,
                    cache_dir=cache_dir,
                    manifest=manifest,
                )
            else:
                self.logger.info(
                    "[需要辨識] %s，共 %d 頁，待處理頁面=%s",
                    pdf_path.name,
                    page_count,
                    missing_pages,
                )

                for page_number in missing_pages:
                    self.run_page_ocr(
                        pdf_path=pdf_path,
                        cache_dir=cache_dir,
                        manifest=manifest,
                        page_number=page_number,
                    )

                all_done = all(
                    self.page_is_cached(manifest, cache_dir, page_number)
                    for page_number in range(1, page_count + 1)
                )

                if all_done:
                    self.assemble_rag_files(
                        pdf_path=pdf_path,
                        cache_dir=cache_dir,
                        manifest=manifest,
                    )
                    self.logger.info(
                        "[完成] %s，共 %d 頁",
                        pdf_path.name,
                        page_count,
                    )
                else:
                    manifest["status"] = "partial"
                    manifest["updated_at"] = utc_now()
                    atomic_write_json(cache_dir / "manifest.json", manifest)
                    self.logger.warning(
                        "[未完整] %s 有頁面辨識失敗；下次執行會續跑。",
                        pdf_path.name,
                    )

            final_manifest = read_json(cache_dir / "manifest.json", manifest)
            return {
                "source_path": str(pdf_path.resolve()),
                "source_file": pdf_path.name,
                "document_id": pdf_sha256,
                "ocr_config_hash": self.ocr_config_hash,
                "cache_dir": str(cache_dir),
                "page_count": page_count,
                "status": final_manifest.get("status", "unknown"),
                "error": None,
            }

        except Exception as error:
            error_text = "".join(
                traceback.format_exception_only(type(error), error)
            ).strip()
            self.logger.exception(
                "[PDF 失敗] %s：%s",
                pdf_path,
                error_text,
            )
            return {
                "source_path": str(pdf_path.resolve()),
                "source_file": pdf_path.name,
                "document_id": None,
                "ocr_config_hash": self.ocr_config_hash,
                "cache_dir": None,
                "page_count": None,
                "status": "error",
                "error": error_text,
            }

    def rebuild_global_rag(self, documents: list[dict[str, Any]]) -> None:
        """
        只彙整目前 input 中的 PDF。
        舊快取保留，但從 input 移除的 PDF 不會進入目前的 all_chunks.jsonl。
        """
        catalog = {
            "generated_at": utc_now(),
            "input_dir": str(self.input_dir),
            "ocr_config_hash": self.ocr_config_hash,
            "rag_format_version": RAG_FORMAT_VERSION,
            "documents": documents,
        }

        # 同內容 PDF 只加入一次全域 RAG，避免重複檢索。
        seen_document_ids: set[str] = set()
        all_chunks_lines: list[str] = []
        all_documents_lines: list[str] = []

        for item in documents:
            document_id = item.get("document_id")
            if (
                item.get("status") != "complete"
                or not document_id
                or document_id in seen_document_ids
            ):
                continue

            seen_document_ids.add(document_id)
            cache_dir = Path(item["cache_dir"])
            chunks_path = cache_dir / "chunks.jsonl"
            document_path = cache_dir / "document.md"

            if chunks_path.is_file():
                content = chunks_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                if content and not content.endswith("\n"):
                    content += "\n"
                all_chunks_lines.append(content)

            if document_path.is_file():
                all_documents_lines.append(
                    json.dumps(
                        {
                            "document_id": document_id,
                            "source_file": item["source_file"],
                            "source_path": item["source_path"],
                            "page_count": item["page_count"],
                            "cache_dir": item["cache_dir"],
                            "document_markdown": str(document_path),
                            "chunks_jsonl": str(chunks_path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        atomic_write_json(self.rag_root / "catalog.json", catalog)
        atomic_write_text(
            self.rag_root / "all_chunks.jsonl",
            "".join(all_chunks_lines),
        )
        atomic_write_text(
            self.rag_root / "documents.jsonl",
            "".join(all_documents_lines),
        )

        complete_count = sum(
            1 for item in documents if item.get("status") == "complete"
        )
        self.logger.info(
            "RAG 索引已更新：PDF=%d，完整=%d，去重後文件=%d",
            len(documents),
            complete_count,
            len(seen_document_ids),
        )
        self.logger.info(
            "全域 chunks：%s",
            self.rag_root / "all_chunks.jsonl",
        )

    def run_cycle(self) -> list[dict[str, Any]]:
        pdf_files = list_pdf_files(self.input_dir)

        if not pdf_files:
            self.logger.info("input 內目前沒有 PDF：%s", self.input_dir)
            self.rebuild_global_rag([])
            return []

        self.logger.info("找到 %d 份 PDF", len(pdf_files))
        documents: list[dict[str, Any]] = []

        now = time.time()
        for pdf_path in pdf_files:
            # watch 模式避免 PDF 尚未複製完成就開始讀取。
            if self.args.watch and self.args.stable_seconds > 0:
                age = now - pdf_path.stat().st_mtime
                if age < self.args.stable_seconds:
                    self.logger.info(
                        "[等待檔案穩定] %s，最後修改 %.1f 秒前",
                        pdf_path.name,
                        age,
                    )
                    documents.append(
                        {
                            "source_path": str(pdf_path.resolve()),
                            "source_file": pdf_path.name,
                            "document_id": None,
                            "ocr_config_hash": self.ocr_config_hash,
                            "cache_dir": None,
                            "page_count": None,
                            "status": "waiting_for_stable_file",
                            "error": None,
                        }
                    )
                    continue

            documents.append(self.process_pdf(pdf_path))

        self.rebuild_global_rag(documents)
        return documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unlimited-OCR 自動 PDF 快取與 RAG 整理管線"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"PDF 輸入資料夾，預設：{DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--flash-dir",
        type=Path,
        default=DEFAULT_FLASH_DIR,
        help=f"快取與 RAG 資料夾，預設：{DEFAULT_FLASH_DIR}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face 模型，預設：{DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="模型 revision／commit，預設：main",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="單頁 OCR prompt",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"PDF 轉圖片 DPI，預設：{DEFAULT_DPI}",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"每一頁的模型最大序列長度，預設：{DEFAULT_MAX_LENGTH}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"RAG 每塊目標字元數，預設：{DEFAULT_CHUNK_SIZE}",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"RAG 相鄰塊重疊字元數，預設：{DEFAULT_CHUNK_OVERLAP}",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="持續監看 input 資料夾",
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=15.0,
        help="watch 模式掃描間隔秒數，預設：15",
    )
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=5.0,
        help="watch 模式等待 PDF 停止修改的秒數，預設：5",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略目前設定的 OCR 快取，強制全部重跑",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="顯示除錯紀錄",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.dpi < 72:
        raise ValueError("--dpi 不可小於 72")
    if args.max_length < 512:
        raise ValueError("--max-length 不可小於 512")
    if args.chunk_size < 200:
        raise ValueError("--chunk-size 不可小於 200")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap 不可小於 0")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap 必須小於 --chunk-size")
    if args.scan_interval < 1:
        raise ValueError("--scan-interval 不可小於 1 秒")
    if args.stable_seconds < 0:
        raise ValueError("--stable-seconds 不可小於 0")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))

    args.flash_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.flash_dir, args.verbose)

    lock_path = args.flash_dir / ".ocr_pipeline.lock"
    lock_file = lock_path.open("w", encoding="utf-8")

    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error("已有另一個 OCR 管線正在執行：%s", lock_path)
        return 2

    lock_file.write(str(os.getpid()))
    lock_file.flush()

    pipeline = OCRPipeline(args, logger)

    logger.info("input：%s", pipeline.input_dir)
    logger.info("flash：%s", pipeline.flash_dir)
    logger.info("OCR 設定雜湊：%s", pipeline.ocr_config_hash)

    try:
        if not args.watch:
            pipeline.run_cycle()
            return 0

        logger.info(
            "開始持續監看，每 %.1f 秒掃描一次；Ctrl+C 可停止。",
            args.scan_interval,
        )

        while True:
            cycle_started = time.monotonic()
            pipeline.run_cycle()
            elapsed = time.monotonic() - cycle_started
            sleep_seconds = max(1.0, args.scan_interval - elapsed)
            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        logger.info("收到停止訊號，OCR 管線結束。")
        return 0
    except Exception:
        logger.exception("OCR 管線發生未處理錯誤。")
        return 1
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
