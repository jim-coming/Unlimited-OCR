#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_rag_index.py

將 Unlimited-OCR 產生的 all_chunks.jsonl 建立成兩層 FAISS 索引：

1. chunk.index
   - 用於「設備啟動前要檢查什麼？」等細節問題。
2. document.index
   - 每份 PDF 使用所有 chunk 向量的語意中心（centroid）。
   - 若未來存在 document_summary.json，會自動將摘要向量與 centroid 合併。
   - 用於先找相關文件，或挑選代表性 chunks。

適合目前 Unlimited-OCR 正式管線的資料：
  ./flash/rag/all_chunks.jsonl
  ./flash/rag/documents.jsonl

輸出：
  ./flash/vector/
    chunk.index
    chunk_metadata.jsonl
    document.index
    document_metadata.jsonl
    index_manifest.json
    embedding_cache.sqlite3

設計重點：
- 使用 intfloat/multilingual-e5-base。
- 文件內容使用 "passage: " 前綴。
- 自動將過長 chunk 切成 tokenizer-aware 子分塊，避免超過 512 tokens。
- Embedding 寫入 SQLite 快取；相同內容不重算。
- 每次重建 FAISS 很快，但只對新增／變更內容重新做 embedding。
- FAISS 使用 IndexFlatIP + L2-normalized vectors，等同 cosine similarity。
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHUNKS_FILE = DEFAULT_ROOT / "flash/rag/all_chunks.jsonl"
DEFAULT_DOCUMENTS_FILE = DEFAULT_ROOT / "flash/rag/documents.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "flash/vector"

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_MODEL_REVISION = "main"

# multilingual-e5-base 最長 512 tokens。
# 預留 passage 前綴、文件名稱、標題、頁碼與特殊 tokens。
DEFAULT_MAX_CONTENT_TOKENS = 400
DEFAULT_TOKEN_OVERLAP = 60

# Jetson Orin 16 GB 建議先從 8 開始。
DEFAULT_BATCH_SIZE = 8
DEFAULT_REPRESENTATIVE_CHUNKS = 8

INDEX_FORMAT_VERSION = 1
CACHE_FORMAT_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("向量無法正規化：norm 為 0 或非有限值")
    return vector / norm


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    faiss.normalize_L2(matrix)
    return matrix


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


def atomic_write_faiss(path: Path, index: faiss.Index) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    faiss.write_index(index, str(temp_path))
    os.replace(temp_path, path)


def jsonl_lines(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是有效 JSON：{error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path} 第 {line_number} 行必須是 JSON object"
                )
            yield line_number, value


def atomic_write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in records
    )
    atomic_write_text(path, content)


def setup_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("build_rag_index")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(handler)
    return logger


class EmbeddingCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                cache_format_version INTEGER NOT NULL,
                model_fingerprint TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_embeddings_model
            ON embeddings(model_fingerprint)
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def load_many(
        self,
        cache_keys: Sequence[str],
        expected_dimension: int,
    ) -> dict[str, np.ndarray]:
        if not cache_keys:
            return {}

        result: dict[str, np.ndarray] = {}
        unique_keys = list(dict.fromkeys(cache_keys))

        # 保守避開 SQLite 參數數量限制。
        for start in range(0, len(unique_keys), 500):
            batch = unique_keys[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            query = (
                "SELECT cache_key, dimension, vector "
                f"FROM embeddings WHERE cache_key IN ({placeholders})"
            )
            for cache_key, dimension, vector_blob in self.connection.execute(
                query,
                batch,
            ):
                if int(dimension) != expected_dimension:
                    continue
                vector = np.frombuffer(vector_blob, dtype="<f4").copy()
                if vector.shape != (expected_dimension,):
                    continue
                if not np.all(np.isfinite(vector)):
                    continue
                result[str(cache_key)] = vector
        return result

    def save_many(
        self,
        rows: Sequence[tuple[str, str, str, np.ndarray]],
    ) -> None:
        if not rows:
            return

        now = utc_now()
        values = []
        for cache_key, model_fingerprint, text_hash, vector in rows:
            vector = np.asarray(vector, dtype="<f4")
            values.append(
                (
                    cache_key,
                    CACHE_FORMAT_VERSION,
                    model_fingerprint,
                    text_hash,
                    int(vector.shape[0]),
                    sqlite3.Binary(vector.tobytes()),
                    now,
                )
            )

        with self.connection:
            self.connection.executemany(
                """
                INSERT OR REPLACE INTO embeddings (
                    cache_key,
                    cache_format_version,
                    model_fingerprint,
                    text_sha256,
                    dimension,
                    vector,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )


def load_documents_catalog(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}

    documents: dict[str, dict[str, Any]] = {}
    for _, record in jsonl_lines(path):
        document_id = str(record.get("document_id") or "").strip()
        if document_id:
            documents[document_id] = record
    return documents


def load_optional_document_summary(
    document_record: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not document_record:
        return None

    cache_dir_raw = document_record.get("cache_dir")
    if not cache_dir_raw:
        return None

    path = Path(str(cache_dir_raw)) / "document_summary.json"
    if not path.is_file():
        return None

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return value if isinstance(value, dict) else None


def summary_to_embedding_text(
    source_file: str,
    title: str,
    summary: dict[str, Any],
) -> str | None:
    fields: list[str] = []

    summary_title = str(summary.get("title") or title).strip()
    summary_text = str(summary.get("summary") or "").strip()
    purpose = str(summary.get("purpose") or "").strip()
    intended_audience = str(summary.get("intended_audience") or "").strip()

    topics_raw = summary.get("topics")
    topics: list[str] = []
    if isinstance(topics_raw, list):
        topics = [
            str(item).strip()
            for item in topics_raw
            if str(item).strip()
        ]

    fields.append(f"passage: 文件名稱：{source_file}")
    if summary_title:
        fields.append(f"文件標題：{summary_title}")
    if summary_text:
        fields.append(f"文件摘要：{summary_text}")
    if purpose:
        fields.append(f"文件目的：{purpose}")
    if topics:
        fields.append(f"主要主題：{'、'.join(topics)}")
    if intended_audience:
        fields.append(f"適用對象：{intended_audience}")

    # 若檔案只有空殼 summary，不使用。
    if not summary_text and not topics and not purpose:
        return None
    return "\n".join(fields)


def split_text_by_token_offsets(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    """
    用 tokenizer offset mapping 切割，盡量保留原始 OCR 文字。
    若 tokenizer 不提供 offsets，才退回 decode token IDs。
    """
    text = text.strip()
    if not text:
        return []

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        token_ids = encoded["input_ids"]
        offsets = encoded.get("offset_mapping")
    except Exception:
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        offsets = None

    token_count = len(token_ids)
    if token_count <= max_tokens:
        return [
            {
                "content": text,
                "token_count": token_count,
                "token_start": 0,
                "token_end": token_count,
            }
        ]

    pieces: list[dict[str, Any]] = []
    start = 0

    while start < token_count:
        end = min(start + max_tokens, token_count)

        if offsets and end > start:
            char_start = int(offsets[start][0])
            char_end = int(offsets[end - 1][1])
            piece = text[char_start:char_end].strip()
        else:
            piece = tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

        if piece:
            pieces.append(
                {
                    "content": piece,
                    "token_count": end - start,
                    "token_start": start,
                    "token_end": end,
                }
            )

        if end >= token_count:
            break

        next_start = end - overlap_tokens
        if next_start <= start:
            next_start = end
        start = next_start

    return pieces


def build_passage_text(record: dict[str, Any]) -> str:
    source_file = str(record.get("source_file") or "未知文件")
    title = str(record.get("title") or Path(source_file).stem)
    page = record.get("page")
    content = str(record.get("content") or "").strip()

    lines = [
        f"passage: 文件名稱：{source_file}",
        f"文件標題：{title}",
    ]
    if page is not None:
        lines.append(f"頁碼：第 {page} 頁")
    lines.append("")
    lines.append(content)
    return "\n".join(lines)


def prepare_index_records(
    chunks_file: Path,
    tokenizer: Any,
    max_content_tokens: int,
    overlap_tokens: int,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records_by_id: dict[str, dict[str, Any]] = {}
    source_count = 0
    skipped_empty = 0
    split_source_chunks = 0

    for line_number, source in jsonl_lines(chunks_file):
        source_count += 1

        parent_chunk_id = str(source.get("chunk_id") or "").strip()
        document_id = str(source.get("document_id") or "").strip()
        content = str(source.get("content") or "").strip()

        if not parent_chunk_id:
            raise ValueError(
                f"{chunks_file} 第 {line_number} 行缺少 chunk_id"
            )
        if not document_id:
            raise ValueError(
                f"{chunks_file} 第 {line_number} 行缺少 document_id"
            )
        if not content:
            skipped_empty += 1
            continue

        metadata = source.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        source_file = str(source.get("source_file") or "未知文件")
        title = str(metadata.get("title") or Path(source_file).stem)
        page = int(source.get("page") or source.get("page_start") or 0)

        pieces = split_text_by_token_offsets(
            content,
            tokenizer=tokenizer,
            max_tokens=max_content_tokens,
            overlap_tokens=overlap_tokens,
        )

        if len(pieces) > 1:
            split_source_chunks += 1

        for sub_index, piece in enumerate(pieces, start=1):
            if len(pieces) == 1:
                vector_chunk_id = parent_chunk_id
            else:
                vector_chunk_id = f"{parent_chunk_id}-s{sub_index:03d}"

            record = {
                "vector_chunk_id": vector_chunk_id,
                "parent_chunk_id": parent_chunk_id,
                "document_id": document_id,
                "source_file": source_file,
                "source_path": str(source.get("source_path") or ""),
                "title": title,
                "page": page,
                "page_start": int(source.get("page_start") or page),
                "page_end": int(source.get("page_end") or page),
                "chunk_index": source.get("chunk_index"),
                "page_chunk_index": source.get("page_chunk_index"),
                "sub_chunk_index": sub_index,
                "sub_chunk_count": len(pieces),
                "token_count": int(piece["token_count"]),
                "token_start": int(piece["token_start"]),
                "token_end": int(piece["token_end"]),
                "content": str(piece["content"]),
                "content_sha256": sha256_text(str(piece["content"])),
                "source_metadata": metadata,
            }
            record["embedding_text"] = build_passage_text(record)
            record["embedding_text_sha256"] = sha256_text(
                record["embedding_text"]
            )

            # 若重複 ID 出現，最後一筆覆蓋並記錄警告。
            if vector_chunk_id in records_by_id:
                logger.warning(
                    "重複 vector_chunk_id，採用最後一筆：%s",
                    vector_chunk_id,
                )
            records_by_id[vector_chunk_id] = record

    records = sorted(
        records_by_id.values(),
        key=lambda item: (
            item["document_id"],
            int(item["page"]),
            int(item.get("chunk_index") or 0),
            int(item["sub_chunk_index"]),
            item["vector_chunk_id"],
        ),
    )

    stats = {
        "source_chunk_count": source_count,
        "indexed_chunk_count": len(records),
        "skipped_empty_count": skipped_empty,
        "split_source_chunk_count": split_source_chunks,
    }
    return records, stats


def build_model_fingerprint(
    model_name: str,
    model_revision: str,
    max_content_tokens: int,
) -> str:
    data = {
        "model_name": model_name,
        "model_revision": model_revision,
        "max_content_tokens": max_content_tokens,
        "normalize_embeddings": True,
        "prefix": "passage:",
        "index_format_version": INDEX_FORMAT_VERSION,
    }
    return sha256_text(
        json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def embedding_cache_key(
    model_fingerprint: str,
    embedding_text_sha256: str,
) -> str:
    return sha256_text(
        f"{model_fingerprint}:{embedding_text_sha256}"
    )


def encode_records_with_cache(
    model: SentenceTransformer,
    records: list[dict[str, Any]],
    cache: EmbeddingCache,
    model_fingerprint: str,
    dimension: int,
    batch_size: int,
    force_reembed: bool,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[str, int]]:
    for record in records:
        record["embedding_cache_key"] = embedding_cache_key(
            model_fingerprint,
            record["embedding_text_sha256"],
        )

    cache_keys = [
        record["embedding_cache_key"]
        for record in records
    ]

    cached_vectors = (
        {}
        if force_reembed
        else cache.load_many(cache_keys, expected_dimension=dimension)
    )

    missing_by_key: dict[str, str] = {}
    missing_hash_by_key: dict[str, str] = {}

    for record in records:
        cache_key = record["embedding_cache_key"]
        if cache_key not in cached_vectors:
            missing_by_key.setdefault(cache_key, record["embedding_text"])
            missing_hash_by_key.setdefault(
                cache_key,
                record["embedding_text_sha256"],
            )

    missing_keys = list(missing_by_key)
    logger.info(
        "Embedding 快取：命中=%d，需計算=%d",
        len(records) - sum(
            1
            for record in records
            if record["embedding_cache_key"] not in cached_vectors
        ),
        len(missing_keys),
    )

    new_cache_rows: list[tuple[str, str, str, np.ndarray]] = []

    if missing_keys:
        missing_texts = [
            missing_by_key[cache_key]
            for cache_key in missing_keys
        ]

        logger.info(
            "開始計算 %d 筆新 embedding，batch_size=%d",
            len(missing_texts),
            batch_size,
        )

        with torch.inference_mode():
            new_vectors = model.encode(
                missing_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        new_vectors = np.asarray(new_vectors, dtype=np.float32)
        if new_vectors.ndim == 1:
            new_vectors = new_vectors.reshape(1, -1)

        if new_vectors.shape != (len(missing_keys), dimension):
            raise RuntimeError(
                "Embedding 維度不符："
                f"預期 {(len(missing_keys), dimension)}，"
                f"實際 {new_vectors.shape}"
            )

        new_vectors = normalize_matrix(new_vectors)

        for cache_key, vector in zip(missing_keys, new_vectors):
            cached_vectors[cache_key] = vector
            new_cache_rows.append(
                (
                    cache_key,
                    model_fingerprint,
                    missing_hash_by_key[cache_key],
                    vector,
                )
            )

        cache.save_many(new_cache_rows)

    matrix = np.vstack(
        [
            cached_vectors[record["embedding_cache_key"]]
            for record in records
        ]
    ).astype(np.float32, copy=False)
    matrix = normalize_matrix(matrix)

    return matrix, {
        "cache_hit_record_count": len(records)
        - sum(
            1
            for record in records
            if record["embedding_cache_key"] in {
                row[0] for row in new_cache_rows
            }
        ),
        "new_unique_embedding_count": len(new_cache_rows),
        "total_indexed_vector_count": len(records),
    }


def encode_texts_with_cache(
    model: SentenceTransformer,
    texts: Sequence[str],
    cache: EmbeddingCache,
    model_fingerprint: str,
    dimension: int,
    batch_size: int,
) -> list[np.ndarray]:
    if not texts:
        return []

    hashes = [sha256_text(text) for text in texts]
    keys = [
        embedding_cache_key(model_fingerprint, text_hash)
        for text_hash in hashes
    ]
    cached = cache.load_many(keys, expected_dimension=dimension)

    missing_indices = [
        index
        for index, key in enumerate(keys)
        if key not in cached
    ]

    if missing_indices:
        missing_texts = [texts[index] for index in missing_indices]
        with torch.inference_mode():
            vectors = model.encode(
                missing_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        vectors = normalize_matrix(vectors)

        rows = []
        for local_index, source_index in enumerate(missing_indices):
            key = keys[source_index]
            vector = vectors[local_index]
            cached[key] = vector
            rows.append(
                (
                    key,
                    model_fingerprint,
                    hashes[source_index],
                    vector,
                )
            )
        cache.save_many(rows)

    return [cached[key] for key in keys]


def select_representative_chunks(
    document_vector: np.ndarray,
    record_indices: Sequence[int],
    records: Sequence[dict[str, Any]],
    chunk_vectors: np.ndarray,
    count: int,
) -> list[str]:
    if not record_indices or count <= 0:
        return []

    candidate_indices = list(record_indices)
    scores = chunk_vectors[candidate_indices] @ document_vector

    score_order = [
        candidate_indices[index]
        for index in np.argsort(-scores)
    ]

    selected: list[int] = []
    selected_ids: set[str] = set()
    per_page_count: dict[int, int] = defaultdict(int)

    # 文件開頭通常包含標題、目的與適用範圍，先保留最多兩塊。
    first_order = sorted(
        candidate_indices,
        key=lambda index: (
            int(records[index]["page"]),
            int(records[index].get("chunk_index") or 0),
            int(records[index]["sub_chunk_index"]),
        ),
    )
    for index in first_order[:2]:
        chunk_id = records[index]["vector_chunk_id"]
        if chunk_id not in selected_ids:
            selected.append(index)
            selected_ids.add(chunk_id)
            per_page_count[int(records[index]["page"])] += 1

    # 再選最接近文件語意中心的 chunks，限制每頁最多兩塊。
    for index in score_order:
        if len(selected) >= count:
            break
        page = int(records[index]["page"])
        chunk_id = records[index]["vector_chunk_id"]
        if chunk_id in selected_ids:
            continue
        if per_page_count[page] >= 2:
            continue
        selected.append(index)
        selected_ids.add(chunk_id)
        per_page_count[page] += 1

    # 文件很短或頁面限制導致不足時，再補滿。
    for index in score_order:
        if len(selected) >= count:
            break
        chunk_id = records[index]["vector_chunk_id"]
        if chunk_id in selected_ids:
            continue
        selected.append(index)
        selected_ids.add(chunk_id)

    return [
        records[index]["vector_chunk_id"]
        for index in selected[:count]
    ]


def build_document_vectors(
    records: list[dict[str, Any]],
    chunk_vectors: np.ndarray,
    documents_catalog: dict[str, dict[str, Any]],
    model: SentenceTransformer,
    cache: EmbeddingCache,
    model_fingerprint: str,
    dimension: int,
    batch_size: int,
    representative_count: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped_indices[record["document_id"]].append(index)

    document_ids = sorted(grouped_indices)
    document_vectors: list[np.ndarray] = []
    document_metadata: list[dict[str, Any]] = []

    summary_texts: list[str] = []
    summary_doc_ids: list[str] = []
    summaries: dict[str, dict[str, Any]] = {}

    for document_id in document_ids:
        summary = load_optional_document_summary(
            documents_catalog.get(document_id)
        )
        if summary:
            first_record = records[grouped_indices[document_id][0]]
            summary_text = summary_to_embedding_text(
                source_file=first_record["source_file"],
                title=first_record["title"],
                summary=summary,
            )
            if summary_text:
                summaries[document_id] = summary
                summary_doc_ids.append(document_id)
                summary_texts.append(summary_text)

    summary_vectors: dict[str, np.ndarray] = {}
    if summary_texts:
        logger.info(
            "找到 %d 份 document_summary.json，建立摘要向量",
            len(summary_texts),
        )
        vectors = encode_texts_with_cache(
            model=model,
            texts=summary_texts,
            cache=cache,
            model_fingerprint=model_fingerprint,
            dimension=dimension,
            batch_size=batch_size,
        )
        summary_vectors = dict(zip(summary_doc_ids, vectors))

    for faiss_id, document_id in enumerate(document_ids):
        indices = grouped_indices[document_id]
        vectors = chunk_vectors[indices]

        # 稍微提高內容較完整 chunk 的權重，但避免超長 chunk 壟斷。
        weights = np.asarray(
            [
                math.sqrt(max(1, min(len(records[index]["content"]), 2000)))
                for index in indices
            ],
            dtype=np.float32,
        )
        centroid = np.average(vectors, axis=0, weights=weights)
        centroid = normalize_vector(centroid)

        vector_source = "chunk_centroid"
        document_vector = centroid

        if document_id in summary_vectors:
            # 摘要較能代表「整份文件」，centroid 保留原文語意。
            document_vector = normalize_vector(
                0.65 * summary_vectors[document_id]
                + 0.35 * centroid
            )
            vector_source = "summary_65_centroid_35"

        first_record = records[indices[0]]
        pages = sorted(
            {
                int(records[index]["page"])
                for index in indices
            }
        )
        page_count_from_metadata = max(
            (
                int(
                    records[index]
                    .get("source_metadata", {})
                    .get("page_count")
                    or 0
                )
                for index in indices
            ),
            default=0,
        )
        page_count = page_count_from_metadata or (max(pages) if pages else 0)

        representative_chunk_ids = select_representative_chunks(
            document_vector=document_vector,
            record_indices=indices,
            records=records,
            chunk_vectors=chunk_vectors,
            count=representative_count,
        )

        summary = summaries.get(document_id, {})
        document_metadata.append(
            {
                "faiss_id": faiss_id,
                "document_id": document_id,
                "source_file": first_record["source_file"],
                "source_path": first_record["source_path"],
                "title": str(
                    summary.get("title")
                    or first_record["title"]
                ),
                "page_count": page_count,
                "indexed_page_numbers": pages,
                "indexed_chunk_count": len(indices),
                "vector_source": vector_source,
                "summary_available": bool(document_id in summary_vectors),
                "summary": str(summary.get("summary") or ""),
                "topics": (
                    summary.get("topics")
                    if isinstance(summary.get("topics"), list)
                    else []
                ),
                "representative_chunk_ids": representative_chunk_ids,
            }
        )
        document_vectors.append(document_vector)

    if not document_vectors:
        return np.empty((0, dimension), dtype=np.float32), []

    matrix = np.vstack(document_vectors).astype(np.float32, copy=False)
    matrix = normalize_matrix(matrix)
    return matrix, document_metadata


def build_faiss_flat_ip(vectors: np.ndarray) -> faiss.IndexFlatIP:
    if vectors.ndim != 2:
        raise ValueError(f"向量矩陣必須是二維，實際：{vectors.shape}")

    dimension = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dimension)
    if vectors.shape[0] > 0:
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="從 Unlimited-OCR all_chunks.jsonl 建立增量快取與雙層 FAISS 索引"
    )
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=DEFAULT_CHUNKS_FILE,
        help=f"輸入 chunks JSONL，預設：{DEFAULT_CHUNKS_FILE}",
    )
    parser.add_argument(
        "--documents-file",
        type=Path,
        default=DEFAULT_DOCUMENTS_FILE,
        help=f"文件目錄 JSONL，預設：{DEFAULT_DOCUMENTS_FILE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"索引輸出資料夾，預設：{DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding 模型，預設：{DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help="模型 revision 或 commit hash，預設：main",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Embedding 裝置，預設：auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help="CUDA 模型精度，預設：float16；CPU 一律 float32",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size，預設：{DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--max-content-tokens",
        type=int,
        default=DEFAULT_MAX_CONTENT_TOKENS,
        help=(
            "每個索引子分塊最多內容 tokens，"
            f"預設：{DEFAULT_MAX_CONTENT_TOKENS}"
        ),
    )
    parser.add_argument(
        "--token-overlap",
        type=int,
        default=DEFAULT_TOKEN_OVERLAP,
        help=f"子分塊 token 重疊，預設：{DEFAULT_TOKEN_OVERLAP}",
    )
    parser.add_argument(
        "--representative-chunks",
        type=int,
        default=DEFAULT_REPRESENTATIVE_CHUNKS,
        help=(
            "每份文件保留的代表性 chunk IDs，"
            f"預設：{DEFAULT_REPRESENTATIVE_CHUNKS}"
        ),
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="忽略 SQLite embedding 快取，全部重新計算",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="顯示詳細紀錄",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size 必須大於 0")
    if not 64 <= args.max_content_tokens <= 480:
        raise ValueError("--max-content-tokens 建議介於 64 到 480")
    if args.token_overlap < 0:
        raise ValueError("--token-overlap 不可小於 0")
    if args.token_overlap >= args.max_content_tokens:
        raise ValueError(
            "--token-overlap 必須小於 --max-content-tokens"
        )
    if args.representative_chunks < 1:
        raise ValueError("--representative-chunks 必須大於 0")


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 --device cuda，但 torch.cuda.is_available() 為 False")
    return requested


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))

    logger = setup_logging(args.verbose)

    chunks_file = args.chunks_file.expanduser().resolve()
    documents_file = args.documents_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not chunks_file.is_file():
        logger.error("找不到 chunks 檔案：%s", chunks_file)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    lock_path = output_dir / ".build_rag_index.lock"
    lock_file = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(
            lock_file.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
    except BlockingIOError:
        logger.error("已有另一個 build_rag_index.py 正在執行")
        lock_file.close()
        return 2

    lock_file.write(str(os.getpid()))
    lock_file.flush()

    cache: EmbeddingCache | None = None
    started = time.monotonic()

    try:
        device = resolve_device(args.device)
        logger.info("chunks：%s", chunks_file)
        logger.info("output：%s", output_dir)
        logger.info("model：%s", args.model)
        logger.info("device：%s", device)

        model_kwargs: dict[str, Any] = {}
        if device == "cuda" and args.dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16

        logger.info("離線載入 Embedding 模型（只使用本機快取）")
        model_init_kwargs: dict[str, Any] = {
            "revision": args.model_revision,
            "device": device,
            "local_files_only": True,
        }
        if model_kwargs:
            model_init_kwargs["model_kwargs"] = model_kwargs

        model = SentenceTransformer(
            args.model,
            **model_init_kwargs,
        )

        # 明確限制在 E5 的 512 tokens。
        model.max_seq_length = min(
            int(getattr(model, "max_seq_length", 512) or 512),
            512,
        )
        tokenizer = model.tokenizer

        dimension_raw = model.get_sentence_embedding_dimension()
        if dimension_raw is None:
            raise RuntimeError("無法取得 embedding 維度")
        dimension = int(dimension_raw)

        logger.info(
            "模型載入完成：dimension=%d，max_seq_length=%d",
            dimension,
            model.max_seq_length,
        )

        records, preparation_stats = prepare_index_records(
            chunks_file=chunks_file,
            tokenizer=tokenizer,
            max_content_tokens=args.max_content_tokens,
            overlap_tokens=args.token_overlap,
            logger=logger,
        )

        if not records:
            logger.error("沒有可建立索引的有效 chunks")
            return 1

        logger.info(
            "來源 chunks=%d，索引 chunks=%d，過長而拆分的來源 chunks=%d",
            preparation_stats["source_chunk_count"],
            preparation_stats["indexed_chunk_count"],
            preparation_stats["split_source_chunk_count"],
        )

        model_fingerprint = build_model_fingerprint(
            model_name=args.model,
            model_revision=args.model_revision,
            max_content_tokens=args.max_content_tokens,
        )

        cache = EmbeddingCache(
            output_dir / "embedding_cache.sqlite3"
        )

        chunk_vectors, embedding_stats = encode_records_with_cache(
            model=model,
            records=records,
            cache=cache,
            model_fingerprint=model_fingerprint,
            dimension=dimension,
            batch_size=args.batch_size,
            force_reembed=args.force_reembed,
            logger=logger,
        )

        chunk_metadata: list[dict[str, Any]] = []
        for faiss_id, record in enumerate(records):
            metadata = {
                key: value
                for key, value in record.items()
                if key
                not in {
                    "embedding_text",
                    "embedding_cache_key",
                }
            }
            metadata["faiss_id"] = faiss_id
            chunk_metadata.append(metadata)

        documents_catalog = load_documents_catalog(
            documents_file
        )

        document_vectors, document_metadata = build_document_vectors(
            records=records,
            chunk_vectors=chunk_vectors,
            documents_catalog=documents_catalog,
            model=model,
            cache=cache,
            model_fingerprint=model_fingerprint,
            dimension=dimension,
            batch_size=args.batch_size,
            representative_count=args.representative_chunks,
            logger=logger,
        )

        chunk_index = build_faiss_flat_ip(chunk_vectors)
        document_index = build_faiss_flat_ip(document_vectors)

        atomic_write_faiss(
            output_dir / "chunk.index",
            chunk_index,
        )
        atomic_write_jsonl(
            output_dir / "chunk_metadata.jsonl",
            chunk_metadata,
        )
        atomic_write_faiss(
            output_dir / "document.index",
            document_index,
        )
        atomic_write_jsonl(
            output_dir / "document_metadata.jsonl",
            document_metadata,
        )

        elapsed = time.monotonic() - started
        manifest = {
            "format_version": INDEX_FORMAT_VERSION,
            "created_at": utc_now(),
            "elapsed_seconds": round(elapsed, 3),
            "source": {
                "chunks_file": str(chunks_file),
                "chunks_file_sha256": sha256_file(chunks_file),
                "documents_file": (
                    str(documents_file)
                    if documents_file.is_file()
                    else None
                ),
            },
            "embedding": {
                "model": args.model,
                "model_revision": args.model_revision,
                "model_fingerprint": model_fingerprint,
                "device": device,
                "dtype": (
                    args.dtype
                    if device == "cuda"
                    else "float32"
                ),
                "dimension": dimension,
                "max_seq_length": model.max_seq_length,
                "max_content_tokens": args.max_content_tokens,
                "token_overlap": args.token_overlap,
                "batch_size": args.batch_size,
                "passage_prefix": "passage:",
                "normalized": True,
            },
            "faiss": {
                "index_type": "IndexFlatIP",
                "metric": "inner_product_on_L2_normalized_vectors",
                "chunk_index": "chunk.index",
                "chunk_metadata": "chunk_metadata.jsonl",
                "chunk_count": int(chunk_index.ntotal),
                "document_index": "document.index",
                "document_metadata": "document_metadata.jsonl",
                "document_count": int(document_index.ntotal),
            },
            "preparation_stats": preparation_stats,
            "embedding_stats": embedding_stats,
        }
        atomic_write_json(
            output_dir / "index_manifest.json",
            manifest,
        )

        logger.info("建立完成")
        logger.info(
            "chunk index：%s（%d vectors）",
            output_dir / "chunk.index",
            chunk_index.ntotal,
        )
        logger.info(
            "document index：%s（%d vectors）",
            output_dir / "document.index",
            document_index.ntotal,
        )
        logger.info(
            "embedding cache：%s",
            output_dir / "embedding_cache.sqlite3",
        )
        logger.info("耗時：%.1f 秒", elapsed)

        if chunk_index.ntotal > 100_000:
            logger.warning(
                "chunk 已超過 100,000；之後可考慮 HNSW／IVF。"
                "目前 30～50 頁文件仍適合 IndexFlatIP。"
            )

        return 0

    except torch.OutOfMemoryError:
        logger.exception(
            "CUDA 記憶體不足。請先停止 OCR 模型，"
            "或改用 --batch-size 4／--dtype float16。"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 3
    except Exception:
        logger.exception("建立 RAG 索引失敗")
        return 1
    finally:
        if cache is not None:
            cache.close()
        with contextlib.suppress(OSError):
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
