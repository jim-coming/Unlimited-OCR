#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_search.py — E5 + FAISS 召回，再以 BGE CrossEncoder reranker 判斷相關性。

流程：
1. multilingual-e5-base + FAISS 召回候選 chunks。
2. BAAI/bge-reranker-v2-m3 同時讀取「問題 + 段落」重新排序。
3. detail 模式以 reranker 分數判斷是否有足夠內容。
4. overview 模式使用指定文件的代表性 chunks。
5. documents 模式搜尋可能相關的 PDF。

範例：
  python rag_search.py "這份 PDF 在講什麼？" --source-file SOP.pdf
  python rag_search.py "設備啟動前需要檢查什麼？" --source-file SOP.pdf
  python rag_search.py "明天的天氣如何？" --source-file SOP.pdf
  python rag_search.py --interactive --source-file SOP.pdf
  python rag_search.py --list-documents
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
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_VECTOR_DIR = DEFAULT_ROOT / "flash/vector"

DEFAULT_TOP_K = 6
DEFAULT_DENSE_CANDIDATES = 20
DEFAULT_DOCUMENT_TOP_K = 3

# E5 分數集中在高分區，這個門檻只用於「跨文件搜尋」。
DEFAULT_DOCUMENT_MIN_SCORE = 0.55

# 細節問答最後以 reranker 分數判斷。
DEFAULT_MIN_RERANK_SCORE = 0.50
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_BATCH_SIZE = 2
DEFAULT_RERANKER_MAX_LENGTH = 1024

DEFAULT_MAX_CONTEXT_CHARS = 9000


OVERVIEW_PATTERNS = (
    # 文件名稱／指示詞在前：
    # 「這份 PDF 在講什麼」、「這份文件幫我講解一下」
    r"(?:這份|此份|該份|目前這份|這個|此個)?"
    r"(?:pdf|文件|文檔|檔案|文章)"
    r".*(?:講什麼|說什麼|主要內容|主旨|重點|摘要|概要|概述|"
    r"講解|解說|解讀|導讀|說明|介紹|分析|整理|總結)",

    # 動作在前：
    # 「講解一下這份 PDF」、「請幫我分析這個文件」
    r"(?:幫我|請|麻煩|可以)?(?:先)?"
    r"(?:講解|解說|解讀|導讀|說明|介紹|分析|整理|摘要|概述|總結)"
    r"(?:一下|一下子|給我|看看|下)?"
    r".*(?:這份|此份|該份|目前這份|這個|此個)?"
    r"(?:pdf|文件|文檔|檔案|文章)",

    # 不一定明說 PDF，但已在 GUI 選定文件：
    # 「全文講解」、「整份整理一下」
    r"(?:全文|整份|整個文件|整篇)"
    r".*(?:講解|解說|解讀|導讀|說明|介紹|分析|整理|摘要|概述|總結)",

    r"(?:這份|此份|該份).*(?:核心內容|文件目的|用途|適用對象)",
    r"(?:有哪些|列出|整理).*(?:章節|主題|重點)",
    r"(?:what is|what's).*(?:this|the).*(?:pdf|document).*(?:about|discuss)",
    r"(?:explain|summarize|summary|overview|introduce|analyze)"
    r".*(?:this|the)?.*(?:pdf|document|file)",
    r"(?:main topic|main idea|purpose).*(?:pdf|document|file)",
)

DOCUMENT_DISCOVERY_PATTERNS = (
    r"哪份(?:pdf|文件|文檔|檔案)",
    r"哪些(?:pdf|文件|文檔|檔案)",
    r"哪一份",
    r"which (?:pdf|document|file)",
    r"what (?:pdf|document|file)",
)


@dataclass(frozen=True)
class SearchConfig:
    vector_dir: Path
    top_k: int
    dense_candidates: int
    document_top_k: int
    document_min_score: float
    min_rerank_score: float
    max_context_chars: int
    device: str
    dtype: str
    reranker_model: str
    reranker_device: str
    reranker_batch_size: int
    reranker_max_length: int
    disable_reranker: bool


def setup_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("rag_search")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"找不到檔案：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON 格式錯誤：{path}：{error}") from error

    if not isinstance(value, dict):
        raise ValueError(f"JSON 必須是 object：{path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        file = path.open("r", encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"找不到檔案：{path}") from error

    with file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path} 第 {line_number} 行 JSON 錯誤：{error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path} 第 {line_number} 行必須是 JSON object"
                )
            records.append(value)
    return records


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "指定 CUDA，但 torch.cuda.is_available() 為 False"
        )
    return requested


def normalize_query(question: str) -> str:
    question = re.sub(r"\s+", " ", question).strip()
    if not question:
        raise ValueError("問題不可為空")
    return question


def detect_query_mode(question: str, has_selected_document: bool) -> str:
    normalized = question.lower()

    for pattern in DOCUMENT_DISCOVERY_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return "documents"

    for pattern in OVERVIEW_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return "overview"

    if has_selected_document:
        generic_overview_words = (
            "內容是什麼",
            "內容為何",
            "主要內容",
            "文件目的",
            "這是什麼",
            "用途是什麼",
            "講什麼",
            "說什麼",
            "講解一下",
            "幫我講解",
            "解說一下",
            "幫我解說",
            "解讀一下",
            "幫我解讀",
            "導讀一下",
            "幫我導讀",
            "說明一下",
            "幫我說明",
            "介紹一下",
            "幫我介紹",
            "分析一下",
            "幫我分析",
            "整理一下",
            "幫我整理",
            "總結一下",
            "幫我總結",
            "全文摘要",
            "整份摘要",
            "全文重點",
            "整份重點",
        )
        if any(word in normalized for word in generic_overview_words):
            return "overview"

    return "detail"


def load_all_vectors(index: faiss.Index) -> np.ndarray:
    if index.ntotal == 0:
        return np.empty((0, index.d), dtype=np.float32)
    vectors = np.empty((index.ntotal, index.d), dtype=np.float32)
    index.reconstruct_n(0, index.ntotal, vectors)
    return vectors


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("向量無法正規化")
    return vector / norm


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


# 語意路由範例不是「必須完全相同的關鍵字」。
# E5 會比較使用者整句話與以下意圖的向量相似度，
# 因此「帶我看一下這份資料」等未列出的說法也能被辨識。
OVERVIEW_INTENT_EXAMPLES = (
    "請摘要並講解目前選取的整份文件",
    "這份 PDF 整體在說什麼",
    "幫我整理這份手冊的主要內容與重點",
    "請對整份文件做導讀與概述",
    "介紹這份資料的章節、目的和核心內容",
    "我想先了解整份文件的全貌",
    "請總結全文而不是查某一個細節",
    "帶我快速看懂目前選取的 PDF",
)

DETAIL_INTENT_EXAMPLES = (
    "根據文件回答一個具體問題",
    "查詢文件中的特定數值、規格或庫存",
    "某一頁或某一條規定寫了什麼",
    "找出設備操作的特定步驟",
    "文件中某個零件的規格是多少",
    "請根據相關段落回答，而不是摘要整份文件",
    "查詢某個專有名詞、條文或項目",
    "比較文件中的兩個具體項目",
)

SEMANTIC_ROUTER_MARGIN = 0.035


QUESTION_STOP_PHRASES = (
    "請問", "麻煩", "告訴我", "幫我查", "幫我找",
    "文件中", "表格中", "這份文件", "這份pdf", "這個pdf",
    "庫存量是多少", "庫存有多少", "數量是多少", "有多少",
    "是多少", "多少", "庫存量", "庫存", "數量",
    "規格是什麼", "規格", "編號是什麼", "編號",
    "哪一個", "哪個", "什麼", "為何", "如何", "怎麼",
    "嗎", "呢", "？", "?", "。", "，", ",", "：", ":",
)


def compact_search_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"&quot;|&#34;", '"', text)
    text = re.sub(
        r"[\\s|`*_#\\-—–/\\\\()\\[\\]{}<>\\\"'，。；：、！？?;:]+",
        "",
        text,
    )
    return text


def lexical_query_key(question: str) -> str:
    text = str(question or "").lower()
    for phrase in QUESTION_STOP_PHRASES:
        text = text.replace(phrase, "")
    return compact_search_text(text)


def lexical_match_score(question: str, content: str) -> float:
    """短表格列、料號與規格的精確字詞後援。"""
    key = lexical_query_key(question)
    if len(key) < 2:
        return 0.0

    compact_content = compact_search_text(content)
    if key in compact_content:
        return 1.0

    pieces = re.findall(
        r"[a-z0-9.]+|[\\u3400-\\u9fff]{2,}",
        key,
    )
    pieces = [piece for piece in pieces if len(piece) >= 2]
    if pieces and all(piece in compact_content for piece in pieces):
        return 0.95

    return 0.0


class RAGSearcher:
    def __init__(self, config: SearchConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.vector_dir = config.vector_dir.resolve()

        self.manifest = read_json(
            self.vector_dir / "index_manifest.json"
        )
        self.chunk_metadata = read_jsonl(
            self.vector_dir / "chunk_metadata.jsonl"
        )
        self.document_metadata = read_jsonl(
            self.vector_dir / "document_metadata.jsonl"
        )
        self.chunk_index = faiss.read_index(
            str(self.vector_dir / "chunk.index")
        )
        self.document_index = faiss.read_index(
            str(self.vector_dir / "document.index")
        )
        self._validate_index_alignment()

        self.chunk_by_id = {
            str(item["vector_chunk_id"]): item
            for item in self.chunk_metadata
            if item.get("vector_chunk_id")
        }
        self.documents_by_id = {
            str(item["document_id"]): item
            for item in self.document_metadata
            if item.get("document_id")
        }

        self.chunk_ids_by_document: dict[str, list[int]] = {}
        for faiss_id, item in enumerate(self.chunk_metadata):
            document_id = str(item.get("document_id") or "")
            self.chunk_ids_by_document.setdefault(
                document_id, []
            ).append(faiss_id)

        self.device = resolve_device(config.device)
        self.reranker_device = resolve_device(
            config.reranker_device
        )

        embedding_info = self.manifest.get("embedding", {})
        self.model_name = str(
            embedding_info.get(
                "model", "intfloat/multilingual-e5-base"
            )
        )
        self.model_revision = str(
            embedding_info.get("model_revision", "main")
        )
        self.dimension = int(
            embedding_info.get("dimension", self.chunk_index.d)
        )

        model_kwargs: dict[str, Any] = {}
        if self.device == "cuda" and config.dtype == "float16":
            # transformers 5.x 使用 dtype，避免 torch_dtype deprecated 警告。
            model_kwargs["dtype"] = torch.float16

        self.logger.info(
            "載入 Embedding 模型：%s，device=%s",
            self.model_name,
            self.device,
        )
        self.model = SentenceTransformer(
            self.model_name,
            revision=self.model_revision,
            device=self.device,
            model_kwargs=model_kwargs or None,
            local_files_only=True,
        )
        self.model.max_seq_length = min(
            int(embedding_info.get("max_seq_length", 512)),
            512,
        )

        # 使用同一個 E5 建立意圖原型，不載入額外分類模型。
        self._semantic_route_vectors = (
            self._build_semantic_route_vectors()
        )

        self.reranker: CrossEncoder | None = None
        self._chunk_vectors: np.ndarray | None = None
        self._document_vectors: np.ndarray | None = None

    def _encode_passages(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """離線編碼一組語意路由原型。"""
        passages = [
            f"passage: {normalize_query(text)}"
            for text in texts
        ]

        with torch.inference_mode():
            vectors_tensor = self.model.encode(
                passages,
                show_progress_bar=False,
                convert_to_numpy=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

        if isinstance(vectors_tensor, torch.Tensor):
            vectors = np.asarray(
                vectors_tensor.detach().float().cpu().tolist(),
                dtype=np.float32,
            )
        else:
            vectors = np.asarray(
                vectors_tensor,
                dtype=np.float32,
            )

        if vectors.ndim != 2:
            raise RuntimeError(
                f"語意路由向量維度錯誤：{vectors.shape}"
            )
        return vectors

    def _build_semantic_route_vectors(
        self,
    ) -> dict[str, np.ndarray]:
        """建立 overview/detail 兩組語意意圖向量。"""
        return {
            "overview": self._encode_passages(
                list(OVERVIEW_INTENT_EXAMPLES)
            ),
            "detail": self._encode_passages(
                list(DETAIL_INTENT_EXAMPLES)
            ),
        }

    def semantic_query_mode(
        self,
        question: str,
        has_selected_document: bool,
    ) -> tuple[str, dict[str, Any]]:
        """
        以 E5 語意判斷 overview/detail。

        固定規則只保留：
        - 文件清單／找文件等明確 documents 指令。
        - 已知的 overview 說法作為接近分數時的 tie-breaker。

        其餘由問題整句的 embedding 自動判斷，不要求命中固定字詞。
        """
        normalized = normalize_query(question)

        for pattern in DOCUMENT_DISCOVERY_PATTERNS:
            if re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            ):
                return (
                    "documents",
                    {
                        "method": "rule_documents",
                        "overview_score": None,
                        "detail_score": None,
                        "margin": None,
                    },
                )

        query_vector = self.embed_query(normalized)
        overview_scores = (
            self._semantic_route_vectors["overview"]
            @ query_vector
        )
        detail_scores = (
            self._semantic_route_vectors["detail"]
            @ query_vector
        )

        overview_score = float(
            np.max(overview_scores)
        )
        detail_score = float(
            np.max(detail_scores)
        )
        score_margin = overview_score - detail_score

        # 舊規則不再是必要條件，只在兩種語意非常接近時協助打破平手。
        rule_hint = detect_query_mode(
            normalized,
            has_selected_document=has_selected_document,
        )

        if (
            has_selected_document
            and score_margin >= SEMANTIC_ROUTER_MARGIN
        ):
            mode = "overview"
            method = "semantic"
        elif (
            has_selected_document
            and rule_hint == "overview"
            and score_margin >= -SEMANTIC_ROUTER_MARGIN
        ):
            mode = "overview"
            method = "semantic_with_rule_tiebreak"
        else:
            # 不確定時採 detail 比較安全，避免無故摘要整份文件。
            mode = "detail"
            method = "semantic"

        return (
            mode,
            {
                "method": method,
                "overview_score": overview_score,
                "detail_score": detail_score,
                "margin": score_margin,
                "required_margin": SEMANTIC_ROUTER_MARGIN,
                "rule_hint": rule_hint,
                "has_selected_document": (
                    has_selected_document
                ),
            },
        )

    def _validate_index_alignment(self) -> None:
        if self.chunk_index.ntotal != len(self.chunk_metadata):
            raise RuntimeError(
                "chunk.index 與 chunk_metadata.jsonl 數量不一致"
            )
        if self.document_index.ntotal != len(
            self.document_metadata
        ):
            raise RuntimeError(
                "document.index 與 document_metadata.jsonl 數量不一致"
            )
        if self.chunk_index.d != self.document_index.d:
            raise RuntimeError("兩個 FAISS 索引維度不一致")

    @property
    def chunk_vectors(self) -> np.ndarray:
        if self._chunk_vectors is None:
            self._chunk_vectors = load_all_vectors(
                self.chunk_index
            )
        return self._chunk_vectors

    @property
    def document_vectors(self) -> np.ndarray:
        if self._document_vectors is None:
            self._document_vectors = load_all_vectors(
                self.document_index
            )
        return self._document_vectors

    def load_reranker_if_needed(self) -> CrossEncoder:
        if self.reranker is not None:
            return self.reranker

        if self.config.disable_reranker:
            raise RuntimeError("reranker 已停用")

        model_kwargs: dict[str, Any] = {}
        if (
            self.reranker_device == "cuda"
            and self.config.dtype == "float16"
        ):
            model_kwargs["dtype"] = torch.float16

        self.logger.info(
            "載入 Reranker：%s，device=%s",
            self.config.reranker_model,
            self.reranker_device,
        )
        self.reranker = CrossEncoder(
            self.config.reranker_model,
            device=self.reranker_device,
            max_length=self.config.reranker_max_length,
            activation_fn=torch.nn.Sigmoid(),
            model_kwargs=model_kwargs or None,
            local_files_only=True,
        )
        return self.reranker

    def embed_query(self, question: str) -> np.ndarray:
        text = f"query: {normalize_query(question)}"
        with torch.inference_mode():
            vector_tensor = self.model.encode(
                [text],
                show_progress_bar=False,
                convert_to_numpy=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

        # 避免 Jetson PyTorch wheel 的 Tensor.numpy() ABI 問題。
        if isinstance(vector_tensor, torch.Tensor):
            vector = np.asarray(
                vector_tensor.detach().float().cpu().tolist(),
                dtype=np.float32,
            )
        else:
            vector = np.asarray(vector_tensor, dtype=np.float32)

        if vector.ndim == 2:
            vector = vector[0]
        if vector.shape != (self.dimension,):
            raise RuntimeError(
                f"查詢向量維度錯誤：{vector.shape}"
            )
        return normalize_vector(vector)

    def list_documents(self) -> list[dict[str, Any]]:
        return [
            {
                "document_id": item.get("document_id"),
                "source_file": item.get("source_file"),
                "title": item.get("title"),
                "page_count": item.get("page_count"),
                "indexed_chunk_count": item.get(
                    "indexed_chunk_count"
                ),
                "summary_available": item.get(
                    "summary_available", False
                ),
            }
            for item in self.document_metadata
        ]

    def resolve_document(
        self,
        document_id: str | None,
        source_file: str | None,
    ) -> dict[str, Any] | None:
        if document_id:
            document_id = document_id.strip()
            exact = self.documents_by_id.get(document_id)
            if exact:
                return exact
            matches = [
                item
                for key, item in self.documents_by_id.items()
                if key.startswith(document_id)
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"document_id 前綴不唯一：{document_id}"
                )
            raise ValueError(
                f"找不到 document_id：{document_id}"
            )

        if source_file:
            query = source_file.strip().lower()
            exact = [
                item
                for item in self.document_metadata
                if str(item.get("source_file") or "").lower()
                == query
            ]
            if len(exact) == 1:
                return exact[0]

            matches = [
                item
                for item in self.document_metadata
                if query
                in str(item.get("source_file") or "").lower()
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                names = ", ".join(
                    str(item.get("source_file"))
                    for item in matches
                )
                raise ValueError(
                    f"檔名比對到多份文件：{names}"
                )
            raise ValueError(
                f"找不到 source_file：{source_file}"
            )
        return None

    def search_documents(
        self, question: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        if self.document_index.ntotal == 0:
            return []

        query_vector = self.embed_query(question)
        count = min(
            int(top_k or self.config.document_top_k),
            self.document_index.ntotal,
        )
        scores = self.document_vectors @ query_vector
        order = np.argsort(-scores)[:count]

        output: list[dict[str, Any]] = []
        for rank, faiss_id in enumerate(order, start=1):
            item = dict(self.document_metadata[int(faiss_id)])
            item.update(
                {
                    "rank": rank,
                    "dense_score": float(scores[int(faiss_id)]),
                    "faiss_id": int(faiss_id),
                }
            )
            output.append(item)
        return output

    def dense_retrieve_chunks(
        self,
        question: str,
        document_id: str | None,
        candidate_count: int,
    ) -> list[dict[str, Any]]:
        if self.chunk_index.ntotal == 0:
            return []

        query_vector = self.embed_query(question)

        if document_id:
            allowed_ids = self.chunk_ids_by_document.get(
                document_id, []
            )
            if not allowed_ids:
                return []
            allowed = np.asarray(allowed_ids, dtype=np.int64)
            local_scores = self.chunk_vectors[allowed] @ query_vector
            local_order = np.argsort(-local_scores)
            pairs = [
                (
                    int(allowed[index]),
                    float(local_scores[index]),
                )
                for index in local_order[:candidate_count]
            ]
        else:
            scores = self.chunk_vectors @ query_vector
            order = np.argsort(-scores)[:candidate_count]
            pairs = [
                (int(index), float(scores[int(index)]))
                for index in order
            ]

        results: list[dict[str, Any]] = []
        used_vector_ids: set[str] = set()
        used_content_hashes: set[str] = set()

        for faiss_id, dense_score in pairs:
            item = dict(self.chunk_metadata[faiss_id])
            vector_id = str(
                item.get("vector_chunk_id") or faiss_id
            )
            content_hash = str(item.get("content_sha256") or "")

            # 不在召回階段排除同一 parent 的子分塊，
            # 因為真正答案可能只在其中一個子分塊。
            if vector_id in used_vector_ids:
                continue
            if content_hash and content_hash in used_content_hashes:
                continue

            used_vector_ids.add(vector_id)
            if content_hash:
                used_content_hashes.add(content_hash)

            item.update(
                {
                    "dense_rank": len(results) + 1,
                    "dense_score": dense_score,
                    "faiss_id": faiss_id,
                }
            )
            results.append(item)

        return results

    def lexical_retrieve_chunks(
        self,
        question: str,
        document_id: str | None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """掃描 metadata，補回短表格列與料號等精確匹配。"""
        output: list[dict[str, Any]] = []

        if document_id:
            faiss_ids = self.chunk_ids_by_document.get(
                document_id,
                [],
            )
        else:
            faiss_ids = range(len(self.chunk_metadata))

        for faiss_id in faiss_ids:
            item = dict(self.chunk_metadata[int(faiss_id)])
            score = lexical_match_score(
                question,
                str(item.get("content") or ""),
            )
            if score <= 0:
                continue

            item.update(
                {
                    "lexical_score": float(score),
                    "lexical_match": True,
                    "faiss_id": int(faiss_id),
                    "dense_score": safe_float(
                        item.get("dense_score"),
                        0.0,
                    ),
                    "dense_rank": None,
                }
            )
            output.append(item)

        output.sort(
            key=lambda item: safe_float(
                item.get("lexical_score"),
                0.0,
            ),
            reverse=True,
        )
        return output[:limit]

    def rerank_chunks(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        if self.config.disable_reranker:
            output = candidates[:top_k]
            for rank, item in enumerate(output, start=1):
                item["rank"] = rank
                item["rerank_score"] = None
            return output

        reranker = self.load_reranker_if_needed()
        pairs = [
            [
                question,
                str(item.get("content") or ""),
            ]
            for item in candidates
        ]

        with torch.inference_mode():
            score_tensor = reranker.predict(
                pairs,
                batch_size=self.config.reranker_batch_size,
                show_progress_bar=False,
                activation_fn=torch.nn.Sigmoid(),
                convert_to_numpy=False,
                convert_to_tensor=True,
            )

        if isinstance(score_tensor, torch.Tensor):
            scores = np.asarray(
                score_tensor.detach().float().cpu().tolist(),
                dtype=np.float32,
            ).reshape(-1)
        else:
            scores = np.asarray(
                score_tensor,
                dtype=np.float32,
            ).reshape(-1)
        if len(scores) != len(candidates):
            raise RuntimeError(
                "Reranker 回傳筆數與候選筆數不一致"
            )

        order = sorted(
            range(len(candidates)),
            key=lambda index: (
                safe_float(
                    candidates[index].get("lexical_score"),
                    0.0,
                ),
                float(scores[index]),
            ),
            reverse=True,
        )
        output: list[dict[str, Any]] = []
        used_parent_ids: set[str] = set()

        for index in order:
            item = dict(candidates[int(index)])
            parent_id = str(
                item.get("parent_chunk_id")
                or item.get("vector_chunk_id")
                or item.get("faiss_id")
            )

            # rerank 後才去除同一原始 chunk 的重疊子分塊。
            if parent_id in used_parent_ids:
                continue
            used_parent_ids.add(parent_id)

            item["rerank_score"] = float(scores[int(index)])
            item["rank"] = len(output) + 1
            output.append(item)

            if len(output) >= top_k:
                break

        return output

    def representative_chunks(
        self, document: dict[str, Any]
    ) -> list[dict[str, Any]]:
        chunk_ids = document.get("representative_chunk_ids")
        if not isinstance(chunk_ids, list):
            return []

        output: list[dict[str, Any]] = []
        for chunk_id in chunk_ids:
            item = self.chunk_by_id.get(str(chunk_id))
            if not item:
                continue
            record = dict(item)
            record["rank"] = len(output) + 1
            record["dense_score"] = None
            record["rerank_score"] = None
            output.append(record)
        return output

    def overview_context(
        self,
        question: str,
        selected_document: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if selected_document is not None:
            documents = [dict(selected_document)]
        else:
            documents = self.search_documents(
                question, self.config.document_top_k
            )

        if not documents:
            return {
                "status": "insufficient_context",
                "mode": "overview",
                "question": question,
                "reason": "沒有可用文件。",
                "documents": [],
                "contexts": [],
            }

        contexts: list[dict[str, Any]] = []
        best_dense_score: float | None = None

        for document_rank, document in enumerate(
            documents, start=1
        ):
            dense_score = document.get("dense_score")
            if dense_score is not None:
                dense_score = safe_float(dense_score)
                best_dense_score = (
                    dense_score
                    if best_dense_score is None
                    else max(best_dense_score, dense_score)
                )

            representative = self.representative_chunks(
                document
            )
            if not representative:
                document_id = str(document.get("document_id"))
                for faiss_id in self.chunk_ids_by_document.get(
                    document_id, []
                )[: self.config.top_k]:
                    item = dict(self.chunk_metadata[faiss_id])
                    item["rank"] = len(representative) + 1
                    item["dense_score"] = None
                    item["rerank_score"] = None
                    representative.append(item)

            contexts.append(
                {
                    "document_rank": document_rank,
                    "document": {
                        key: value
                        for key, value in document.items()
                        if key != "representative_chunk_ids"
                    },
                    "summary": str(
                        document.get("summary") or ""
                    ).strip(),
                    "topics": document.get("topics") or [],
                    "chunks": representative,
                }
            )

        if (
            selected_document is None
            and best_dense_score is not None
            and best_dense_score
            < self.config.document_min_score
        ):
            status = "insufficient_context"
            reason = "找不到明確相關的文件。"
        else:
            status = "ok"
            reason = None

        return {
            "status": status,
            "mode": "overview",
            "question": question,
            "selected_document_id": (
                selected_document.get("document_id")
                if selected_document
                else None
            ),
            "best_dense_score": best_dense_score,
            "document_min_score": (
                self.config.document_min_score
            ),
            "reason": reason,
            "documents": documents,
            "contexts": contexts,
        }

    def detail_context(
        self,
        question: str,
        selected_document: dict[str, Any] | None,
    ) -> dict[str, Any]:
        document_id = (
            str(selected_document.get("document_id"))
            if selected_document
            else None
        )

        candidates = self.dense_retrieve_chunks(
            question=question,
            document_id=document_id,
            candidate_count=self.config.dense_candidates,
        )

        lexical_candidates = self.lexical_retrieve_chunks(
            question=question,
            document_id=document_id,
            limit=self.config.dense_candidates,
        )

        seen_ids = {
            int(item.get("faiss_id"))
            for item in candidates
            if item.get("faiss_id") is not None
        }
        for item in lexical_candidates:
            faiss_id = int(item["faiss_id"])
            if faiss_id in seen_ids:
                for existing in candidates:
                    if int(existing.get("faiss_id", -1)) == faiss_id:
                        existing["lexical_score"] = item["lexical_score"]
                        existing["lexical_match"] = True
                        break
                continue
            candidates.append(item)
            seen_ids.add(faiss_id)

        results = self.rerank_chunks(
            question=question,
            candidates=candidates,
            top_k=self.config.top_k,
        )

        best_dense_score = (
            max(
                safe_float(item.get("dense_score"))
                for item in candidates
            )
            if candidates
            else None
        )
        best_rerank_score = (
            safe_float(results[0].get("rerank_score"))
            if results
            and results[0].get("rerank_score") is not None
            else None
        )
        best_lexical_score = (
            max(
                safe_float(item.get("lexical_score"))
                for item in results
            )
            if results
            else 0.0
        )

        if not results:
            status = "insufficient_context"
            reason = "沒有搜尋到任何候選段落。"
        elif self.config.disable_reranker:
            status = "ok"
            reason = (
                "reranker 已停用；此結果只經過 E5 召回，"
                "不可可靠判斷答案是否存在。"
            )
        elif best_lexical_score >= 0.95:
            status = "ok"
            reason = (
                "表格／料號精確字詞匹配；"
                "即使語意分數偏低，文件中仍明確存在查詢項目。"
            )
        elif (
            best_rerank_score is None
            or best_rerank_score
            < self.config.min_rerank_score
        ):
            status = "insufficient_context"
            reason = (
                "Reranker 判斷最相關段落仍不足以回答問題。"
            )
        else:
            status = "ok"
            reason = None

        return {
            "status": status,
            "mode": "detail",
            "question": question,
            "selected_document_id": document_id,
            "selected_source_file": (
                selected_document.get("source_file")
                if selected_document
                else None
            ),
            "best_dense_score": best_dense_score,
            "best_rerank_score": best_rerank_score,
            "best_lexical_score": best_lexical_score,
            "lexical_fallback_used": (
                best_lexical_score >= 0.95
            ),
            "min_rerank_score": (
                self.config.min_rerank_score
            ),
            "reranker_model": (
                None
                if self.config.disable_reranker
                else self.config.reranker_model
            ),
            "reason": reason,
            "candidate_count": len(candidates),
            "results": results,
        }

    def document_discovery_context(
        self, question: str
    ) -> dict[str, Any]:
        documents = self.search_documents(
            question, self.config.document_top_k
        )
        best_score = (
            safe_float(documents[0].get("dense_score"))
            if documents
            else None
        )

        if not documents:
            status = "insufficient_context"
            reason = "沒有可搜尋的文件。"
        elif (
            best_score is not None
            and best_score < self.config.document_min_score
        ):
            status = "insufficient_context"
            reason = "文件層最高相似度低於門檻。"
        else:
            status = "ok"
            reason = None

        return {
            "status": status,
            "mode": "documents",
            "question": question,
            "best_dense_score": best_score,
            "document_min_score": (
                self.config.document_min_score
            ),
            "reason": reason,
            "documents": documents,
        }

    def search(
        self,
        question: str,
        mode: str,
        selected_document: dict[str, Any] | None,
    ) -> dict[str, Any]:
        question = normalize_query(question)
        actual_mode = mode

        routing_info: dict[str, Any] | None = None

        if mode == "auto":
            actual_mode, routing_info = (
                self.semantic_query_mode(
                    question,
                    has_selected_document=(
                        selected_document is not None
                    ),
                )
            )

        if actual_mode == "overview":
            result = self.overview_context(
                question, selected_document
            )
        elif actual_mode == "documents":
            result = self.document_discovery_context(
                question
            )
        elif actual_mode == "detail":
            result = self.detail_context(
                question, selected_document
            )
        else:
            raise ValueError(
                f"不支援的搜尋模式：{actual_mode}"
            )

        result["requested_mode"] = mode
        result["actual_mode"] = actual_mode
        if routing_info is not None:
            result["routing"] = routing_info
        return self.limit_context(result)

    def limit_context(
        self, result: dict[str, Any]
    ) -> dict[str, Any]:
        remaining = self.config.max_context_chars

        if result.get("actual_mode") == "detail":
            limited: list[dict[str, Any]] = []
            for original in result.get("results", []):
                if remaining <= 0:
                    break
                item = dict(original)
                content = str(item.get("content") or "")
                if len(content) > remaining:
                    content = content[:remaining].rstrip()
                    item["content_truncated"] = True
                else:
                    item["content_truncated"] = False
                item["content"] = content
                remaining -= len(content)
                limited.append(item)
            result["results"] = limited

        elif result.get("actual_mode") == "overview":
            limited_contexts: list[dict[str, Any]] = []
            for original_context in result.get(
                "contexts", []
            ):
                context = dict(original_context)
                chunks: list[dict[str, Any]] = []
                for original in context.get("chunks", []):
                    if remaining <= 0:
                        break
                    item = dict(original)
                    content = str(item.get("content") or "")
                    if len(content) > remaining:
                        content = content[:remaining].rstrip()
                        item["content_truncated"] = True
                    else:
                        item["content_truncated"] = False
                    item["content"] = content
                    remaining -= len(content)
                    chunks.append(item)
                context["chunks"] = chunks
                limited_contexts.append(context)
                if remaining <= 0:
                    break
            result["contexts"] = limited_contexts

        result["context_character_limit"] = (
            self.config.max_context_chars
        )
        result["context_characters_used"] = (
            self.config.max_context_chars - remaining
        )
        return result


def format_score(value: Any) -> str:
    if value is None:
        return "-"
    return f"{safe_float(value):.4f}"


def print_documents(
    documents: Sequence[dict[str, Any]]
) -> None:
    if not documents:
        print("目前沒有文件。")
        return

    for index, document in enumerate(documents, start=1):
        print(
            f"[{index}] {document.get('source_file')} "
            f"｜標題：{document.get('title')} "
            f"｜頁數：{document.get('page_count')} "
            f"｜chunks：{document.get('indexed_chunk_count')}"
        )
        print(
            f"    document_id：{document.get('document_id')}"
        )


def print_search_result(result: dict[str, Any]) -> None:
    print(f"模式：{result.get('actual_mode')}")
    print(f"狀態：{result.get('status')}")

    if result.get("best_dense_score") is not None:
        print(
            "最高 E5 分數："
            f"{format_score(result.get('best_dense_score'))}"
        )

    if result.get("best_rerank_score") is not None:
        print(
            "最高 Reranker 分數："
            f"{format_score(result.get('best_rerank_score'))} "
            f"（門檻 "
            f"{format_score(result.get('min_rerank_score'))}）"
        )

    if result.get("reason"):
        print(f"原因：{result['reason']}")

    print()
    mode = result.get("actual_mode")

    if mode == "detail":
        records = result.get("results", [])
        if not records:
            print("沒有可用內容。")
            return

        for item in records:
            print(
                f"--- 結果 {item.get('rank')} "
                f"｜rerank="
                f"{format_score(item.get('rerank_score'))} "
                f"｜e5={format_score(item.get('dense_score'))} "
                f"｜{item.get('source_file')} "
                f"｜第 {item.get('page')} 頁 ---"
            )
            print(item.get("content") or "")
            print()

    elif mode == "overview":
        contexts = result.get("contexts", [])
        if not contexts:
            print("沒有可用文件內容。")
            return

        for context in contexts:
            document = context.get("document", {})
            print(
                f"=== 文件：{document.get('source_file')} "
                f"｜標題：{document.get('title')} ==="
            )
            summary = str(
                context.get("summary") or ""
            ).strip()
            topics = context.get("topics") or []
            if summary:
                print(f"既有摘要：{summary}")
            if topics:
                print(
                    f"主題：{'、'.join(map(str, topics))}"
                )
            print("代表性內容：")
            for item in context.get("chunks", []):
                print(
                    f"[第 {item.get('page')} 頁] "
                    f"{item.get('content') or ''}"
                )
                print()

    elif mode == "documents":
        documents = result.get("documents", [])
        if not documents:
            print("沒有相關文件。")
            return
        for item in documents:
            print(
                f"[{item.get('rank')}] "
                f"e5={format_score(item.get('dense_score'))} "
                f"｜{item.get('source_file')} "
                f"｜標題：{item.get('title')} "
                f"｜頁數：{item.get('page_count')}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E5 + FAISS + BGE Reranker 的 RAG 搜尋器"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="要搜尋的問題",
    )
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=DEFAULT_VECTOR_DIR,
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "overview", "detail", "documents"),
        default="auto",
    )
    parser.add_argument("--source-file")
    parser.add_argument("--document-id")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )
    parser.add_argument(
        "--dense-candidates",
        type=int,
        default=DEFAULT_DENSE_CANDIDATES,
        help="E5 先召回多少候選給 reranker",
    )
    parser.add_argument(
        "--document-top-k",
        type=int,
        default=DEFAULT_DOCUMENT_TOP_K,
    )
    parser.add_argument(
        "--document-min-score",
        type=float,
        default=DEFAULT_DOCUMENT_MIN_SCORE,
        help="只用於跨文件搜尋，不用於細節拒答",
    )
    parser.add_argument(
        "--min-rerank-score",
        type=float,
        default=DEFAULT_MIN_RERANK_SCORE,
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=DEFAULT_RERANKER_BATCH_SIZE,
    )
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=DEFAULT_RERANKER_MAX_LENGTH,
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="僅供除錯；正式使用不建議",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--reranker-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--interactive", action="store_true"
    )
    parser.add_argument(
        "--list-documents", action="store_true"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k < 1:
        raise ValueError("--top-k 必須大於 0")
    if args.dense_candidates < args.top_k:
        raise ValueError(
            "--dense-candidates 不可小於 --top-k"
        )
    if args.document_top_k < 1:
        raise ValueError(
            "--document-top-k 必須大於 0"
        )
    if not 0.0 <= args.min_rerank_score <= 1.0:
        raise ValueError(
            "--min-rerank-score 必須介於 0 到 1"
        )
    if args.reranker_batch_size < 1:
        raise ValueError(
            "--reranker-batch-size 必須大於 0"
        )
    if args.reranker_max_length < 128:
        raise ValueError(
            "--reranker-max-length 不可小於 128"
        )
    if args.max_context_chars < 500:
        raise ValueError(
            "--max-context-chars 不可小於 500"
        )
    if (
        not args.question
        and not args.interactive
        and not args.list_documents
    ):
        raise ValueError(
            "請提供問題，或使用 --interactive／--list-documents"
        )


def run_interactive(
    searcher: RAGSearcher,
    args: argparse.Namespace,
    selected_document: dict[str, Any] | None,
) -> int:
    print("互動模式：輸入 :quit 離開，:docs 查看文件。")
    if selected_document:
        print(
            f"目前文件：{selected_document.get('source_file')}"
        )
    else:
        print("目前搜尋全部 PDF。")
    print()

    while True:
        try:
            question = input("問題> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question:
            continue
        if question in {
            ":quit", ":q", "quit", "exit"
        }:
            return 0
        if question == ":docs":
            print_documents(searcher.list_documents())
            print()
            continue

        try:
            result = searcher.search(
                question,
                args.mode,
                selected_document,
            )
            if args.json:
                print(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print_search_result(result)
        except Exception as error:
            print(f"搜尋失敗：{error}", file=sys.stderr)
        print()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))

    logger = setup_logging(args.verbose)
    config = SearchConfig(
        vector_dir=args.vector_dir.expanduser(),
        top_k=args.top_k,
        dense_candidates=args.dense_candidates,
        document_top_k=args.document_top_k,
        document_min_score=args.document_min_score,
        min_rerank_score=args.min_rerank_score,
        max_context_chars=args.max_context_chars,
        device=args.device,
        dtype=args.dtype,
        reranker_model=args.reranker_model,
        reranker_device=args.reranker_device,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        disable_reranker=args.no_reranker,
    )

    try:
        searcher = RAGSearcher(config, logger)

        if args.list_documents:
            documents = searcher.list_documents()
            if args.json:
                print(
                    json.dumps(
                        {"documents": documents},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print_documents(documents)
            if not args.question and not args.interactive:
                return 0

        selected_document = searcher.resolve_document(
            args.document_id,
            args.source_file,
        )

        if args.interactive:
            return run_interactive(
                searcher, args, selected_document
            )

        result = searcher.search(
            args.question,
            args.mode,
            selected_document,
        )

        if args.json:
            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print_search_result(result)
        return 0

    except torch.OutOfMemoryError:
        logger.exception(
            "CUDA 記憶體不足。可用 "
            "--reranker-device cpu 或降低 "
            "--reranker-batch-size 1。"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 3
    except Exception:
        logger.exception("RAG 搜尋失敗")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
