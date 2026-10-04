#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_chat_engine.py

給最終 LLM GUI 使用的內部 RAG 引擎。

重要：
- 不會透過 subprocess 執行 rag_search.py。
- Embedding、FAISS、Reranker 只載入一次。
- GUI 直接 import RAGChatEngine 並呼叫 chat()。
- LLM 由 GUI 傳入 generate_fn，因此可接 Transformers、llama.cpp、
  TensorRT-LLM 或其他本機模型，不綁定特定載入方式。

專案結構建議：
  ./
    rag_search_reranker.py
    rag_chat_engine.py
    flash/vector/...

GUI 使用概念：

    from rag_chat_engine import RAGChatEngine

    engine = RAGChatEngine(source_file="SOP.pdf")

    def my_llm_generate(messages):
        # 在這裡呼叫你的本機 LLM。
        return local_llm.generate(messages)

    response = engine.chat(
        question="設備啟動前要檢查什麼？",
        history=[],
        generate_fn=my_llm_generate,
    )

    print(response["answer"])
    print(response["sources"])
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_VECTOR_DIR = DEFAULT_ROOT / "flash/vector"

GenerateFunction = Callable[[list[dict[str, str]]], str]


@dataclass(frozen=True)
class RAGEngineSettings:
    vector_dir: Path = DEFAULT_VECTOR_DIR
    source_file: str | None = None
    document_id: str | None = None

    top_k: int = 6
    dense_candidates: int = 20
    document_top_k: int = 3

    document_min_score: float = 0.55
    min_rerank_score: float = 0.25

    max_context_chars: int = 9000
    max_history_turns: int = 6

    embedding_device: str = "auto"
    reranker_device: str = "auto"
    dtype: str = "float16"

    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = 2
    reranker_max_length: int = 1024


class RAGChatEngine:
    """
    最終 GUI 中常駐的 RAG 引擎。

    一個 instance 對應一組已載入的：
    - E5 embedding model
    - chunk/document FAISS indexes
    - BGE reranker（第一次 detail 問題時才延遲載入）
    """

    def __init__(
        self,
        settings: RAGEngineSettings | None = None,
        *,
        source_file: str | None = None,
        document_id: str | None = None,
        logger: logging.Logger | None = None,
    ):
        if settings is None:
            settings = RAGEngineSettings(
                source_file=source_file,
                document_id=document_id,
            )
        elif source_file is not None or document_id is not None:
            raise ValueError(
                "使用 settings 時，不可同時傳入 source_file/document_id"
            )

        self.settings = settings
        self.logger = logger or self._make_logger()
        self._lock = threading.RLock()

        self._search_module = self._load_search_module()
        self._searcher = self._create_searcher()
        self._selected_document = self._searcher.resolve_document(
            self.settings.document_id,
            self.settings.source_file,
        )

    @staticmethod
    def _make_logger() -> logging.Logger:
        logger = logging.getLogger("rag_chat_engine")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    @staticmethod
    def _candidate_search_files() -> list[Path]:
        current_dir = Path(__file__).resolve().parent
        return [
            current_dir / "rag_search_reranker.py",
            current_dir / "rag_search.py",
        ]

    def _load_search_module(self) -> Any:
        search_path = next(
            (
                path
                for path in self._candidate_search_files()
                if path.is_file()
            ),
            None,
        )
        if search_path is None:
            names = ", ".join(
                str(path) for path in self._candidate_search_files()
            )
            raise FileNotFoundError(
                f"找不到 RAG 搜尋模組，預期其中之一存在：{names}"
            )

        module_name = "_unlimited_ocr_rag_search"
        spec = importlib.util.spec_from_file_location(
            module_name,
            search_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"無法載入搜尋模組：{search_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        required = ("RAGSearcher", "SearchConfig")
        missing = [
            name for name in required if not hasattr(module, name)
        ]
        if missing:
            raise ImportError(
                f"{search_path} 缺少新版介面：{', '.join(missing)}"
            )

        # 防止不小心載入只有 E5、沒有 reranker 的舊版。
        config_fields = getattr(
            module.SearchConfig,
            "__dataclass_fields__",
            {},
        )
        if "min_rerank_score" not in config_fields:
            raise ImportError(
                f"{search_path} 是舊版搜尋程式，沒有 reranker。"
            )

        self.logger.info("使用 RAG 搜尋模組：%s", search_path)
        return module

    def _create_searcher(self) -> Any:
        s = self.settings
        config = self._search_module.SearchConfig(
            vector_dir=s.vector_dir.expanduser(),
            top_k=s.top_k,
            dense_candidates=s.dense_candidates,
            document_top_k=s.document_top_k,
            document_min_score=s.document_min_score,
            min_rerank_score=s.min_rerank_score,
            max_context_chars=s.max_context_chars,
            device=s.embedding_device,
            dtype=s.dtype,
            reranker_model=s.reranker_model,
            reranker_device=s.reranker_device,
            reranker_batch_size=s.reranker_batch_size,
            reranker_max_length=s.reranker_max_length,
            disable_reranker=False,
        )
        return self._search_module.RAGSearcher(
            config,
            self.logger,
        )

    @property
    def selected_document(self) -> dict[str, Any] | None:
        if self._selected_document is None:
            return None
        return dict(self._selected_document)

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._searcher.list_documents()

    def select_document(
        self,
        *,
        source_file: str | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any] | None:
        """
        GUI 切換目前 PDF 時呼叫。

        source_file/document_id 都不傳時，取消文件限制，
        後續問題會搜尋全部 PDF。
        """
        with self._lock:
            self._selected_document = (
                self._searcher.resolve_document(
                    document_id,
                    source_file,
                )
                if source_file or document_id
                else None
            )
            return self.selected_document

    def refresh_indexes(self) -> None:
        """
        OCR 或 build_rag_index.py 更新完成後，由 GUI 呼叫。

        會重新讀取 FAISS/metadata。因 embedding 模型名稱相同，
        Hugging Face 本機快取仍會直接使用，但此方法會重新建立
        searcher instance。
        """
        with self._lock:
            old_source_file = (
                self._selected_document.get("source_file")
                if self._selected_document
                else self.settings.source_file
            )
            old_document_id = (
                self._selected_document.get("document_id")
                if self._selected_document
                else self.settings.document_id
            )

            self._searcher = self._create_searcher()

            try:
                self._selected_document = (
                    self._searcher.resolve_document(
                        old_document_id,
                        None,
                    )
                    if old_document_id
                    else (
                        self._searcher.resolve_document(
                            None,
                            old_source_file,
                        )
                        if old_source_file
                        else None
                    )
                )
            except ValueError:
                self._selected_document = None

    def retrieve(
        self,
        question: str,
        *,
        mode: str = "auto",
        source_file: str | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        """
        只做 RAG，不呼叫 LLM。

        GUI 平常不需要直接呼叫此方法；主要供測試或顯示搜尋細節。
        """
        with self._lock:
            selected = self._resolve_request_document(
                source_file=source_file,
                document_id=document_id,
            )
            return self._searcher.search(
                question,
                mode,
                selected,
            )

    def _resolve_request_document(
        self,
        *,
        source_file: str | None,
        document_id: str | None,
    ) -> dict[str, Any] | None:
        if source_file or document_id:
            return self._searcher.resolve_document(
                document_id,
                source_file,
            )
        return self._selected_document

    @staticmethod
    def _clean_history(
        history: Sequence[Any] | None,
        max_turns: int,
    ) -> list[dict[str, str]]:
        """
        接受兩種 GUI 常見 history：

        1. OpenAI messages：
           [{"role": "user", "content": "..."}, ...]

        2. Gradio tuples：
           [("問題", "回答"), ...]
        """
        if not history or max_turns <= 0:
            return []

        normalized: list[dict[str, str]] = []

        for item in history:
            if isinstance(item, dict):
                role = str(item.get("role") or "").strip()
                content = str(item.get("content") or "").strip()
                if role in {"user", "assistant"} and content:
                    normalized.append(
                        {"role": role, "content": content}
                    )
                continue

            if (
                isinstance(item, (tuple, list))
                and len(item) >= 2
            ):
                user_text = str(item[0] or "").strip()
                assistant_text = str(item[1] or "").strip()
                if user_text:
                    normalized.append(
                        {"role": "user", "content": user_text}
                    )
                if assistant_text:
                    normalized.append(
                        {
                            "role": "assistant",
                            "content": assistant_text,
                        }
                    )

        # 一個 turn 約為 user + assistant 兩則訊息。
        return normalized[-max_turns * 2 :]

    @staticmethod
    def _source_key(
        source_file: str,
        page: Any,
    ) -> tuple[str, int]:
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            page_number = 0
        return source_file, page_number

    @classmethod
    def _extract_context_and_sources(
        cls,
        retrieval: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        mode = retrieval.get("actual_mode")
        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        seen_sources: set[tuple[str, int]] = set()

        def add_chunk(item: dict[str, Any]) -> None:
            content = str(item.get("content") or "").strip()
            if not content:
                return

            source_file = str(
                item.get("source_file") or "未知文件"
            )
            page = item.get("page")
            key = cls._source_key(source_file, page)

            if key not in seen_sources:
                seen_sources.add(key)
                sources.append(
                    {
                        "source_file": source_file,
                        "page": key[1],
                    }
                )

            source_number = sources.index(
                {
                    "source_file": source_file,
                    "page": key[1],
                }
            ) + 1

            context_parts.append(
                f"[來源{source_number}｜{source_file}｜第 {key[1]} 頁]\n"
                f"{content}"
            )

        if mode == "detail":
            for item in retrieval.get("results", []):
                add_chunk(item)

        elif mode == "overview":
            for document_context in retrieval.get(
                "contexts",
                [],
            ):
                document = document_context.get(
                    "document",
                    {},
                )
                source_file = str(
                    document.get("source_file")
                    or "未知文件"
                )

                summary = str(
                    document_context.get("summary") or ""
                ).strip()
                topics = document_context.get("topics") or []

                if summary:
                    context_parts.append(
                        f"[文件摘要｜{source_file}]\n{summary}"
                    )
                if topics:
                    context_parts.append(
                        f"[文件主題｜{source_file}]\n"
                        f"{'、'.join(map(str, topics))}"
                    )

                for item in document_context.get(
                    "chunks",
                    [],
                ):
                    add_chunk(item)

        elif mode == "documents":
            for rank, document in enumerate(
                retrieval.get("documents", []),
                start=1,
            ):
                context_parts.append(
                    f"[候選文件{rank}]\n"
                    f"檔名：{document.get('source_file')}\n"
                    f"標題：{document.get('title')}\n"
                    f"頁數：{document.get('page_count')}\n"
                    f"E5 分數：{document.get('dense_score')}"
                )

        return "\n\n".join(context_parts), sources

    @staticmethod
    def _system_prompt(mode: str) -> str:
        common = """
你是文件問答助手。只能根據本次提供的文件內容回答，不可使用未出現在內容中的知識補充或猜測。

回答規則：
1. 使用繁體中文。
2. 先直接回答問題，再補充必要步驟或說明。
3. 每個重要敘述後標示來源，例如：[來源1]。
4. 不要捏造頁碼、規格、步驟或結論。
5. 文件只提供部分答案時，要明確說明哪些部分有資料、哪些部分沒有資料。
6. 不要提到向量、FAISS、embedding、reranker 或內部搜尋流程。
""".strip()

        if mode == "overview":
            return (
                common
                + "\n7. 此題是文件概要題。整理文件目的、主要內容、"
                "重要流程及注意事項，不要只摘要其中一小段。"
            )

        if mode == "documents":
            return (
                common
                + "\n7. 此題是在判斷哪份文件較相關。只根據候選文件"
                "資訊說明，不要假設尚未提供的文件內容。"
            )

        return (
            common
            + "\n7. 此題是文件細節題。優先給出可操作、精確的答案。"
        )

    def build_messages(
        self,
        *,
        question: str,
        retrieval: dict[str, Any],
        history: Sequence[Any] | None = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        """
        將搜尋結果整理成 LLM 可直接使用的 messages。
        """
        context, sources = self._extract_context_and_sources(
            retrieval
        )
        mode = str(retrieval.get("actual_mode") or "detail")

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self._system_prompt(mode),
            }
        ]
        messages.extend(
            self._clean_history(
                history,
                self.settings.max_history_turns,
            )
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"使用者問題：\n{question}\n\n"
                    f"文件內容：\n{context}"
                ),
            }
        )
        return messages, sources

    @staticmethod
    def _insufficient_answer(
        retrieval: dict[str, Any],
    ) -> str:
        selected_file = retrieval.get(
            "selected_source_file"
        )
        if selected_file:
            return (
                f"目前在《{selected_file}》中找不到足夠資訊回答這個問題。"
            )
        return "目前提供的文件中找不到足夠資訊回答這個問題。"

    @staticmethod
    def _append_source_footer(
        answer: str,
        sources: Sequence[dict[str, Any]],
    ) -> str:
        """
        如果 LLM 忘記列來源，仍在答案底部補上實際使用的來源。
        """
        answer = answer.strip()
        if not sources:
            return answer

        footer_lines = ["", "參考來源："]
        for index, source in enumerate(sources, start=1):
            footer_lines.append(
                f"- [來源{index}] "
                f"{source['source_file']}，第 {source['page']} 頁"
            )

        return answer + "\n" + "\n".join(footer_lines)

    def chat(
        self,
        question: str,
        generate_fn: GenerateFunction,
        *,
        history: Sequence[Any] | None = None,
        mode: str = "auto",
        source_file: str | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        """
        GUI 的主要呼叫入口。

        generate_fn 必須接受 OpenAI messages 格式：
            [{"role": "system", "content": "..."}, ...]

        並回傳 LLM 產生的純文字答案。
        """
        if not callable(generate_fn):
            raise TypeError("generate_fn 必須是可呼叫函式")

        question = re.sub(r"\s+", " ", question).strip()
        if not question:
            raise ValueError("問題不可為空")

        with self._lock:
            retrieval = self.retrieve(
                question,
                mode=mode,
                source_file=source_file,
                document_id=document_id,
            )

            if retrieval.get("status") != "ok":
                return {
                    "answer": self._insufficient_answer(
                        retrieval
                    ),
                    "status": "insufficient_context",
                    "mode": retrieval.get("actual_mode"),
                    "sources": [],
                    "retrieval": retrieval,
                    "llm_called": False,
                }

            messages, sources = self.build_messages(
                question=question,
                retrieval=retrieval,
                history=history,
            )

            answer = generate_fn(messages)
            if not isinstance(answer, str):
                raise TypeError(
                    "generate_fn 必須回傳字串"
                )
            answer = answer.strip()
            if not answer:
                raise RuntimeError("LLM 回傳空白答案")

            answer = self._append_source_footer(
                answer,
                sources,
            )

            return {
                "answer": answer,
                "status": "ok",
                "mode": retrieval.get("actual_mode"),
                "sources": sources,
                "retrieval": retrieval,
                "messages": messages,
                "llm_called": True,
            }
