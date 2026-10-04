#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py

Unlimited-OCR + FAISS RAG + Qwen3.5 的單一 Gradio GUI。

啟動：
    conda activate LLM
    cd Unlimited-OCR
    python app.py --model /完整路徑/Qwen3.5-9B.Q4_K_M.gguf

瀏覽器：
    http://127.0.0.1:7860

目前功能：
- 只啟動一個 app.py。
- E5、FAISS、BGE Reranker、Qwen3.5 都在同一 Python 程序內。
- 選擇已完成 OCR 與索引的 PDF。
- 快速回答／深度思考切換。
- Reranker 達標時依文件回答；不足時可切換為模型一般回答。
- 回答以逐字串流方式顯示，並附 PDF 名稱與頁碼。
- 上傳 PDF 後，自動等待 OCR、建立 FAISS 索引並重新載入模型。
- 回答以真正的 llama.cpp 串流逐字更新。
- 仍可手動重新讀取已建立完成的 FAISS 索引。

自動上傳流程：
- PDF 先以暫存檔原子方式放入 input/，避免 OCR 讀到未複製完成的檔案。
- OCR service 執行中時，GUI 等待它完成；未執行時，GUI 自動執行一次 OCR。
- 建立索引前暫時釋放 Qwen、E5 與 Reranker，降低 Jetson 記憶體壓力。
- 自動以目錄／查修步驟切 chunk、建立 E5 索引，完成後重新載入 RAG 與 Qwen。
- 新 PDF 自動出現在文件選單，不需再使用終端機。
"""

from __future__ import annotations

import os

# 必須在匯入 Gradio、Transformers、SentenceTransformers 前設定。
# 預設完全離線：不查 Hugging Face、不送遙測、不建立公開連結。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterator, Sequence

import gradio as gr
import torch

from local_llm import LocalQwenLLM, resolve_model_path
from rag_chat_engine import RAGChatEngine, RAGEngineSettings


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
VECTOR_DIR = Path(os.getenv(
    "YUNTECH_RAG_VECTOR_DIR",
    str(ROOT / "flash/vector"),
)).expanduser()
RAG_DIR = ROOT / "flash/rag"
OCR_CATALOG_PATH = RAG_DIR / "catalog.json"
TOC_CHUNK_SCRIPT = ROOT / "toc_step_aware_chunker.py"
TOC_CHUNKS_FILE = RAG_DIR / "all_chunks_toc_step.jsonl"
BUILD_INDEX_SCRIPT = ROOT / "build_rag_index.py"
OCR_PIPELINE_SCRIPT = ROOT / "ocr_pipeline.py"

LOGGER = logging.getLogger("unlimited_ocr_gui")


def setup_logging() -> None:
    if LOGGER.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def service_status() -> str:
    """回傳 OCR systemd service 的目前狀態，不因失敗中止 GUI。"""
    try:
        completed = subprocess.run(
            [
                "systemctl",
                "is-active",
                "unlimited-ocr.service",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return "未知"

    status = completed.stdout.strip()
    if status == "active":
        return "執行中"
    if status:
        return status
    return "未執行"


def normalize_gradio_history(
    history: Sequence[Any] | None,
) -> list[dict[str, str]]:
    """只保留文字 user/assistant 訊息，供 RAGChatEngine 當歷史。"""
    output: list[dict[str, str]] = []

    for message in history or []:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role") or "")
        content = message.get("content")

        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue

        content = content.strip()
        if content:
            output.append(
                {"role": role, "content": content}
            )

    return output


class AppRuntime:
    """
    GUI 全域模型狀態。

    Gradio 的 queue 會限制同時只處理一件模型工作；
    類別本身再加 RLock，避免按鈕事件同時修改模型。
    """

    def __init__(
        self,
        *,
        model_path: Path,
        n_ctx: int,
        n_batch: int,
        rag_context_chars: int,
        reranker_device: str,
        min_rerank_score: float,
        stream_char_delay: float,
        ocr_timeout: int,
        index_batch_size: int,
        verbose_llama: bool,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.rag_context_chars = rag_context_chars
        self.reranker_device = reranker_device
        self.min_rerank_score = float(min_rerank_score)
        self.stream_char_delay = max(0.0, float(stream_char_delay))
        self.ocr_timeout = max(30, int(ocr_timeout))
        self.index_batch_size = max(1, int(index_batch_size))
        self.verbose_llama = verbose_llama

        self.rag: RAGChatEngine | None = None
        self.llm: LocalQwenLLM | None = None

        self._lock = threading.RLock()
        self._loading = False

    def _rag_settings(
        self,
        source_file: str | None = None,
    ) -> RAGEngineSettings:
        return RAGEngineSettings(
            vector_dir=VECTOR_DIR,
            source_file=source_file,
            max_context_chars=self.rag_context_chars,
            reranker_device=self.reranker_device,
            min_rerank_score=self.min_rerank_score,
        )

    def ensure_loaded(self) -> None:
        with self._lock:
            if self.rag is not None and self.llm is not None:
                return
            if self._loading:
                raise RuntimeError("模型正在載入中")
            self._loading = True

            try:
                LOGGER.info("載入 RAG...")
                self.rag = RAGChatEngine(
                    settings=self._rag_settings()
                )

                LOGGER.info(
                    "載入 Qwen：%s",
                    self.model_path,
                )
                if not hasattr(
                    LocalQwenLLM,
                    "stream_generate_fast",
                ):
                    raise RuntimeError(
                        "目前 local_llm.py 是舊版，"
                        "缺少 stream_generate_fast()。"
                        "請一起替換新版 local_llm.py。"
                    )
                self.llm = LocalQwenLLM(
                    self.model_path,
                    n_ctx=self.n_ctx,
                    n_batch=self.n_batch,
                    n_ubatch=min(self.n_batch, 128),
                    n_gpu_layers=-1,
                    flash_attn=True,
                    default_mode="fast",
                    verbose=self.verbose_llama,
                )
                LOGGER.info("GUI 模型載入完成")
            except Exception:
                self.rag = None
                self.llm = None
                raise
            finally:
                self._loading = False

    def release_models(self) -> None:
        """
        釋放 GUI 內的 Qwen、E5、Reranker 與 FAISS 物件。

        新 PDF 建立索引時會啟動另一個 embedding 程序；
        Jetson 採共用記憶體，先釋放模型可降低 OOM 風險。
        """
        with self._lock:
            old_rag = self.rag
            old_llm = self.llm
            self.rag = None
            self.llm = None

            if old_llm is not None:
                try:
                    model = getattr(old_llm, "model", None)
                    close_fn = getattr(model, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception:
                    LOGGER.warning(
                        "關閉 Qwen context 時發生非致命錯誤",
                        exc_info=True,
                    )

            del old_rag
            del old_llm

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

    def reload_all_models(self) -> list[str]:
        """完整重新載入 RAG 與 Qwen，回傳目前文件名稱。"""
        self.release_models()
        self.ensure_loaded()
        return self.document_names()

    def document_names(self) -> list[str]:
        self.ensure_loaded()
        assert self.rag is not None

        return [
            str(item["source_file"])
            for item in self.rag.list_documents()
            if item.get("source_file")
        ]

    def reload_rag(self) -> list[str]:
        """
        重新讀取 vector_toc_step 中已經存在的新索引。

        先釋放舊 RAG 模型，避免同時保留兩份 E5/Reranker。
        Qwen 保留常駐。
        """
        with self._lock:
            old_rag = self.rag
            self.rag = None
            del old_rag

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            LOGGER.info("重新載入 RAG 索引...")
            self.rag = RAGChatEngine(
                settings=self._rag_settings()
            )
            return self.document_names()

    @staticmethod
    def _build_general_messages(
        question: str,
        history: Sequence[Any] | None,
    ) -> list[dict[str, str]]:
        """
        文件相關度不足時，改用 Qwen 一般知識回答。

        這組 prompt 明確禁止模型假裝答案來自 PDF，也不允許使用
        [來源X] 或虛構頁碼。
        """
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "你是一個一般知識助手。"
                    "這次文件檢索沒有找到足夠相關的內容，"
                    "因此只能根據模型的一般知識回答。"
                    "請使用繁體中文直接回答。"
                    "不要聲稱答案來自 PDF、SOP 或任何文件；"
                    "不要輸出 [來源1]、頁碼或虛構引用。"
                    "若無法確定，必須清楚說明不確定，不能猜測。"
                    "若問題涉及安全、醫療、法律、財務或設備控制，"
                    "請提醒使用者依正式文件或專業人員確認。"
                ),
            }
        ]
        messages.extend(
            normalize_gradio_history(history)[-12:]
        )
        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )
        return messages

    def _normalize_query_with_llm(
        self,
        question: str,
        retrieval: dict[str, Any],
    ) -> str | None:
        """
        Reranker 判斷檢索信心不足時，用已載入的 Qwen 檢查：那些分數不足的
        候選段落，會不會其實就是使用者要問的內容，只是講法不同（語音辨識
        錯字、同音字誤聽、口語化描述）。如果是，改寫成更精確、更適合拿去
        重新檢索的查詢句；不相關就回答「無」，維持原本「找不到」的行為。

        只在檢索信心不足時才呼叫，不影響原本檢索成功的情況，也沒有額外的
        模型載入成本（沿用 GUI 常駐的 self.llm）。
        """
        assert self.llm is not None
        candidates = list(retrieval.get("results") or [])[:5]
        if not candidates:
            return None

        excerpts = "\n\n".join(
            f"[段落{i}] {str(item.get('content') or '').strip()[:200]}"
            for i, item in enumerate(candidates, start=1)
            if str(item.get("content") or "").strip()
        )
        if not excerpts:
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "你是搜尋查詢校正器。系統剛才用使用者的原始問題檢索文件，"
                    "但信心不足，以下是分數不足的候選段落。"
                    "使用者的問題可能包含語音辨識錯字、同音字誤聽或口語化描述，"
                    "導致用詞和文件不一致。"
                    "如果某個候選段落其實就是使用者要問的內容，"
                    "請用該段落的實際用詞，把使用者的問題改寫成一句更精確、"
                    "更適合重新檢索的查詢句；如果都不相關，只回答「無」。"
                    "只能輸出改寫後的查詢句或「無」，不要輸出其他文字或解釋。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"候選段落：\n{excerpts}\n\n使用者問題：{question}"
                ),
            },
        ]
        try:
            result = self.llm.generate_result(
                messages,
                mode="fast",
                max_tokens=60,
            )
            answer = str(result.get("answer") or "").strip()
        except Exception:
            LOGGER.warning("查詢正規化呼叫失敗", exc_info=True)
            return None

        if not answer or answer in {"無", "無。", "無相關"}:
            return None
        if answer == question.strip():
            return None
        return answer

    @staticmethod
    def _build_line_indexed_context(
        retrieval: dict[str, Any],
    ) -> tuple[str, set[str], list[dict[str, Any]], str]:
        """
        把檢索結果切成逐行、附上可引用 ID（例如 [S1-3]＝來源1第3行），
        取代原本整段落當一個來源的做法。有了逐行 ID，才有辦法在生成後
        驗證每個引用是否真的存在，而不是只信任 LLM 自己標的 [來源1]。

        回傳：(帶引用 ID 的文件內容文字, 合法 ID 集合, 來源清單, 備援原文)
        """
        results = list(retrieval.get("results") or [])
        context_blocks: list[str] = []
        valid_ids: set[str] = set()
        sources: list[dict[str, Any]] = []
        seen_sources: set[tuple[str, int]] = set()
        fallback_lines: list[str] = []

        for chunk_index, item in enumerate(results, start=1):
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            source_file = str(item.get("source_file") or "未知文件")
            try:
                page = int(item.get("page") or 0)
            except (TypeError, ValueError):
                page = 0
            key = (source_file, page)
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append({"source_file": source_file, "page": page})

            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            numbered = []
            for line_index, line_text in enumerate(lines, start=1):
                cid = f"S{chunk_index}-{line_index}"
                valid_ids.add(cid)
                numbered.append(f"[{cid}] {line_text}")
                if chunk_index == 1:
                    fallback_lines.append(line_text)
            context_blocks.append(
                f"【來源{chunk_index}｜{source_file}｜第{page}頁】\n" + "\n".join(numbered)
            )

        context_text = "\n\n".join(context_blocks)
        fallback_text = "\n".join(fallback_lines)
        return context_text, valid_ids, sources, fallback_text

    @staticmethod
    def _grounded_system_prompt() -> str:
        return (
            "你是文件依據型問答助手，只能根據下面提供的內容回答，不可以使用其他知識。\n"
            "內容前面都標有 [S編號-行號] 這種標記。請在答案的關鍵步驟、數值、結論"
            "後面盡量標上對應的引用，方便使用者對照原文；標記必須是下面文件裡"
            "實際出現過的，不可以自己編號，但不需要每一句都加引用。\n"
            "只要提供的內容裡有跟問題相關的資訊，就根據這些資訊回答，"
            "即使只有部分相關也照樣回答，並說明哪些細節文件沒有明確提到；"
            "只有在提供的內容完全沒有相關資訊時，才回答「文件內容不足以回答這個問題」，"
            "不要因為不確定細節就直接說不足。\n"
            "使用繁體中文，不使用 Markdown 標題、星號或表格。"
        )

    def _build_grounded_messages(
        self,
        *,
        question: str,
        retrieval: dict[str, Any],
        history: Sequence[Any] | None,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]], set[str], str]:
        context_text, valid_ids, sources, fallback_text = (
            self._build_line_indexed_context(retrieval)
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._grounded_system_prompt()},
        ]
        messages.extend(normalize_gradio_history(history)[-12:])
        messages.append(
            {
                "role": "user",
                "content": f"文件內容：\n{context_text}\n\n使用者問題：\n{question}",
            }
        )
        return messages, sources, valid_ids, fallback_text

    def prepare_stream(
        self,
        *,
        question: str,
        history: Sequence[Any] | None,
        source_file: str,
    ) -> dict[str, Any]:
        """
        先執行 RAG，再決定回答路徑：

        - Reranker >= 閾值：使用 RAG context。
        - Reranker < 閾值：先嘗試用 Qwen 把問題正規化成文件的實際用詞，
          重新檢索一次；仍然不足才建立一般知識 prompt，交由 GUI 決定
          是否允許 Qwen 自行回答。
        """
        self.ensure_loaded()
        assert self.rag is not None
        assert self.llm is not None

        clean_history = normalize_gradio_history(history)
        retrieval = self.rag.retrieve(
            question,
            source_file=source_file,
        )

        if retrieval.get("status") != "ok":
            normalized_query = self._normalize_query_with_llm(
                question, retrieval
            )
            if normalized_query:
                retried = self.rag.retrieve(
                    normalized_query,
                    source_file=source_file,
                )
                if retried.get("status") == "ok":
                    LOGGER.info(
                        "查詢正規化：%r -> %r",
                        question,
                        normalized_query,
                    )
                    retrieval = retried

        if retrieval.get("status") != "ok":
            return {
                "status": "general_answer",
                "answer_source": "general",
                "mode": retrieval.get("actual_mode"),
                "retrieval": retrieval,
                "messages": self._build_general_messages(
                    question,
                    clean_history,
                ),
                "sources": [],
            }

        messages, sources, valid_ids, fallback_text = self._build_grounded_messages(
            question=question,
            retrieval=retrieval,
            history=clean_history,
        )

        return {
            "status": "ok",
            "answer_source": "rag",
            "mode": retrieval.get("actual_mode"),
            "retrieval": retrieval,
            "messages": messages,
            "sources": sources,
            "valid_citation_ids": valid_ids,
            "fallback_text": fallback_text,
        }


RUNTIME: AppRuntime | None = None


def require_runtime() -> AppRuntime:
    if RUNTIME is None:
        raise RuntimeError("GUI Runtime 尚未建立")
    return RUNTIME


def initialize_ui() -> tuple[Any, str]:
    """頁面開啟時載入模型並更新文件下拉選單。"""
    runtime = require_runtime()

    try:
        runtime.ensure_loaded()
        names = runtime.document_names()
        selected = names[0] if names else None

        status = (
            "### 系統已就緒\n"
            f"- 執行模式：`完全離線`\n"
            f"- Qwen：`{runtime.model_path.name}`\n"
            f"- GPU offload：`{runtime.llm.gpu_offload_supported}`\n"
            f"- 已索引文件：`{len(names)}` 份\n"
            f"- Reranker 閾值：`{runtime.min_rerank_score:.2f}`\n"
            f"- OCR 服務：`{service_status()}`"
        )

        return (
            gr.Dropdown(
                choices=names,
                value=selected,
                interactive=True,
            ),
            status,
        )
    except Exception as error:
        LOGGER.exception("GUI 初始化失敗")
        return (
            gr.Dropdown(
                choices=[],
                value=None,
                interactive=False,
            ),
            (
                "### 初始化失敗\n"
                f"```\n{type(error).__name__}: {error}\n```"
            ),
        )


def make_debug_info(result: dict[str, Any]) -> dict[str, Any]:
    retrieval = result.get("retrieval") or {}
    info: dict[str, Any] = {
        "status": result.get("status"),
        "rag_mode": result.get("mode"),
        "llm_called": result.get("llm_called"),
        "answer_source": result.get("answer_source"),
        "sources": result.get("sources") or [],
        "best_dense_score": retrieval.get(
            "best_dense_score"
        ),
        "best_rerank_score": retrieval.get(
            "best_rerank_score"
        ),
        "retrieval_reason": retrieval.get("reason"),
    }

    runtime = require_runtime()
    if (
        result.get("llm_called")
        and runtime.llm is not None
        and runtime.llm.last_result is not None
    ):
        llm_info = runtime.llm.last_result
        info["llm"] = {
            "mode": llm_info.get("mode"),
            "finish_reason": llm_info.get(
                "finish_reason"
            ),
            "usage": llm_info.get("usage"),
            "elapsed_seconds": round(
                float(
                    llm_info.get(
                        "elapsed_seconds",
                        0.0,
                    )
                ),
                2,
            ),
            "tokens_per_second": (
                round(
                    float(
                        llm_info.get(
                            "tokens_per_second",
                            0.0,
                        )
                    ),
                    2,
                )
                if llm_info.get("tokens_per_second")
                is not None
                else None
            ),
            "hidden_reasoning_characters": len(
                str(llm_info.get("reasoning") or "")
            ),
        }

    return info



def append_source_footer(
    answer: str,
    sources: Sequence[dict[str, Any]],
) -> str:
    answer = answer.rstrip()
    if not sources:
        return answer

    lines = ["", "參考來源："]
    for index, source in enumerate(sources, start=1):
        lines.append(
            f"- [來源{index}] "
            f"{source.get('source_file')}，"
            f"第 {source.get('page')} 頁"
        )
    return answer + "\n" + "\n".join(lines)


_CITATION_ID_PATTERN = re.compile(r"S(\d+)-(\d+)")
_CITATION_BRACKET_PATTERN = re.compile(r"\[([^\[\]]*S\d+-\d+[^\[\]]*)\]")
_INSUFFICIENT_PHRASE = "文件內容不足以回答這個問題"


def validate_citations(
    answer: str,
    valid_ids: set[str],
) -> tuple[bool, list[str]]:
    """
    驗證答案裡的 [S編號-行號] 引用是否都是文件裡真的存在的行。

    這是事後把關，不是信任 LLM 自己說有標來源就算數：
    - 模型誠實承認「文件內容不足以回答」（system prompt 要求的固定說法）：
      視為合法答案，不需要引用，也不該被備援原文覆蓋掉。
    - 完全沒有引用標記（也不是上面的誠實拒答）：視為沒有依據，不通過。
    - 引用了不存在的 S/行號組合（等於引用是編出來的）：不通過。
    通過只代表「每個引用的行確實存在於提供的內容裡」，不保證那句話的
    語意真的等於那一行在講的東西——那需要更貴的逐句語意檢查，這裡先
    擋掉最基本、也最常見的「引用格式對但編號是編的」與「完全沒引用」。

    引用括號允許同時包含多個 ID，例如 [S1-15, S1-23] 或 [S1-15][S1-23]，
    兩種寫法都會被抓出來逐一驗證。
    """
    if _INSUFFICIENT_PHRASE in answer:
        return True, []

    brackets = _CITATION_BRACKET_PATTERN.findall(answer)
    if not brackets:
        return False, []
    cited_ids = [
        f"S{s}-{l}"
        for bracket in brackets
        for s, l in _CITATION_ID_PATTERN.findall(bracket)
    ]
    if not cited_ids:
        return False, []
    invalid = [cid for cid in cited_ids if cid not in valid_ids]
    return (len(invalid) == 0), invalid


def insufficient_answer(
    source_file: str | None,
) -> str:
    if source_file:
        return (
            f"目前在《{source_file}》中找不到足夠資訊回答這個問題。"
        )
    return "目前提供的文件中找不到足夠資訊回答這個問題。"


def chat_submit(
    message: str,
    history: Sequence[Any] | None,
    source_file: str | None,
    ui_mode: str,
    allow_general_answer: bool,
) -> Iterator[
    tuple[
        list[dict[str, str]],
        str,
        str,
        dict[str, Any],
    ]
]:
    """
    RAG 分數達標時依文件回答；分數不足時可改由 Qwen 一般回答。
    兩種路徑都使用 llama.cpp stream=True 逐字更新聊天泡泡。
    """
    question = (message or "").strip()
    previous_history = normalize_gradio_history(history)

    if not question:
        yield (
            previous_history,
            "",
            "請先輸入問題。",
            {},
        )
        return

    if not source_file:
        yield (
            previous_history,
            question,
            "請先從左側選擇 PDF。",
            {},
        )
        return

    user_history = previous_history + [
        {"role": "user", "content": question}
    ]

    yield (
        user_history,
        "",
        (
            f"正在搜尋 **{source_file}**，"
            f"回答模式：**{ui_mode}**…"
        ),
        {},
    )

    try:
        runtime = require_runtime()
        prepared = runtime.prepare_stream(
            question=question,
            history=previous_history,
            source_file=source_file,
        )

        answer_source = prepared.get("answer_source")
        is_general = answer_source == "general"

        if is_general and not allow_general_answer:
            answer = insufficient_answer(source_file)
            final_history = user_history + [
                {
                    "role": "assistant",
                    "content": answer,
                }
            ]
            result = {
                "status": "insufficient_context",
                "answer_source": "none",
                "mode": prepared.get("mode"),
                "llm_called": False,
                "sources": [],
                "retrieval": prepared["retrieval"],
            }
            yield (
                final_history,
                "",
                (
                    "文件相關度低於閾值，且一般回答已關閉，"
                    "因此未呼叫 Qwen。"
                ),
                make_debug_info(result),
            )
            return

        assert runtime.llm is not None

        stream = (
            runtime.llm.stream_generate_thinking(
                prepared["messages"]
            )
            if ui_mode == "深度思考"
            else runtime.llm.stream_generate_fast(
                prepared["messages"]
            )
        )

        general_prefix = (
            "⚠️ **文件中沒有找到足夠相關內容，"
            "以下為模型一般知識回答，並非來自目前 PDF：**\n\n"
            if is_general
            else ""
        )

        yield (
            user_history
            + [
                {
                    "role": "assistant",
                    "content": general_prefix + "▌",
                }
            ],
            "",
            (
                "RAG 分數低於閾值，Qwen 正在以一般知識回答…"
                if is_general
                else "RAG 完成，Qwen 正在根據文件回答…"
            ),
            {},
        )

        latest_answer = ""
        for partial_answer in stream:
            latest_answer = partial_answer
            if runtime.stream_char_delay > 0:
                time.sleep(runtime.stream_char_delay)
            yield (
                user_history
                + [
                    {
                        "role": "assistant",
                        "content": (
                            general_prefix
                            + latest_answer
                            + "▌"
                        ),
                    }
                ],
                "",
                (
                    "Qwen 正在逐字產生一般知識回答…"
                    if is_general
                    else "Qwen 正在逐字產生文件回答…"
                ),
                {},
            )

        citation_ok = True
        invalid_citations: list[str] = []
        retry_count = 0
        if is_general:
            final_answer = general_prefix + latest_answer
        else:
            citation_ok, invalid_citations = validate_citations(
                latest_answer,
                prepared.get("valid_citation_ids") or set(),
            )

            # Citation format compliance varies run-to-run at this model's
            # sampling temperature (confirmed empirically: same question,
            # same prompt, ~40% pass rate across independent tries). A
            # couple of quick non-streaming retries recovers most of those
            # instead of immediately giving up on a good-content answer that
            # just forgot to tag itself this one time.
            max_retries = 2
            gen_mode = "thinking" if ui_mode == "深度思考" else "fast"
            while not citation_ok and retry_count < max_retries:
                retry_count += 1
                yield (
                    user_history
                    + [
                        {
                            "role": "assistant",
                            "content": (
                                latest_answer
                                + f"\n\n_（引用驗證未通過，正在重試 {retry_count}/{max_retries}…）_"
                            ),
                        }
                    ],
                    "",
                    f"引用驗證未通過，正在重試（第 {retry_count} 次）…",
                    {},
                )
                retry_result = runtime.llm.generate_result(
                    prepared["messages"],
                    mode=gen_mode,
                )
                latest_answer = str(retry_result.get("answer") or "")
                citation_ok, invalid_citations = validate_citations(
                    latest_answer,
                    prepared.get("valid_citation_ids") or set(),
                )

            if citation_ok:
                final_answer = append_source_footer(
                    latest_answer,
                    prepared["sources"],
                )
            else:
                fallback_text = str(prepared.get("fallback_text") or "").strip()
                final_answer = (
                    "⚠️ **系統回答未通過來源驗證，以下改為顯示手冊原文摘錄：**\n\n"
                    + (fallback_text or "（沒有可用的原文摘錄）")
                )
                final_answer = append_source_footer(
                    final_answer,
                    prepared["sources"],
                )
                LOGGER.warning(
                    "答案未通過引用驗證（重試 %d 次後仍失敗），改用原文備援：invalid_citations=%s",
                    retry_count,
                    invalid_citations,
                )

        final_history = user_history + [
            {
                "role": "assistant",
                "content": final_answer,
            }
        ]

        result = {
            "status": (
                "general_answer"
                if is_general
                else "ok"
            ),
            "answer_source": answer_source,
            "mode": prepared.get("mode"),
            "llm_called": True,
            "sources": prepared["sources"],
            "retrieval": prepared["retrieval"],
            "citation_verified": None if is_general else citation_ok,
            "invalid_citations": invalid_citations,
            "citation_retries": retry_count,
        }
        debug = make_debug_info(result)
        llm_info = debug.get("llm") or {}

        if is_general:
            retrieval = prepared["retrieval"]
            score = retrieval.get("best_rerank_score")
            threshold = retrieval.get(
                "min_rerank_score",
                runtime.min_rerank_score,
            )
            score_text = (
                f"{float(score):.4f}"
                if score is not None
                else "無"
            )
            status = (
                f"完成｜一般知識回答"
                f"｜Reranker：`{score_text}`"
                f" < 閾值：`{float(threshold):.2f}`"
            )
        else:
            verify_text = "通過" if citation_ok else "未通過，已改用原文"
            if retry_count:
                verify_text += f"（重試 {retry_count} 次）"
            status = (
                f"完成｜RAG：`{prepared.get('mode')}`"
                f"｜LLM：`{ui_mode}`"
                f"｜來源驗證：`{verify_text}`"
            )

        if llm_info.get("elapsed_seconds") is not None:
            status += (
                f"｜耗時：`{llm_info['elapsed_seconds']} 秒`"
            )
        if llm_info.get("tokens_per_second") is not None:
            status += (
                f"｜速度：`{llm_info['tokens_per_second']} tokens/s`"
            )

        yield (
            final_history,
            "",
            status,
            debug,
        )

    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        error_history = user_history + [
            {
                "role": "assistant",
                "content": (
                    "GPU／共用記憶體不足，這次問題無法完成。"
                    "請改用快速回答，或降低 n_ctx、n_batch。"
                ),
            }
        ]
        yield (
            error_history,
            "",
            "CUDA 記憶體不足。",
            {"error": "CUDA out of memory"},
        )

    except Exception as error:
        LOGGER.exception("GUI 問答失敗")
        error_history = user_history + [
            {
                "role": "assistant",
                "content": (
                    "處理問題時發生錯誤：\n\n"
                    f"`{type(error).__name__}: {error}`"
                ),
            }
        ]
        yield (
            error_history,
            "",
            "問答失敗，請查看終端機紀錄。",
            {
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )


def select_document(
    source_file: str | None,
) -> tuple[list[dict[str, str]], str]:
    if not source_file:
        return [], "目前未選擇文件。"

    try:
        runtime = require_runtime()
        runtime.ensure_loaded()
        assert runtime.rag is not None
        runtime.rag.select_document(
            source_file=source_file
        )
        return (
            [],
            f"已切換至 **{source_file}**，對話已清除。",
        )
    except Exception as error:
        return (
            [],
            f"切換文件失敗：`{error}`",
        )


def refresh_indexes() -> tuple[Any, str, list]:
    try:
        names = require_runtime().reload_rag()
        selected = names[0] if names else None

        return (
            gr.Dropdown(
                choices=names,
                value=selected,
                interactive=True,
            ),
            (
                f"已重新載入索引，共找到 "
                f"**{len(names)}** 份文件。"
            ),
            [],
        )
    except Exception as error:
        LOGGER.exception("重新載入索引失敗")
        return (
            gr.Dropdown(),
            f"重新載入失敗：`{error}`",
            [],
        )


def _read_ocr_catalog_record(
    source_file: str,
) -> tuple[dict[str, Any] | None, float | None]:
    """從 OCR catalog.json 找出指定 PDF 的最新狀態。"""
    if not OCR_CATALOG_PATH.is_file():
        return None, None

    try:
        catalog_mtime = OCR_CATALOG_PATH.stat().st_mtime
        payload = json.loads(
            OCR_CATALOG_PATH.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    except Exception:
        return None, None

    for item in payload.get("documents") or []:
        if str(item.get("source_file") or "") == source_file:
            return item, catalog_mtime

    return None, catalog_mtime


def _short_log(lines: Sequence[str], limit: int = 12) -> str:
    cleaned = [line.rstrip() for line in lines if line.strip()]
    return "\n".join(cleaned[-limit:])


def _run_command_stream(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
):
    """
    執行子程序並逐行回傳紀錄。

    yield: (line, return_code)
    執行中 return_code 為 None；結束時再 yield 一次最終 return code。
    """
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )

    started = time.monotonic()
    assert process.stdout is not None

    try:
        while True:
            line = process.stdout.readline()
            if line:
                yield line.rstrip(), None

            code = process.poll()
            if code is not None:
                # 把 pipe 中剩餘輸出讀完。
                remainder = process.stdout.read()
                if remainder:
                    for remaining_line in remainder.splitlines():
                        yield remaining_line.rstrip(), None
                yield "", int(code)
                return

            if time.monotonic() - started > timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise TimeoutError(
                    f"指令執行超過 {timeout_seconds} 秒："
                    + " ".join(command)
                )

            if not line:
                time.sleep(0.2)
    finally:
        if process.poll() is None:
            process.terminate()


def _wait_for_service_ocr(
    *,
    source_file: str,
    copied_mtime: float,
    timeout_seconds: int,
):
    """
    等待 systemd OCR watch 完成指定 PDF。

    catalog 必須在本次複製後重新寫入，避免把舊紀錄誤認為完成。
    """
    started = time.monotonic()
    last_status = None

    while time.monotonic() - started <= timeout_seconds:
        record, catalog_mtime = _read_ocr_catalog_record(
            source_file
        )

        if record is not None:
            status = str(record.get("status") or "")
            if status != last_status:
                last_status = status
                yield (
                    f"OCR 狀態：{status}",
                    None,
                )

            catalog_is_new = (
                catalog_mtime is not None
                and catalog_mtime >= copied_mtime - 0.5
            )

            if status == "complete" and catalog_is_new:
                yield ("", record)
                return

            if status == "error" and catalog_is_new:
                raise RuntimeError(
                    "OCR 失敗："
                    + str(record.get("error") or "未知錯誤")
                )

        elapsed = int(time.monotonic() - started)
        yield (
            f"等待 OCR service 完成：{elapsed} 秒",
            None,
        )
        time.sleep(2)

    raise TimeoutError(
        f"OCR 等待超過 {timeout_seconds} 秒：{source_file}"
    )


def upload_pdf_and_build_index(
    file_value: Any,
    current_document: str | None,
):
    """
    GUI 單鍵完成：
    上傳 → OCR → TOC／查修步驟切塊 → E5 索引 → 重載 RAG/Qwen。

    輸出：
    upload_status, document dropdown, system_status, chatbot, answer_status
    """
    runtime = require_runtime()

    try:
        existing_names = runtime.document_names()
    except Exception:
        existing_names = []

    selected_before = (
        current_document
        if current_document in existing_names
        else (existing_names[0] if existing_names else None)
    )

    def dropdown(
        names: Sequence[str],
        value: str | None,
    ):
        return gr.Dropdown(
            choices=list(names),
            value=value,
            interactive=True,
        )

    if file_value is None:
        yield (
            "請先選擇 PDF。",
            dropdown(existing_names, selected_before),
            "系統未變更。",
            [],
            "等待問題。",
        )
        return

    source = extract_uploaded_path(file_value)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".pdf":
        raise ValueError("只接受 PDF 檔案")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = INPUT_DIR / source.name
    temporary = INPUT_DIR / (
        f".{source.name}.uploading-{os.getpid()}"
    )

    log_lines: list[str] = []
    models_released = False

    try:
        yield (
            f"準備加入 **{source.name}**…",
            dropdown(existing_names, selected_before),
            "正在準備新 PDF。",
            [],
            "新 PDF 處理中，暫停問答。",
        )

        # 原子複製，避免 watch service 讀到半份 PDF。
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        copied_mtime = destination.stat().st_mtime

        yield (
            (
                f"已將 **{source.name}** 放入 `input/`。\n\n"
                "正在釋放 Qwen、E5 與 Reranker，"
                "為 OCR／索引保留記憶體…"
            ),
            dropdown(existing_names, selected_before),
            "正在釋放模型。",
            [],
            "新 PDF 處理中，暫停問答。",
        )

        runtime.release_models()
        models_released = True

        ocr_is_active = service_status() == "執行中"
        ocr_record: dict[str, Any] | None = None

        if ocr_is_active:
            yield (
                (
                    f"OCR 服務正在處理 **{source.name}**。\n\n"
                    "等待 OCR 完成…"
                ),
                dropdown(existing_names, selected_before),
                "OCR 執行中。",
                [],
                "新 PDF 處理中，暫停問答。",
            )

            for status_line, record in _wait_for_service_ocr(
                source_file=source.name,
                copied_mtime=copied_mtime,
                timeout_seconds=runtime.ocr_timeout,
            ):
                if record is not None:
                    ocr_record = record
                    break
                yield (
                    (
                        f"正在 OCR **{source.name}**…\n\n"
                        f"`{status_line}`"
                    ),
                    dropdown(existing_names, selected_before),
                    "OCR 執行中。",
                    [],
                    "新 PDF 處理中，暫停問答。",
                )
        else:
            yield (
                (
                    "OCR systemd 服務未執行，"
                    "GUI 將自動執行一次 `ocr_pipeline.py`。"
                ),
                dropdown(existing_names, selected_before),
                "啟動單次 OCR。",
                [],
                "新 PDF 處理中，暫停問答。",
            )

            ocr_command = [
                sys.executable,
                str(OCR_PIPELINE_SCRIPT),
            ]
            return_code = None
            for line, code in _run_command_stream(
                ocr_command,
                cwd=ROOT,
                timeout_seconds=runtime.ocr_timeout,
            ):
                if line:
                    log_lines.append(line)
                    yield (
                        (
                            f"正在 OCR **{source.name}**…\n\n"
                            "```text\n"
                            f"{_short_log(log_lines)}\n"
                            "```"
                        ),
                        dropdown(existing_names, selected_before),
                        "OCR 執行中。",
                        [],
                        "新 PDF 處理中，暫停問答。",
                    )
                if code is not None:
                    return_code = code

            if return_code != 0:
                raise RuntimeError(
                    "ocr_pipeline.py 執行失敗，"
                    f"return code={return_code}\n"
                    f"{_short_log(log_lines)}"
                )

            ocr_record, _ = _read_ocr_catalog_record(
                source.name
            )
            if (
                ocr_record is None
                or ocr_record.get("status") != "complete"
            ):
                raise RuntimeError(
                    "OCR 指令已結束，但 catalog 中沒有找到"
                    f" {source.name} 的 complete 紀錄。"
                )

        page_count = (
            ocr_record.get("page_count")
            if ocr_record
            else None
        )

        log_lines.clear()
        yield (
            (
                f"**{source.name}** OCR 完成"
                + (
                    f"，共 {page_count} 頁。"
                    if page_count
                    else "。"
                )
                + "\n\n正在依目錄章節與查修步驟切塊…"
            ),
            dropdown(existing_names, selected_before),
            "切分文件中。",
            [],
            "新 PDF 處理中，暫停問答。",
        )

        chunk_command = [
            sys.executable,
            str(TOC_CHUNK_SCRIPT),
            "--output-jsonl",
            str(TOC_CHUNKS_FILE),
            "--no-embed",
        ]

        return_code = None
        for line, code in _run_command_stream(
            chunk_command,
            cwd=ROOT,
            timeout_seconds=runtime.ocr_timeout,
        ):
            if line:
                log_lines.append(line)
                yield (
                    (
                        "正在依目錄章節與查修步驟切塊…\n\n"
                        "```text\n"
                        f"{_short_log(log_lines)}\n"
                        "```"
                    ),
                    dropdown(existing_names, selected_before),
                    "切分文件中。",
                    [],
                    "新 PDF 處理中，暫停問答。",
                )
            if code is not None:
                return_code = code

        if return_code != 0:
            raise RuntimeError(
                "toc_step_aware_chunker.py 執行失敗，"
                f"return code={return_code}\n"
                f"{_short_log(log_lines)}"
            )

        log_lines.clear()
        yield (
            "文件切分完成，正在建立／更新 E5 FAISS 索引…",
            dropdown(existing_names, selected_before),
            "建立向量索引中。",
            [],
            "新 PDF 處理中，暫停問答。",
        )

        index_command = [
            sys.executable,
            str(BUILD_INDEX_SCRIPT),
            "--chunks-file",
            str(TOC_CHUNKS_FILE),
            "--output-dir",
            str(VECTOR_DIR),
            "--device",
            "auto",
            "--batch-size",
            str(runtime.index_batch_size),
        ]

        return_code = None
        for line, code in _run_command_stream(
            index_command,
            cwd=ROOT,
            timeout_seconds=runtime.ocr_timeout,
        ):
            if line:
                log_lines.append(line)
                yield (
                    (
                        "正在建立向量索引…\n\n"
                        "```text\n"
                        f"{_short_log(log_lines)}\n"
                        "```"
                    ),
                    dropdown(existing_names, selected_before),
                    "建立向量索引中。",
                    [],
                    "新 PDF 處理中，暫停問答。",
                )
            if code is not None:
                return_code = code

        if return_code != 0:
            raise RuntimeError(
                "build_rag_index.py 執行失敗，"
                f"return code={return_code}\n"
                f"{_short_log(log_lines)}"
            )

        yield (
            (
                "向量索引建立完成。\n\n"
                "正在重新載入 RAG 與 Qwen…"
            ),
            dropdown(existing_names, selected_before),
            "重新載入模型中。",
            [],
            "新 PDF 處理中，暫停問答。",
        )

        runtime.ensure_loaded()
        models_released = False
        names = runtime.document_names()
        selected = (
            source.name
            if source.name in names
            else (names[0] if names else None)
        )

        duplicate_note = ""
        if source.name not in names:
            duplicate_note = (
                "\n\n注意：此 PDF 可能與既有文件內容完全相同，"
                "OCR 全域資料會依 document_id 去重，"
                "因此選單可能保留既有檔名。"
            )

        yield (
            (
                f"✅ **{source.name}** 已完成："
                "上傳 → OCR → 建立索引 → 重載模型。"
                f"{duplicate_note}"
            ),
            dropdown(names, selected),
            (
                "### 系統已就緒\n"
                f"- 執行模式：`完全離線`\n"
                f"- Qwen：`{runtime.model_path.name}`\n"
                f"- 已索引文件：`{len(names)}` 份\n"
                f"- Reranker 閾值："
                f"`{runtime.min_rerank_score:.2f}`\n"
                f"- OCR 服務：`{service_status()}`"
            ),
            [],
            (
                f"已選擇 **{selected}**，現在可以直接提問。"
                if selected
                else "索引中尚無可選文件。"
            ),
        )

    except Exception as error:
        LOGGER.exception("自動 OCR／索引失敗")

        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass

        # 無論哪一步失敗，都盡量恢復原本 GUI 模型。
        if models_released:
            try:
                runtime.ensure_loaded()
                models_released = False
            except Exception:
                LOGGER.exception("錯誤後重新載入模型失敗")

        try:
            names = runtime.document_names()
        except Exception:
            names = existing_names

        selected = (
            current_document
            if current_document in names
            else (names[0] if names else None)
        )

        yield (
            (
                "❌ 自動處理失敗：\n\n"
                f"`{type(error).__name__}: {error}`\n\n"
                "詳細紀錄請查看啟動 GUI 的終端機。"
            ),
            dropdown(names, selected),
            (
                "系統已嘗試恢復模型；"
                "請查看終端機確認目前狀態。"
            ),
            [],
            "新 PDF 處理失敗。",
        )


def extract_uploaded_path(file_value: Any) -> Path:
    if isinstance(file_value, (str, os.PathLike)):
        return Path(file_value)

    if isinstance(file_value, dict):
        for key in ("path", "name"):
            value = file_value.get(key)
            if value:
                return Path(value)

    for attribute in ("path", "name"):
        value = getattr(file_value, attribute, None)
        if value:
            return Path(value)

    raise ValueError("無法取得上傳檔案路徑")


def clear_chat() -> tuple[list, str, dict]:
    return [], "對話已清除。", {}


CSS = """
#app-title {
    text-align: center;
    margin-bottom: 0.25rem;
}
#app-subtitle {
    text-align: center;
    opacity: 0.75;
    margin-bottom: 1rem;
}
#chatbot {
    min-height: 620px;
}
.status-card {
    border-radius: 12px;
    padding: 4px 8px;
}
"""


def build_gui() -> gr.Blocks:
    with gr.Blocks(
        title="Unlimited-OCR 文件問答",
        css=CSS,
    ) as demo:
        gr.Markdown(
            "# Unlimited-OCR 文件問答",
            elem_id="app-title",
        )
        gr.Markdown(
            "PDF OCR・FAISS RAG・BGE Reranker・Qwen3.5",
            elem_id="app-subtitle",
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                document = gr.Dropdown(
                    label="目前文件",
                    choices=[],
                    value=None,
                    interactive=True,
                    info="選擇這次要詢問的 PDF",
                )

                answer_mode = gr.Radio(
                    label="回答模式",
                    choices=["快速回答", "深度思考"],
                    value="快速回答",
                    info=(
                        "一般 SOP 問答建議快速回答；"
                        "跨段落複雜分析再使用深度思考。"
                    ),
                )

                allow_general_answer = gr.Checkbox(
                    label="文件不足時允許模型自行回答",
                    value=True,
                    info=(
                        "Reranker 低於閾值時，不使用 PDF 內容，"
                        "改由 Qwen 一般知識回答並顯示警告。"
                    ),
                )

                refresh_button = gr.Button(
                    "重新載入索引",
                    variant="secondary",
                )

                with gr.Accordion(
                    "加入新 PDF",
                    open=False,
                ):
                    upload = gr.File(
                        label="選擇 PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    upload_button = gr.Button(
                        "上傳並自動建立索引",
                        variant="primary",
                    )
                    upload_status = gr.Markdown()

                system_status = gr.Markdown(
                    "模型尚未載入。",
                    elem_classes=["status-card"],
                )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="文件問答",
                    type="messages",
                    layout="bubble",
                    height=620,
                    elem_id="chatbot",
                    placeholder=(
                        "模型載入完成後，選擇 PDF 並輸入問題。"
                    ),
                )

                question = gr.Textbox(
                    label="問題",
                    placeholder=(
                        "例如：設備啟動前需要檢查什麼？"
                    ),
                    lines=2,
                    max_lines=6,
                )

                with gr.Row():
                    send_button = gr.Button(
                        "送出",
                        variant="primary",
                    )
                    clear_button = gr.Button(
                        "清除對話",
                    )

                answer_status = gr.Markdown(
                    "等待問題。",
                    elem_classes=["status-card"],
                )

                with gr.Accordion(
                    "檢索與效能資訊",
                    open=False,
                ):
                    debug_output = gr.JSON(
                        label="Debug",
                        value={},
                    )

        demo.load(
            fn=initialize_ui,
            inputs=None,
            outputs=[document, system_status],
            concurrency_limit=1,
        )

        send_event = send_button.click(
            fn=chat_submit,
            inputs=[
                question,
                chatbot,
                document,
                answer_mode,
                allow_general_answer,
            ],
            outputs=[
                chatbot,
                question,
                answer_status,
                debug_output,
            ],
            concurrency_limit=1,
        )

        question.submit(
            fn=chat_submit,
            inputs=[
                question,
                chatbot,
                document,
                answer_mode,
                allow_general_answer,
            ],
            outputs=[
                chatbot,
                question,
                answer_status,
                debug_output,
            ],
            concurrency_limit=1,
        )

        document.change(
            fn=select_document,
            inputs=[document],
            outputs=[chatbot, answer_status],
            concurrency_limit=1,
        )

        refresh_button.click(
            fn=refresh_indexes,
            inputs=None,
            outputs=[
                document,
                system_status,
                chatbot,
            ],
            concurrency_limit=1,
        )

        upload_button.click(
            fn=upload_pdf_and_build_index,
            inputs=[upload, document],
            outputs=[
                upload_status,
                document,
                system_status,
                chatbot,
                answer_status,
            ],
            concurrency_limit=1,
        )

        clear_button.click(
            fn=clear_chat,
            inputs=None,
            outputs=[
                chatbot,
                answer_status,
                debug_output,
            ],
            concurrency_limit=1,
        )

    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unlimited-OCR RAG + Qwen 單一 GUI"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help=(
            "Qwen3.5-9B GGUF 完整路徑；"
            "省略時會讀取 QWEN_GGUF_PATH 或自動搜尋。"
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Gradio 綁定位置。只在本機使用建議 127.0.0.1；"
            "區網使用可設 0.0.0.0。"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--n-batch",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--rag-context-chars",
        type=int,
        default=7000,
    )
    parser.add_argument(
        "--reranker-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--min-rerank-score",
        type=float,
        default=0.25,
        help=(
            "細節問題使用 RAG 的最低 Reranker 分數；"
            "低於此值時改走一般知識回答。預設 0.25"
        ),
    )
    parser.add_argument(
        "--stream-char-delay",
        type=float,
        default=0.01,
        help=(
            "GUI 每個字元更新的最小延遲秒數。"
            "預設 0.01；設 0 可取消額外延遲。"
        ),
    )
    parser.add_argument(
        "--ocr-timeout",
        type=int,
        default=1800,
        help="自動 OCR／索引每個階段的逾時秒數，預設 1800。",
    )
    parser.add_argument(
        "--index-batch-size",
        type=int,
        default=4,
        help="自動建立 embedding 索引的 batch size，預設 4。",
    )
    parser.add_argument(
        "--verbose-llama",
        action="store_true",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        help="啟動後自動開啟瀏覽器",
    )
    return parser


def main() -> int:
    global RUNTIME

    setup_logging()
    args = build_parser().parse_args()
    model_path = resolve_model_path(args.model)

    RUNTIME = AppRuntime(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        rag_context_chars=args.rag_context_chars,
        reranker_device=args.reranker_device,
        min_rerank_score=args.min_rerank_score,
        stream_char_delay=args.stream_char_delay,
        ocr_timeout=args.ocr_timeout,
        index_batch_size=args.index_batch_size,
        verbose_llama=args.verbose_llama,
    )

    demo = build_gui()
    demo.queue(
        max_size=8,
        default_concurrency_limit=1,
    )
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        inbrowser=args.inbrowser,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
