#!/usr/bin/env python3
"""REST API for the existing Unlimited-OCR RAG + local Qwen runtime."""

from __future__ import annotations

import json
import importlib.util
import os
import re
import shutil
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from local_llm import resolve_model_path


ROOT = Path(__file__).resolve().parent
_app_spec = importlib.util.spec_from_file_location("unlimited_ocr_gui_app", ROOT / "app.py")
if _app_spec is None or _app_spec.loader is None:
    raise ImportError(f"無法載入 {ROOT / 'app.py'}")
_app_module = importlib.util.module_from_spec(_app_spec)
_app_spec.loader.exec_module(_app_module)
AppRuntime = _app_module.AppRuntime


SOP_FILE = os.getenv("PCB_SOP_FILE", "PCB異常處理SOP_v1.pdf")
_runtime: AppRuntime | None = None
_runtime_lock = threading.Lock()
_upload_jobs: dict[str, dict[str, Any]] = {}
_upload_jobs_lock = threading.Lock()
MAX_UPLOAD_BYTES = int(os.getenv("SOP_UPLOAD_MAX_BYTES", str(30 * 1024 * 1024)))


class PcbAnalyzeRequest(BaseModel):
    event_id: str = Field(min_length=1, max_length=200)
    pcb_id: str = Field(default="未知 PCB", max_length=200)
    missing_items: list[str] = Field(default_factory=list)
    defect_types: list[str] = Field(default_factory=list)
    positions: list[str] = Field(default_factory=list)
    location: str = Field(default="", max_length=200)
    detected_at: str | None = None


class SourceRef(BaseModel):
    file: str
    page: int | None = None


class DispatchDecision(BaseModel):
    should_dispatch: bool
    severity: Literal["critical", "warning", "info"]
    sop_reference: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=300)
    department: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2000)
    actions: list[str] = Field(min_length=1)
    required_evidence: list[str] = Field(min_length=1)
    verification: list[str] = Field(min_length=1)
    deadline_minutes: int = Field(ge=1, le=1440)
    sources: list[SourceRef] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)
    source_file: str | None = Field(default=None, max_length=255)


class ChatResponse(BaseModel):
    answer: str
    answer_source: Literal["rag", "none"]
    sources: list[SourceRef] = Field(default_factory=list)


def require_key(value: str | None) -> None:
    expected = os.getenv("SOP_API_KEY", "")
    if expected and value != expected:
        raise HTTPException(401, "無效的 X-API-Key")


def get_runtime() -> AppRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = AppRuntime(
                model_path=resolve_model_path(os.getenv("QWEN_GGUF_PATH") or None),
                n_ctx=int(os.getenv("SOP_N_CTX", "8192")),
                n_batch=int(os.getenv("SOP_N_BATCH", "256")),
                rag_context_chars=int(os.getenv("SOP_RAG_CONTEXT_CHARS", "7000")),
                reranker_device=os.getenv("SOP_RERANKER_DEVICE", "auto"),
                min_rerank_score=float(os.getenv("SOP_MIN_RERANK_SCORE", "0.50")),
                stream_char_delay=0,
                ocr_timeout=1800,
                index_batch_size=4,
                verbose_llama=False,
            )
            _runtime.ensure_loaded()
    return _runtime


def _indexed_documents() -> list[dict[str, Any]]:
    metadata_path = ROOT / "flash" / "vector" / "document_metadata.jsonl"
    documents: list[dict[str, Any]] = []
    if not metadata_path.is_file():
        return documents
    for line in metadata_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        documents.append({
            "document_id": item.get("document_id"),
            "source_file": item.get("source_file"),
            "title": item.get("title") or item.get("source_file"),
            "page_count": item.get("page_count"),
            "indexed_chunk_count": item.get("indexed_chunk_count"),
        })
    return documents


def _active_upload_job() -> dict[str, Any] | None:
    with _upload_jobs_lock:
        for job in reversed(list(_upload_jobs.values())):
            if job.get("status") not in {"completed", "failed"}:
                return dict(job)
    return None


def _update_upload_job(job_id: str, **changes: Any) -> None:
    with _upload_jobs_lock:
        job = _upload_jobs[job_id]
        job.update(changes)


def _run_upload_job(job_id: str, staged_file: Path, filename: str) -> None:
    destination = ROOT / "input" / filename
    temporary = destination.with_name(f".{filename}.uploading-{job_id}")
    models_released = False
    try:
        runtime = get_runtime()
        _update_upload_job(job_id, status="preparing", progress=8, message="正在準備 PDF")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged_file, temporary)
        os.replace(temporary, destination)
        copied_mtime = destination.stat().st_mtime

        _update_upload_job(job_id, status="preparing", progress=12, message="正在釋放模型記憶體")
        runtime.release_models()
        models_released = True

        ocr_record: dict[str, Any] | None = None
        if _app_module.service_status() == "執行中":
            _update_upload_job(job_id, status="ocr", progress=18, message="OCR 文字辨識中")
            for status_line, record in _app_module._wait_for_service_ocr(
                source_file=filename,
                copied_mtime=copied_mtime,
                timeout_seconds=runtime.ocr_timeout,
            ):
                if record is not None:
                    ocr_record = record
                    break
                _update_upload_job(job_id, status="ocr", message=status_line or "OCR 文字辨識中")
        else:
            _update_upload_job(job_id, status="ocr", progress=18, message="OCR 文字辨識中")
            return_code = None
            for line, code in _app_module._run_command_stream(
                [sys.executable, str(_app_module.OCR_PIPELINE_SCRIPT)],
                cwd=ROOT,
                timeout_seconds=runtime.ocr_timeout,
            ):
                if line:
                    _update_upload_job(job_id, status="ocr", message=line[-300:])
                if code is not None:
                    return_code = code
            if return_code != 0:
                raise RuntimeError(f"OCR 執行失敗（return code={return_code}）")
            ocr_record, _ = _app_module._read_ocr_catalog_record(filename)

        if not ocr_record or ocr_record.get("status") != "complete":
            raise RuntimeError("OCR 已結束，但沒有找到完成紀錄")
        page_count = int(ocr_record.get("page_count") or 0)

        _update_upload_job(
            job_id,
            status="indexing",
            progress=68,
            page_count=page_count,
            message=f"OCR 完成（{page_count} 頁），正在建立向量索引",
        )
        return_code = None
        for line, code in _app_module._run_command_stream(
            [
                sys.executable,
                str(_app_module.BUILD_INDEX_SCRIPT),
                "--device",
                "auto",
                "--batch-size",
                str(runtime.index_batch_size),
            ],
            cwd=ROOT,
            timeout_seconds=runtime.ocr_timeout,
        ):
            if line:
                _update_upload_job(job_id, status="indexing", message=line[-300:])
            if code is not None:
                return_code = code
        if return_code != 0:
            raise RuntimeError(f"向量索引建立失敗（return code={return_code}）")

        _update_upload_job(job_id, status="reloading", progress=92, message="正在重新載入 RAG 與 Qwen")
        runtime.ensure_loaded()
        models_released = False
        documents = _indexed_documents()
        _update_upload_job(
            job_id,
            status="completed",
            progress=100,
            message=f"{filename} 已完成 OCR 並加入知識庫",
            document_count=len(documents),
        )
    except Exception as exc:
        if models_released:
            try:
                get_runtime().ensure_loaded()
            except Exception:
                pass
        _update_upload_job(
            job_id,
            status="failed",
            progress=100,
            message=f"處理失敗：{exc}",
            error=str(exc),
        )
    finally:
        for path in (staged_file, temporary):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S | re.I)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("LLM 回覆不是 JSON object")
    return value


def build_question(event: PcbAnalyzeRequest) -> str:
    event_json = json.dumps(event.model_dump(), ensure_ascii=False, indent=2)
    return f"""請只根據所選 PCB SOP 分析以下異常，禁止使用文件以外的處理規格。

異常事件：
{event_json}

只輸出一個精簡的合法 JSON object，不要 Markdown、前言或補充文字。
所有文字都用繁體中文短句；summary 最多 80 字；actions 最多 5 項；
required_evidence 最多 3 項；verification 最多 2 項。不要重述輸入資料，
不要解釋判斷過程，只保留建立派工單所需內容。格式必須完整符合：
{{
  "should_dispatch": true,
  "severity": "critical|warning|info",
  "sop_reference": "SOP 編號",
  "title": "派工標題",
  "department": "負責單位",
  "summary": "異常摘要",
  "actions": ["SOP 處理步驟短句，最多 5 項"],
  "required_evidence": ["必填證據短句，最多 3 項"],
  "verification": ["結案條件短句，最多 2 項"],
  "deadline_minutes": 15
}}
若文件不足，仍不可自行編造；should_dispatch 設為 true、引用 PCB-SOP-001-B，並將動作設為隔離及升級工程師確認。"""


app = FastAPI(title="Unlimited-OCR PCB SOP API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": _runtime is not None,
        "sop_file": SOP_FILE,
    }


@app.get("/api/knowledge")
def knowledge(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    require_key(x_api_key)
    active = _active_upload_job()
    with _upload_jobs_lock:
        latest = dict(list(_upload_jobs.values())[-1]) if _upload_jobs else None
    return {
        "documents": _indexed_documents(),
        "active": active is not None,
        "job": active or latest,
    }


@app.post("/api/knowledge/upload", status_code=202)
async def upload_knowledge(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
) -> dict[str, Any]:
    require_key(x_api_key)
    if _active_upload_job() is not None:
        raise HTTPException(409, "已有文件正在進行 OCR/RAG，請等待完成")

    filename = re.sub(r'[\r\n"]', "_", Path(file.filename or "").name)
    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(415, "只接受 PDF 文件")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not content:
        raise HTTPException(400, "PDF 文件是空的")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"PDF 不可超過 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(415, "檔案內容不是有效的 PDF")

    job_id = uuid.uuid4().hex
    staged_file = Path("/tmp") / f"unlimited-ocr-{job_id}.pdf"
    staged_file.write_bytes(content)
    job = {
        "id": job_id,
        "filename": filename,
        "status": "queued",
        "progress": 2,
        "message": "PDF 已上傳，等待 OCR",
    }
    with _upload_jobs_lock:
        _upload_jobs[job_id] = job
        while len(_upload_jobs) > 10:
            _upload_jobs.pop(next(iter(_upload_jobs)))
    threading.Thread(
        target=_run_upload_job,
        args=(job_id, staged_file, filename),
        name=f"ocr-rag-{job_id[:8]}",
        daemon=True,
    ).start()
    return {"accepted": True, "job": dict(job)}


@app.post("/api/analyze", response_model=DispatchDecision)
def analyze(
    event: PcbAnalyzeRequest,
    x_api_key: str | None = Header(default=None),
) -> DispatchDecision:
    require_key(x_api_key)
    runtime = get_runtime()
    prepared = runtime.prepare_stream(
        question=build_question(event),
        history=[],
        source_file=SOP_FILE,
    )
    if prepared.get("status") != "ok" or prepared.get("answer_source") != "rag":
        raise HTTPException(422, "PCB SOP 檢索相關度不足，未呼叫 LLM")

    assert runtime.llm is not None
    generated = runtime.llm.generate_result(
        prepared["messages"],
        mode="fast",
        max_tokens=400,
    )
    try:
        decision_data = extract_json(generated["answer"])
        sources = []
        for source in prepared.get("sources", []):
            sources.append({
                "file": str(source.get("source_file") or SOP_FILE),
                "page": source.get("page") or source.get("page_start"),
            })
        decision_data["sources"] = sources
        decision = DispatchDecision.model_validate(decision_data)
    except Exception as exc:
        raise HTTPException(502, f"LLM 結構化回覆驗證失敗：{exc}") from exc

    if not decision.sop_reference.startswith("PCB-SOP-001"):
        raise HTTPException(502, "LLM 未引用允許的 PCB SOP")
    return decision


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    x_api_key: str | None = Header(default=None),
) -> ChatResponse:
    """Answer operator questions using only the configured PCB SOP."""
    require_key(x_api_key)
    if _active_upload_job() is not None:
        raise HTTPException(409, "OCR/RAG 正在更新知識庫，完成後即可提問")
    runtime = get_runtime()
    history = [item.model_dump() for item in payload.history]
    prepared = runtime.prepare_stream(
        question=payload.question.strip(),
        history=history,
        source_file=payload.source_file,
    )
    if prepared.get("status") != "ok" or prepared.get("answer_source") != "rag":
        return ChatResponse(
            answer="目前知識庫中找不到足夠相關的內容，請補充問題，或先上傳相關 PDF。",
            answer_source="none",
            sources=[],
        )

    assert runtime.llm is not None
    generated = runtime.llm.generate_result(
        prepared["messages"],
        mode="fast",
        max_tokens=900,
    )
    answer = str(generated.get("answer") or "").strip()
    if not answer:
        raise HTTPException(502, "LLM 回傳空白答案")
    sources = [
        SourceRef(
            file=str(source.get("source_file") or SOP_FILE),
            page=source.get("page") or source.get("page_start"),
        )
        for source in prepared.get("sources", [])
    ]
    return ChatResponse(answer=answer, answer_source="rag", sources=sources)


@app.post("/api/chat/stream")
def chat_stream(
    payload: ChatRequest,
    x_api_key: str | None = Header(default=None),
) -> StreamingResponse:
    """Stream cumulative SOP-grounded answer text as NDJSON."""
    require_key(x_api_key)

    # Async generator keeps llama.cpp's thread-owned RLock on one thread while
    # StreamingResponse yields characters. A sync generator may hop between
    # Starlette worker threads when the client disconnects.
    async def emit():
        try:
            active_job = _active_upload_job()
            if active_job is not None:
                yield json.dumps({"type": "error", "detail": "OCR/RAG 正在更新知識庫，完成後即可提問"}, ensure_ascii=False) + "\n"
                return
            yield json.dumps({"type": "status", "message": "正在檢索 OCR/RAG 知識庫…"}, ensure_ascii=False) + "\n"
            runtime = get_runtime()
            history = [item.model_dump() for item in payload.history]
            prepared = runtime.prepare_stream(
                question=payload.question.strip(),
                history=history,
                source_file=payload.source_file,
            )
            if prepared.get("status") != "ok" or prepared.get("answer_source") != "rag":
                yield json.dumps({
                    "type": "done",
                    "answer": "目前知識庫中找不到足夠相關的內容，請補充問題，或先上傳相關 PDF。",
                    "answer_source": "none",
                    "sources": [],
                }, ensure_ascii=False) + "\n"
                return

            assert runtime.llm is not None
            latest = ""
            for partial in runtime.llm.stream_generate_result(
                prepared["messages"], mode="fast", max_tokens=900
            ):
                latest = partial
                yield json.dumps({"type": "delta", "answer": latest}, ensure_ascii=False) + "\n"
            sources = [
                {
                    "file": str(source.get("source_file") or SOP_FILE),
                    "page": source.get("page") or source.get("page_start"),
                }
                for source in prepared.get("sources", [])
            ]
            yield json.dumps({
                "type": "done",
                "answer": latest,
                "answer_source": "rag",
                "sources": sources,
            }, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}, ensure_ascii=False) + "\n"

    return StreamingResponse(
        emit(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
