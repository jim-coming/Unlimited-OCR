#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local_llm.py

Qwen3.5 GGUF 的本機 llama-cpp-python 包裝器。

用途：
- 與 rag_chat_engine.RAGChatEngine 直接在同一 Python 程序中整合。
- 不啟動 HTTP server，不使用 subprocess，不開另一個命令視窗。
- 支援快速回答與深度思考兩種模式。
- 快速模式直接建立 Qwen3.5 text-only ChatML prompt，於 assistant 開頭放入
  空的 <think></think> 區塊，避免模型先輸出冗長 Thinking Process。
- 深度模式保留 reasoning，但只把 </think> 後的最終答案回傳給 GUI。
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from llama_cpp import Llama, llama_cpp


ThinkingMode = Literal["fast", "thinking"]


@dataclass(frozen=True)
class GenerationSettings:
    fast_max_tokens: int = 768
    thinking_max_tokens: int = 2048

    fast_temperature: float = 0.7
    fast_top_p: float = 0.8
    fast_top_k: int = 20
    fast_min_p: float = 0.0
    fast_presence_penalty: float = 1.5
    fast_repeat_penalty: float = 1.0

    thinking_temperature: float = 1.0
    thinking_top_p: float = 0.95
    thinking_top_k: int = 20
    thinking_min_p: float = 0.0
    thinking_presence_penalty: float = 1.5
    thinking_repeat_penalty: float = 1.0


class LocalQwenLLM:
    """可常駐於最終 GUI 的 Qwen3.5 GGUF 模型。"""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 8192,
        n_batch: int = 256,
        n_ubatch: int = 128,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        flash_attn: bool = True,
        use_mmap: bool = True,
        use_mlock: bool = False,
        default_mode: ThinkingMode = "fast",
        generation: GenerationSettings | None = None,
        verbose: bool = False,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"找不到 Qwen GGUF 模型：{self.model_path}"
            )

        if n_ctx < 2048:
            raise ValueError("n_ctx 建議至少設為 2048")
        if n_batch < 1 or n_ubatch < 1:
            raise ValueError("n_batch 與 n_ubatch 必須大於 0")
        if default_mode not in {"fast", "thinking"}:
            raise ValueError("default_mode 必須是 fast 或 thinking")

        self.n_ctx = int(n_ctx)
        self.default_mode: ThinkingMode = default_mode
        self.generation = generation or GenerationSettings()
        self._lock = threading.RLock()
        self.last_result: dict[str, Any] | None = None

        supports_gpu = bool(
            llama_cpp.llama_supports_gpu_offload()
        )
        if n_gpu_layers != 0 and not supports_gpu:
            raise RuntimeError(
                "目前 llama-cpp-python 不支援 GPU offload。"
                "請先以 GGML_CUDA=ON 重新編譯。"
            )

        llama_kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": self.n_ctx,
            "n_batch": int(n_batch),
            "n_ubatch": int(n_ubatch),
            "n_gpu_layers": int(n_gpu_layers),
            "flash_attn": bool(flash_attn),
            "use_mmap": bool(use_mmap),
            "use_mlock": bool(use_mlock),
            "verbose": bool(verbose),
        }
        if n_threads is not None:
            llama_kwargs["n_threads"] = int(n_threads)
            llama_kwargs["n_threads_batch"] = int(n_threads)

        self.model = Llama(**llama_kwargs)

    @property
    def gpu_offload_supported(self) -> bool:
        return bool(llama_cpp.llama_supports_gpu_offload())

    def set_mode(self, mode: ThinkingMode) -> None:
        if mode not in {"fast", "thinking"}:
            raise ValueError("mode 必須是 fast 或 thinking")
        self.default_mode = mode

    @staticmethod
    def _normalize_messages(
        messages: Sequence[dict[str, Any]],
    ) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(
                    f"messages[{index}] 必須是 dict"
                )

            role = str(message.get("role") or "").strip()
            content = message.get("content")

            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"不支援的 role：{role!r}")
            if not isinstance(content, str):
                raise TypeError(
                    f"messages[{index}].content 必須是字串"
                )

            content = content.strip()
            if content:
                output.append(
                    {"role": role, "content": content}
                )

        if not output:
            raise ValueError("messages 不可為空")

        return output

    @staticmethod
    def _strip_previous_reasoning(text: str) -> str:
        text = text.strip()

        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1].strip()

        text = re.sub(
            r"(?is)^\s*<think>.*?</think>\s*",
            "",
            text,
        )
        return text.strip()

    def _build_qwen_prompt(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        thinking: bool,
    ) -> str:
        normalized = self._normalize_messages(messages)
        parts: list[str] = []

        for message in normalized:
            role = message["role"]
            content = message["content"]

            if role == "assistant":
                content = self._strip_previous_reasoning(content)

            parts.append(
                f"<|im_start|>{role}\n"
                f"{content}"
                f"<|im_end|>\n"
            )

        parts.append("<|im_start|>assistant\n")

        if thinking:
            parts.append("<think>\n")
        else:
            parts.append("<think>\n\n</think>\n\n")

        return "".join(parts)

    def _count_prompt_tokens(self, prompt: str) -> int:
        return len(
            self.model.tokenize(
                prompt.encode("utf-8"),
                add_bos=False,
                special=True,
            )
        )

    @staticmethod
    def _split_reasoning(
        generated_text: str,
        *,
        thinking: bool,
    ) -> tuple[str, str]:
        text = generated_text.strip()
        reasoning = ""

        if "</think>" in text:
            before, _, after = text.partition("</think>")
            reasoning = before
            answer = after
        else:
            answer = text

        reasoning = re.sub(
            r"(?is)^\s*<think>\s*",
            "",
            reasoning,
        ).strip()
        answer = re.sub(
            r"(?is)^\s*<think>.*?</think>\s*",
            "",
            answer,
        ).strip()
        answer = re.sub(
            r"(?is)^\s*(?:ot\s*)?</think>\s*",
            "",
            answer,
        ).strip()

        if not thinking and answer.lower().startswith(
            "thinking process:"
        ):
            final_markers = (
                "\nFinal Answer:",
                "\nFinal answer:",
                "\n最終回答：",
                "\n正式回答：",
            )
            for marker in final_markers:
                if marker in answer:
                    answer = answer.rsplit(
                        marker, 1
                    )[1].strip()
                    break

        return answer.strip(), reasoning.strip()

    def generate_result(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        mode: ThinkingMode | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        selected_mode = mode or self.default_mode
        if selected_mode not in {"fast", "thinking"}:
            raise ValueError("mode 必須是 fast 或 thinking")

        thinking = selected_mode == "thinking"
        prompt = self._build_qwen_prompt(
            messages,
            thinking=thinking,
        )
        prompt_tokens = self._count_prompt_tokens(prompt)

        requested_max_tokens = int(
            max_tokens
            if max_tokens is not None
            else (
                self.generation.thinking_max_tokens
                if thinking
                else self.generation.fast_max_tokens
            )
        )

        available_tokens = self.n_ctx - prompt_tokens - 16
        if available_tokens < 64:
            raise RuntimeError(
                "送入 LLM 的 prompt 太長："
                f"{prompt_tokens} tokens，n_ctx={self.n_ctx}。"
                "請減少 RAG context、聊天紀錄，或增加 n_ctx。"
            )

        actual_max_tokens = min(
            requested_max_tokens,
            available_tokens,
        )

        if thinking:
            temperature = self.generation.thinking_temperature
            top_p = self.generation.thinking_top_p
            top_k = self.generation.thinking_top_k
            min_p = self.generation.thinking_min_p
            presence_penalty = (
                self.generation.thinking_presence_penalty
            )
            repeat_penalty = (
                self.generation.thinking_repeat_penalty
            )
        else:
            temperature = self.generation.fast_temperature
            top_p = self.generation.fast_top_p
            top_k = self.generation.fast_top_k
            min_p = self.generation.fast_min_p
            presence_penalty = (
                self.generation.fast_presence_penalty
            )
            repeat_penalty = (
                self.generation.fast_repeat_penalty
            )

        started = time.perf_counter()

        with self._lock:
            response = self.model.create_completion(
                prompt=prompt,
                max_tokens=actual_max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                stop=[
                    "<|im_end|>",
                    "<|endoftext|>",
                ],
                echo=False,
                stream=False,
            )

        elapsed = time.perf_counter() - started
        choice = response["choices"][0]
        raw_text = str(choice.get("text") or "")
        answer, reasoning = self._split_reasoning(
            raw_text,
            thinking=thinking,
        )

        if not answer:
            if choice.get("finish_reason") == "length":
                raise RuntimeError(
                    "模型的輸出 token 全部用在思考內容，"
                    "尚未產生正式答案。請增加 thinking_max_tokens，"
                    "或改用 fast 模式。"
                )
            raise RuntimeError("Qwen 沒有產生正式答案")

        usage = response.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        tokens_per_second = None
        if (
            isinstance(completion_tokens, int)
            and elapsed > 0
        ):
            tokens_per_second = completion_tokens / elapsed

        result = {
            "answer": answer,
            "reasoning": reasoning,
            "mode": selected_mode,
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_per_second,
            "prompt_tokens_counted": prompt_tokens,
            "max_tokens_requested": requested_max_tokens,
            "max_tokens_used": actual_max_tokens,
            "raw_text": raw_text,
        }
        self.last_result = result
        return result


    def stream_generate_result(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        mode: ThinkingMode | None = None,
        max_tokens: int | None = None,
    ):
        """
        串流產生回答。

        每次 yield 都是「目前完整答案」，GUI 可直接覆蓋聊天泡泡。
        為了達到逐字顯示，模型每回傳一段 token 後，會再逐字 yield。

        深度思考模式會隱藏 </think> 前的 reasoning，只串流正式答案。
        """
        selected_mode = mode or self.default_mode
        if selected_mode not in {"fast", "thinking"}:
            raise ValueError("mode 必須是 fast 或 thinking")

        thinking = selected_mode == "thinking"
        prompt = self._build_qwen_prompt(
            messages,
            thinking=thinking,
        )
        prompt_tokens = self._count_prompt_tokens(prompt)

        requested_max_tokens = int(
            max_tokens
            if max_tokens is not None
            else (
                self.generation.thinking_max_tokens
                if thinking
                else self.generation.fast_max_tokens
            )
        )

        available_tokens = self.n_ctx - prompt_tokens - 16
        if available_tokens < 64:
            raise RuntimeError(
                "送入 LLM 的 prompt 太長："
                f"{prompt_tokens} tokens，n_ctx={self.n_ctx}。"
                "請減少 RAG context、聊天紀錄，或增加 n_ctx。"
            )

        actual_max_tokens = min(
            requested_max_tokens,
            available_tokens,
        )

        if thinking:
            temperature = self.generation.thinking_temperature
            top_p = self.generation.thinking_top_p
            top_k = self.generation.thinking_top_k
            min_p = self.generation.thinking_min_p
            presence_penalty = (
                self.generation.thinking_presence_penalty
            )
            repeat_penalty = (
                self.generation.thinking_repeat_penalty
            )
        else:
            temperature = self.generation.fast_temperature
            top_p = self.generation.fast_top_p
            top_k = self.generation.fast_top_k
            min_p = self.generation.fast_min_p
            presence_penalty = (
                self.generation.fast_presence_penalty
            )
            repeat_penalty = (
                self.generation.fast_repeat_penalty
            )

        started = time.perf_counter()
        raw_parts: list[str] = []
        answer = ""
        reasoning = ""
        finish_reason = None

        # thinking 模式一開始就在 reasoning 區。
        # fast 模式先保留少量前綴，防止某些 GGUF 又輸出 Thinking Process。
        state = "reasoning" if thinking else "probe"
        probe = ""

        def emit_text(text: str):
            nonlocal answer
            text = (
                text.replace("<|im_end|>", "")
                .replace("<|endoftext|>", "")
            )
            for character in text:
                answer += character
                yield answer

        with self._lock:
            response_stream = self.model.create_completion(
                prompt=prompt,
                max_tokens=actual_max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                stop=[
                    "<|im_end|>",
                    "<|endoftext|>",
                ],
                echo=False,
                stream=True,
            )

            for chunk in response_stream:
                choice = chunk["choices"][0]
                delta = str(choice.get("text") or "")
                if choice.get("finish_reason") is not None:
                    finish_reason = choice.get("finish_reason")
                if not delta:
                    continue

                raw_parts.append(delta)

                if state == "probe":
                    probe += delta
                    stripped = probe.lstrip()
                    looks_like_reasoning = (
                        stripped.startswith("<think>")
                        or stripped.lower().startswith(
                            "thinking process:"
                        )
                    )

                    if looks_like_reasoning:
                        state = "reasoning"
                        reasoning += probe
                        probe = ""
                        continue

                    # 最多延遲約 32 個字元，確認不是 reasoning 後開始逐字顯示。
                    if len(probe) >= 32 or "\n" in probe:
                        state = "answer"
                        for partial in emit_text(probe):
                            yield partial
                        probe = ""
                    continue

                if state == "reasoning":
                    reasoning += delta
                    if "</think>" not in reasoning:
                        continue

                    before, _, after = reasoning.partition(
                        "</think>"
                    )
                    reasoning = before
                    state = "answer"
                    if after:
                        for partial in emit_text(after):
                            yield partial
                    continue

                for partial in emit_text(delta):
                    yield partial

        # 非思考模式的回答很短時，probe 可能尚未釋放。
        if state == "probe" and probe:
            for partial in emit_text(probe):
                yield partial

        raw_text = "".join(raw_parts)

        # 某些模型若沒有依預期輸出 closing tag，結束後再做一次解析。
        if not answer:
            final_answer, parsed_reasoning = self._split_reasoning(
                raw_text,
                thinking=thinking,
            )
            if parsed_reasoning and not reasoning:
                reasoning = parsed_reasoning

            if final_answer:
                for partial in emit_text(final_answer):
                    yield partial

        answer = answer.strip()
        reasoning = re.sub(
            r"(?is)^\s*<think>\s*",
            "",
            reasoning,
        ).strip()

        if not answer:
            if finish_reason == "length":
                raise RuntimeError(
                    "模型的輸出 token 全部用在思考內容，"
                    "尚未產生正式答案。請增加 thinking_max_tokens，"
                    "或改用 fast 模式。"
                )
            raise RuntimeError("Qwen 沒有產生正式答案")

        elapsed = time.perf_counter() - started

        # llama.cpp 串流回應不一定附 usage，因此在本機自行計算。
        completion_tokens = len(
            self.model.tokenize(
                raw_text.encode("utf-8"),
                add_bos=False,
                special=True,
            )
        )
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        tokens_per_second = (
            completion_tokens / elapsed
            if elapsed > 0
            else None
        )

        self.last_result = {
            "answer": answer,
            "reasoning": reasoning,
            "mode": selected_mode,
            "finish_reason": finish_reason,
            "usage": usage,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_per_second,
            "prompt_tokens_counted": prompt_tokens,
            "max_tokens_requested": requested_max_tokens,
            "max_tokens_used": actual_max_tokens,
            "raw_text": raw_text,
        }

    def stream_generate(
        self,
        messages: Sequence[dict[str, Any]],
    ):
        return self.stream_generate_result(messages)

    def stream_generate_fast(
        self,
        messages: Sequence[dict[str, Any]],
    ):
        return self.stream_generate_result(
            messages,
            mode="fast",
        )

    def stream_generate_thinking(
        self,
        messages: Sequence[dict[str, Any]],
    ):
        return self.stream_generate_result(
            messages,
            mode="thinking",
        )

    def generate(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> str:
        return self.generate_result(messages)["answer"]

    def generate_fast(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> str:
        return self.generate_result(
            messages,
            mode="fast",
        )["answer"]

    def generate_thinking(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> str:
        return self.generate_result(
            messages,
            mode="thinking",
        )["answer"]


def resolve_model_path(
    explicit_path: str | Path | None = None,
) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"找不到模型：{path}")

    env_path = os.getenv("QWEN_GGUF_PATH", "").strip()
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(
            f"QWEN_GGUF_PATH 指向不存在的檔案：{path}"
        )

    roots = [
        Path(__file__).resolve().parent / "models",
        Path(__file__).resolve().parent,
        Path.home() / "models",
        Path.home(),
    ]
    patterns = (
        "Qwen3.5-9B*.gguf",
        "Qwen3.5_9B*.gguf",
        "*Qwen3.5*9B*Q4_K_M*.gguf",
    )

    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches.extend(root.glob(pattern))
            if root.name == "models":
                matches.extend(root.glob(f"**/{pattern}"))

    unique = sorted(
        {
            match.resolve()
            for match in matches
            if match.is_file()
        }
    )

    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise FileNotFoundError(
            "找不到 Qwen3.5-9B GGUF。請傳入 model_path，"
            "或設定環境變數 QWEN_GGUF_PATH。"
        )

    choices = "\n".join(f"- {path}" for path in unique)
    raise RuntimeError(
        "找到多個模型，請明確指定其中一個：\n"
        f"{choices}"
    )
