from __future__ import annotations

import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from src.generation.base import BaseLLMProvider
from src.generation.models import GenerationRequest, GenerationResponse, GenerationUsage
from src.generation.prompt_builder import RAGPromptBuilder


DEFAULT_QWEN_MODEL = "qwen3.8-max"
DEFAULT_QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenProvider(BaseLLMProvider):
    """Grounded QwenCloud provider using DashScope's OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        timeout_seconds: float = 60.0,
        enable_thinking: bool = True,
        thinking_budget: int | None = 128,
        prompt_builder: RAGPromptBuilder | None = None,
    ) -> None:
        load_dotenv(override=True)
        selected_key = str(api_key or os.getenv("DASHSCOPE_API_KEY", "")).strip()
        self._api_key_configured = bool(selected_key)
        self._model_name = str(model_name or "").strip()
        if not self._model_name:
            raise ValueError("A Qwen model name is required.")
        self._enable_thinking = bool(enable_thinking)
        self._thinking_budget = (
            None if thinking_budget is None else int(thinking_budget)
        )
        if self._thinking_budget is not None and self._thinking_budget <= 0:
            raise ValueError("thinking_budget must be positive or null.")
        self._prompt_builder = prompt_builder or RAGPromptBuilder()
        normalized_timeout = float(timeout_seconds)
        if normalized_timeout <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self._client = OpenAI(
            api_key=selected_key or "missing-dashscope-key",
            base_url=str(base_url).rstrip("/"),
            timeout=normalized_timeout,
            max_retries=0,
        )

    @property
    def provider_name(self) -> str:
        return "qwen"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest.")
        if not self._api_key_configured:
            raise RuntimeError("DASHSCOPE_API_KEY is missing.")

        extra_body: dict[str, Any] = {
            "enable_thinking": self._enable_thinking,
        }
        if self._enable_thinking and self._thinking_budget is not None:
            extra_body["thinking_budget"] = self._thinking_budget

        messages = self._prompt_builder.build_messages(request)
        if self._enable_thinking:
            messages[0]["content"] += (
                " You may reason internally, but return the user-visible final "
                "answer only between <final> and </final>. Put the required "
                "authorized citations inside that block. Do not place any "
                "analysis or reasoning inside the final block."
            )

        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": int(request.max_output_tokens),
            "temperature": float(request.temperature),
            "stream": False,
            "extra_body": extra_body,
        }

        started = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**kwargs)
        except AuthenticationError as error:
            raise RuntimeError("Qwen authentication failed.") from error
        except RateLimitError as error:
            raise RuntimeError("Qwen rate limit or quota reached.") from error
        except APITimeoutError as error:
            raise RuntimeError("Qwen request timed out.") from error
        except APIConnectionError as error:
            raise RuntimeError("Qwen API is unreachable.") from error
        except APIStatusError as error:
            status_code = getattr(error, "status_code", None)
            raise RuntimeError(f"Qwen API returned status {status_code}.") from error
        except Exception as error:
            raise RuntimeError(
                f"Unexpected Qwen generation error: {type(error).__name__}."
            ) from error

        choices = getattr(completion, "choices", None)
        if not choices:
            raise RuntimeError("Qwen returned no completion choice.")
        message = getattr(choices[0], "message", None)
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            raise RuntimeError("Qwen returned empty final content.")
        if self._enable_thinking:
            final_match = re.search(
                r"<final>\s*(.*?)\s*</final>",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if final_match is None or not final_match.group(1).strip():
                raise RuntimeError("Qwen returned no isolated final answer.")
            content = final_match.group(1).strip()

        usage = getattr(completion, "usage", None)
        return GenerationResponse(
            answer=content,
            provider=self.provider_name,
            model_name=self.model_name,
            citations=(),
            usage=GenerationUsage(
                input_tokens=self._usage_value(usage, "prompt_tokens"),
                output_tokens=self._usage_value(usage, "completion_tokens"),
                total_tokens=self._usage_value(usage, "total_tokens"),
            ),
            generation_time_ms=round((time.perf_counter() - started) * 1000, 4),
            raw_response_id=(
                str(getattr(completion, "id", "") or "").strip() or None
            ),
        )

    @staticmethod
    def _usage_value(usage: Any, name: str) -> int | None:
        value = getattr(usage, name, None) if usage is not None else None
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def health_check(self) -> bool:
        return bool(self._api_key_configured and self._client and self._model_name)
