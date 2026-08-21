from __future__ import annotations

from src.generation import (
    BaseLLMProvider,
    GenerationContext,
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
    LLMManager,
)


class FailingQwen(BaseLLMProvider):
    calls = 0

    @property
    def provider_name(self) -> str:
        return "qwen"

    @property
    def model_name(self) -> str:
        return "qwen3.8-max"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        raise TimeoutError("simulated Qwen timeout")


class SuccessfulOpenAI(BaseLLMProvider):
    calls = 0

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return "gpt-5-nano"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        citation = request.contexts[0].citation_id
        return GenerationResponse(
            answer=f"Fallback answer {citation}",
            provider=self.provider_name,
            model_name=self.model_name,
            citations=(citation,),
            usage=GenerationUsage(),
            generation_time_ms=1.0,
        )


class SuccessfulQwen(BaseLLMProvider):
    calls = 0

    @property
    def provider_name(self) -> str:
        return "qwen"

    @property
    def model_name(self) -> str:
        return "qwen3.8-max"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        citation = request.contexts[0].citation_id
        return GenerationResponse(
            answer=f"Primary answer {citation}",
            provider=self.provider_name,
            model_name=self.model_name,
            citations=(citation,),
            usage=GenerationUsage(),
            generation_time_ms=1.0,
        )


class CitationFreeQwen(SuccessfulQwen):
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        return GenerationResponse(
            answer="I don't know based on the available internal documents.",
            provider=self.provider_name,
            model_name=self.model_name,
            citations=(),
            usage=GenerationUsage(),
            generation_time_ms=1.0,
        )


def main() -> None:
    request = GenerationRequest(
        question="Test provider chain",
        contexts=(
            GenerationContext(
                rank=1,
                chunk_id="chunk",
                source="source.md",
                content="Grounded context.",
            ),
        ),
    )

    successful_qwen = SuccessfulQwen()
    unused_openai = SuccessfulOpenAI()
    primary_manager = LLMManager(
        primary_provider=successful_qwen,
        fallback_providers=(unused_openai,),
        max_attempts_per_provider=1,
        retry_delay_seconds=0.0,
    )
    primary_response = primary_manager.generate(request)
    assert successful_qwen.calls == 1
    assert unused_openai.calls == 0
    assert primary_response.provider == "qwen"

    qwen = FailingQwen()
    openai = SuccessfulOpenAI()
    manager = LLMManager(
        primary_provider=qwen,
        fallback_providers=(openai,),
        max_attempts_per_provider=1,
        retry_delay_seconds=0.0,
    )
    response = manager.generate(request)
    assert qwen.calls == 1
    assert openai.calls == 1
    assert response.provider == "openai"
    assert [attempt.success for attempt in manager.last_attempts] == [False, True]

    citation_free_qwen = CitationFreeQwen()
    validated_openai = SuccessfulOpenAI()
    validation_manager = LLMManager(
        primary_provider=citation_free_qwen,
        fallback_providers=(validated_openai,),
        max_attempts_per_provider=2,
        retry_delay_seconds=0.0,
    )
    validated_response = validation_manager.generate(
        request,
        response_validator=lambda candidate: bool(candidate.citations),
    )
    assert citation_free_qwen.calls == 1
    assert validated_openai.calls == 1
    assert validated_response.provider == "openai"
    assert [attempt.success for attempt in validation_manager.last_attempts] == [False, True]
    print("Qwen primary short-circuit and OpenAI fallback tests passed.")


if __name__ == "__main__":
    main()
