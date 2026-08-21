from __future__ import annotations

import os

from dotenv import load_dotenv

from src.generation import GenerationContext, GenerationRequest, QwenProvider


def main() -> None:
    load_dotenv(override=True)
    if not os.getenv("DASHSCOPE_API_KEY", "").strip():
        raise RuntimeError("Set DASHSCOPE_API_KEY in .env before running this test.")

    provider = QwenProvider(model_name="qwen3.8-max")
    request = GenerationRequest(
        question="What does the supplied context say about cloud reliability?",
        contexts=(
            GenerationContext(
                rank=1,
                chunk_id="qwen_test_chunk",
                source="qwen_test.md",
                content="Cloud reliability improves through redundancy and tested recovery.",
            ),
        ),
        language="en",
        max_output_tokens=180,
        temperature=0.0,
    )
    response = provider.generate(request)
    assert response.answer.strip()
    assert response.provider == "qwen"
    assert response.model_name == "qwen3.8-max"
    assert "[qwen_test.md:qwen_test_chunk]" in response.answer
    print("Qwen provider test passed.")
    print("Provider:", response.provider)
    print("Model:", response.model_name)
    print("Answer:", response.answer)


if __name__ == "__main__":
    main()

