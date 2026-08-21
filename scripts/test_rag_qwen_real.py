from __future__ import annotations

import os

from dotenv import load_dotenv

from src.pipeline import RAGPipeline


def main() -> None:
    load_dotenv()
    if not os.getenv("DASHSCOPE_API_KEY", "").strip():
        raise RuntimeError(
            "DASHSCOPE_API_KEY is missing. Add it to .env before running this real API test."
        )

    rag = RAGPipeline.from_settings(language="en")
    try:
        response = rag.answer("How can I improve Azure reliability?")
    finally:
        rag.unload_models()

    assert response.accepted, response.refusal_reason
    assert response.provider == "qwen", response.provider
    assert response.model_name == "qwen3.8-max", response.model_name
    assert response.answer.strip()
    assert response.citations
    assert response.citation_validation is not None
    assert response.citation_validation.valid

    print("Real Qwen RAG test passed.")
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model_name}")
    print(f"Citations: {len(response.citations)}")


if __name__ == "__main__":
    main()
