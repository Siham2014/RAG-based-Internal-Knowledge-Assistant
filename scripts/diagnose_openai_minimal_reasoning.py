from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv(override=True)

    api_key = str(
        os.getenv(
            "OPENAI_API_KEY",
            "",
        )
    ).strip()

    model_name = str(
        os.getenv(
            "OPENAI_MODEL",
            "gpt-5-nano",
        )
    ).strip()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY est absent."
        )

    client = OpenAI(
        api_key=api_key,
        timeout=60.0,
        max_retries=0,
    )

    print("=" * 100)
    print("DIAGNOSTIC OPENAI — MINIMAL REASONING")
    print("=" * 100)
    print("Model :", model_name)
    print("Reasoning effort : minimal")
    print("Max completion tokens : 100")

    start = time.perf_counter()

    try:
        response = (
            client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Reply exactly: "
                            "The test works."
                        ),
                    }
                ],
                reasoning_effort="minimal",
                max_completion_tokens=100,
            )
        )

    except Exception as error:
        print()
        print("=" * 100)
        print("ERREUR")
        print("=" * 100)
        print(
            type(error).__name__,
            ":",
            error,
        )
        raise SystemExit(1) from error

    elapsed = (
        time.perf_counter()
        - start
    )

    choice = response.choices[0]
    message = choice.message

    print()
    print("=" * 100)
    print("RÉPONSE")
    print("=" * 100)
    print(
        repr(
            message.content
        )
    )

    print()
    print("Finish reason :")
    print(
        choice.finish_reason
    )

    print()
    print("=" * 100)
    print("USAGE")
    print("=" * 100)

    usage = response.usage
    print(usage)

    if usage is not None:
        print(
            "Prompt tokens :",
            usage.prompt_tokens,
        )

        print(
            "Completion tokens :",
            usage.completion_tokens,
        )

        details = getattr(
            usage,
            "completion_tokens_details",
            None,
        )

        reasoning_tokens = (
            getattr(
                details,
                "reasoning_tokens",
                None,
            )
            if details is not None
            else None
        )

        print(
            "Reasoning tokens :",
            reasoning_tokens,
        )

        print(
            "Total tokens :",
            usage.total_tokens,
        )

    print()
    print(
        f"Temps : {elapsed:.2f} s"
    )


if __name__ == "__main__":
    main()