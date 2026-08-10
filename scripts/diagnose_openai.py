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
    print("DIAGNOSTIC OPENAI")
    print("=" * 100)

    print("Model :", model_name)
    print("Max completion tokens : 200")

    start = time.perf_counter()

    response = (
        client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Reply with exactly this sentence: "
                        "The test works."
                    ),
                }
            ],
            max_completion_tokens=200,
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    choice = response.choices[0]
    message = choice.message

    print()
    print("=" * 100)
    print("RÉPONSE VISIBLE")
    print("=" * 100)

    print(
        repr(
            message.content
        )
    )

    print()
    print("=" * 100)
    print("MESSAGE COMPLET")
    print("=" * 100)

    try:
        print(
            message.model_dump()
        )
    except Exception:
        print(message)

    print()
    print("=" * 100)
    print("FINISH REASON")
    print("=" * 100)

    print(
        choice.finish_reason
    )

    print()
    print("=" * 100)
    print("USAGE")
    print("=" * 100)

    print(response.usage)

    if response.usage is not None:
        print(
            "Prompt tokens :",
            response.usage.prompt_tokens,
        )

        print(
            "Completion tokens :",
            response.usage.completion_tokens,
        )

        print(
            "Total tokens :",
            response.usage.total_tokens,
        )

        details = getattr(
            response.usage,
            "completion_tokens_details",
            None,
        )

        if details is not None:
            print(
                "Reasoning tokens :",
                getattr(
                    details,
                    "reasoning_tokens",
                    None,
                ),
            )

    print()
    print(
        f"Temps : {elapsed:.2f} s"
    )


if __name__ == "__main__":
    main()