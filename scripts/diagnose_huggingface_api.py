from __future__ import annotations

import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError


MODEL_NAME = "google/gemma-2-2b-it"
INFERENCE_PROVIDER = "featherless-ai"


def main() -> None:
    load_dotenv()

    token = str(
        os.getenv(
            "HF_TOKEN",
            "",
        )
    ).strip()

    if not token:
        raise RuntimeError(
            "HF_TOKEN est absent du fichier .env."
        )

    client = InferenceClient(
        provider=INFERENCE_PROVIDER,
        api_key=token,
        timeout=120,
    )

    print("=" * 90)
    print("DIAGNOSTIC HUGGING FACE")
    print("=" * 90)
    print("Modèle :", MODEL_NAME)
    print("Provider :", INFERENCE_PROVIDER)
    print()

    try:
        response = client.chat_completion(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Answer in one short sentence: "
                        "Who shares responsibility for "
                        "cloud sustainability?"
                    ),
                }
            ],
            max_tokens=40,
            stream=False,
        )

        print("RÉPONSE")
        print("-" * 90)
        print(
            response.choices[0]
            .message.content
        )

        print()
        print("Usage :", response.usage)

    except HfHubHTTPError as error:
        print("ERREUR HTTP")
        print("-" * 90)

        response = getattr(
            error,
            "response",
            None,
        )

        print(
            "Type :",
            type(error).__name__,
        )

        print(
            "Statut :",
            getattr(
                response,
                "status_code",
                None,
            ),
        )

        print(
            "Message :",
            str(error),
        )

        print(
            "Server message :",
            getattr(
                error,
                "server_message",
                None,
            ),
        )

        print(
            "Request ID :",
            getattr(
                error,
                "request_id",
                None,
            ),
        )

        if response is not None:
            print()
            print("Corps HTTP :")
            print(
                getattr(
                    response,
                    "text",
                    "",
                )
            )

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()