from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "pdf"
PDF_DIR.mkdir(parents=True, exist_ok=True)


PDF_SOURCES = [
    {
        "name": "enterprise_cloud_strategy.pdf",
        "url": (
            "https://info.microsoft.com/rs/157-GQE-382/images/"
            "EN-US-CNTNT-ebook-Enterprise_Cloud_Strategy_2nd_Edition_"
            "AzureInfrastructure.pdf"
        ),
    },
    {
        "name": "azure_serverless_computing_cookbook.pdf",
        "url": (
            "https://info.microsoft.com/rs/157-GQE-382/images/"
            "Azure%20Serverless%20Computing%20Cookbook.1.pdf"
        ),
    },
]


def download_pdf(name: str, url: str) -> bool:
    output_path = PDF_DIR / name

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Déjà présent : {name}")
        return True

    print(f"Téléchargement : {name}")

    try:
        response = requests.get(
            url,
            timeout=90,
            headers={
                "User-Agent": "Mozilla/5.0 Azure-RAG-Student-Project/1.0"
            },
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
            print(f"Erreur : le contenu reçu n'est pas un PDF : {name}")
            return False

        output_path.write_bytes(response.content)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Enregistré : {name} — {size_mb:.2f} Mo")
        return True

    except requests.RequestException as error:
        print(f"Erreur réseau pour {name} : {error}")
        return False


def main() -> None:
    successes = 0

    for source in PDF_SOURCES:
        if download_pdf(source["name"], source["url"]):
            successes += 1

    print(
        f"\nTerminé : {successes}/{len(PDF_SOURCES)} "
        "PDF téléchargés."
    )


if __name__ == "__main__":
    main()