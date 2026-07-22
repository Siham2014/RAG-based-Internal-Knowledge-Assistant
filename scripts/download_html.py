from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "html_web"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

URLS = [
    "https://learn.microsoft.com/en-us/azure/architecture/",
    "https://learn.microsoft.com/en-us/azure/architecture/browse/",
    "https://learn.microsoft.com/en-us/azure/architecture/patterns/",
    "https://learn.microsoft.com/en-us/azure/architecture/ai-ml/",
    "https://learn.microsoft.com/en-us/azure/architecture/data-guide/",
]


def create_filename(url: str) -> str:
    path = urlparse(url).path.strip("/")

    if not path:
        return "index.html"

    filename = path.replace("/", "_")
    return f"{filename}.html"


def download_page(url: str) -> None:
    print(f"Téléchargement : {url}")

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Azure-RAG-Student-Project/1.0"
        },
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Sur Microsoft Learn, le contenu utile se trouve normalement ici.
    main_content = soup.find("main")

    if main_content is None:
        raise ValueError("La balise principale <main> est introuvable.")

    # Suppression des éléments inutiles pour le futur RAG.
    for element in main_content.select(
        "script, style, nav, form, button, "
        ".feedback-section, .metadata, .contributors"
    ):
        element.decompose()

    output_path = OUTPUT_DIR / create_filename(url)

    output_path.write_text(
        str(main_content),
        encoding="utf-8",
    )

    print(f"Enregistré : {output_path.name}")


def main() -> None:
    successes = 0

    for url in URLS:
        try:
            download_page(url)
            successes += 1
        except requests.RequestException as error:
            print(f"Erreur réseau pour {url} : {error}")
        except Exception as error:
            print(f"Erreur pour {url} : {error}")

    print(f"\nTerminé : {successes}/{len(URLS)} pages téléchargées.")


if __name__ == "__main__":
    main()