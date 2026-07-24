from pathlib import Path
import shutil
import random

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\pc\Desktop\Azure corpus")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark_dataset"

MARKDOWN_SOURCE = RAW_DIR / "markdown" / "well-architected"
HTML_SOURCE = RAW_DIR / "html"
PDF_SOURCE = RAW_DIR / "pdf"

N_MARKDOWN = 10
RANDOM_SEED = 42

# Dossiers ou fichiers à exclure
EXCLUDED_FOLDER_NAMES = {
    "images",
    "image",
    "_images",
    "media",
    ".github",
    ".docutune",
}

EXCLUDED_MARKDOWN_NAMES = {
    "README.md",
    "SECURITY.md",
    "AGENTS.md",
    "ThirdPartyNotices.md",
    "LICENSE",
    "LICENSE-CODE",
    "CODEOWNERS",
}


# ============================================================
# FONCTIONS
# ============================================================

def reset_directory(directory: Path) -> None:
    """
    Supprime puis recrée le dossier du benchmark.
    """
    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(parents=True, exist_ok=True)


def is_valid_markdown(path: Path) -> bool:
    """
    Vérifie qu'un fichier Markdown peut être utilisé
    dans le benchmark.
    """
    if path.suffix.lower() != ".md":
        return False

    if path.name in EXCLUDED_MARKDOWN_NAMES:
        return False

    if any(part.lower() in EXCLUDED_FOLDER_NAMES for part in path.parts):
        return False

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    # Évite les fichiers presque vides
    if len(text.strip()) < 500:
        return False

    return True


def collect_markdown_files() -> list[Path]:
    """
    Récupère tous les fichiers Markdown valides.
    """
    files = [
        path
        for path in MARKDOWN_SOURCE.rglob("*.md")
        if is_valid_markdown(path)
    ]

    return sorted(files)


def select_representative_markdown(files: list[Path]) -> list[Path]:
    """
    Sélectionne un échantillon reproductible de fichiers Markdown.
    """
    if len(files) < N_MARKDOWN:
        raise ValueError(
            f"Seulement {len(files)} fichiers Markdown valides trouvés, "
            f"alors que {N_MARKDOWN} sont demandés."
        )

    random.seed(RANDOM_SEED)
    return sorted(random.sample(files, N_MARKDOWN))


def copy_files(files: list[Path], destination: Path, base_source: Path) -> None:
    """
    Copie les fichiers dans le dossier du benchmark.

    Le nom du fichier inclut une partie de son chemin afin
    d'éviter les doublons de noms.
    """
    destination.mkdir(parents=True, exist_ok=True)

    for source_file in files:
        relative_path = source_file.relative_to(base_source)

        safe_name = "__".join(relative_path.parts)
        target_file = destination / safe_name

        shutil.copy2(source_file, target_file)


def collect_files_by_extension(source: Path, extension: str) -> list[Path]:
    """
    Collecte tous les fichiers d'une extension donnée.
    """
    if not source.exists():
        return []

    return sorted(source.glob(f"*{extension}"))


def print_summary(
    markdown_files: list[Path],
    html_files: list[Path],
    pdf_files: list[Path],
) -> None:
    """
    Affiche un résumé du dataset produit.
    """
    print("=" * 65)
    print("CRÉATION DU DATASET DE BENCHMARK")
    print("=" * 65)

    print(f"\nMarkdown sélectionnés : {len(markdown_files)}")
    for file in markdown_files:
        print(f"  - {file.relative_to(MARKDOWN_SOURCE)}")

    print(f"\nHTML sélectionnés : {len(html_files)}")
    for file in html_files:
        print(f"  - {file.name}")

    print(f"\nPDF sélectionnés : {len(pdf_files)}")
    for file in pdf_files:
        print(f"  - {file.name}")

    total = len(markdown_files) + len(html_files) + len(pdf_files)

    print("\n" + "-" * 65)
    print(f"Total de documents : {total}")
    print(f"Dossier créé : {BENCHMARK_DIR}")
    print("=" * 65)


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable : {RAW_DIR}")

    reset_directory(BENCHMARK_DIR)

    markdown_output = BENCHMARK_DIR / "markdown"
    html_output = BENCHMARK_DIR / "html"
    pdf_output = BENCHMARK_DIR / "pdf"

    all_markdown_files = collect_markdown_files()
    selected_markdown_files = select_representative_markdown(
        all_markdown_files
    )

    html_files = collect_files_by_extension(HTML_SOURCE, ".html")
    pdf_files = collect_files_by_extension(PDF_SOURCE, ".pdf")

    copy_files(
        files=selected_markdown_files,
        destination=markdown_output,
        base_source=MARKDOWN_SOURCE,
    )

    copy_files(
        files=html_files,
        destination=html_output,
        base_source=HTML_SOURCE,
    )

    copy_files(
        files=pdf_files,
        destination=pdf_output,
        base_source=PDF_SOURCE,
    )

    print_summary(
        markdown_files=selected_markdown_files,
        html_files=html_files,
        pdf_files=pdf_files,
    )


if __name__ == "__main__":
    main()