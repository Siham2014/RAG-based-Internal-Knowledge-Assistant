import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def main() -> None:
    output_directory = (
        PROJECT_ROOT
        / "data"
        / "staging"
        / "parser_test"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "azure_security.pdf"
    )

    pdf = canvas.Canvas(
        str(output_path),
        pagesize=A4,
    )

    pdf.setTitle(
        "Azure Security Guide"
    )

    pdf.setAuthor(
        "Microsoft"
    )

    width, height = A4

    pdf.setFont(
        "Helvetica-Bold",
        18,
    )

    pdf.drawString(
        70,
        height - 80,
        "Azure Security Guide",
    )

    pdf.setFont(
        "Helvetica",
        11,
    )

    pdf.drawString(
        70,
        height - 120,
        "Azure security protects cloud workloads and data.",
    )

    pdf.drawString(
        70,
        height - 145,
        "Use identity management, encryption and monitoring.",
    )

    pdf.setFont(
        "Helvetica-Bold",
        14,
    )

    pdf.drawString(
        70,
        height - 190,
        "Security recommendations",
    )

    pdf.setFont(
        "Helvetica",
        11,
    )

    pdf.drawString(
        90,
        height - 220,
        "- Enable multifactor authentication",
    )

    pdf.drawString(
        90,
        height - 245,
        "- Apply least privilege",
    )

    pdf.drawString(
        90,
        height - 270,
        "- Monitor security events",
    )

    pdf.showPage()

    pdf.setFont(
        "Helvetica-Bold",
        18,
    )

    pdf.drawString(
        70,
        height - 80,
        "Incident response",
    )

    pdf.setFont(
        "Helvetica",
        11,
    )

    pdf.drawString(
        70,
        height - 120,
        "Define a documented incident response process.",
    )

    pdf.drawString(
        70,
        height - 145,
        "Test the recovery process regularly.",
    )

    pdf.save()

    print(
        "PDF créé :",
        output_path,
    )


if __name__ == "__main__":
    main()