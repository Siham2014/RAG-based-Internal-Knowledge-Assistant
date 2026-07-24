from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(slots=True)
class ToolStatus:
    name: str
    command: list[str]
    available: bool
    output: str


def execute_command(
    name: str,
    command: list[str],
) -> ToolStatus:
    executable = command[0]

    if shutil.which(executable) is None:
        return ToolStatus(
            name=name,
            command=command,
            available=False,
            output=f"Commande introuvable : {executable}",
        )

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return ToolStatus(
            name=name,
            command=command,
            available=False,
            output="La commande a dépassé 30 secondes.",
        )
    except OSError as exc:
        return ToolStatus(
            name=name,
            command=command,
            available=False,
            output=str(exc),
        )

    combined_output = (
        result.stdout.strip()
        or result.stderr.strip()
        or "Commande exécutée sans sortie."
    )

    return ToolStatus(
        name=name,
        command=command,
        available=result.returncode == 0,
        output=combined_output,
    )


def display_status(status: ToolStatus) -> None:
    symbol = "OK" if status.available else "ERREUR"

    print("-" * 70)
    print(f"{status.name} : {symbol}")
    print(f"Commande : {' '.join(status.command)}")
    print(status.output)


def main() -> None:
    print("=" * 70)
    print("DIAGNOSTIC DE L'ENVIRONNEMENT DU BENCHMARK RAG")
    print("=" * 70)

    print(f"Système : {platform.system()} {platform.release()}")
    print(f"Architecture : {platform.machine()}")
    print(f"Python : {sys.version}")
    print(f"Exécutable Python : {sys.executable}")

    tools = [
        execute_command(
            name="Python",
            command=["python", "--version"],
        ),
        execute_command(
            name="pip",
            command=[
                sys.executable,
                "-m",
                "pip",
                "--version",
            ],
        ),
        execute_command(
            name="Docker",
            command=["docker", "--version"],
        ),
        execute_command(
            name="Docker Compose",
            command=[
                "docker",
                "compose",
                "version",
            ],
        ),
        execute_command(
            name="Git",
            command=["git", "--version"],
        ),
    ]

    for tool in tools:
        display_status(tool)

    required_tools = {
        "Python",
        "pip",
        "Docker",
        "Docker Compose",
    }

    missing_required_tools = [
        tool.name
        for tool in tools
        if (
            tool.name in required_tools
            and not tool.available
        )
    ]

    print()
    print("=" * 70)

    if missing_required_tools:
        print("ENVIRONNEMENT NON PRET")
        print(
            "Outils manquants ou non fonctionnels : "
            + ", ".join(missing_required_tools)
        )
    else:
        print("ENVIRONNEMENT PRET POUR POSTGRESQL + PGVECTOR")

    print("=" * 70)


if __name__ == "__main__":
    main()