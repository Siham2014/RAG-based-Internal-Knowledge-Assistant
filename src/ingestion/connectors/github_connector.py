import shutil
import subprocess
from pathlib import Path
from typing import Any

from src.ingestion.connectors.base import (
    BaseConnector,
    ConnectorValidationError,
    IngestionResult,
)
from src.ingestion.registry import ConnectorRegistry


@ConnectorRegistry.register("github")
class GitHubConnector(BaseConnector):
    """
    Connecteur chargé de récupérer un dépôt GitHub.

    Fonctionnement :
    - clone le dépôt s'il n'existe pas encore ;
    - met à jour le dépôt s'il existe déjà ;
    - récupère le commit Git courant ;
    - filtre les fichiers selon la configuration ;
    - retourne les fichiers bruts dans IngestionResult.
    """

    def validate(self) -> None:
        """
        Vérifie la configuration et la disponibilité de Git.
        """

        if shutil.which("git") is None:
            raise ConnectorValidationError(
                "Git n'est pas installé ou n'est pas disponible "
                "dans la variable PATH."
            )

        input_config = self.source.input

        repository_url = input_config.get("repository_url")

        if not repository_url:
            raise ConnectorValidationError(
                f"La source '{self.source.id}' doit définir "
                "input.repository_url."
            )

        destination = input_config.get("destination")

        if not destination:
            raise ConnectorValidationError(
                f"La source '{self.source.id}' doit définir "
                "input.destination."
            )

        branch = input_config.get("branch", "main")

        if not isinstance(branch, str) or not branch.strip():
            raise ConnectorValidationError(
                f"input.branch est invalide pour "
                f"la source '{self.source.id}'."
            )

        extensions = self.source.filters.get(
            "extensions",
            [],
        )

        if extensions and not isinstance(
            extensions,
            list,
        ):
            raise ConnectorValidationError(
                "filters.extensions doit être une liste."
            )

        excluded_directories = self.source.filters.get(
            "exclude_directories",
            [],
        )

        if excluded_directories and not isinstance(
            excluded_directories,
            list,
        ):
            raise ConnectorValidationError(
                "filters.exclude_directories doit être une liste."
            )

    def ingest(self) -> IngestionResult:
        """
        Clone ou met à jour le dépôt, puis retourne
        les fichiers correspondant aux filtres.
        """

        try:
            self.validate()

            input_config = self.source.input

            repository_url = str(
                input_config["repository_url"]
            )

            branch = str(
                input_config.get("branch", "main")
            )

            destination = self.resolve_path(
                input_config["destination"]
            )

            repository_subpath = str(
                input_config.get("repository_path", "")
            ).strip()

            if self._is_git_repository(destination):
                action = "updated"

                self._update_repository(
                    repository_directory=destination,
                    branch=branch,
                )
            else:
                action = "cloned"

                self._prepare_destination(destination)

                self._clone_repository(
                    repository_url=repository_url,
                    branch=branch,
                    destination=destination,
                )

            commit_sha = self._get_commit_sha(destination)

            commit_date = self._get_commit_date(destination)

            search_directory = destination

            if repository_subpath:
                search_directory = (
                    destination / repository_subpath
                ).resolve()

            if not search_directory.exists():
                raise ConnectorValidationError(
                    "Le sous-dossier GitHub configuré "
                    f"n'existe pas : {search_directory}"
                )

            files = self._discover_files(
                search_directory=search_directory,
            )

            total_size = sum(
                file_path.stat().st_size
                for file_path in files
            )

            extension_counts = self._count_extensions(
                files
            )

            status = "success" if files else "empty"

            return IngestionResult(
                source_id=self.source.id,
                status=status,
                files=files,
                metadata={
                    "connector": "github",
                    "repository_url": repository_url,
                    "branch": branch,
                    "commit_sha": commit_sha,
                    "commit_date": commit_date,
                    "destination": str(destination),
                    "repository_path": repository_subpath,
                    "action": action,
                    "file_count": len(files),
                    "total_size_bytes": total_size,
                    "extension_counts": extension_counts,
                },
            )

        except Exception as error:
            return IngestionResult(
                source_id=self.source.id,
                status="failed",
                errors=[str(error)],
                metadata={
                    "connector": "github",
                },
            )

    def _clone_repository(
        self,
        repository_url: str,
        branch: str,
        destination: Path,
    ) -> None:
        """
        Clone uniquement la dernière version de la branche.
        """

        command = [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            branch,
            "--single-branch",
            repository_url,
            str(destination),
        ]

        self._run_git_command(
            command=command,
            working_directory=self.project_root,
        )

    def _update_repository(
        self,
        repository_directory: Path,
        branch: str,
    ) -> None:
        """
        Synchronise le dépôt local avec la branche distante.

        Le reset garantit que le Raw correspond exactement
        au contenu distant.
        """

        self._run_git_command(
            command=[
                "git",
                "fetch",
                "--depth",
                "1",
                "origin",
                branch,
            ],
            working_directory=repository_directory,
        )

        self._run_git_command(
            command=[
                "git",
                "checkout",
                branch,
            ],
            working_directory=repository_directory,
        )

        self._run_git_command(
            command=[
                "git",
                "reset",
                "--hard",
                f"origin/{branch}",
            ],
            working_directory=repository_directory,
        )

    def _discover_files(
        self,
        search_directory: Path,
    ) -> list[Path]:
        """
        Découvre les fichiers autorisés dans le dépôt.
        """

        extensions = self._normalize_extensions(
            self.source.filters.get(
                "extensions",
                [],
            )
        )

        excluded_directories = {
            str(directory).strip().lower()
            for directory in self.source.filters.get(
                "exclude_directories",
                [],
            )
            if str(directory).strip()
        }

        excluded_directories.add(".git")

        recursive = bool(
            self.source.input.get("recursive", True)
        )

        iterator = (
            search_directory.rglob("*")
            if recursive
            else search_directory.glob("*")
        )

        files: list[Path] = []

        for path in iterator:
            if not path.is_file():
                continue

            relative_path = path.relative_to(
                search_directory
            )

            parent_directories = {
                part.lower()
                for part in relative_path.parts[:-1]
            }

            if (
                parent_directories
                & excluded_directories
            ):
                continue

            if (
                extensions
                and path.suffix.lower() not in extensions
            ):
                continue

            files.append(path.resolve())

        return sorted(
            files,
            key=lambda file_path: str(
                file_path
            ).lower(),
        )

    @staticmethod
    def _is_git_repository(
        destination: Path,
    ) -> bool:
        """
        Vérifie si le dossier contient un dépôt Git.
        """

        return (
            destination.exists()
            and destination.is_dir()
            and (destination / ".git").exists()
        )

    @staticmethod
    def _prepare_destination(
        destination: Path,
    ) -> None:
        """
        Prépare le dossier avant le clonage.
        """

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            if any(destination.iterdir()):
                raise ConnectorValidationError(
                    "Le dossier de destination existe mais "
                    "n'est pas un dépôt Git et n'est pas vide : "
                    f"{destination}"
                )

            destination.rmdir()

    def _get_commit_sha(
        self,
        repository_directory: Path,
    ) -> str:
        """
        Retourne le SHA du commit courant.
        """

        return self._run_git_command(
            command=[
                "git",
                "rev-parse",
                "HEAD",
            ],
            working_directory=repository_directory,
        )

    def _get_commit_date(
        self,
        repository_directory: Path,
    ) -> str:
        """
        Retourne la date ISO du commit courant.
        """

        return self._run_git_command(
            command=[
                "git",
                "show",
                "-s",
                "--format=%cI",
                "HEAD",
            ],
            working_directory=repository_directory,
        )

    @staticmethod
    def _run_git_command(
        command: list[str],
        working_directory: Path,
    ) -> str:
        """
        Exécute une commande Git et transforme les erreurs
        en messages lisibles.
        """

        process = subprocess.run(
            command,
            cwd=working_directory,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        if process.returncode != 0:
            error_message = (
                process.stderr.strip()
                or process.stdout.strip()
                or "Erreur Git inconnue."
            )

            raise ConnectorValidationError(
                f"Échec de la commande "
                f"'{' '.join(command)}' : "
                f"{error_message}"
            )

        return process.stdout.strip()

    @staticmethod
    def _normalize_extensions(
        extensions: list[Any],
    ) -> set[str]:
        """
        Transforme md, MD ou .md en .md.
        """

        normalized_extensions: set[str] = set()

        for extension in extensions:
            value = str(extension).strip().lower()

            if not value:
                continue

            if not value.startswith("."):
                value = f".{value}"

            normalized_extensions.add(value)

        return normalized_extensions

    @staticmethod
    def _count_extensions(
        files: list[Path],
    ) -> dict[str, int]:
        """
        Compte les fichiers par extension.
        """

        counts: dict[str, int] = {}

        for file_path in files:
            extension = (
                file_path.suffix.lower()
                if file_path.suffix
                else "[sans extension]"
            )

            counts[extension] = (
                counts.get(extension, 0) + 1
            )

        return dict(sorted(counts.items()))