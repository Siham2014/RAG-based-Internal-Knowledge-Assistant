import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from src.ingestion.connectors.base import (
    BaseConnector,
    ConnectorValidationError,
    IngestionResult,
)
from src.ingestion.registry import ConnectorRegistry


@ConnectorRegistry.register("html")
class HtmlConnector(BaseConnector):
    """
    Connecteur chargé de télécharger une page HTML distante.

    Responsabilités :
    - valider l'URL ;
    - télécharger le document ;
    - vérifier que la réponse contient du HTML ;
    - sauvegarder le document brut ;
    - retourner un résultat d'ingestion standardisé.
    """

    def validate(self) -> None:
        url = self.source.input.get("url")

        if not url:
            raise ConnectorValidationError(
                f"La source '{self.source.id}' doit définir input.url."
            )

        parsed_url = urlparse(str(url))

        if parsed_url.scheme not in {"http", "https"}:
            raise ConnectorValidationError(
                f"L'URL de la source '{self.source.id}' doit utiliser "
                "http ou https."
            )

        if not parsed_url.netloc:
            raise ConnectorValidationError(
                f"L'URL de la source '{self.source.id}' est invalide : "
                f"{url}"
            )

        output_directory = self.source.output.get("directory")

        if not output_directory:
            raise ConnectorValidationError(
                f"La source '{self.source.id}' doit définir "
                "output.directory."
            )

    def ingest(self) -> IngestionResult:
        try:
            self.validate()

            url = str(self.source.input["url"])
            output_directory = self.resolve_path(
                self.source.output["directory"]
            )

            output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            response = self._download_with_retries(url)

            self._validate_response(
                response=response,
                url=url,
            )

            file_path = output_directory / self._build_filename(url)

            overwrite_existing = bool(
                self.ingestion_config
                .get("ingestion", {})
                .get("overwrite_existing", False)
            )

            if file_path.exists() and not overwrite_existing:
                status = "skipped"
            else:
                file_path.write_bytes(response.content)
                status = "success"

            return IngestionResult(
                source_id=self.source.id,
                status=status,
                files=[file_path.resolve()],
                metadata={
                    "connector": self.connector_type,
                    "url": url,
                    "output_file": str(file_path.resolve()),
                    "http_status": response.status_code,
                    "content_type": response.headers.get(
                        "Content-Type",
                        "",
                    ),
                    "content_length_bytes": len(response.content),
                    "source_metadata": self.source.metadata,
                },
            )

        except Exception as error:
            return IngestionResult(
                source_id=self.source.id,
                status="failed",
                errors=[str(error)],
                metadata={
                    "connector": self.connector_type,
                    "url": self.source.input.get("url"),
                },
            )

    def _download_with_retries(
        self,
        url: str,
    ) -> requests.Response:
        """
        Télécharge une page avec timeout et tentatives automatiques.
        """

        network_config = self.ingestion_config.get(
            "network",
            {},
        )

        timeout_seconds = int(
            network_config.get("timeout_seconds", 60)
        )

        max_retries = int(
            network_config.get("max_retries", 3)
        )

        retry_delay_seconds = float(
            network_config.get("retry_delay_seconds", 2)
        )

        user_agent = str(
            network_config.get(
                "user_agent",
                "Configurable-RAG-Ingestion/1.0",
            )
        )

        headers = {
            "User-Agent": user_agent,
            "Accept": (
                "text/html,"
                "application/xhtml+xml,"
                "application/xml;q=0.9,"
                "*/*;q=0.8"
            ),
        }

        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=timeout_seconds,
                    allow_redirects=True,
                )

                response.raise_for_status()

                return response

            except requests.RequestException as error:
                last_error = error

                if attempt < max_retries:
                    time.sleep(retry_delay_seconds)

        raise RuntimeError(
            f"Échec du téléchargement après {max_retries} tentative(s) : "
            f"{url}. Dernière erreur : {last_error}"
        )

    def _validate_response(
        self,
        response: requests.Response,
        url: str,
    ) -> None:
        """
        Vérifie que la réponse reçue correspond à une page HTML valide.
        """

        content_type = response.headers.get(
            "Content-Type",
            "",
        ).lower()

        if (
            "text/html" not in content_type
            and "application/xhtml+xml" not in content_type
        ):
            raise ValueError(
                f"La ressource ne semble pas être une page HTML : "
                f"{url}. Content-Type reçu : {content_type}"
            )

        minimum_content_length = int(
            self.ingestion_config
            .get("validation", {})
            .get("minimum_content_length", 100)
        )

        if len(response.content) < minimum_content_length:
            raise ValueError(
                f"Le contenu HTML téléchargé est trop court : "
                f"{len(response.content)} octets."
            )

        soup = BeautifulSoup(
            response.content,
            "html.parser",
        )

        if soup.find("html") is None:
            raise ValueError(
                f"Le document téléchargé ne contient pas de balise "
                f"<html> : {url}"
            )

    def _build_filename(
        self,
        url: str,
    ) -> str:
        """
        Produit un nom stable et unique à partir de l'URL.

        Exemple :
        azure-architecture-a1b2c3d4e5f6.html
        """

        parsed_url = urlparse(url)

        path_name = parsed_url.path.strip("/")

        if path_name:
            readable_name = path_name.replace("/", "-")
        else:
            readable_name = parsed_url.netloc

        readable_name = self._sanitize_filename(
            readable_name
        )

        url_hash = hashlib.sha256(
            url.encode("utf-8")
        ).hexdigest()[:12]

        return f"{readable_name}-{url_hash}.html"

    @staticmethod
    def _sanitize_filename(
        value: str,
    ) -> str:
        """
        Supprime les caractères non adaptés à un nom de fichier Windows.
        """

        forbidden_characters = '<>:"/\\|?*'

        sanitized_value = value

        for character in forbidden_characters:
            sanitized_value = sanitized_value.replace(
                character,
                "-",
            )

        sanitized_value = sanitized_value.strip(
            " .-_"
        )

        while "--" in sanitized_value:
            sanitized_value = sanitized_value.replace(
                "--",
                "-",
            )

        return sanitized_value or "document"