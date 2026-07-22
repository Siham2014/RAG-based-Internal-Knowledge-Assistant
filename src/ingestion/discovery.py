import importlib
import pkgutil
from types import ModuleType

import src.ingestion.connectors as connectors_package


class ConnectorDiscoveryError(Exception):
    """Erreur pendant la découverte automatique des connecteurs."""


def discover_connectors() -> list[ModuleType]:
    """
    Importe automatiquement tous les modules présents dans
    src.ingestion.connectors dont le nom se termine par
    '_connector'.

    L'import déclenche les décorateurs :
        @ConnectorRegistry.register(...)
    """

    imported_modules: list[ModuleType] = []

    package_path = connectors_package.__path__
    package_name = connectors_package.__name__

    discovered_modules = pkgutil.iter_modules(
        package_path,
        prefix=f"{package_name}.",
    )

    for module_info in discovered_modules:
        module_name = module_info.name

        if not module_name.endswith("_connector"):
            continue

        try:
            imported_module = importlib.import_module(
                module_name
            )

            imported_modules.append(imported_module)

        except Exception as error:
            raise ConnectorDiscoveryError(
                f"Impossible de charger le connecteur "
                f"'{module_name}' : {error}"
            ) from error

    return imported_modules