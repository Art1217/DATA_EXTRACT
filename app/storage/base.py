"""Puerto de persistencia de artefactos.

`Storage` es la única interfaz que los servicios conocen. La implementan
`LocalStorage` sobre el sistema de archivos y `GCSStorage` sobre un bucket de
Google Cloud Storage, y ningún servicio sabe cuál está corriendo.

Dos decisiones que condicionan el diseño:

* **Escrituras atómicas.** Un job interrumpido a mitad de un `write_json` no
  debe dejar un artefacto truncado que una etapa posterior lea como válido.
* **`materialize()`.** PyMuPDF y Document AI necesitan una ruta de archivo
  real, no un stream. En local eso es la ruta directa; en un bucket implicará
  descargar a un temporal. El contrato es el mismo en ambos casos, así que
  quien lo usa no necesita saber cuál está corriendo.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO, Any

from app.domain.errors import ArtifactNotFoundError


def validate_key(key: str) -> str:
    """Rechaza claves que puedan escapar de la raíz del storage.

    Las claves se construyen en `app/storage/keys.py`, pero algunas incorporan
    identificadores que vienen del exterior; esto evita que un `document_id`
    manipulado se convierta en un salto de directorio.
    """
    if not key or key.startswith("/") or key.endswith("/"):
        raise ValueError(f"Clave de storage inválida: {key!r}")
    parts = key.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Clave de storage inválida: {key!r}")
    if "\\" in key or ":" in key:
        raise ValueError(f"Clave de storage inválida: {key!r}")
    return key


class Storage(ABC):
    """Almacén de artefactos direccionados por clave."""

    # --- Lectura ----------------------------------------------------------

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """Lanza `ArtifactNotFoundError` si la clave no existe."""

    @abstractmethod
    def open(self, key: str) -> IO[bytes]:
        """Abre el artefacto para lectura en streaming."""

    @abstractmethod
    def size(self, key: str) -> int: ...

    @abstractmethod
    def list(self, prefix: str) -> Iterator[str]:
        """Claves existentes bajo un prefijo, en orden alfabético."""

    # --- Escritura --------------------------------------------------------

    @abstractmethod
    def write_bytes(self, key: str, data: bytes) -> str:
        """Escribe de forma atómica y devuelve la clave."""

    @abstractmethod
    def import_file(self, source: Path, key: str, *, move: bool = False) -> str:
        """Incorpora al storage un archivo producido en disco local.

        `move=True` para temporales que ya no se van a reutilizar.
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """Borra la clave. No falla si no existe."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Borra todo lo que cuelgue del prefijo. Devuelve cuántas claves."""

    # --- Acceso como archivo local ---------------------------------------

    @abstractmethod
    def materialize(self, key: str) -> AbstractContextManager[Path]:
        """Expone el artefacto como una ruta de archivo real y legible.

        Necesario para las librerías que solo aceptan rutas (PyMuPDF,
        Document AI). En implementaciones remotas el archivo temporal se
        limpia al salir del contexto.
        """

    # --- Conveniencias JSON ----------------------------------------------
    # Implementadas sobre las primitivas: no hay que reescribirlas por backend.

    def read_json(self, key: str) -> Any:
        return json.loads(self.read_bytes(key).decode("utf-8"))

    def read_json_or_none(self, key: str) -> Any | None:
        """Igual que `read_json` pero devuelve `None` si el artefacto no está.

        Es el acceso natural para las etapas que reutilizan resultados
        previos cuando existen.
        """
        try:
            return self.read_json(key)
        except ArtifactNotFoundError:
            return None

    def write_json(self, key: str, payload: Any) -> str:
        data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        return self.write_bytes(key, data)
