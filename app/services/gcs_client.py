"""Cliente de Google Cloud Storage.

Envoltorio delgado sobre `google.cloud.storage`: aísla el import perezoso de la
librería y da un punto único que los tests sustituyen sin tocar la red. Lo
usan la publicación de entregables y el backend de storage `gcs`.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from app.config import PublishTarget

logger = logging.getLogger(__name__)

CONTENT_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".json": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".zip": "application/zip",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

_DEFAULT_CONTENT_TYPE = "application/octet-stream"

# El cliente es caro de construir (lee credenciales, abre una sesión HTTP) y
# los jobs corren en varios hilos: se cachea uno por proyecto y credencial.
_clients: dict[tuple[str | None, str | None], Any] = {}
_lock = threading.Lock()


def content_type_for(key: str) -> str:
    suffix = key[key.rfind(".") :].lower() if "." in key else ""
    return CONTENT_TYPES.get(suffix, _DEFAULT_CONTENT_TYPE)


def object_name(target: PublishTarget, key: str) -> str:
    """Nombre del objeto en el bucket para una clave: antepone `GCS_PREFIX`."""
    prefix = target.prefix.strip("/")
    return f"{prefix}/{key}" if prefix else key


def object_uri(target: PublishTarget, key: str) -> str:
    return f"gs://{target.bucket}/{object_name(target, key)}"


def upload_bytes(target: PublishTarget, key: str, data: bytes) -> str:
    """Sube el contenido bajo la clave dada y devuelve su URI `gs://`."""
    if not target.bucket:
        raise ValueError("No hay GCS_BUCKET configurado.")

    blob = bucket(target).blob(object_name(target, key))
    blob.upload_from_string(data, content_type=content_type_for(key))
    uri = object_uri(target, key)
    logger.info("Subido a %s (%d bytes)", uri, len(data))
    return uri


def bucket(target: PublishTarget) -> Any:
    """Handle del bucket del destino, con el cliente cacheado."""
    return client(target.project, target.credentials_path).bucket(target.bucket)


def client(project: str | None, credentials_path: Path | None) -> Any:
    from google.cloud import storage

    from app.services.gcp_auth import credentials_and_project

    cache_key = (project, str(credentials_path) if credentials_path else None)
    with _lock:
        cached = _clients.get(cache_key)
    if cached is not None:
        return cached

    credentials, resolved_project = credentials_and_project(credentials_path, project)
    built = storage.Client(project=resolved_project, credentials=credentials)
    with _lock:
        return _clients.setdefault(cache_key, built)


def reset_clients() -> None:
    """Olvida los clientes cacheados. Solo para tests y recarga de settings."""
    with _lock:
        _clients.clear()
