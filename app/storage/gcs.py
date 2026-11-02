"""Adaptador de storage sobre un bucket de Google Cloud Storage.

Con `STORAGE_BACKEND=gcs` todos los artefactos —el PDF, las clasificaciones,
los recortes, las extracciones, los jobs— viven en el bucket, y el disco del
servidor queda como espacio de trabajo desechable. Es lo que permite correr
la API en Cloud Run, donde el disco es efímero.

Las claves de `app/storage/keys.py` son literalmente el nombre del objeto (con
`GCS_PREFIX` delante si lo hay): el diseño por claves POSIX existía para que
este adaptador no tuviera que traducir nada.

Dos diferencias con el disco que conviene tener presentes:

* **Escrituras atómicas gratis.** Un objeto de GCS aparece entero o no
  aparece: no hace falta el baile de temporal + rename de `LocalStorage`.
* **`materialize()` descarga.** PyMuPDF y Document AI necesitan una ruta de
  archivo, así que el objeto se baja a un temporal que se borra al salir del
  contexto.
"""

from __future__ import annotations

import io
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from app.domain.errors import ArtifactNotFoundError
from app.services.gcs_client import content_type_for
from app.storage.base import Storage, validate_key


class GCSStorage(Storage):
    def __init__(self, bucket: Any, prefix: str = "") -> None:
        """`bucket` es un `google.cloud.storage.Bucket` (o un doble en tests)."""
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    def _name(self, key: str) -> str:
        key = validate_key(key)
        return f"{self._prefix}/{key}" if self._prefix else key

    def _key(self, name: str) -> str:
        return name[len(self._prefix) + 1 :] if self._prefix else name

    def _blob(self, key: str) -> Any:
        return self._bucket.blob(self._name(key))

    def _require(self, key: str) -> Any:
        blob = self._bucket.get_blob(self._name(key))
        if blob is None:
            raise ArtifactNotFoundError(key)
        return blob

    # --- Lectura ----------------------------------------------------------

    def exists(self, key: str) -> bool:
        return self._blob(key).exists()

    def read_bytes(self, key: str) -> bytes:
        return self._require(key).download_as_bytes()

    def open(self, key: str) -> IO[bytes]:
        return io.BytesIO(self.read_bytes(key))

    def size(self, key: str) -> int:
        return int(self._require(key).size or 0)

    def list(self, prefix: str) -> Iterator[str]:
        name = self._name(prefix).rstrip("/") + "/"
        keys = sorted(self._key(blob.name) for blob in self._bucket.list_blobs(prefix=name))
        yield from keys

    # --- Escritura --------------------------------------------------------

    def write_bytes(self, key: str, data: bytes) -> str:
        self._blob(key).upload_from_string(data, content_type=content_type_for(key))
        return key

    def import_file(self, source: Path, key: str, *, move: bool = False) -> str:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(f"No existe el archivo a importar: {source}")
        self._blob(key).upload_from_filename(str(source), content_type=content_type_for(key))
        if move:
            source.unlink(missing_ok=True)
        return key

    def delete(self, key: str) -> None:
        blob = self._bucket.get_blob(self._name(key))
        if blob is not None:
            blob.delete()

    def delete_prefix(self, prefix: str) -> int:
        name = self._name(prefix).rstrip("/") + "/"
        removed = 0
        for blob in list(self._bucket.list_blobs(prefix=name)):
            blob.delete()
            removed += 1
        return removed

    # --- Acceso como archivo local ---------------------------------------

    @contextmanager
    def materialize(self, key: str) -> Iterator[Path]:
        """Baja el objeto a un temporal y lo borra al salir."""
        data = self.read_bytes(key)
        # `delete=False`: en Windows un temporal abierto no se puede reabrir
        # por nombre, y PyMuPDF necesita abrirlo él mismo.
        with tempfile.NamedTemporaryFile(suffix=Path(key).suffix, delete=False) as handle:
            handle.write(data)
            path = Path(handle.name)
        try:
            yield path
        finally:
            path.unlink(missing_ok=True)
