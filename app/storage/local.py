"""Adaptador de storage sobre el sistema de archivos local."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from app.domain.errors import ArtifactNotFoundError
from app.storage.base import Storage, validate_key


class LocalStorage(Storage):
    """Guarda cada clave como un archivo bajo un directorio raíz.

    Las escrituras son atómicas: se escribe en un temporal del mismo
    directorio y se renombra con `os.replace`, que en Windows y POSIX es
    atómico dentro del mismo volumen. Un job interrumpido deja el artefacto
    anterior intacto en lugar de uno a medias.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, key: str) -> Path:
        return self._root.joinpath(*validate_key(key).split("/"))

    def _require(self, key: str) -> Path:
        path = self._resolve(key)
        if not path.is_file():
            raise ArtifactNotFoundError(key)
        return path

    # --- Lectura ----------------------------------------------------------

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def read_bytes(self, key: str) -> bytes:
        return self._require(key).read_bytes()

    def open(self, key: str) -> IO[bytes]:
        return self._require(key).open("rb")

    def size(self, key: str) -> int:
        return self._require(key).stat().st_size

    def list(self, prefix: str) -> Iterator[str]:
        base = self._root.joinpath(*validate_key(prefix).split("/"))
        if not base.is_dir():
            return
        for path in sorted(base.rglob("*")):
            if path.is_file():
                yield path.relative_to(self._root).as_posix()

    # --- Escritura --------------------------------------------------------

    def write_bytes(self, key: str, data: bytes) -> str:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(target, data)
        return key

    def import_file(self, source: Path, key: str, *, move: bool = False) -> str:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(f"No existe el archivo a importar: {source}")
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        if move:
            # `Path.replace` falla entre volúmenes distintos; `shutil.move` no.
            shutil.move(str(source), str(target))
        else:
            shutil.copy2(source, target)
        return key

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        path.unlink(missing_ok=True)
        self._prune(path.parent)

    def delete_prefix(self, prefix: str) -> int:
        base = self._root.joinpath(*validate_key(prefix).split("/"))
        if not base.exists():
            return 0
        removed = sum(1 for path in base.rglob("*") if path.is_file())
        shutil.rmtree(base, ignore_errors=True)
        self._prune(base.parent)
        return removed

    def _prune(self, folder: Path) -> None:
        """Borra hacia arriba las carpetas que se hayan quedado vacías.

        Sin esto, borrar cincuenta documentos dejaría un centenar de carpetas
        `quality/<análisis>/` vacías, que no ocupan nada pero hacen ilegible el
        árbol y aparentan que no se limpió nada.

        `rmdir` falla si la carpeta no está vacía, y ese fallo es justo la
        condición de parada: en cuanto una tiene contenido —o la está
        escribiendo otro hilo— se deja de subir.
        """
        root = self._root.resolve()
        while folder.resolve() != root and root in folder.resolve().parents:
            try:
                folder.rmdir()
            except OSError:
                return
            folder = folder.parent

    # --- Acceso como archivo local ---------------------------------------

    @contextmanager
    def materialize(self, key: str) -> Iterator[Path]:
        """En local el artefacto ya es un archivo: se entrega su ruta.

        No se copia nada, así que quien lo reciba debe tratarlo como solo
        lectura.
        """
        yield self._require(key)

    # --- Interno ----------------------------------------------------------

    _REPLACE_MAX_ATTEMPTS = 5
    _REPLACE_RETRY_SECONDS = 0.05

    @classmethod
    def _atomic_write(cls, target: Path, data: bytes) -> None:
        """Escribe a un temporal y lo renombra al destino, todo o nada.

        En Windows, `os.replace()` puede fallar de forma transitoria con
        `PermissionError` si el destino tiene un handle abierto de otro hilo
        en ese instante exacto — por ejemplo, un job leyendo el progreso de
        otro justo cuando ese otro intenta actualizarlo. A diferencia de
        POSIX, donde el rename es atómico sin importar handles abiertos,
        Windows puede rechazar el reemplazo con una violación de uso
        compartido. Bajo carga real (cientos de jobs escribiendo su progreso
        a la vez) esto se observó ocurriendo de verdad — no es un caso
        hipotético — y sin reintento, esa escritura se pierde en silencio y
        el job puede quedar congelado en el estado anterior para siempre.
        Se reintenta unas pocas veces con una espera mínima porque la causa
        es por definición transitoria: el lector suelta el handle en
        milisegundos.
        """
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

            for attempt in range(cls._REPLACE_MAX_ATTEMPTS):
                try:
                    os.replace(tmp_name, target)
                    return
                except PermissionError:
                    if attempt == cls._REPLACE_MAX_ATTEMPTS - 1:
                        raise
                    time.sleep(cls._REPLACE_RETRY_SECONDS)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
