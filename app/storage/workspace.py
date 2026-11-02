"""Espacio de trabajo temporal, privado por unidad de ejecución.

Resuelve un defecto de corrección del pipeline anterior: las páginas
renderizadas se escribían en `Temp/page_<n>.jpg`, con la clave formada
únicamente por el número de página, y al terminar se borraban *todos* los JPG
del directorio compartido. Con un solo documento a la vez pasaba inadvertido;
en un backend, dos jobs simultáneos se pisaban las imágenes y le enviaban al
modelo páginas del documento equivocado.

Cada job recibe ahora su propio directorio y solo borra el suyo.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Workspace:
    """Directorio temporal aislado, propiedad de una única ejecución."""

    root: Path

    def path(self, *parts: str) -> Path:
        """Ruta dentro del workspace, creando los directorios intermedios."""
        target = self.root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def subdir(self, name: str) -> Path:
        target = self.root / name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@contextmanager
def workspace(parent: Path, *, prefix: str = "job") -> Iterator[Workspace]:
    """Crea un workspace y lo elimina al salir, incluso si hubo excepción."""
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=parent))
    try:
        yield Workspace(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def purge_stale(parent: Path, *, older_than_hours: float = 24.0) -> int:
    """Borra workspaces huérfanos de ejecuciones que murieron sin limpiar.

    Se llama al arrancar el servidor: un corte de luz o un `kill -9` dejan
    directorios que nadie más va a reclamar.
    """
    import time

    if not parent.is_dir():
        return 0
    cutoff = time.time() - older_than_hours * 3600
    removed = 0
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError as exc:  # el directorio puede desaparecer entre medias
            logger.debug("No se pudo purgar %s: %s", child, exc)
    return removed
