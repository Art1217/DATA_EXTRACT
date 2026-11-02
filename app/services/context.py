"""Contexto de ejecución compartido por todas las etapas.

Agrupa lo que cualquier etapa necesita —configuración, storage, directorio
temporal propio, reporte de progreso y señal de cancelación— en un solo objeto
que se pasa hacia abajo. Evita que cada servicio reciba media docena de
parámetros sueltos y, sobre todo, evita que los servicios busquen esas
dependencias por su cuenta con imports globales.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from app.config import Settings
from app.domain.errors import JobCancelledError
from app.storage.base import Storage
from app.storage.workspace import Workspace

ProgressCallback = Callable[[str, float], None]
"""Firma heredada del pipeline original: (mensaje, avance entre 0 y 1)."""


class Progress:
    """Reporta avance dentro de una franja del total.

    Cada etapa informa su propio avance de 0 a 1 sin saber qué porción del
    trabajo global representa. Quien la compone le entrega una subfranja con
    `slice()`. Así desaparecen los números mágicos del pipeline anterior
    (`0.10 + 0.70 * (completed / n_blocks)`), que había que recalcular a mano
    cada vez que se añadía un paso.
    """

    __slots__ = ("_callback", "_start", "_span")

    def __init__(
        self, callback: ProgressCallback | None = None, start: float = 0.0, end: float = 1.0
    ) -> None:
        self._callback = callback
        self._start = start
        self._span = max(0.0, end - start)

    def report(self, message: str, fraction: float = 0.0) -> None:
        if self._callback is None:
            return
        local = min(max(fraction, 0.0), 1.0)
        self._callback(message, self._start + self._span * local)

    def slice(self, start: float, end: float) -> Progress:
        """Sub-progreso que ocupa `[start, end]` de la franja actual."""
        return Progress(
            self._callback,
            self._start + self._span * start,
            self._start + self._span * end,
        )

    @staticmethod
    def none() -> Progress:
        """Progreso que no informa a nadie: útil en tests y scripts."""
        return Progress(None)


class CancellationToken:
    """Señal cooperativa de cancelación.

    Cooperativa porque no se puede matar un hilo a medias sin dejar
    artefactos corruptos: el trabajo comprueba la señal entre bloques, que es
    donde el estado es consistente.
    """

    __slots__ = ("_event",)

    def __init__(self, event: threading.Event | None = None) -> None:
        self._event = event or threading.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise JobCancelledError("El job fue cancelado.")

    @staticmethod
    def none() -> CancellationToken:
        return CancellationToken()


@dataclass(frozen=True, slots=True)
class StageContext:
    """Todo lo que una etapa necesita para ejecutarse."""

    settings: Settings
    storage: Storage
    workspace: Workspace
    job_id: str | None = None
    """Job que originó la ejecución, si la hubo. Se registra en la fila de
    trazabilidad de BigQuery para poder volver del dato al trabajo que lo
    produjo; es `None` cuando una etapa se invoca desde un script o un test."""
    batch: str | None = None
    """Corrida que agrupa varios documentos, para publicarlos juntos.

    Un lote no existe en el backend: cada documento es un job independiente.
    Quien los agrupa —un cliente que sube 30 archivos de una vez— genera el
    identificador una sola vez y lo manda en cada job; así los 30 caen bajo la
    misma carpeta del bucket. Sin él, la publicación usa la fecha y hora del
    momento, que para un job suelto es exactamente lo mismo."""
    # `default_factory` y no un valor directo: un `CancellationToken()` como
    # default se evaluaría una sola vez y todos los contextos compartirían el
    # mismo `Event`, de modo que cancelar un job cancelaría a los demás.
    progress: Progress = field(default_factory=Progress.none)
    cancellation: CancellationToken = field(default_factory=CancellationToken.none)

    def with_progress(self, start: float, end: float) -> StageContext:
        """Copia del contexto acotada a una subfranja de progreso."""
        return replace(self, progress=self.progress.slice(start, end))
