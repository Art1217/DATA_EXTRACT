"""Difusión en vivo del progreso de los jobs.

El pipeline ya emitía progreso mediante un callback síncrono
`(mensaje, avance)`. Aquí ese callback se convierte en eventos que los
clientes HTTP consumen por SSE, sin que los servicios tengan que enterarse.

El bus vive en memoria: si el proceso se reinicia, los suscriptores se caen y
tienen que volver a sondear el estado, que sí está persistido. Es la elección
correcta para un despliegue de un solo proceso; con varias réplicas habría que
sustituirlo por un bus externo, y este módulo es el único punto a cambiar.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from app.jobs.models import JobEvent

logger = logging.getLogger(__name__)

_QUEUE_LIMIT = 256


class EventBus:
    """Publicación por job con múltiples suscriptores."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[queue.Queue[JobEvent | None]]] = {}

    def publish(self, event: JobEvent) -> None:
        with self._lock:
            listeners = list(self._subscribers.get(event.job_id, ()))
        for listener in listeners:
            try:
                listener.put_nowait(event)
            except queue.Full:
                # Un cliente lento no debe frenar el pipeline: se descarta el
                # evento. El estado real siempre se puede consultar por GET.
                logger.debug("Suscriptor saturado en el job %s; evento descartado.", event.job_id)

    def close(self, job_id: str) -> None:
        """Señala el fin del stream a los suscriptores de un job."""
        with self._lock:
            listeners = list(self._subscribers.get(job_id, ()))
        for listener in listeners:
            try:
                listener.put_nowait(None)
            except queue.Full:
                pass

    @contextmanager
    def subscribe(self, job_id: str) -> Iterator[queue.Queue[JobEvent | None]]:
        channel: queue.Queue[JobEvent | None] = queue.Queue(maxsize=_QUEUE_LIMIT)
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(channel)
        try:
            yield channel
        finally:
            with self._lock:
                listeners = self._subscribers.get(job_id, [])
                if channel in listeners:
                    listeners.remove(channel)
                if not listeners:
                    self._subscribers.pop(job_id, None)
