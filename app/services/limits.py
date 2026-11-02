"""Techo global de llamadas simultáneas al modelo.

El paralelismo efectivo del backend es multiplicativo: `job_workers` jobs a la
vez, cada uno abriendo su propio `ThreadPoolExecutor` de `busqueda_workers`
hilos. Con los valores por defecto son seis llamadas concurrentes, y sube en
cuanto alguien toca la configuración. Este semáforo pone un límite absoluto,
independiente de cuántos jobs haya en vuelo, para no agotar la cuota de la API.

Es un semáforo de proceso. Cuando el backend se despliegue con varias réplicas
habrá que sustituirlo por un limitador compartido; el punto de intervención
queda acotado a este módulo.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from app.config import get_settings

_lock = threading.Lock()
_semaphore: threading.BoundedSemaphore | None = None


def _get_semaphore() -> threading.BoundedSemaphore:
    global _semaphore
    if _semaphore is None:
        with _lock:
            if _semaphore is None:
                _semaphore = threading.BoundedSemaphore(get_settings().max_concurrent_llm_calls)
    return _semaphore


@contextmanager
def llm_slot() -> Iterator[None]:
    """Reserva un cupo de llamada al modelo mientras dure el bloque."""
    semaphore = _get_semaphore()
    semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()


def reset() -> None:
    """Descarta el semáforo para que se reconstruya con la configuración nueva.

    Solo para tests.
    """
    global _semaphore
    with _lock:
        _semaphore = None
