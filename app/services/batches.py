"""Corridas: agrupar los jobs de un mismo lote y poder volver sobre ellos.

Un lote no existe como entidad en el backend —cada documento es un job
independiente— pero **sí queda registrado**: quien agrupa manda el mismo
identificador en cada job y ahí se queda, en `payload.batch`. Esto lo lee al
revés: recorre los jobs, los agrupa por ese identificador y reconstruye la
corrida.

Reconstruir en vez de guardar una tabla de lotes es deliberado. Una tabla
sería un segundo registro que mantener en sincronía con los jobs, con la
posibilidad de que digan cosas distintas; aquí el job es la única fuente y una
corrida es una vista sobre ella. El costo es recorrer `Storage/jobs/`, que es
un archivo pequeño por job y solo se paga al abrir la pantalla de resultados.

Sirve para elegir una corrida de hace tres días y bajarse su ZIP.
"""

from __future__ import annotations

import logging
from datetime import datetime

from app.jobs.models import Job, JobType
from app.jobs.store import JobStore
from app.storage import keys
from app.storage.base import Storage

logger = logging.getLogger(__name__)

PROCESSING_TYPES = frozenset({JobType.PIPELINE, JobType.AUTO})
"""Los jobs que procesan un documento entero. Los demás —empaquetar— también
llevan la etiqueta del lote, pero contarlos como documentos inflaría el total."""

_SCAN_LIMIT = 5000


def list_batches(store: JobStore, storage: Storage, *, limit: int = 50) -> list[dict]:
    """Corridas conocidas, de la más reciente a la más antigua."""
    corridas: dict[str, dict[str, Job]] = {}
    momentos: dict[str, datetime] = {}

    for job in store.list(limit=_SCAN_LIMIT):
        batch = (job.payload or {}).get("batch")
        if not batch or job.type not in PROCESSING_TYPES:
            continue
        _anotar(corridas.setdefault(batch, {}), job)
        momentos[batch] = min(momentos.get(batch, _naive(job.created_at)), _naive(job.created_at))

    salida = [
        _resumen(batch, intentos, momentos[batch], storage) for batch, intentos in corridas.items()
    ]
    salida.sort(key=lambda c: c["created_at"], reverse=True)
    return salida[:limit]


def batch_analyses(store: JobStore, batch: str) -> dict[str, list[str]]:
    """`{documento: [análisis]}` que produjo esa corrida y solo esa.

    Un documento acumula en el storage todo lo que se le haya hecho alguna
    vez: el mismo informe puede tener `resultados` de una corrida y
    `inventarios` de otra. Empaquetar "lo que hay" metería en el ZIP de una
    corrida análisis que nadie pidió en ella. Los jobs sí saben qué se pidió
    en cada corrida, así que se lee de ahí.
    """
    salida: dict[str, set[str]] = {}
    for job in store.list(limit=_SCAN_LIMIT):
        if (job.payload or {}).get("batch") != batch or job.type not in PROCESSING_TYPES:
            continue
        if job.document_id:
            salida.setdefault(job.document_id, set()).update(_analyses_of(job))
    return {documento: sorted(claves) for documento, claves in salida.items()}


def _analyses_of(job: Job) -> set[str]:
    """Análisis que cubrió un job, sea de un tipo concreto o del modo automático."""
    resultado = job.result or {}
    if "results" in resultado:  # automático: varios análisis por documento
        return {
            item.get("analysis_type")
            for item in (resultado.get("results") or [])
            if item.get("analysis_type")
        }
    return {job.analysis_type} if job.analysis_type else set()


def batch_detail(store: JobStore, storage: Storage, batch: str) -> dict:
    """Documentos de una corrida, con el desenlace de cada uno."""
    intentos: dict[str, Job] = {}
    momento = datetime.max

    for job in store.list(limit=_SCAN_LIMIT):
        if (job.payload or {}).get("batch") != batch or job.type not in PROCESSING_TYPES:
            continue
        _anotar(intentos, job)
        momento = min(momento, _naive(job.created_at))

    resumen = _resumen(batch, intentos, momento, storage)
    return resumen | {"items": [_documento(storage, job) for job in intentos.values()]}


# --- Piezas ---------------------------------------------------------------


def _anotar(intentos: dict[str, Job], job: Job) -> None:
    """Guarda el intento **más reciente** de cada documento de la corrida.

    Reprocesar un documento crea un job nuevo con la misma etiqueta de lote, y
    eso es lo correcto: el historial tiene que conservar el intento fallido con
    su hora y su motivo. Pero una corrida se cuenta por documentos, no por
    intentos. Contando jobs, un lote de 18 documentos del que se relanzaron 9
    acabaría diciendo "18 completados y 9 con error" — 27 desenlaces para 18
    documentos, y un lote enteramente correcto pintado de rojo.

    Aquí solo cuenta el último: es el que dice cómo quedó el documento. Los
    anteriores siguen en el historial de jobs, que es donde se preguntan.
    """
    if not job.document_id:
        return
    previo = intentos.get(job.document_id)
    if previo is None or job.created_at > previo.created_at:
        intentos[job.document_id] = job


def _resumen(batch: str, intentos: dict[str, Job], momento: datetime, storage: Storage) -> dict:
    estados = [job.status for job in intentos.values()]
    return {
        "batch": batch,
        "created_at": momento.isoformat(),
        "documents": len(intentos),
        "succeeded": sum(1 for e in estados if e.value == "succeeded"),
        "failed": sum(1 for e in estados if e.is_terminal and e.value != "succeeded"),
        "bundle_ready": storage.exists(keys.batch_bundle(batch)),
    }


def _documento(storage: Storage, job: Job) -> dict:
    resultado = job.result or {}
    extraction = resultado.get("extraction") or {}
    return {
        "document_id": job.document_id,
        "filename": filename_of(storage, job.document_id or ""),
        "analysis_type": job.analysis_type,
        "status": job.status.value,
        "row_count": extraction.get("row_count"),
        "quality_status": (resultado.get("quality") or {}).get("status"),
        "error": job.error,
    }


def filename_of(storage: Storage, document_id: str) -> str | None:
    """Nombre original del archivo, leído de su ficha."""
    if not document_id:
        return None
    meta = storage.read_json_or_none(keys.document_meta(document_id))
    return (meta or {}).get("filename")


def _naive(value: datetime) -> datetime:
    """Sin zona horaria, para poder comparar con `datetime.max`."""
    return value.replace(tzinfo=None) if value.tzinfo else value
