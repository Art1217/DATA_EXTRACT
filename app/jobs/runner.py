"""Cola de trabajos y trabajadores en proceso.

Modelo de ejecución: una cola FIFO y N hilos trabajadores dentro del mismo
proceso del API. Es suficiente porque el trabajo pesado ocurre esperando
respuestas de red del modelo, no en la CPU de Python, y evita montar una
infraestructura de cola externa para un despliegue de un solo servidor.

Los dos puntos que hay que respetar para que esto sea correcto:

* cada job recibe su **workspace propio**, porque las etapas escriben
  imágenes intermedias con nombres que solo dependen del número de página;
* la cancelación es **cooperativa**: el trabajo comprueba la señal entre
  bloques, que es donde el estado en disco es consistente.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from langsmith.run_helpers import trace

from app.config import Settings
from app.domain.errors import JobCancelledError, JobNotFoundError
from app.jobs.events import EventBus
from app.jobs.models import Job, JobEvent, JobStatus, JobType
from app.jobs.store import JobStore
from app.services.context import CancellationToken, Progress, StageContext
from app.storage.base import Storage
from app.storage.workspace import workspace

# Nombres legibles para la traza raíz de cada tipo de job en LangSmith.
_TRACE_NAMES: dict[JobType, str] = {
    JobType.CLASSIFY_GROUP: "Clasificación de Grupos (Fase 1)",
    JobType.CLASSIFY_ANALYSIS: "Clasificación de Análisis (Fase 2)",
    JobType.PLAN: "Plan de Procesamiento (Fase 1+2)",
    JobType.IDENTITY: "Identidad del Documento",
    JobType.SEARCH: "Búsqueda de Páginas",
    JobType.CUT: "Recorte Manual",
    JobType.EXTRACTION: "Extracción",
    JobType.QUALITY: "Control de Calidad",
    JobType.PIPELINE: "Pipeline Completo",
    JobType.AUTO: "Flujo Automático (Documento)",
    JobType.BUNDLE: "Empaquetado ZIP",
    JobType.BUNDLE_BATCH: "Empaquetado ZIP (corrida)",
}

logger = logging.getLogger(__name__)

Handler = Callable[[Job, StageContext], Any]

_PERSIST_INTERVAL_SECONDS = 1.0
_SHUTDOWN = object()


class JobRunner:
    """Encola trabajos y los ejecuta en hilos trabajadores."""

    def __init__(
        self,
        *,
        store: JobStore,
        storage: Storage,
        settings: Settings,
        handlers: dict[JobType, Handler],
        events: EventBus | None = None,
    ) -> None:
        self._store = store
        self._storage = storage
        self._settings = settings
        self._handlers = handlers
        self._events = events or EventBus()
        self._queue: queue.Queue[Any] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._tokens: dict[str, CancellationToken] = {}
        self._lock = threading.Lock()
        self._running = False

    @property
    def events(self) -> EventBus:
        return self._events

    # --- Ciclo de vida ----------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for index in range(self._settings.job_workers):
            worker = threading.Thread(target=self._work, name=f"job-worker-{index}", daemon=True)
            worker.start()
            self._workers.append(worker)
        logger.info("Runner iniciado con %d trabajadores.", len(self._workers))

    def shutdown(self, *, wait: bool = True, timeout: float = 10.0) -> None:
        if not self._running:
            return
        self._running = False
        for _ in self._workers:
            self._queue.put(_SHUTDOWN)
        if wait:
            for worker in self._workers:
                worker.join(timeout=timeout)
        self._workers.clear()
        logger.info("Runner detenido.")

    # --- API pública ------------------------------------------------------

    def submit(
        self,
        job_type: JobType,
        *,
        document_id: str | None = None,
        analysis_type: str | None = None,
        payload: dict[str, Any] | None = None,
        parent_job_id: str | None = None,
    ) -> Job:
        """Encola un trabajo y devuelve su ficha en estado `queued`."""
        if job_type not in self._handlers:
            raise ValueError(f"No hay handler registrado para '{job_type}'.")
        job = Job(
            job_id=uuid.uuid4().hex,
            type=job_type,
            document_id=document_id,
            analysis_type=analysis_type,
            payload=payload or {},
            parent_job_id=parent_job_id,
        )
        self._store.save(job)
        with self._lock:
            self._tokens[job.job_id] = CancellationToken()
        self._queue.put(job.job_id)
        logger.info("Job encolado %s (%s)", job.job_id, job_type)
        return job

    def cancel(self, job_id: str) -> Job:
        """Pide la cancelación. Surte efecto en el siguiente punto de control."""
        job = self._store.get(job_id)
        if job.is_terminal:
            return job

        with self._lock:
            token = self._tokens.get(job_id)
        if token is not None:
            token.cancel()

        if job.status is JobStatus.QUEUED:
            # Todavía no lo tomó ningún trabajador: se cierra ya, y el
            # trabajador lo descartará al sacarlo de la cola.
            self._finish(job, JobStatus.CANCELLED, message="Cancelado antes de empezar.")
        return self._store.get(job_id)

    # --- Bucle del trabajador --------------------------------------------

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SHUTDOWN:
                    return
                self._execute(str(item))
            except Exception:
                logger.exception("Fallo inesperado en el trabajador de jobs.")
            finally:
                self._queue.task_done()

    def _execute(self, job_id: str) -> None:
        try:
            job = self._store.get(job_id)
        except JobNotFoundError:
            logger.warning("El job %s desapareció antes de ejecutarse.", job_id)
            return

        if job.is_terminal:
            return

        with self._lock:
            token = self._tokens.get(job_id) or CancellationToken()
        if token.is_cancelled:
            self._finish(job, JobStatus.CANCELLED, message="Cancelado antes de empezar.")
            return

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        job.message = "En ejecución."
        self._store.save(job)
        self._publish(job)

        reporter = _ProgressReporter(job, self._store, self._events)
        try:
            with workspace(self._settings.temp_dir, prefix=f"job_{job.job_id[:8]}") as space:
                context = StageContext(
                    settings=self._settings,
                    storage=self._storage,
                    workspace=space,
                    job_id=job.job_id,
                    batch=job.payload.get("batch"),
                    progress=Progress(reporter),
                    cancellation=token,
                )
                # Una sola traza raíz por job: las llamadas al modelo dentro de
                # los servicios (cada una ya decorada con su propio
                # `@traceable`) quedan anidadas bajo ella en LangSmith, en vez
                # de aparecer como trazas sueltas sin relación entre sí.
                with trace(
                    _TRACE_NAMES.get(job.type, job.type.value),
                    run_type="chain",
                    metadata={
                        "job_id": job.job_id,
                        "document_id": job.document_id,
                        "analysis_type": job.analysis_type,
                    },
                    tags=[job.type.value],
                ):
                    result = self._handlers[job.type](job, context)
            self._finish(job, JobStatus.SUCCEEDED, result=result, message="Completado.")
        except JobCancelledError:
            self._finish(job, JobStatus.CANCELLED, message="Cancelado.")
        except Exception as exc:
            logger.exception("Job %s (%s) falló.", job.job_id, job.type)
            self._finish(
                job,
                JobStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                message="El job falló.",
            )
        finally:
            with self._lock:
                self._tokens.pop(job_id, None)

    # --- Internos ---------------------------------------------------------

    def _finish(
        self,
        job: Job,
        status: JobStatus,
        *,
        result: Any = None,
        error: str | None = None,
        message: str = "",
    ) -> None:
        job.status = status
        job.result = result
        job.error = error
        job.message = message or job.message
        job.progress = 1.0 if status is JobStatus.SUCCEEDED else job.progress
        job.finished_at = datetime.now(UTC)
        self._store.save(job)
        self._publish(job)
        self._events.close(job.job_id)

    def _publish(self, job: Job) -> None:
        self._events.publish(
            JobEvent(
                job_id=job.job_id,
                status=job.status,
                progress=job.progress,
                message=job.message,
            )
        )


class _ProgressReporter:
    """Traduce el callback de progreso del pipeline a estado persistido y eventos.

    Los eventos se emiten siempre —son baratos y en memoria— pero el job solo
    se reescribe a disco una vez por segundo: un documento largo genera cientos
    de avances y no aporta nada guardarlos todos.
    """

    __slots__ = ("_job", "_store", "_events", "_last_persist")

    def __init__(self, job: Job, store: JobStore, events: EventBus) -> None:
        self._job = job
        self._store = store
        self._events = events
        self._last_persist = 0.0

    def __call__(self, message: str, progress: float) -> None:
        self._job.message = message
        self._job.progress = min(max(progress, 0.0), 1.0)
        self._events.publish(
            JobEvent(
                job_id=self._job.job_id,
                status=self._job.status,
                progress=self._job.progress,
                message=message,
            )
        )
        now = time.monotonic()
        if now - self._last_persist >= _PERSIST_INTERVAL_SECONDS:
            self._last_persist = now
            self._store.save(self._job)
