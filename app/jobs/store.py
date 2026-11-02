"""Persistencia del estado de los jobs.

Se guarda un JSON por job en el storage, con la misma escritura atómica que
el resto de artefactos. La interfaz está separada de la implementación porque
este es el segundo punto —junto con el storage— donde un despliegue
distribuido exigirá otra tecnología (Firestore, Postgres): el resto del código
solo conoce `JobStore`.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta

from app.domain.errors import JobNotFoundError
from app.jobs.models import Job, JobStatus
from app.storage import keys
from app.storage.base import Storage

logger = logging.getLogger(__name__)


class JobStore(ABC):
    @abstractmethod
    def save(self, job: Job) -> Job: ...

    @abstractmethod
    def get(self, job_id: str) -> Job: ...

    @abstractmethod
    def find(self, job_id: str) -> Job | None: ...

    @abstractmethod
    def list(
        self,
        *,
        job_ids: list[str] | None = None,
        document_id: str | None = None,
        status: JobStatus | None = None,
        job_type: str | None = None,
        limit: int = 100,
    ) -> list[Job]: ...

    @abstractmethod
    def delete(self, job_id: str) -> None: ...


class FileJobStore(JobStore):
    """Almacén de jobs sobre la capa de storage.

    El cerrojo serializa las escrituras dentro del proceso: varios hilos
    trabajadores actualizan el progreso de sus jobs a la vez, y aunque cada
    job tenga su archivo, el bloqueo mantiene la caché de listado coherente.
    """

    def __init__(self, storage: Storage) -> None:
        self._storage = storage
        self._lock = threading.Lock()

    def save(self, job: Job) -> Job:
        with self._lock:
            self._storage.write_json(keys.job(job.job_id), job.model_dump(mode="json"))
        return job

    def get(self, job_id: str) -> Job:
        job = self.find(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def find(self, job_id: str) -> Job | None:
        payload = self._storage.read_json_or_none(keys.job(job_id))
        if payload is None:
            return None
        try:
            return Job.model_validate(payload)
        except ValueError as exc:
            logger.warning("Registro de job ilegible %s: %s", job_id, exc)
            return None

    def list(
        self,
        *,
        job_ids: list[str] | None = None,
        document_id: str | None = None,
        status: JobStatus | None = None,
        job_type: str | None = None,
        limit: int = 100,
    ) -> list[Job]:
        # Con `job_ids` se hacen lecturas directas por clave en vez de
        # recorrer todo `Storage/jobs/` — necesario para que consultar el
        # estado de un lote de N documentos sea una sola llamada barata y no
        # dependa de cuántos jobs existan en total en el historial.
        if job_ids:
            jobs = [job for job in (self.find(job_id) for job_id in job_ids) if job]
            jobs.sort(key=lambda job: job.created_at, reverse=True)
            return jobs[:limit]

        jobs = []
        for key in self._storage.list(keys.JOBS):
            if not key.endswith(".json"):
                continue
            payload = self._storage.read_json_or_none(key)
            if payload is None:
                continue
            try:
                job = Job.model_validate(payload)
            except ValueError:
                continue
            if document_id and job.document_id != document_id:
                continue
            if status and job.status != status:
                continue
            if job_type and job.type != job_type:
                continue
            jobs.append(job)
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs[:limit]

    def delete(self, job_id: str) -> None:
        self._storage.delete(keys.job(job_id))

    def purge_older_than(self, days: int) -> int:
        """Borra los jobs terminados más antiguos que `days`."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        removed = 0
        for job in self.list(limit=100_000):
            if job.is_terminal and job.created_at < cutoff:
                self.delete(job.job_id)
                removed += 1
        return removed

    def recover_interrupted(self) -> int:
        """Cierra los jobs que quedaron a medias por un reinicio del proceso.

        Un job en `running` sin proceso que lo ejecute nunca va a avanzar. Se
        marcan como fallidos al arrancar para que un cliente que estaba
        esperando reciba una respuesta en vez de sondear indefinidamente.
        """
        recovered = 0
        for job in self.list(limit=100_000):
            if job.is_terminal:
                continue
            job.status = JobStatus.FAILED
            job.error = "El servidor se reinició mientras el job estaba en ejecución."
            job.finished_at = datetime.now(UTC)
            self.save(job)
            recovered += 1
        if recovered:
            logger.warning("Se cerraron %d jobs interrumpidos por el reinicio.", recovered)
        return recovered
