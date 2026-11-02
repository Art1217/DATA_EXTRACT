"""Consulta y control de los trabajos asíncronos."""

from __future__ import annotations

import json
import queue
from collections.abc import Iterator

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from app.api.deps import JobStoreDep, RunnerDep
from app.api.schemas import JobResponse
from app.jobs.models import Job, JobStatus, JobType

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])

_HEARTBEAT_SECONDS = 15.0


def _to_response(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        type=job.type,
        status=job.status,
        document_id=job.document_id,
        analysis_type=job.analysis_type,
        progress=job.progress,
        message=job.message,
        result=job.result,
        error=job.error,
        parent_job_id=job.parent_job_id,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        duration_seconds=job.duration_seconds,
    )


@router.get("", response_model=list[JobResponse])
def list_jobs(
    store: JobStoreDep,
    job_id: list[str] | None = Query(default=None),
    document_id: str | None = None,
    status: JobStatus | None = None,
    type: JobType | None = None,
    limit: int = 100,
) -> list[JobResponse]:
    """`job_id` es repetible (`?job_id=a&job_id=b&...`) — pedir el estado de
    un lote conocido de jobs en una sola llamada, en vez de una por job.
    """
    jobs = store.list(
        job_ids=job_id, document_id=document_id, status=status, job_type=type, limit=limit
    )
    return [_to_response(job) for job in jobs]


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, store: JobStoreDep) -> JobResponse:
    return _to_response(store.get(job_id))


@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: str, runner: RunnerDep) -> JobResponse:
    """Solicita la cancelación.

    Es cooperativa: surte efecto en el siguiente punto de control del trabajo,
    que es donde el estado en disco queda consistente.
    """
    return _to_response(runner.cancel(job_id))


@router.get("/{job_id}/events")
def stream_events(job_id: str, request: Request, store: JobStoreDep, runner: RunnerDep):
    """Progreso en vivo por Server-Sent Events.

    Si el job ya terminó, emite su estado final y cierra: así un cliente que
    se suscribe tarde no se queda esperando eventos que nunca llegarán.
    """
    job = store.get(job_id)

    def stream() -> Iterator[str]:
        if job.is_terminal:
            yield _sse(_snapshot(job))
            return

        with runner.events.subscribe(job_id) as channel:
            # Estado actual antes de escuchar, para no perder lo ocurrido
            # entre la consulta y la suscripción.
            yield _sse(_snapshot(store.get(job_id)))
            while True:
                try:
                    event = channel.get(timeout=_HEARTBEAT_SECONDS)
                except queue.Empty:
                    # Comentario SSE: mantiene viva la conexión a través de
                    # proxies que cortan por inactividad.
                    yield ": heartbeat\n\n"
                    continue
                if event is None:
                    yield _sse(_snapshot(store.get(job_id)))
                    return
                yield _sse(event.model_dump(mode="json"))

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _snapshot(job: Job) -> dict:
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
