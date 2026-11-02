"""Operaciones compuestas: pipeline por análisis, flujo automático y paquete.

No añaden capacidades nuevas: orquestan las etapas que ya existen como
endpoints propios. Están para el caso habitual —"procesa este documento"— sin
obligar a encadenar seis llamadas a mano.
"""

from __future__ import annotations

import re
from datetime import datetime

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.api.deps import JobStoreDep, RunnerDep, StorageDep
from app.api.routers.common import accepted, ensure_analysis, ensure_document
from app.api.schemas import (
    AutoRequest,
    BatchBundleRequest,
    BundleRequest,
    JobAccepted,
    PipelineRequest,
)
from app.domain.errors import ArtifactNotFoundError, InvalidDocumentError
from app.jobs.models import JobType
from app.services import batches
from app.storage import keys

router = APIRouter(prefix="/v1/documents/{document_id}", tags=["compuestos"])

# Los ZIP de una corrida no cuelgan de un documento: agrupan a varios.
batch_router = APIRouter(prefix="/v1/bundles", tags=["compuestos"])

_BATCH_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def _safe_batch_id(batch: str | None) -> str:
    """El identificador acaba siendo una ruta: no puede traer sorpresas."""
    if batch is None:
        return datetime.now().strftime("%Y%m%d-%H%M")
    if not _BATCH_ID.fullmatch(batch):
        raise InvalidDocumentError(
            f"Identificador de corrida inválido: {batch!r}. "
            "Solo letras, números, punto, guion y guion bajo."
        )
    return batch


@batch_router.post("", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
def build_batch_bundle(
    request: BatchBundleRequest, storage: StorageDep, store: JobStoreDep, runner: RunnerDep
) -> JobAccepted:
    """Empaqueta varios documentos en un solo ZIP: el de una corrida.

    Qué análisis entra por documento se resuelve **aquí**, leyendo los jobs de
    la corrida: es lo que distingue "el ZIP de esta corrida" de "todo lo que
    estos documentos hayan acumulado alguna vez" (ver
    `app/services/batches.py`). Se hace en el router porque es donde está el
    almacén de jobs, y el handler solo recibe el resultado en su payload.
    """
    for document_id in request.document_ids:
        ensure_document(storage, document_id)
    batch_id = _safe_batch_id(request.batch)
    job = runner.submit(
        JobType.BUNDLE_BATCH,
        payload={
            "document_ids": request.document_ids,
            "batch_id": batch_id,
            "analyses": batches.batch_analyses(store, batch_id) or None,
        },
    )
    return accepted(job)


@batch_router.get("/{batch_id}")
def download_batch_bundle(batch_id: str, storage: StorageDep) -> StreamingResponse:
    key = keys.batch_bundle(_safe_batch_id(batch_id))
    if not storage.exists(key):
        raise ArtifactNotFoundError(key)
    return StreamingResponse(
        storage.open(key),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{batch_id}.zip"'},
    )


@router.post("/pipeline", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
def run_pipeline(
    document_id: str, request: PipelineRequest, storage: StorageDep, runner: RunnerDep
) -> JobAccepted:
    """Búsqueda, extracción y calidad para un análisis concreto."""
    ensure_document(storage, document_id)
    analysis_type = ensure_analysis(request.analysis_type)
    payload = request.as_payload()
    payload.pop("analysis_type", None)
    payload |= {
        "reuse": request.reuse,
        "stages": [stage.value for stage in request.stages] if request.stages else None,
        "batch": request.batch,
    }
    job = runner.submit(
        JobType.PIPELINE,
        document_id=document_id,
        analysis_type=analysis_type,
        payload=payload,
    )
    return accepted(job)


@router.post("/auto", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
def run_auto(
    document_id: str, request: AutoRequest, storage: StorageDep, runner: RunnerDep
) -> JobAccepted:
    """Clasifica el documento y procesa todos los análisis que encuentre.

    Con `plan_only=true` se detiene tras la clasificación, sin gastar en
    extracción: sirve para revisar el plan antes de autorizar el gasto.
    """
    ensure_document(storage, document_id)
    payload = request.as_payload() | {
        "plan_only": request.plan_only,
        "reuse": request.reuse,
        "stages": [stage.value for stage in request.stages] if request.stages else None,
        "batch": request.batch,
    }
    job = runner.submit(JobType.AUTO, document_id=document_id, payload=payload)
    return accepted(job)


@router.post("/bundle", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
def build_bundle(
    document_id: str, request: BundleRequest, storage: StorageDep, runner: RunnerDep
) -> JobAccepted:
    """Empaqueta en un ZIP todos los artefactos del documento."""
    ensure_document(storage, document_id)
    job = runner.submit(
        JobType.BUNDLE,
        document_id=document_id,
        payload={"analysis_types": request.analysis_types},
    )
    return accepted(job)


@router.get("/bundles/{bundle_id}")
def download_bundle(document_id: str, bundle_id: str, storage: StorageDep) -> StreamingResponse:
    key = keys.bundle(document_id, bundle_id)
    if not storage.exists(key):
        raise ArtifactNotFoundError(key)
    return StreamingResponse(
        storage.open(key),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{document_id}_{bundle_id}.zip"'},
    )
