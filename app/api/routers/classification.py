"""Etapa de clasificación: Fase 1 (grupos) y Fase 2 (análisis).

Fase 1 y Fase 2 son endpoints independientes. Fase 2 recibe el grupo en el
cuerpo de la petición y no requiere que Fase 1 se haya ejecutado: quien ya
sabe que su documento trae los estados principales puede llamarla
directamente y ahorrarse una pasada completa por el documento, que es lo que
cuesta dinero. `/plan` encadena ambas para quien no lo sabe de antemano.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import RunnerDep, StorageDep
from app.api.routers.common import accepted, ensure_document
from app.api.schemas import (
    ClassifyAnalysisRequest,
    ClassifyGroupRequest,
    JobAccepted,
    PlanRequest,
)
from app.domain.analysis import parse_group
from app.domain.errors import StageNotReadyError
from app.jobs.models import JobType
from app.storage import keys
from app.storage.base import Storage

router = APIRouter(prefix="/v1/documents/{document_id}", tags=["clasificación"])


@router.post(
    "/classification/groups",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def classify_groups(
    document_id: str, request: ClassifyGroupRequest, storage: StorageDep, runner: RunnerDep
) -> JobAccepted:
    """Fase 1: detecta qué grupos (estados principales, notas) hay en el documento."""
    ensure_document(storage, document_id)
    job = runner.submit(
        JobType.CLASSIFY_GROUP, document_id=document_id, payload=request.as_payload()
    )
    return accepted(job)


@router.get("/classification/groups")
def get_group_classification(document_id: str, storage: StorageDep) -> dict:
    return _require(storage, keys.classification_group(document_id), "classification/groups")


@router.post(
    "/classification/analysis",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def classify_analysis(
    document_id: str,
    request: ClassifyAnalysisRequest,
    storage: StorageDep,
    runner: RunnerDep,
) -> JobAccepted:
    """Fase 2: detecta qué análisis contiene el documento dentro de un grupo.

    Es una llamada por grupo. Si la Fase 1 devolvió Estados y Notas, son dos
    peticiones a este endpoint.
    """
    ensure_document(storage, document_id)
    group = parse_group(request.group)
    payload = request.as_payload() | {"group": group.value}
    job = runner.submit(JobType.CLASSIFY_ANALYSIS, document_id=document_id, payload=payload)
    return accepted(job)


@router.get("/classification/analysis/{group}")
def get_analysis_classification(document_id: str, group: str, storage: StorageDep) -> dict:
    parsed = parse_group(group)
    return _require(
        storage,
        keys.classification_analysis(document_id, parsed.value),
        f"classification/analysis/{parsed.value}",
    )


@router.post("/plan", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
def build_plan(
    document_id: str, request: PlanRequest, storage: StorageDep, runner: RunnerDep
) -> JobAccepted:
    """Encadena Fase 1 y una Fase 2 por cada grupo encontrado."""
    ensure_document(storage, document_id)
    job = runner.submit(JobType.PLAN, document_id=document_id, payload=request.as_payload())
    return accepted(job)


@router.get("/plan")
def get_plan(document_id: str, storage: StorageDep) -> dict:
    return _require(storage, keys.plan(document_id), "plan")


def _require(storage: Storage, key: str, stage: str) -> dict:
    """Devuelve el artefacto guardado o indica qué hay que ejecutar antes."""
    payload = storage.read_json_or_none(key)
    if payload is None:
        raise StageNotReadyError(
            stage, requires=stage, hint="Lanza el POST del mismo recurso para generarlo."
        )
    return payload
