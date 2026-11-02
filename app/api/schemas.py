"""Esquemas de petición y respuesta de la API.

Se validan los bordes: lo que entra por HTTP y lo que sale. Las filas
extraídas viajan como `dict` sin esquema porque su forma la define cada
prompt y cambia por análisis; imponerles un modelo obligaría a modificarlo
cada vez que se ajusta un prompt.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.analysis import Stage
from app.jobs.models import JobStatus, JobType


class ScanParamsBody(BaseModel):
    """Parámetros de recorrido. Omitir un campo usa el valor configurado."""

    dpi: int | None = Field(default=None, ge=72, le=600)
    block_size: int | None = Field(default=None, ge=1, le=200)
    workers: int | None = Field(default=None, ge=1, le=16)
    max_pages: int | None = Field(
        default=None, ge=1, description="Limita el recorrido a las primeras N páginas."
    )

    def as_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


# --- Peticiones por etapa -------------------------------------------------


class ClassifyGroupRequest(ScanParamsBody):
    """Fase 1: detectar qué grupos de contenido tiene el documento."""


class ClassifyAnalysisRequest(ScanParamsBody):
    """Fase 2: detectar qué análisis hay dentro de un grupo.

    No exige haber ejecutado la Fase 1: quien ya sabe qué grupo tiene puede
    ir directo y ahorrarse una pasada completa por el documento.
    """

    group: str = Field(description="Estados o Notas.")


class PlanRequest(ScanParamsBody):
    force: bool = Field(
        default=False, description="Reclasifica aunque ya haya un resultado guardado."
    )


class IdentityRequest(BaseModel):
    dpi: int | None = Field(default=None, ge=72, le=600)
    force: bool = False


class SearchRequest(ScanParamsBody):
    analysis_type: str
    use_identity: bool = Field(
        default=True,
        description="Usa el nombre de la empresa detectado como ancla de contexto.",
    )
    cut: bool = Field(default=True, description="Genera el PDF recortado.")


class CutRequest(BaseModel):
    """Recorte manual, para corregir la selección de la búsqueda."""

    analysis_type: str
    pages: list[int] = Field(min_length=1, description="Páginas 1-indexadas.")


class ExtractionRequest(BaseModel):
    analysis_type: str
    include_identity: bool = True
    source_key: str | None = Field(
        default=None,
        description="Recorte alternativo a extraer. Por defecto, el de la búsqueda.",
    )


class QualityRequest(BaseModel):
    analysis_type: str


class QualityCheckRequest(BaseModel):
    """Validación de filas provistas por el cliente, sin documento asociado."""

    analysis_type: str
    rows: list[dict]


class PipelineRequest(ScanParamsBody):
    analysis_type: str
    stages: list[Stage] | None = Field(
        default=None, description="Por defecto: búsqueda, extracción y calidad."
    )
    reuse: bool = Field(
        default=True, description="Reutiliza resultados ya calculados en vez de repetirlos."
    )
    batch: str | None = Field(
        default=None,
        description=(
            "Identificador de la corrida que agrupa varios documentos. Todos "
            "los que lo compartan se publican bajo la misma carpeta del bucket. "
            "Si se omite, cada job usa su propia fecha y hora."
        ),
    )


class AutoRequest(ScanParamsBody):
    plan_only: bool = Field(
        default=False, description="Se detiene tras el plan, sin gastar en extracción."
    )
    stages: list[Stage] | None = None
    reuse: bool = True
    batch: str | None = Field(
        default=None,
        description=(
            "Identificador de la corrida que agrupa varios documentos. Todos "
            "los que lo compartan se publican bajo la misma carpeta del bucket. "
            "Si se omite, cada job usa su propia fecha y hora."
        ),
    )


class BundleRequest(BaseModel):
    analysis_types: list[str] | None = None


class BatchBundleRequest(BaseModel):
    """Un ZIP con varios documentos: el de una corrida completa."""

    document_ids: list[str] = Field(min_length=1)
    batch: str | None = Field(
        default=None,
        description=(
            "Identificador de la corrida, el mismo que se mandó al procesarla. "
            "Da nombre al ZIP; si se omite se genera de la fecha y hora."
        ),
    )


# --- Respuestas -----------------------------------------------------------


class JobAccepted(BaseModel):
    """Respuesta de toda operación asíncrona."""

    job_id: str
    status: JobStatus
    type: JobType
    document_id: str | None = None
    analysis_type: str | None = None
    status_url: str
    events_url: str


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    job_id: str
    type: JobType
    status: JobStatus
    document_id: str | None
    analysis_type: str | None
    progress: float
    message: str
    result: Any | None
    error: str | None
    parent_job_id: str | None
    created_at: Any
    started_at: Any
    finished_at: Any
    duration_seconds: float | None


class AnalysisTypeResponse(BaseModel):
    """Entrada del catálogo publicado."""

    key: str
    label: str
    group: str | None
    code: str | None
    supports_extraction: bool
    supports_quality: bool
    llm_quality_review: bool
    accounting_rules: list[str]
    needs_identity: bool
    prefer_docai_json: bool


class GroupCatalogResponse(BaseModel):
    group: str
    codes: list[str]
    analysis_types: list[str]


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    filename: str
    sha256: str
    pages: int
    size_bytes: int
    is_scanned: bool
    created_at: Any


class DocumentDetailResponse(DocumentResponse):
    stages: Any


class ErrorResponse(BaseModel):
    detail: str
    code: str
