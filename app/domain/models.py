"""Contratos de datos que cruzan las fronteras entre etapas.

Se tipa lo que es estable: identificadores, páginas, estados, conteos. No se
tipa la forma de las filas extraídas: cada `.prompt` define su propio esquema
de salida y varía por análisis, así que declararlas como `dict` es el contrato
honesto. Fingir una estructura fija ahí solo produciría validaciones que hay
que desactivar en cuanto se ajusta un prompt.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(UTC)


class DocumentMeta(BaseModel):
    """Ficha del PDF ingresado."""

    document_id: str
    filename: str
    sha256: str
    pages: int
    size_bytes: int
    is_scanned: bool
    """Determina si la extracción pasa por Document AI antes del modelo."""
    created_at: datetime = Field(default_factory=_now)


class PageFinding(BaseModel):
    """Una página etiquetada por el clasificador."""

    model_config = ConfigDict(extra="allow")

    page_number: int
    value: str
    """Grupo en la Fase 1, código de análisis en la Fase 2."""
    confidence: str | None = None
    evidence: str | None = None


class GroupClassification(BaseModel):
    """Fase 1: qué grupos de contenido tiene el documento."""

    groups: list[str]
    detail: list[PageFinding] = Field(default_factory=list)
    pages_analyzed: int = 0
    blocks_failed: int = 0


class AnalysisClassification(BaseModel):
    """Fase 2: qué análisis hay dentro de un grupo concreto."""

    group: str
    analyses: list[str]
    analysis_types: list[str] = Field(default_factory=list)
    """Claves del catálogo equivalentes a `analyses`, ya listas para usar."""
    detail: list[PageFinding] = Field(default_factory=list)
    blocks_failed: int = 0


class Plan(BaseModel):
    """Fase 1 + Fase 2: qué se puede procesar en este documento."""

    groups: dict[str, list[str]] = Field(default_factory=dict)
    analysis_types: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class SearchResult(BaseModel):
    """Páginas relevantes para un análisis y el recorte resultante."""

    analysis_type: str
    relevant_pages: list[int] = Field(default_factory=list)
    sections: list[dict] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    blocks_total: int = 0
    blocks_failed: int = 0
    cut_pdf_key: str | None = None
    created_at: datetime = Field(default_factory=_now)


class ExtractionResult(BaseModel):
    """Filas extraídas del recorte, más la identidad del documento."""

    analysis_type: str
    rows: list[dict] = Field(default_factory=list)
    identity: dict | None = None
    row_count: int = 0
    source_pages: list[int] = Field(default_factory=list)
    docai_key: str | None = None
    created_at: datetime = Field(default_factory=_now)


class QualityResult(BaseModel):
    """Veredicto del control de calidad sobre una extracción."""

    analysis_type: str
    status: str
    """`Aprobado`, `Rechazado` o `Error`."""
    errors: list = Field(default_factory=list)
    """Observaciones de la revisión con el modelo y descuadres de las reglas."""
    checks: list[dict] = Field(default_factory=list)
    """Resultado de cada regla contable determinista, período a período."""
    llm_review: bool = False
    """Si además de las reglas se pidió una revisión al modelo."""
    rows_checked: int = 0
    created_at: datetime = Field(default_factory=_now)


class StageInventory(BaseModel):
    """Qué se ha ejecutado ya sobre un documento.

    Permite a un cliente saber qué endpoints tiene sentido llamar sin
    tantear uno por uno.
    """

    identity: bool = False
    classification_group: bool = False
    classification_analysis: list[str] = Field(default_factory=list)
    plan: bool = False
    search: list[str] = Field(default_factory=list)
    extraction: list[str] = Field(default_factory=list)
    quality: list[str] = Field(default_factory=list)
