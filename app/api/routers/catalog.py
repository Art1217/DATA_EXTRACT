"""Catálogo de análisis y grupos.

Se sirve directamente del registro que usa el pipeline, así que lo publicado
no puede divergir de lo que el backend sabe hacer.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import AnalysisTypeResponse, GroupCatalogResponse
from app.domain.accounting import RULES
from app.domain.analysis import (
    ANALYSIS_TYPES,
    AnalysisType,
    Group,
    analyses_for_group,
    get_analysis_type,
)

router = APIRouter(prefix="/v1/catalog", tags=["catálogo"])


def _to_response(analysis: AnalysisType) -> AnalysisTypeResponse:
    return AnalysisTypeResponse(
        key=analysis.key,
        label=analysis.label,
        group=analysis.group.value if analysis.group else None,
        code=analysis.code,
        supports_extraction=analysis.supports_extraction,
        supports_quality=analysis.supports_quality,
        llm_quality_review=analysis.quality_prompt is not None,
        accounting_rules=[rule.description for rule in RULES.get(analysis.key, ())],
        needs_identity=analysis.needs_identity,
        prefer_docai_json=analysis.prefer_docai_json,
    )


@router.get("/analysis-types", response_model=list[AnalysisTypeResponse])
def list_analysis_types(group: str | None = None) -> list[AnalysisTypeResponse]:
    """Todos los tipos de análisis, opcionalmente filtrados por grupo."""
    return [
        _to_response(analysis)
        for analysis in ANALYSIS_TYPES.values()
        if group is None or (analysis.group and analysis.group.value.casefold() == group.casefold())
    ]


@router.get("/analysis-types/{analysis_type}", response_model=AnalysisTypeResponse)
def get_analysis_type_detail(analysis_type: str) -> AnalysisTypeResponse:
    return _to_response(get_analysis_type(analysis_type))


@router.get("/groups", response_model=list[GroupCatalogResponse])
def list_groups() -> list[GroupCatalogResponse]:
    """Grupos de contenido y los análisis que cada uno admite."""
    return [
        GroupCatalogResponse(
            group=group.value,
            codes=[a.code for a in analyses_for_group(group) if a.code],
            analysis_types=[a.key for a in analyses_for_group(group)],
        )
        for group in Group
    ]
