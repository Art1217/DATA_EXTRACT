"""Enlace entre tipos de job y servicios de etapa.

Cada handler traduce el `payload` del job a los argumentos del servicio y
devuelve un resultado serializable a JSON. Es una capa deliberadamente fina:
en cuanto un handler necesite lógica propia, esa lógica pertenece a un
servicio.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from app.domain.analysis import Stage, get_analysis_type, parse_group
from app.jobs.models import Job, JobType
from app.jobs.runner import Handler
from app.services import (
    auto,
    bundle,
    classification,
    extraction,
    identity,
    pipeline,
    quality,
    search,
)
from app.services.context import StageContext
from app.services.params import ScanParams


def _scan_params(job: Job) -> ScanParams:
    payload = job.payload
    return ScanParams(
        dpi=payload.get("dpi"),
        block_size=payload.get("block_size"),
        workers=payload.get("workers"),
        max_pages=payload.get("max_pages"),
    )


def _stages(job: Job) -> frozenset[Stage]:
    requested = job.payload.get("stages")
    if not requested:
        return pipeline.DEFAULT_STAGES
    return frozenset(Stage(value) for value in requested)


def _serialize(value: Any) -> Any:
    """Convierte resultados de servicio a estructuras JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return str(value)


def _document_id(job: Job) -> str:
    if not job.document_id:
        raise ValueError(f"El job '{job.type}' requiere un document_id.")
    return job.document_id


def _analysis(job: Job):
    if not job.analysis_type:
        raise ValueError(f"El job '{job.type}' requiere un analysis_type.")
    return get_analysis_type(job.analysis_type)


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------


def handle_classify_group(job: Job, ctx: StageContext) -> Any:
    return _serialize(
        classification.classify_groups(ctx, _document_id(job), params=_scan_params(job))
    )


def handle_classify_analysis(job: Job, ctx: StageContext) -> Any:
    group = parse_group(job.payload["group"])
    return _serialize(
        classification.classify_analysis(ctx, _document_id(job), group, params=_scan_params(job))
    )


def handle_plan(job: Job, ctx: StageContext) -> Any:
    return _serialize(
        classification.build_plan(
            ctx,
            _document_id(job),
            params=_scan_params(job),
            force=bool(job.payload.get("force")),
        )
    )


def handle_identity(job: Job, ctx: StageContext) -> Any:
    return _serialize(
        identity.extract_identity(
            ctx,
            _document_id(job),
            dpi=job.payload.get("dpi"),
            force=bool(job.payload.get("force")),
        )
    )


def handle_search(job: Job, ctx: StageContext) -> Any:
    document_id = _document_id(job)
    analysis = _analysis(job)
    entity_context = identity.UNKNOWN_ENTITY
    if job.payload.get("use_identity", True) and analysis.needs_identity:
        entity_context = identity.entity_name(
            identity.extract_identity(ctx.with_progress(0.0, 0.05), document_id)
        )
    return _serialize(
        search.search_pages(
            ctx.with_progress(0.05, 1.0),
            document_id,
            analysis,
            params=_scan_params(job),
            entity_context=entity_context,
            cut=bool(job.payload.get("cut", True)),
        )
    )


def handle_cut(job: Job, ctx: StageContext) -> Any:
    key = search.cut_pages(ctx, _document_id(job), _analysis(job), job.payload["pages"])
    return {"cut_pdf_key": key, "pages": job.payload["pages"]}


def handle_extraction(job: Job, ctx: StageContext) -> Any:
    document_id = _document_id(job)
    stored_identity = (
        identity.get_identity(ctx, document_id)
        if job.payload.get("include_identity", True)
        else None
    )
    return _serialize(
        extraction.extract_data(
            ctx,
            document_id,
            _analysis(job),
            identity=stored_identity,
            source_key=job.payload.get("source_key"),
        )
    )


def handle_quality(job: Job, ctx: StageContext) -> Any:
    return _serialize(quality.check_document(ctx, _document_id(job), _analysis(job)))


def handle_pipeline(job: Job, ctx: StageContext) -> Any:
    return _serialize(
        pipeline.run_pipeline(
            ctx,
            _document_id(job),
            job.analysis_type or "",
            stages=_stages(job),
            params=_scan_params(job),
            reuse=bool(job.payload.get("reuse", True)),
        )
    )


def handle_auto(job: Job, ctx: StageContext) -> Any:
    return _serialize(
        auto.run_auto(
            ctx,
            _document_id(job),
            params=_scan_params(job),
            stages=_stages(job),
            plan_only=bool(job.payload.get("plan_only")),
            reuse=bool(job.payload.get("reuse", True)),
        )
    )


def handle_bundle(job: Job, ctx: StageContext) -> Any:
    key = bundle.build_bundle(
        ctx, _document_id(job), analysis_types=job.payload.get("analysis_types")
    )
    return {"bundle_key": key}


def handle_bundle_batch(job: Job, ctx: StageContext) -> Any:
    batch_id = job.payload["batch_id"]
    key = bundle.build_batch_bundle(
        ctx, job.payload["document_ids"], batch_id, job.payload.get("analyses")
    )
    return {"bundle_key": key, "batch_id": batch_id}


HANDLERS: dict[JobType, Handler] = {
    JobType.CLASSIFY_GROUP: handle_classify_group,
    JobType.CLASSIFY_ANALYSIS: handle_classify_analysis,
    JobType.PLAN: handle_plan,
    JobType.IDENTITY: handle_identity,
    JobType.SEARCH: handle_search,
    JobType.CUT: handle_cut,
    JobType.EXTRACTION: handle_extraction,
    JobType.QUALITY: handle_quality,
    JobType.PIPELINE: handle_pipeline,
    JobType.AUTO: handle_auto,
    JobType.BUNDLE: handle_bundle,
    JobType.BUNDLE_BATCH: handle_bundle_batch,
}
