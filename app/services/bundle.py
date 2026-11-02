"""Empaquetado de artefactos en un ZIP, para descargar.

No se reconstruye nada al empaquetar: se copia lo que las etapas dejaron en
el storage. El ZIP es una vista de los artefactos, no una segunda fuente de
verdad que pueda quedar desincronizada.

Dos formas: el de un documento y el de una corrida entera
(`build_batch_bundle`), que agrupa varios bajo una carpeta por documento.

Los archivos se nombran igual que en el bucket —`<documento>_<análisis>_<qué
es>`— y no `extraccion.xlsx` a secas: un ZIP se descomprime, y en cuanto el
Excel sale de su carpeta su nombre es lo único que dice de dónde vino.
"""

from __future__ import annotations

import logging
import uuid
import zipfile

from app.domain.analysis import get_analysis_type
from app.domain.errors import DocumentNotFoundError
from app.domain.models import ExtractionResult
from app.services import export
from app.services.context import StageContext
from app.storage import keys

logger = logging.getLogger(__name__)

_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("search/{key}/result.json", "busqueda.json"),
    ("search/{key}/corte.pdf", "corte.pdf"),
    ("search/{key}/corte_searchable.pdf", "corte_ocr.pdf"),
    ("extraction/{key}/result.json", "extraccion.json"),
    ("extraction/{key}/docai.json", "docai.json"),
    ("quality/{key}/result.json", "calidad.json"),
)


def _named(slug: str, *partes: str) -> str:
    """`<documento>_<análisis>_<qué es>`, saltando lo que venga vacío."""
    return "_".join(parte for parte in (slug, *partes) if parte)


def build_bundle(
    ctx: StageContext, document_id: str, *, analysis_types: list[str] | None = None
) -> str:
    """Empaqueta los artefactos del documento. Devuelve la clave del ZIP."""
    if not ctx.storage.exists(keys.document_meta(document_id)):
        raise DocumentNotFoundError(document_id)

    from app.services.documents import inventory

    stages = inventory(ctx.storage, document_id)
    selected = analysis_types or sorted(
        set(stages.search) | set(stages.extraction) | set(stages.quality)
    )

    bundle_id = uuid.uuid4().hex[:12]
    archive_path = ctx.workspace.path("bundles", f"{bundle_id}.zip")

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        _write_document(ctx, archive, document_id, selected)

    ctx.progress.report("Paquete generado.", 1.0)
    return ctx.storage.import_file(archive_path, keys.bundle(document_id, bundle_id), move=True)


def build_batch_bundle(
    ctx: StageContext,
    document_ids: list[str],
    batch_id: str,
    analyses: dict[str, list[str]] | None = None,
) -> str:
    """Empaqueta una corrida entera en un ZIP, una carpeta por documento.

    Subir treinta documentos y bajarlos de uno en uno no es una descarga, es
    treinta. `analyses` acota qué análisis de cada documento entran, para no
    arrastrar los de otras corridas del mismo documento.
    """
    from app.services.documents import inventory

    archive_path = ctx.workspace.path("bundles", f"{batch_id}.zip")
    total = len(document_ids) or 1

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, document_id in enumerate(document_ids):
            ctx.cancellation.raise_if_cancelled()
            if not ctx.storage.exists(keys.document_meta(document_id)):
                logger.warning("Se omite %s del paquete: no está registrado.", document_id)
                continue
            stages = inventory(ctx.storage, document_id)
            presentes = set(stages.search) | set(stages.extraction) | set(stages.quality)
            pedidos = (analyses or {}).get(document_id)
            selected = (
                sorted(presentes & set(pedidos)) if pedidos is not None else sorted(presentes)
            )
            ctx.progress.report(f"Empaquetando documento {index + 1}/{total}...", index / total)
            _write_document(
                ctx, archive, document_id, selected, prefix=_document_folder(ctx, document_id)
            )

    ctx.progress.report("Paquete de la corrida generado.", 1.0)
    return ctx.storage.import_file(archive_path, keys.batch_bundle(batch_id), move=True)


def _document_folder(ctx: StageContext, document_id: str) -> str:
    """Carpeta del documento dentro del ZIP de la corrida.

    La misma que en el bucket, para que descargar y navegar el bucket den la
    misma estructura.
    """
    from app.services.publish import document_folder

    meta = ctx.storage.read_json_or_none(keys.document_meta(document_id)) or {}
    return document_folder(document_id, meta.get("filename"))


def _write_document(
    ctx: StageContext,
    archive: zipfile.ZipFile,
    document_id: str,
    selected: list[str],
    *,
    prefix: str = "",
) -> None:
    """Escribe los artefactos de un documento dentro del ZIP abierto."""
    from app.services.publish import document_slug

    raiz = f"{prefix}/" if prefix else ""
    root = keys.document_root(document_id)
    meta = ctx.storage.read_json_or_none(keys.document_meta(document_id)) or {}
    slug = document_slug(meta.get("filename"))

    archive.writestr(
        f"{raiz}{_named(slug, 'manifiesto.json')}", _manifest(ctx, document_id, selected)
    )
    plan_text_key = keys.plan_text(document_id)
    if ctx.storage.exists(plan_text_key):
        archive.writestr(f"{raiz}{_named(slug, 'plan.txt')}", ctx.storage.read_bytes(plan_text_key))

    for analysis_key in selected:
        folder = f"{raiz}{_folder_for(analysis_key)}"
        for template, name in _ARTIFACTS:
            key = f"{root}/{template.format(key=analysis_key)}"
            if ctx.storage.exists(key):
                archive.writestr(
                    f"{folder}/{_named(slug, analysis_key, name)}", ctx.storage.read_bytes(key)
                )
        _add_excel(ctx, archive, document_id, analysis_key, folder, slug)


def _folder_for(analysis_key: str) -> str:
    """Ruta dentro del ZIP. La define el catálogo (`AnalysisType.folder`)."""
    try:
        return get_analysis_type(analysis_key).folder
    except Exception:
        return analysis_key


def _add_excel(
    ctx: StageContext, archive, document_id: str, analysis_key: str, folder: str, slug: str
):
    nombre = f"{folder}/{_named(slug, analysis_key, 'extraccion.xlsx')}"
    stored = keys.extraction_xlsx(document_id, analysis_key)
    if ctx.storage.exists(stored):
        # Ya lo generó la publicación: se copia, para que el del ZIP, el de la
        # descarga y el del bucket sean el mismo archivo.
        archive.writestr(nombre, ctx.storage.read_bytes(stored))
        return

    payload = ctx.storage.read_json_or_none(keys.extraction_result(document_id, analysis_key))
    if not payload:
        return
    result = ExtractionResult.model_validate(payload)
    workbook = export.to_excel_bytes(result.rows, result.identity)
    if workbook:
        archive.writestr(nombre, workbook)


def _manifest(ctx: StageContext, document_id: str, analysis_types: list[str]) -> str:
    """La misma ficha que se publica al bucket."""
    from app.services.documents import manifest

    return manifest(ctx.storage, document_id, analysis_types)
