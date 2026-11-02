"""Publicación de entregables: bucket de GCS y fila de trazabilidad en BigQuery.

Publicar es un paso **adicional**, no una condición de éxito del pipeline. El
storage sigue siendo la fuente de verdad operativa; lo que va al bucket son
los entregables que consume quien no tiene acceso al backend: el PDF
recortado, el Excel de extracción y los JSON de búsqueda y de calidad.

El árbol del bucket agrupa por corrida, documento, grupo y análisis — como se
busca un dato, no como el pipeline lo produjo:

    run_20260811-1215/                           ← la corrida (ver `batch_folder`)
      eeff-2024_146c38d4/                        ← el documento
        eeff-2024_manifiesto.json
        eeff-2024_plan.txt
        Estados/SitFin/
          eeff-2024_situacion_financiera_corte.pdf
          eeff-2024_situacion_financiera_extraccion.xlsx
          eeff-2024_situacion_financiera_busqueda.json
          eeff-2024_situacion_financiera_calidad.json
        Notas/Invent/
          ...

En BigQuery queda una fila por (documento, análisis) con el número de filas,
el veredicto de calidad, las URIs de los entregables y las filas extraídas en
JSON, lista para consultar con SQL.

Nada de esto puede tumbar un job: si el bucket rechaza una subida o BigQuery
la fila, se registra el error y se sigue. La extracción ya está pagada y
guardada en el storage.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from app.config import PublishTarget
from app.domain.analysis import AnalysisType
from app.domain.models import ExtractionResult
from app.services import documents as documents_service
from app.services import export
from app.services.bigquery_client import insert_extraction_row
from app.services.context import StageContext
from app.services.gcs_client import upload_bytes
from app.storage import keys

if TYPE_CHECKING:  # Solo para el tipo: importarlo en runtime sería un ciclo.
    from app.services.pipeline import PipelineOutcome

logger = logging.getLogger(__name__)

_FOLDER_MAX_CHARS = 60
"""Tope del nombre legible: el objeto completo no debe pasar de 1024 bytes."""

_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("pdf_uri", "corte"),
    ("excel_uri", "extraccion"),
    ("search_json_uri", "busqueda"),
    ("quality_json_uri", "calidad"),
)
"""`(columna de BigQuery, qué es el archivo)`.

Los nombres del storage (`result.json`) funcionan ahí porque la carpeta los
distingue, pero fuera del bucket no: el JSON de búsqueda y el de calidad se
llamarían igual y el segundo pisaría al primero en la carpeta de Descargas. Se
usa el mismo vocabulario que el ZIP (`app/services/bundle.py`)."""


@dataclass(slots=True)
class PublishResult:
    """Qué acabó publicado y dónde. Se adjunta al resultado del job."""

    document_id: str
    analysis_type: str
    bucket: str | None = None
    batch: str = ""
    uris: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    """Entregables que no existían en el storage y por eso no se subieron."""
    bigquery_row: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "analysis_type": self.analysis_type,
            "bucket": self.bucket,
            "batch": self.batch,
            "uris": dict(self.uris),
            "missing": list(self.missing),
            "bigquery_row": self.bigquery_row,
            "error": self.error,
        }


# --------------------------------------------------------------------------
# Entregables de un análisis
# --------------------------------------------------------------------------


def publish_deliverables(
    ctx: StageContext,
    document_id: str,
    analysis: AnalysisType,
    outcome: PipelineOutcome,
) -> PublishResult | None:
    """Sube los entregables del análisis y registra su fila en BigQuery.

    **También publica cuando no se encontró nada.** "Se revisó y no había" es
    un resultado: se registra con `row_count` en cero, y desde fuera se
    distingue de un documento que nunca se procesó.

    Devuelve `None` si la publicación está apagada o si no llegó a ejecutarse
    ninguna etapa que contar.
    """
    if outcome.extraction is None and outcome.search is None:
        return None
    target = ctx.settings.publish_target()
    if not target.enabled:
        return None
    if not target.ready:
        logger.warning(
            "PUBLISH_ENABLED está activo pero no hay GCS_BUCKET: no se publica %s.",
            analysis.key,
        )
        return None

    meta = _document_meta(ctx, document_id)
    corrida = batch_folder(ctx.batch)
    result = PublishResult(
        document_id=document_id,
        analysis_type=analysis.key,
        bucket=target.bucket,
        batch=corrida,
    )

    extraction = outcome.extraction
    if extraction:
        store_excel(ctx, document_id, analysis.key, extraction)
    _upload_artifacts(target, ctx, document_id, analysis, meta.get("filename"), corrida, result)
    _upload_manifest(target, ctx, document_id, meta.get("filename"), corrida, result)

    try:
        insert_extraction_row(
            target, _build_row(ctx, document_id, analysis, outcome, result.uris, meta, corrida)
        )
        result.bigquery_row = True
    except Exception as exc:
        logger.exception("No se pudo registrar en BigQuery %s de %s", analysis.key, document_id)
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def store_excel(
    ctx: StageContext, document_id: str, analysis_key: str, extraction: ExtractionResult
) -> str | None:
    """Genera el Excel una vez y lo guarda en el storage.

    Así la descarga, el ZIP y el bucket sirven exactamente el mismo archivo.
    """
    workbook = export.to_excel_bytes(extraction.rows, extraction.identity)
    if workbook is None:
        return None
    return ctx.storage.write_bytes(keys.extraction_xlsx(document_id, analysis_key), workbook)


def _upload_artifacts(
    target: PublishTarget,
    ctx: StageContext,
    document_id: str,
    analysis: AnalysisType,
    filename: str | None,
    corrida: str,
    result: PublishResult,
) -> None:
    sources = {
        "pdf_uri": keys.search_pdf(document_id, analysis.key),
        "excel_uri": keys.extraction_xlsx(document_id, analysis.key),
        "search_json_uri": keys.search_result(document_id, analysis.key),
        "quality_json_uri": keys.quality_result(document_id, analysis.key),
    }
    slug = document_slug(filename)
    for column, name in _ARTIFACTS:
        key = sources[column]
        if not ctx.storage.exists(key):
            result.missing.append(column)
            continue
        objeto = object_key(
            batch=corrida,
            documento=document_folder(document_id, filename),
            subcarpeta=analysis.folder,
            nombre=_named(slug, analysis.key, name),
            storage_key=key,
        )
        try:
            result.uris[column] = upload_bytes(target, objeto, ctx.storage.read_bytes(key))
        except Exception as exc:
            logger.exception("No se pudo subir %s al bucket", key)
            result.missing.append(column)
            result.error = f"{type(exc).__name__}: {exc}"


def _upload_manifest(
    target: PublishTarget,
    ctx: StageContext,
    document_id: str,
    filename: str | None,
    corrida: str,
    result: PublishResult,
) -> None:
    """Sube la ficha del documento a su carpeta de la corrida.

    Es lo que hace que la carpeta se explique sola: sin ella, quien abre el
    bucket ve análisis sueltos sin saber de qué informe salieron.
    """
    objeto = object_key(
        batch=corrida,
        documento=document_folder(document_id, filename),
        subcarpeta="",
        nombre=_named(document_slug(filename), "manifiesto"),
        storage_key="x.json",
    )
    etapas = documents_service.inventory(ctx.storage, document_id)
    procesados = sorted(set(etapas.search) | set(etapas.extraction) | set(etapas.quality))
    try:
        result.uris["manifest_uri"] = upload_bytes(
            target,
            objeto,
            documents_service.manifest(ctx.storage, document_id, procesados).encode("utf-8"),
        )
    except Exception:
        logger.exception("No se pudo subir el manifiesto de %s", document_id)


# --------------------------------------------------------------------------
# Nombres de objeto: el bucket es para personas
# --------------------------------------------------------------------------


def document_slug(filename: str | None) -> str:
    """Nombre del documento apto para un nombre de objeto.

    Sin extensión, sin tildes, sin espacios ni barras —una barra crearía
    niveles de carpeta que nadie pidió— y acotado en longitud. Devuelve cadena
    vacía si no queda nada utilizable.
    """
    return _clean(PurePosixPath((filename or "").replace("\\", "/")).stem)


def document_folder(document_id: str, filename: str | None) -> str:
    """Carpeta del documento en el bucket: `<nombre del archivo>_<id corto>`.

    En el storage el documento se direcciona por su `document_id`, el hash de
    su contenido: eso da deduplicación y evita choques entre dos archivos que
    se llamen igual. Pero una carpeta llamada `146c38d430d2...` no le dice
    nada a nadie. El nombre la hace navegable y los 8 primeros caracteres del
    hash conservan la unicidad.
    """
    slug = document_slug(filename)
    return f"{slug}_{document_id[:8]}" if slug else document_id


def batch_folder(batch: str | None) -> str:
    """Carpeta de la corrida: `run_<fecha>-<hora>`.

    Subir treinta documentos de una vez no debería dejar treinta carpetas
    sueltas en la raíz del bucket, sino una corrida con treinta documentos
    dentro. El identificador lo genera quien agrupa y lo manda en cada job
    (ver `StageContext.batch`); un job suelto usa la fecha y hora del momento.
    """
    marca = _clean(batch) if batch else datetime.now().strftime("%Y%m%d-%H%M")
    return f"run_{marca}"


def object_key(
    *, batch: str, documento: str, subcarpeta: str, nombre: str, storage_key: str
) -> str:
    """Ruta completa del objeto en el bucket.

        <corrida>/<documento>/<grupo>/<código>/<documento>_<análisis>_<qué es>.<ext>

    El nombre del archivo repite documento y análisis a propósito: una vez
    descargado y suelto en una carpeta cualquiera, la ruta ya no existe y el
    nombre es lo único que queda para saber qué es.
    """
    extension = PurePosixPath(storage_key).suffix
    partes = [parte for parte in (batch, documento, subcarpeta) if parte]
    return f"{'/'.join(partes)}/{nombre}{extension}"


def _named(*partes: str) -> str:
    return "_".join(parte for parte in partes if parte)


def _clean(valor: str) -> str:
    """Limpia un identificador que viene de fuera antes de que sea una ruta."""
    plano = unicodedata.normalize("NFKD", valor).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", plano).strip("-._")[:_FOLDER_MAX_CHARS]


def _build_row(
    ctx: StageContext,
    document_id: str,
    analysis: AnalysisType,
    outcome: PipelineOutcome,
    uris: dict[str, str],
    meta: dict,
    corrida: str,
) -> dict:
    extraction = outcome.extraction
    identity = (extraction.identity if extraction else None) or outcome.identity or {}
    row = {
        "batch": corrida,
        "document_id": document_id,
        "filename": meta.get("filename"),
        "sha256": meta.get("sha256"),
        "company": identity.get("empresa"),
        "fiscal_period": identity.get("periodo"),
        "currency": identity.get("moneda"),
        "analysis_type": analysis.key,
        "analysis_group": analysis.group.value if analysis.group else None,
        "row_count": extraction.row_count if extraction else 0,
        "quality_status": outcome.quality.status if outcome.quality else None,
        "job_id": ctx.job_id,
        "extracted_at": _timestamp(extraction.created_at) if extraction else None,
        "published_at": _timestamp(datetime.now(UTC)),
        "rows_json": json.dumps(extraction.rows if extraction else [], ensure_ascii=False),
    }
    row.update({column: uris.get(column) for column, _ in _ARTIFACTS})
    return row


def _document_meta(ctx: StageContext, document_id: str) -> dict:
    try:
        return documents_service.get_document(ctx.storage, document_id).model_dump(mode="json")
    except Exception:
        logger.warning("Sin ficha de documento para %s al publicar.", document_id)
        return {}


def _timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


# --------------------------------------------------------------------------
# Plan del modo automático
# --------------------------------------------------------------------------


def publish_plan(ctx: StageContext, document_id: str) -> dict:
    """Sube el `plan.txt` del documento a su carpeta del bucket.

    Devuelve la URI subida; vacío si la publicación está apagada o no hay
    bucket. Los errores se registran y no se propagan.
    """
    target = ctx.settings.publish_target()
    if not target.ready:
        return {}

    key = keys.plan_text(document_id)
    if not ctx.storage.exists(key):
        return {}

    meta = _document_meta(ctx, document_id)
    published: dict = {"bucket": target.bucket, "batch": batch_folder(ctx.batch)}
    objeto = object_key(
        batch=published["batch"],
        documento=document_folder(document_id, meta.get("filename")),
        subcarpeta="",
        nombre=_named(document_slug(meta.get("filename")), "plan"),
        storage_key=key,
    )
    try:
        published["plan_txt_uri"] = upload_bytes(target, objeto, ctx.storage.read_bytes(key))
    except Exception:
        logger.exception("No se pudo subir %s al bucket", key)
    return published
