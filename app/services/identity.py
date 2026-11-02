"""Etapa de identidad: metadata general del documento.

Empresa, RUC, período, moneda, unidad y auditor: lo que dice la portada y
vale para todas las tablas del informe. Se calcula una sola vez por documento
y la reutilizan todos los análisis; al persistirla bajo una clave del
documento, la reutilización deja de ser algo que haya que acordarse de
activar.
"""

from __future__ import annotations

import logging

import pymupdf

from app.adapters.ai_searcher import extract_document_identity
from app.adapters.preprocessing import preprocess_pdf_page
from app.domain.analysis import IDENTITY_PROMPT
from app.services.context import StageContext
from app.services.limits import llm_slot
from app.storage import keys

logger = logging.getLogger(__name__)

ENTITY_NAME_KEY = "empresa"
UNKNOWN_ENTITY = "Desconocido"


def extract_identity(
    ctx: StageContext,
    document_id: str,
    *,
    dpi: int | None = None,
    force: bool = False,
) -> dict:
    """Extrae la identidad del documento a partir de sus primeras páginas."""
    if not force:
        cached = ctx.storage.read_json_or_none(keys.identity(document_id))
        if cached is not None:
            return cached

    settings = ctx.settings
    render_dpi = dpi or settings.dpi
    prompt = str(settings.prompt_path(IDENTITY_PROMPT))
    workspace_dir = ctx.workspace.subdir("identity")

    ctx.progress.report("Preparando páginas iniciales...", 0.2)
    with ctx.storage.materialize(keys.source_pdf(document_id)) as pdf_path:
        with pymupdf.open(str(pdf_path)) as document:
            page_count = min(settings.identity_pages, document.page_count)
            images = [
                preprocess_pdf_page(document, index, dpi=render_dpi, out_dir=workspace_dir)
                for index in range(page_count)
            ]

        ctx.progress.report("Detectando identidad y metadata general...", 0.5)
        with llm_slot():
            identity = extract_document_identity(images, prompt)

    if not isinstance(identity, dict):
        identity = {}

    ctx.storage.write_json(keys.identity(document_id), identity)
    ctx.progress.report("Identidad detectada.", 1.0)
    logger.info("Identidad de %s: %d campos", document_id, len(identity))
    return identity


def get_identity(ctx: StageContext, document_id: str) -> dict | None:
    """Identidad ya calculada, o `None` si la etapa no se ha ejecutado."""
    return ctx.storage.read_json_or_none(keys.identity(document_id))


def save_identity(ctx: StageContext, document_id: str, identity: dict) -> dict:
    """Sustituye la identidad por una corregida a mano.

    Los datos de identidad se copian a cada fila del Excel final, así que un
    error aquí (una moneda o una unidad mal leída) se propaga a todo el
    resultado. Poder arreglarlo sin repetir la
    extracción entera es el motivo de que esta operación exista.
    """
    ctx.storage.write_json(keys.identity(document_id), identity)
    return identity


def entity_name(identity: dict | None) -> str:
    """Nombre de la empresa usado como ancla de contexto durante la búsqueda."""
    if not identity:
        return UNKNOWN_ENTITY
    return str(identity.get(ENTITY_NAME_KEY) or UNKNOWN_ENTITY)
