"""Búsqueda de páginas relevantes e identidad del documento."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langsmith import traceable
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from app.adapters import llm
from app.adapters.prompts import labeled_page_parts, page_image_parts, read_prompt
from app.config import get_settings

logger = logging.getLogger(__name__)

_NUMBERING_RULE = (
    "\n--- REGLA DE NUMERACIÓN CRÍTICA ---\n"
    "Identifica la ubicación de los datos usando ÚNICAMENTE el número proporcionado en la "
    "etiqueta 'IMAGEN PÁGINA X'.\n"
    "Ignora cualquier numeración impresa en el papel si hay discrepancia con la etiqueta enviada.\n"
    "------------------------------------\n\n"
)

_UNKNOWN = {"", "desconocido", "unknown"}


@traceable(run_type="llm", name="Búsqueda de bloque")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_not_exception_type(FileNotFoundError),
)
def search_pages_block(image_data, prompt_path, entity_context=None):
    """Pregunta qué páginas del bloque contienen la tabla del análisis.

    `entity_context` es el nombre de la empresa, si ya se conoce: un informe
    consolidado trae los estados de la matriz y a veces los de subsidiarias, y
    el ancla evita que el modelo mezcle los de otra entidad.
    """
    settings = get_settings()
    anchor = ""
    if entity_context and entity_context.strip().lower() not in _UNKNOWN:
        anchor = (
            "\n--- CONTEXTO DEL DOCUMENTO ---\n"
            f"La entidad que reporta es: {entity_context}.\n"
            "Asegúrate de que las tablas encontradas correspondan a esta entidad.\n"
            "-------------------------------\n\n"
        )

    parts = [
        llm.text(_NUMBERING_RULE + anchor + read_prompt(prompt_path)),
        *labeled_page_parts(image_data),
    ]
    response = llm.generate(
        task="search",
        prompt_name=Path(prompt_path).stem,
        model=settings.model_name_classify,
        parts=parts,
    )
    try:
        return json.loads(response.text or "")
    except Exception as error:
        logger.warning("Error parseando JSON de búsqueda: %s", error)
        return {"error": str(error), "raw_response": response.text}


@traceable(run_type="llm", name="Identidad del documento")
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
def extract_document_identity(image_paths, prompt_path):
    """Lee las primeras páginas y devuelve la ficha del documento."""
    settings = get_settings()
    parts = [llm.text(read_prompt(prompt_path)), *page_image_parts(image_paths)]
    response = llm.generate(
        task="identity",
        prompt_name=Path(prompt_path).stem,
        model=settings.model_name_classify,
        parts=parts,
    )
    try:
        identity = json.loads(response.text or "")
        if isinstance(identity, list):
            identity = identity[0] if identity else {}
        return identity
    except Exception as error:
        logger.warning("Error parseando JSON de identidad: %s", error)
        return {"error": "No se pudo leer la identidad del documento"}
