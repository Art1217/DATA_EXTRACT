"""Clasificación de páginas por bloques (Fase 1 y Fase 2).

Las dos fases usan el mismo mecanismo: un bloque de imágenes etiquetadas por
número de página y un prompt que define el vocabulario de etiquetas. Lo único
que cambia entre ellas es el prompt.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential

from app.adapters import llm
from app.adapters.prompts import labeled_page_parts, read_prompt
from app.config import get_settings

logger = logging.getLogger(__name__)

_NUMBERING_RULE = (
    "\n--- REGLA DE NUMERACIÓN CRÍTICA ---\n"
    "Clasifica cada página usando ÚNICAMENTE el número de la etiqueta 'IMAGEN PÁGINA X'.\n"
    "Ignora cualquier numeración impresa en el papel si hay discrepancia con la etiqueta enviada.\n"
    "------------------------------------\n\n"
)


@traceable(run_type="llm", name="Clasificación de bloque")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def classify_pages_block(image_data, prompt_path):
    """Clasifica un bloque de páginas con el prompt dado.

    image_data: lista de tuplas `(page_number, image_path)`.
    Devuelve: `{"clasificaciones": [{"page_number", <etiqueta>, "confianza", "evidencia"}]}`.
    """
    settings = get_settings()
    parts = [llm.text(_NUMBERING_RULE + read_prompt(prompt_path)), *labeled_page_parts(image_data)]
    response = llm.generate(
        task="classify",
        prompt_name=Path(prompt_path).stem,
        model=settings.model_name_classify,
        parts=parts,
    )
    return merge_classifications(response.text)


def merge_classifications(text: str | None) -> dict:
    """Parser tolerante para las respuestas de clasificación por bloques.

    Acepta un objeto `{"clasificaciones": [...]}`, una lista directa o VARIOS
    objetos JSON concatenados, que Gemini a veces devuelve. Une todas las
    entradas en una sola lista.
    """
    if not text:
        return {"clasificaciones": []}
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
    decoder = json.JSONDecoder()
    merged = []
    idx, n = 0, len(s)
    while idx < n:
        while idx < n and s[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(s, idx)
        except json.JSONDecodeError:
            logger.warning("Respuesta de clasificación con JSON inválido desde el carácter %d", idx)
            break
        if isinstance(obj, dict) and "clasificaciones" in obj:
            merged.extend(obj.get("clasificaciones") or [])
        elif isinstance(obj, list):
            merged.extend(obj)
        elif isinstance(obj, dict) and "page_number" in obj:
            merged.append(obj)
        idx = end
    return {"clasificaciones": merged}
