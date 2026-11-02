"""Piezas comunes para armar peticiones al modelo."""

from __future__ import annotations

import logging
import os

from app.adapters import llm

logger = logging.getLogger(__name__)


def read_prompt(prompt_path) -> str:
    with open(prompt_path, encoding="utf-8") as handle:
        return handle.read()


def labeled_page_parts(image_data) -> list[llm.Part]:
    """Imágenes de página envueltas en etiquetas con su número.

    El modelo responde con el número de la etiqueta, no con el impreso en el
    papel: en un informe con portada y índice sin numerar, ambos difieren.
    Una imagen que no se puede leer se omite en vez de tumbar el bloque.
    """
    parts: list[llm.Part] = []
    for page_num, path in image_data:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError as error:
            logger.warning(
                "No se pudo leer la imagen de la página %s: %s. Se omite.", page_num, error
            )
            continue
        parts.append(llm.text(f"--- IMAGEN PÁGINA {page_num} START ---"))
        parts.append(llm.blob(data, "image/jpeg"))
        parts.append(llm.text(f"--- IMAGEN PÁGINA {page_num} END ---"))
    return parts


def page_image_parts(image_paths) -> list[llm.Part]:
    """Imágenes sin etiqueta, para las etapas que miran el documento entero."""
    parts: list[llm.Part] = []
    for path in image_paths:
        with open(path, "rb") as handle:
            parts.append(llm.blob(handle.read(), "image/jpeg"))
    return parts
