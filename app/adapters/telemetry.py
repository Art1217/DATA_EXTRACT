"""Consumo de tokens hacia la traza de LangSmith.

**Nunca puede tumbar una extracción.** `getattr(usage, name, 0)` no basta: el
atributo existe y a veces vale `None`, que es lo que Gemini devuelve de vez en
cuando, y sumar `None` lanza `TypeError` **después** de que el modelo
respondiera, con la extracción ya pagada. Por eso los contadores se
normalizan y todo va dentro de un `try`: reportar cuánto costó una llamada no
puede costar la llamada.
"""

from __future__ import annotations

import logging
from typing import Any

from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)


def report_usage(response: Any, model: str) -> None:
    """Adjunta a la traza en curso los tokens que consumió `response`."""
    try:
        run_tree = get_current_run_tree()
        usage = getattr(response, "usage_metadata", None)
        if run_tree is None or usage is None:
            return

        razonamiento = _count(usage, "thoughts_token_count")
        run_tree.extra.setdefault("metadata", {}).update(
            {
                "ls_model_name": model,
                "usage_metadata": {
                    "input_tokens": _count(usage, "prompt_token_count"),
                    "output_tokens": _count(usage, "candidates_token_count") + razonamiento,
                    "output_token_details": {"reasoning": razonamiento},
                },
            }
        )
        run_tree.patch()
    except Exception:
        # La traza es observabilidad, no resultado: si falla, se anota y ya.
        logger.debug("No se pudo reportar el consumo de tokens.", exc_info=True)


def _count(usage: Any, name: str) -> int:
    """Contador de tokens como entero, aunque el atributo valga `None`."""
    return int(getattr(usage, name, 0) or 0)
