"""Contar lo que se hace no puede impedir hacerlo.

Un `print` en código de librería escribe en stdout, y en Windows stdout suele
ser cp1252. Un `print(f"📄 PDF {tipo} ...")` justo antes de trocear el PDF
lanza `UnicodeEncodeError` en cuanto el backend arranca con la salida
redirigida, y se lleva por delante una extracción ya pagada por un mensaje
puramente informativo — el progreso real viaja por `ctx.progress`.

El logger no tiene ese problema: `StreamHandler.emit` atrapa sus propios
errores y sigue. Así que la regla es que en `app/` no se imprime, se registra.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"


def _prints(archivo: Path) -> list[int]:
    """Líneas con una llamada a `print` de primer nivel."""
    arbol = ast.parse(archivo.read_text(encoding="utf-8"), filename=str(archivo))
    return [
        nodo.lineno
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Name)
        and nodo.func.id == "print"
    ]


def test_en_el_backend_no_se_imprime_se_registra():
    culpables = [
        f"{archivo.relative_to(APP)}:{linea}"
        for archivo in sorted(APP.rglob("*.py"))
        for linea in _prints(archivo)
    ]

    assert not culpables, (
        "Un print puede lanzar UnicodeEncodeError y tumbar el job que lo emite. "
        "Usa logger.info/warning:\n  " + "\n  ".join(culpables)
    )


def test_registrar_un_emoji_en_una_consola_cp1252_no_lanza(caplog):
    """Una consola cp1252 estricta, sobre el logger."""
    import io

    flujo = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    handler = logging.StreamHandler(flujo)
    logger = logging.getLogger("prueba.cp1252")
    logger.addHandler(handler)
    try:
        logger.warning("📄 PDF digital — 12 páginas.")  # no debe lanzar
    finally:
        logger.removeHandler(handler)


def test_la_configuracion_deja_la_salida_en_utf8():
    """Para que el registro sea legible sin depender de la terminal."""
    import sys

    from app.main import _configure_logging

    _configure_logging("INFO")

    for flujo in (sys.stdout, sys.stderr):
        codificacion = getattr(flujo, "encoding", None)
        if codificacion is not None:
            assert codificacion.lower().replace("-", "") == "utf8", flujo


@pytest.mark.parametrize(
    "modulo", ["ai_extractor", "ai_classifier", "ai_searcher", "pdf_utils", "prompts"]
)
def test_los_adaptadores_tienen_su_propio_logger(modulo):
    import importlib

    adaptador = importlib.import_module(f"app.adapters.{modulo}")

    assert adaptador.logger.name == f"app.adapters.{modulo}"
