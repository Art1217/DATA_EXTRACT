"""Los adaptadores por dentro, con el SDK de Gemini simulado.

El resto de la suite corre con el proveedor `fake`, que responde ya con la
forma final. Estas pruebas bajan un nivel: sustituyen solo el cliente de
`google-genai` dentro de `GoogleGenAIProvider`, de modo que corre el código
real del proveedor y del adaptador —troceado del PDF, elección de modelo,
anchor de continuidad, telemetría— sin llamar al modelo ni gastar un centavo.
"""

from __future__ import annotations

import json

import pytest

from app.adapters import ai_extractor, ai_searcher, llm
from app.domain.analysis import get_analysis_type

FILAS = [
    {"concepto": "Ventas netas", "periodo": "2024", "valor": 100},
    {"concepto": "Costo de ventas", "periodo": "2024", "valor": -60},
]


class _RespuestaFalsa:
    """Lo mínimo que el proveedor lee de una respuesta de Gemini."""

    def __init__(self, payload):
        self.text = json.dumps(payload, ensure_ascii=False)
        self.usage_metadata = None
        self.candidates = []


class _ClienteFalso:
    def __init__(self, payload):
        self._payload = payload
        self.llamadas: list[dict] = []
        self.models = self

    def generate_content(self, model, contents, config=None):
        self.llamadas.append({"model": model, "partes": len(contents), "config": config})
        return _RespuestaFalsa(self._payload)


@pytest.fixture
def pdf_en_disco(tmp_path, pdf_bytes):
    ruta = tmp_path / "corte.pdf"
    ruta.write_bytes(pdf_bytes)
    return str(ruta)


@pytest.fixture
def prompt_extraccion(settings):
    return str(settings.prompt_path(get_analysis_type("resultados").extract_prompt))


def _instalar(monkeypatch, cliente):
    """Pone un `GoogleGenAIProvider` real sobre el cliente simulado."""
    proveedor = llm.GoogleGenAIProvider(cliente)
    monkeypatch.setattr(llm, "get_provider", lambda task=None: proveedor)
    return proveedor


# --- Extractor ------------------------------------------------------------


def test_la_extraccion_devuelve_las_filas_del_modelo(
    monkeypatch, settings, pdf_en_disco, prompt_extraccion
):
    cliente = _ClienteFalso({"filas": FILAS})
    _instalar(monkeypatch, cliente)

    filas, docai = ai_extractor.extract_structured_data_chunked(
        pdf_en_disco, prompt_extraccion, rows_key="filas"
    )

    assert filas == FILAS
    assert docai is None, "un PDF digital no pasa por Document AI"
    assert cliente.llamadas[0]["model"] == settings.model_name_extract
    assert cliente.llamadas[0]["config"].response_mime_type == "application/json"


def test_el_tamano_de_chunk_sale_de_la_configuracion(
    monkeypatch, settings, pdf_en_disco, prompt_extraccion
):
    """El PDF tiene 3 páginas: con chunks de 1 son 3 llamadas al modelo."""
    settings.chunk_size_digital = 1
    settings.chunk_overlap_pages = 0
    cliente = _ClienteFalso({"filas": FILAS})
    _instalar(monkeypatch, cliente)
    monkeypatch.setattr(ai_extractor.time, "sleep", lambda _: None)

    filas, _ = ai_extractor.extract_structured_data_chunked(
        pdf_en_disco, prompt_extraccion, rows_key="filas"
    )

    assert len(cliente.llamadas) == 3
    assert len(filas) == 6


def test_un_json_con_comillas_comidas_se_repara():
    """`valor":` en vez de `"valor":` a mitad de la respuesta no tira la extracción."""
    roto = '[\n  {"concepto": "Ventas", "valor": 10},\n  {"concepto": "Costo",\n  valor": -4}\n]'
    assert ai_extractor.repair_truncated_json(roto) == [
        {"concepto": "Ventas", "valor": 10},
        {"concepto": "Costo", "valor": -4},
    ]


def test_un_json_truncado_conserva_las_filas_completas():
    truncado = '[{"concepto": "Ventas", "valor": 10}, {"concepto": "Cos'
    assert ai_extractor.repair_truncated_json(truncado) == [{"concepto": "Ventas", "valor": 10}]


def test_el_anchor_de_continuidad_respeta_la_configuracion(settings):
    """Sin `max_rows` explícito, el tope sale de `chunk_anchor_rows`."""
    settings.chunk_anchor_rows = 1

    nota = ai_extractor._build_continuity_anchor(FILAS)

    assert "Costo de ventas" in nota and "Ventas netas" not in nota


def test_la_revision_de_calidad_usa_el_modelo_grande(monkeypatch, settings):
    cliente = _ClienteFalso({"estado": "Aprobado", "observaciones": []})
    _instalar(monkeypatch, cliente)
    prompt = str(settings.prompt_path(get_analysis_type("resultados").quality_prompt))

    resultado = ai_extractor.run_quality_review(FILAS, prompt)

    assert resultado["estado"] == "Aprobado"
    assert cliente.llamadas[0]["model"] == settings.model_name_extract


# --- Buscador -------------------------------------------------------------


def test_la_busqueda_usa_el_modelo_barato(monkeypatch, settings, tmp_path):
    from PIL import Image

    imagen = tmp_path / "page_1.jpg"
    Image.new("RGB", (10, 10)).save(imagen)

    cliente = _ClienteFalso({"relevant_sections": [{"page_number": 1}], "alerts": []})
    _instalar(monkeypatch, cliente)
    prompt = str(settings.prompt_path(get_analysis_type("resultados").search_prompt))

    resultado = ai_searcher.search_pages_block([(1, str(imagen))], prompt_path=prompt)

    assert resultado["relevant_sections"][0]["page_number"] == 1
    assert cliente.llamadas[0]["model"] == settings.model_name_classify
    # prompt + (etiqueta, imagen, etiqueta) por página
    assert cliente.llamadas[0]["partes"] == 4


# --- Selección de proveedor -----------------------------------------------


def test_por_defecto_el_proveedor_es_el_de_demo(settings):
    from app.adapters.fake_llm import FakeProvider

    assert isinstance(llm.get_provider(), FakeProvider)


def test_gemini_api_sin_clave_falla_al_construirse(settings, monkeypatch):
    from app.config.settings import AIProvider
    from app.domain.errors import ConfigurationError

    settings.ai_provider = AIProvider.GEMINI_API
    settings.gemini_api_key = None
    with pytest.raises(ConfigurationError):
        llm.get_provider()


# --- La telemetría no puede tumbar una extracción -------------------------


class _UsoConNones:
    """Lo que Gemini devuelve de vez en cuando: contadores existentes en None."""

    prompt_token_count = 1200
    candidates_token_count = None
    thoughts_token_count = None


class _RunTreeFalso:
    """Traza de LangSmith de mentira, con lo poco que `report_usage` toca."""

    def __init__(self):
        self.extra: dict = {}
        self.parcheado = False

    def patch(self):
        self.parcheado = True


def test_un_contador_de_tokens_en_none_no_rompe_la_extraccion(
    monkeypatch, settings, pdf_en_disco, prompt_extraccion
):
    """`0 + None` reventaría DESPUÉS de pagar la llamada.

    Hay que forzar la traza a mano: sin LangSmith activo
    `get_current_run_tree()` devuelve `None`, el bloque de telemetría ni se
    ejecuta y la prueba pasaría sin tocar el código que se quiere fijar.
    """
    run_tree = _RunTreeFalso()
    monkeypatch.setattr("app.adapters.telemetry.get_current_run_tree", lambda: run_tree)

    cliente = _ClienteFalso({"filas": FILAS})
    _instalar(monkeypatch, cliente)
    original = _RespuestaFalsa.__init__

    def con_uso(self, payload):
        original(self, payload)
        self.usage_metadata = _UsoConNones()

    monkeypatch.setattr(_RespuestaFalsa, "__init__", con_uso)

    filas, _ = ai_extractor.extract_structured_data_chunked(
        pdf_en_disco, prompt_extraccion, rows_key="filas"
    )

    assert filas == FILAS
    uso = run_tree.extra["metadata"]["usage_metadata"]
    assert uso["input_tokens"] == 1200
    assert uso["output_tokens"] == 0, "los None cuentan como cero, no revientan"
    assert run_tree.parcheado, "la traza se reportó de verdad"


def test_un_fallo_de_la_traza_no_afecta_al_resultado(monkeypatch):
    """Reportar cuánto costó una llamada no puede costar la llamada."""
    from app.adapters.telemetry import report_usage

    def revienta():
        raise RuntimeError("LangSmith caído")

    monkeypatch.setattr("app.adapters.telemetry.get_current_run_tree", revienta)

    class Respuesta:
        usage_metadata = _UsoConNones()

    report_usage(Respuesta(), "gemini-x")  # no debe lanzar


def test_los_tokens_en_none_cuentan_como_cero():
    from app.adapters.telemetry import _count

    assert _count(_UsoConNones(), "candidates_token_count") == 0
    assert _count(_UsoConNones(), "prompt_token_count") == 1200
    assert _count(_UsoConNones(), "no_existe") == 0
