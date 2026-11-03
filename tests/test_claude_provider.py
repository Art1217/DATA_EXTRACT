"""Proveedor de Claude, sin red: un cliente de mentira con la forma del SDK.

Se prueba la traducción de la petición (bloques, tamaño de imágenes, esfuerzo
por etapa), la lectura de la respuesta (texto, cercas de código, tokens,
declinación por política) y la selección de proveedor por etapa.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from app.adapters import llm
from app.adapters.claude_llm import MAX_IMAGE_SIDE, ClaudeProvider
from app.config.settings import AIProvider
from app.domain.errors import ConfigurationError


class _Stream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class _Messages:
    def __init__(self, message):
        self.message = message
        self.calls: list[dict] = []

    def stream(self, **params):
        self.calls.append(params)
        return _Stream(self.message)


class _Client:
    def __init__(self, message):
        self.messages = _Messages(message)
        self.beta = SimpleNamespace(messages=_Messages(message))


def _message(text='{"ok": true}', stop_reason="end_turn", thinking=0):
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        usage=SimpleNamespace(
            input_tokens=100,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=20,
            output_tokens=50 + thinking,
            output_tokens_details=SimpleNamespace(thinking_tokens=thinking),
        ),
    )


def _request(task="extract", parts=None, model="claude-opus-5-5", max_tokens=None):
    return llm.Request(
        task=task,
        prompt_name="situacion_financiera",
        model=model,
        parts=tuple(parts or [llm.text("prompt")]),
        max_output_tokens=max_tokens,
    )


def _jpeg(width, height) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(out, format="JPEG")
    return out.getvalue()


# --- Petición --------------------------------------------------------------


def test_traduce_texto_imagen_y_pdf_a_bloques():
    client = _Client(_message())
    provider = ClaudeProvider(client)
    parts = [
        llm.blob(b"%PDF-1.7 demo", "application/pdf"),
        llm.text("--- IMAGEN PÁGINA 6 START ---"),
        llm.blob(_jpeg(100, 80), "image/jpeg"),
        llm.text("instrucciones"),
    ]

    provider.generate(_request(parts=parts))

    (call,) = client.messages.calls
    blocks = call["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["document", "text", "image", "text"]
    assert blocks[0]["source"]["media_type"] == "application/pdf"
    assert base64.standard_b64decode(blocks[0]["source"]["data"]) == b"%PDF-1.7 demo"
    assert blocks[1]["text"] == "--- IMAGEN PÁGINA 6 START ---"
    assert "temperature" not in call  # los modelos actuales la rechazan


def test_reduce_las_imagenes_grandes_a_2000_px():
    client = _Client(_message())
    ClaudeProvider(client).generate(
        _request(task="search", parts=[llm.blob(_jpeg(2550, 3300), "image/jpeg")])
    )

    data = client.messages.calls[0]["messages"][0]["content"][0]["source"]["data"]
    with Image.open(io.BytesIO(base64.standard_b64decode(data))) as image:
        assert max(image.size) == MAX_IMAGE_SIDE


def test_esfuerzo_distinto_para_etapas_baratas_y_extraccion():
    client = _Client(_message())
    provider = ClaudeProvider(client, effort_classify="medium", effort_extract="high")

    provider.generate(_request(task="search"))
    provider.generate(_request(task="extract"))
    provider.generate(_request(task="quality"))

    efforts = [c["output_config"]["effort"] for c in client.messages.calls]
    assert efforts == ["medium", "high", "high"]


def test_sin_esfuerzo_configurado_no_envia_output_config():
    client = _Client(_message())
    ClaudeProvider(client).generate(_request())
    assert "output_config" not in client.messages.calls[0]


def test_max_tokens_de_la_etapa_o_un_valor_por_defecto():
    client = _Client(_message())
    provider = ClaudeProvider(client)
    provider.generate(_request(max_tokens=65536))
    provider.generate(_request())
    assert [c["max_tokens"] for c in client.messages.calls] == [65536, 32000]


def test_con_fallback_usa_la_api_beta_donde_actua_el_middleware():
    client = _Client(_message())
    ClaudeProvider(client, use_beta=True).generate(_request())
    assert client.beta.messages.calls and not client.messages.calls


def test_un_modelo_gemini_con_proveedor_claude_es_un_error_de_configuracion():
    with pytest.raises(ConfigurationError, match="MODEL_NAME"):
        ClaudeProvider(_Client(_message())).generate(_request(model="gemini-3.8-flash"))


# --- Respuesta -------------------------------------------------------------


def test_devuelve_solo_el_texto_sin_cerca_de_codigo():
    client = _Client(_message(text='```json\n{"filas": []}\n```'))
    response = ClaudeProvider(client).generate(_request())
    assert response.text == '{"filas": []}'
    assert response.finish_reason == "end_turn"


def test_tokens_separan_razonamiento_y_suman_la_cache():
    client = _Client(_message(thinking=30))
    usage = ClaudeProvider(client).generate(_request()).usage
    assert usage == llm.Usage(input_tokens=120, output_tokens=50, reasoning_tokens=30)


def test_una_declinacion_por_politica_vuelve_sin_texto():
    client = _Client(_message(stop_reason="refusal"))
    response = ClaudeProvider(client).generate(_request())
    assert response.text is None
    assert response.finish_reason == "refusal"


# --- Selección por etapa ---------------------------------------------------


def test_ai_provider_extract_solo_cambia_extraccion_y_calidad(settings, monkeypatch):
    built = []

    def fake_build(kind, _settings):
        built.append(kind)
        return SimpleNamespace(kind=kind)

    monkeypatch.setattr(llm, "_build", fake_build)
    settings.ai_provider = AIProvider.VERTEX
    settings.ai_provider_extract = AIProvider.CLAUDE_VERTEX

    assert llm.get_provider("search").kind is AIProvider.VERTEX
    assert llm.get_provider("extract").kind is AIProvider.CLAUDE_VERTEX
    assert llm.get_provider("quality").kind is AIProvider.CLAUDE_VERTEX
    assert llm.get_provider().kind is AIProvider.VERTEX
    # Un proveedor por tipo y por proceso, no uno por llamada.
    assert built == [AIProvider.VERTEX, AIProvider.CLAUDE_VERTEX]


def test_claude_vertex_construye_el_cliente_de_anthropic(settings):
    settings.ai_provider = AIProvider.CLAUDE_VERTEX
    settings.gcp_project_id = "proyecto-demo"
    settings.claude_fallback_model = None

    provider = llm.get_provider()

    assert isinstance(provider, ClaudeProvider)
    assert type(provider._client).__name__ == "AnthropicVertex"
