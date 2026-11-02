"""Puerta única hacia el modelo de lenguaje.

Los adaptadores de etapa (`ai_classifier`, `ai_searcher`, `ai_extractor`)
arman el contenido de la petición y llaman a `generate()`. Qué proveedor
responde lo decide `AI_PROVIDER`:

* `vertex` — Gemini en Vertex AI, autenticado con las Application Default
  Credentials o con una service account. Es la vía de producción en GCP.
* `gemini_api` — Gemini con una API key de Google AI Studio, para probar con
  un modelo real sin montar un proyecto de GCP.
* `fake` — respuestas deterministas sin red (`app/adapters/fake_llm.py`).
  Permite levantar la API y recorrer el pipeline entero sin credenciales.

El contenido viaja como una lista de `Part` neutra, no como tipos de
`google.genai`: así el modo `fake` no necesita importar el SDK y un proveedor
nuevo solo tiene que traducir dos clases.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol

from app.adapters.telemetry import report_usage
from app.config import get_settings
from app.config.settings import AIProvider

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


@dataclass(frozen=True, slots=True)
class Part:
    """Un trozo de la petición: texto, o bytes con su tipo MIME."""

    text: str | None = None
    data: bytes | None = None
    mime_type: str | None = None


def text(value: str) -> Part:
    return Part(text=value)


def blob(data: bytes, mime_type: str) -> Part:
    return Part(data=data, mime_type=mime_type)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str | None
    """`None` cuando el modelo agotó su presupuesto de salida sin escribir nada."""
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class Request:
    """Todo lo que un proveedor necesita para responder."""

    task: str
    """Qué etapa pregunta (`classify`, `search`, `identity`, `extract`, `quality`).
    El proveedor real lo ignora; el `fake` lo usa para elegir su respuesta."""
    prompt_name: str
    """Nombre del prompt sin extensión (`situacion_financiera`)."""
    model: str
    parts: tuple[Part, ...]
    temperature: float = 0.0
    max_output_tokens: int | None = None


class LLMProvider(Protocol):
    def generate(self, request: Request) -> LLMResponse: ...


def generate(
    *,
    task: str,
    prompt_name: str,
    model: str,
    parts: list[Part] | tuple[Part, ...],
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
) -> LLMResponse:
    """Pide una respuesta JSON al proveedor configurado."""
    request = Request(
        task=task,
        prompt_name=prompt_name,
        model=model,
        parts=tuple(parts),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return get_provider().generate(request)


# --------------------------------------------------------------------------
# Selección del proveedor
# --------------------------------------------------------------------------

_providers: dict[str, LLMProvider] = {}
_lock = threading.Lock()


def get_provider() -> LLMProvider:
    """Proveedor de `AI_PROVIDER`, construido una vez por proceso."""
    settings = get_settings()
    kind = settings.ai_provider
    with _lock:
        provider = _providers.get(kind)
    if provider is not None:
        return provider

    provider = _build(kind, settings)
    with _lock:
        return _providers.setdefault(kind, provider)


def _build(kind: AIProvider, settings) -> LLMProvider:
    if kind is AIProvider.FAKE:
        from app.adapters.fake_llm import FakeProvider

        return FakeProvider()
    return GoogleGenAIProvider(_genai_client(kind, settings))


def _genai_client(kind: AIProvider, settings) -> Any:
    from google import genai

    from app.domain.errors import ConfigurationError

    if kind is AIProvider.GEMINI_API:
        if not settings.gemini_api_key:
            raise ConfigurationError("AI_PROVIDER=gemini_api requiere GEMINI_API_KEY.")
        return genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    if settings.credentials_path is None:
        return genai.Client(
            vertexai=True, project=settings.gcp_project_id, location=settings.gcp_location
        )

    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(
        str(settings.credentials_path), scopes=SCOPES
    )
    return genai.Client(
        vertexai=True,
        project=settings.gcp_project_id or credentials.project_id,
        location=settings.gcp_location,
        credentials=credentials,
    )


def reset_providers() -> None:
    """Olvida los proveedores cacheados. Para tests y recarga de settings."""
    with _lock:
        _providers.clear()


# --------------------------------------------------------------------------
# Gemini (Vertex AI o AI Studio): mismo SDK, distinto cliente
# --------------------------------------------------------------------------

# Los informes financieros no tienen nada que los filtros de seguridad deban
# bloquear, pero un falso positivo (una nota sobre "explosivos" en minería)
# tiraría una extracción ya pagada.
_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_HARASSMENT",
)


class GoogleGenAIProvider:
    def __init__(self, client: Any) -> None:
        self._client = client

    def generate(self, request: Request) -> LLMResponse:
        from google.genai import types

        contents = [
            types.Part.from_text(text=part.text)
            if part.text is not None
            else types.Part.from_bytes(data=part.data, mime_type=part.mime_type)
            for part in request.parts
        ]
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            safety_settings=[
                types.SafetySetting(category=category, threshold="BLOCK_NONE")
                for category in _SAFETY_CATEGORIES
            ],
        )
        response = self._client.models.generate_content(
            model=request.model, contents=contents, config=config
        )
        report_usage(response, request.model)
        return LLMResponse(text=response.text, finish_reason=_finish_reason(response))


def _finish_reason(response: Any) -> str | None:
    try:
        return str(response.candidates[0].finish_reason)
    except Exception:
        return None
