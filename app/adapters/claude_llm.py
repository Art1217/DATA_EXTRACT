"""Claude como proveedor: en Vertex AI Model Garden o con la API de Anthropic.

Traduce las `Part` neutras de `llm.py` a bloques de la Messages API:

* texto -> bloque `text` (las etiquetas `--- IMAGEN PÁGINA N ---` incluidas);
* JPEG/PNG -> bloque `image`, reducido a 2000 px por lado como máximo: con
  más de 20 imágenes en una petición, la API rechaza las que pasen de ese
  tamaño, y un bloque de búsqueda trae 30 páginas a 300 DPI (~2500x3300 px);
* PDF -> bloque `document`, que Claude lee entero, texto e imagen.

Diferencias con Gemini que este módulo absorbe:

* **No hay modo JSON forzado.** Los prompts ya piden "solo JSON"; aquí se
  refuerza con un `system` y se quita la cerca de código si el modelo la
  pone. El resto de la tolerancia está en los parsers de cada etapa.
* **Sin `temperature`.** Los modelos actuales la rechazan; la profundidad se
  controla con `effort`, distinto para etapas baratas y para la extracción.
* **Siempre en streaming**, porque la extracción pide hasta 65k tokens de
  salida y el SDK exige streaming con valores así.
* **Declinar por política es un `stop_reason`, no un error.** Se devuelve sin
  texto para que la etapa lo trate como respuesta vacía. Con
  `CLAUDE_FALLBACK_MODEL` el SDK reintenta con ese modelo antes de rendirse.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any

from app.adapters.llm import LLMResponse, Part, Request, Usage
from app.adapters.telemetry import report_tokens
from app.domain.errors import ConfigurationError

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 32000
MAX_IMAGE_SIDE = 2000
_HEAVY_TASKS = {"extract", "quality"}
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")

_SYSTEM = (
    "Responde únicamente con el JSON que piden las instrucciones: sin texto antes ni "
    "después y sin bloques de código."
)


class ClaudeProvider:
    def __init__(
        self,
        client: Any,
        *,
        effort_classify: str | None = None,
        effort_extract: str | None = None,
        use_beta: bool = False,
    ) -> None:
        self._client = client
        self._effort = {"classify": effort_classify, "extract": effort_extract}
        # El reintento por política es un middleware que solo actúa sobre
        # `client.beta.messages`.
        self._messages = client.beta.messages if use_beta else client.messages

    def generate(self, request: Request) -> LLMResponse:
        if request.model.startswith("gemini"):
            raise ConfigurationError(
                f"El proveedor es Claude pero el modelo es '{request.model}'. "
                "Configura MODEL_NAME_CLASSIFY / MODEL_NAME_EXTRACT con un modelo Claude."
            )

        params: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_output_tokens or DEFAULT_MAX_TOKENS,
            "system": _SYSTEM,
            "messages": [{"role": "user", "content": [_block(p) for p in request.parts]}],
        }
        effort = self._effort["extract" if request.task in _HEAVY_TASKS else "classify"]
        if effort:
            params["output_config"] = {"effort": effort}

        with self._messages.stream(**params) as stream:
            message = stream.get_final_message()

        usage = _usage(message)
        if usage:
            report_tokens(
                request.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            )

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            logger.warning(
                "Claude declinó la petición %s/%s (categoría: %s).",
                request.task,
                request.prompt_name,
                getattr(details, "category", None),
            )
            return LLMResponse(text=None, finish_reason="refusal", usage=usage)

        text = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
        return LLMResponse(
            text=_FENCE.sub("", text.strip()) or None,
            finish_reason=message.stop_reason,
            usage=usage,
        )


# --------------------------------------------------------------------------
# Traducción de la petición
# --------------------------------------------------------------------------


def _block(part: Part) -> dict:
    if part.text is not None:
        return {"type": "text", "text": part.text}
    if part.mime_type == "application/pdf":
        return {"type": "document", "source": _base64_source(part.data, "application/pdf")}
    if part.mime_type in {"image/jpeg", "image/png"}:
        data, mime = _fit_image(part.data, part.mime_type)
        return {"type": "image", "source": _base64_source(data, mime)}
    raise ValueError(f"Tipo de contenido no soportado por Claude: {part.mime_type}")


def _base64_source(data: bytes, media_type: str) -> dict:
    return {
        "type": "base64",
        "media_type": media_type,
        "data": base64.standard_b64encode(data).decode("ascii"),
    }


def _fit_image(data: bytes, mime_type: str) -> tuple[bytes, str]:
    """Reduce la imagen si un lado pasa de `MAX_IMAGE_SIDE`; si no, la deja igual."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        if max(image.size) <= MAX_IMAGE_SIDE:
            return data, mime_type
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        out = io.BytesIO()
        image.convert("RGB").save(out, format="JPEG", quality=85)
        return out.getvalue(), "image/jpeg"


def _usage(message: Any) -> Usage | None:
    usage = getattr(message, "usage", None)
    if usage is None:
        return None
    details = getattr(usage, "output_tokens_details", None)
    thinking = int(getattr(details, "thinking_tokens", 0) or 0)
    # La entrada cacheada también se procesa (y se cobra, más barata): se suma
    # para que el total sea comparable con el de Gemini.
    input_tokens = sum(
        int(getattr(usage, name, 0) or 0)
        for name in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    output = int(getattr(usage, "output_tokens", 0) or 0)
    # `output_tokens` ya incluye el razonamiento; `Usage` los lleva separados.
    return Usage(
        input_tokens=input_tokens,
        output_tokens=max(output - thinking, 0),
        reasoning_tokens=thinking,
    )


# --------------------------------------------------------------------------
# Clientes
# --------------------------------------------------------------------------


def build_provider(kind: str, settings) -> ClaudeProvider:
    try:
        import anthropic
    except ImportError as error:
        raise ConfigurationError(
            "Claude requiere el SDK de Anthropic: pip install -e '.[claude]'"
        ) from error

    options: dict[str, Any] = {}
    if settings.claude_fallback_model:
        options["middleware"] = [
            anthropic.BetaRefusalFallbackMiddleware([{"model": settings.claude_fallback_model}])
        ]

    if kind == "claude_vertex":
        client = anthropic.AnthropicVertex(**_vertex_auth(settings), **options)
    else:
        key = settings.anthropic_api_key
        if key:
            options["api_key"] = key.get_secret_value()
        client = anthropic.Anthropic(**options)

    return ClaudeProvider(
        client,
        effort_classify=settings.claude_effort_classify or None,
        effort_extract=settings.claude_effort_extract or None,
        use_beta=bool(settings.claude_fallback_model),
    )


def _vertex_auth(settings) -> dict[str, Any]:
    """Mismas credenciales que Gemini: service account o ADC, y la región de Vertex."""
    auth: dict[str, Any] = {"region": settings.gcp_location}
    if settings.credentials_path is not None:
        from google.oauth2 import service_account

        from app.adapters.llm import SCOPES

        credentials = service_account.Credentials.from_service_account_file(
            str(settings.credentials_path), scopes=SCOPES
        )
        auth["credentials"] = credentials
        auth["project_id"] = settings.gcp_project_id or credentials.project_id
    elif settings.gcp_project_id:
        auth["project_id"] = settings.gcp_project_id
    return auth
