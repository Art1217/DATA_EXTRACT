"""Fixtures compartidas.

Ninguna prueba llama a un modelo real. Corren con `AI_PROVIDER=fake`, el
mismo proveedor determinista del modo demo, envuelto en un contador: así se
ejercita la orquestación real —adaptadores, parseo de respuestas, storage,
jobs, encadenado de etapas y rutas HTTP— y se puede afirmar que una etapa
reutilizó un resultado en vez de volver a pagar por él.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import pymupdf
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.adapters import llm  # noqa: E402
from app.adapters.fake_llm import FakeProvider  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.services.context import Progress, StageContext  # noqa: E402
from app.storage.local import LocalStorage  # noqa: E402
from app.storage.workspace import Workspace  # noqa: E402

# Variables que un `.env` local podría traer y que cambiarían el resultado de
# una prueba según quién la corra.
_ISOLATED_ENV = {
    "AI_PROVIDER": "fake",
    "AI_PROVIDER_EXTRACT": "",
    "ANTHROPIC_API_KEY": "",
    "CLAUDE_FALLBACK_MODEL": "",
    "STORAGE_BACKEND": "local",
    "PUBLISH_ENABLED": "false",
    "GCS_BUCKET": "",
    "API_KEY": "",
    "LANGSMITH_API_KEY": "",
    "DOCAI_PROCESSOR_ID": "",
    "CREDENTIALS_PATH": "",
    "JOB_WORKERS": "1",
}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Configuración aislada, apuntando a directorios desechables.

    `base_dir` se mantiene para que los prompts reales sigan resolviéndose:
    el catálogo se valida contra el disco de verdad.
    """
    for name, value in _ISOLATED_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "temp"))
    get_settings.cache_clear()
    llm.reset_providers()
    resolved = get_settings()
    resolved.ensure_directories()
    yield resolved
    get_settings.cache_clear()
    llm.reset_providers()


@pytest.fixture
def storage(settings) -> LocalStorage:
    return LocalStorage(settings.storage_dir)


@pytest.fixture
def context(settings, storage, tmp_path) -> StageContext:
    workspace_dir = tmp_path / "ws"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return StageContext(
        settings=settings,
        storage=storage,
        workspace=Workspace(workspace_dir),
        progress=Progress.none(),
    )


@pytest.fixture
def pdf_bytes() -> bytes:
    """PDF de tres páginas con texto, para que no se considere escaneado."""
    return build_pdf(
        [
            "Estado de Situacion Financiera " + "x" * 200,
            "Notas a los estados financieros " + "y" * 200,
            "Estado de Resultados " + "z" * 200,
        ]
    )


def build_pdf(pages: list[str]) -> bytes:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 550, 700), text, fontsize=11)
    payload = document.tobytes()
    document.close()
    return payload


# --------------------------------------------------------------------------
# Proveedor de modelo con contador
# --------------------------------------------------------------------------


class CountingProvider:
    """El proveedor `fake`, anotando cada llamada por etapa y por prompt.

    `calls["search"]` cuenta las búsquedas; `calls["classify:grupo"]`, las
    llamadas de la Fase 1. `overrides` permite que una prueba fuerce la
    respuesta de una etapa (por ejemplo, una extracción que descuadra).
    """

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.overrides: dict[str, object] = {}
        self._inner = FakeProvider()

    def generate(self, request: llm.Request) -> llm.LLMResponse:
        self.calls[request.task] += 1
        self.calls[f"{request.task}:{request.prompt_name}"] += 1
        override = self.overrides.get(request.task)
        if override is not None:
            import json

            payload = override(request) if callable(override) else override
            return llm.LLMResponse(text=json.dumps(payload), finish_reason="STOP")
        return self._inner.generate(request)

    def __getitem__(self, key: str) -> int:
        return self.calls[key]

    def __contains__(self, key: str) -> bool:
        return self.calls[key] > 0


@pytest.fixture
def fake_ai(settings, monkeypatch) -> CountingProvider:
    provider = CountingProvider()
    monkeypatch.setattr(llm, "get_provider", lambda task=None: provider)
    return provider


@pytest.fixture
def client(settings, fake_ai):
    """Cliente HTTP con el ciclo de vida de la aplicación activo."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def wait_for_job(client, job_id: str, timeout: float = 30.0) -> dict:
    """Sondea un job hasta que termina. Falla si se agota el tiempo."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"El job {job_id} no terminó en {timeout} s")
