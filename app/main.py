"""Punto de entrada de la API.

Arranque:

1. valida el catálogo contra el disco y falla de inmediato si falta un prompt,
   en vez de descubrirlo a mitad de un job caro;
2. cierra los jobs que un reinicio dejó a medias, para que nadie quede
   sondeando un trabajo que ya nadie ejecuta;
3. purga los workspaces huérfanos de ejecuciones anteriores.

Ejecutar con:  `python -m uvicorn app.main:app --reload --port 8000`
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import require_api_key
from app.api.errors import register_error_handlers
from app.api.routers import (
    batches,
    catalog,
    classification,
    documents,
    extraction,
    identity,
    jobs,
    pipeline,
    quality,
    search,
)
from app.config import get_settings
from app.domain.analysis import missing_prompts
from app.domain.errors import ConfigurationError
from app.jobs.handlers import HANDLERS
from app.jobs.runner import JobRunner
from app.jobs.store import FileJobStore
from app.storage import build_storage
from app.storage.workspace import purge_stale

logger = logging.getLogger(__name__)

API_TITLE = "Document Extraction AI"
API_VERSION = "1.0.0"


def _configure_logging(level: str) -> None:
    """Nivel, formato y —en Windows— una salida que aguante el castellano.

    La consola de Windows suele ser cp1252, que no sabe codificar ni un emoji
    ni, según la página de códigos, algunas tildes. Escribir ahí un mensaje
    que no quepa lanza `UnicodeEncodeError`; si eso ocurre dentro de un job,
    se pierde una extracción ya pagada por no poder contar lo que estaba
    haciendo.

    En `app/` no hay `print` —lo fija una prueba— y el logger se traga los
    errores de su propio volcado. Aun así se fuerza UTF-8 aquí: que el
    registro sea legible no debería depender de la página de códigos con la
    que arrancó la terminal.
    """
    for flujo in (sys.stdout, sys.stderr):
        reconfigurar = getattr(flujo, "reconfigure", None)
        if reconfigurar is not None:
            try:
                reconfigurar(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass  # stdout redirigido a algo que no admite reconfiguración

    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    settings.ensure_directories()

    absent = missing_prompts(settings.prompt_path)
    if absent:
        raise ConfigurationError(
            "Faltan prompts declarados en el catálogo:\n  " + "\n  ".join(absent)
        )

    storage = build_storage(settings)
    job_store = FileJobStore(storage)
    job_store.recover_interrupted()
    purge_stale(settings.temp_dir)

    runner = JobRunner(store=job_store, storage=storage, settings=settings, handlers=HANDLERS)
    runner.start()

    app.state.settings = settings
    app.state.storage = storage
    app.state.job_store = job_store
    app.state.runner = runner
    logger.info(
        "API lista (modelo: %s, storage: %s)", settings.ai_provider, settings.storage_backend
    )

    try:
        yield
    finally:
        runner.shutdown()


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    summary=(
        "Clasificación, búsqueda, extracción y control de calidad de tablas en PDFs "
        "con Gemini (Vertex AI) y Document AI."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)

for module in (
    catalog,
    documents,
    classification,
    identity,
    search,
    extraction,
    quality,
    pipeline,
    jobs,
    batches,
):
    app.include_router(module.router, dependencies=[Depends(require_api_key)])

app.include_router(pipeline.batch_router, dependencies=[Depends(require_api_key)])


@app.get("/healthz", tags=["sistema"])
def health() -> dict:
    return {"status": "ok", "version": API_VERSION}


@app.get("/v1/config", tags=["sistema"], dependencies=[Depends(require_api_key)])
def effective_config() -> dict:
    """Parámetros con los que está corriendo el backend.

    No expone credenciales: solo los valores que afectan al comportamiento y
    al costo del procesamiento.
    """
    settings = get_settings()
    return {
        "ai_provider": settings.ai_provider,
        "models": {
            "classify": settings.model_name_classify,
            "extract": settings.model_name_extract,
            "location": settings.gcp_location,
        },
        "document_ai": bool(settings.docai_processor_id),
        "storage": settings.storage_backend,
        "processing": {
            "dpi": settings.dpi,
            "clasif_dpi": settings.clasif_dpi,
            "block_size": settings.block_size,
            "busqueda_workers": settings.busqueda_workers,
            "identity_pages": settings.identity_pages,
            "chunk_size_digital": settings.chunk_size_digital,
            "chunk_size_scanned": settings.chunk_size_scanned,
            "chunk_size_docai_json": settings.chunk_size_docai_json,
        },
        "jobs": {
            "workers": settings.job_workers,
            "max_concurrent_llm_calls": settings.max_concurrent_llm_calls,
        },
        "publish": {
            "enabled": settings.publish_enabled,
            "target": settings.publish_target().describe(),
        },
        "credentials": "service_account" if settings.credentials_path else "adc",
    }
