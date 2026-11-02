"""Dependencias compartidas por los routers.

Los componentes de larga vida (storage, almacén de jobs, runner) se crean una
vez durante el arranque y viven en `app.state`. Estas funciones son el único
punto donde los routers los obtienen, de modo que un test puede sustituirlos
sin tocar el código de las rutas.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.config import Settings
from app.jobs.runner import JobRunner
from app.jobs.store import JobStore
from app.services.context import Progress, StageContext
from app.storage.base import Storage
from app.storage.workspace import Workspace


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def get_runner(request: Request) -> JobRunner:
    return request.app.state.runner


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


_api_key_scheme = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description=(
        "Clave de acceso a esta API. Solo se exige si el despliegue define "
        "API_KEY. No tiene relación con las credenciales de Google Cloud, que "
        "el servidor usa por su cuenta para hablar con Vertex AI."
    ),
)


def require_api_key(
    request: Request,
    api_key: Annotated[str | None, Security(_api_key_scheme)] = None,
) -> None:
    """Comprueba la clave de acceso si el despliegue la tiene configurada.

    Sin `API_KEY` en la configuración la API queda abierta, que es lo
    razonable en una instalación local dentro de una red privada. En cuanto se
    define, todo endpoint `/v1` la exige.

    `auto_error=False` en el esquema es deliberado: sin él, FastAPI rechazaría
    con 403 toda petición sin cabecera aunque no haya clave configurada.
    """
    expected = request.app.state.settings.api_key
    if expected and not secrets.compare_digest(
        (api_key or "").encode(), expected.get_secret_value().encode()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Falta la cabecera X-API-Key o es incorrecta.",
        )


def get_read_context(
    request: Request,
    storage: Annotated[Storage, Depends(get_storage)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> StageContext:
    """Contexto para operaciones síncronas de solo lectura.

    Usa un workspace apuntando al temporal general porque estas operaciones no
    escriben archivos intermedios. Las que sí lo hacen corren como job y
    reciben un workspace propio del runner.
    """
    return StageContext(
        settings=settings,
        storage=storage,
        workspace=Workspace(settings.temp_dir),
        progress=Progress.none(),
    )


StorageDep = Annotated[Storage, Depends(get_storage)]
JobStoreDep = Annotated[JobStore, Depends(get_job_store)]
RunnerDep = Annotated[JobRunner, Depends(get_runner)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
ReadContextDep = Annotated[StageContext, Depends(get_read_context)]
