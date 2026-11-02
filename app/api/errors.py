"""Traducción de errores del dominio a respuestas HTTP.

Concentrar el mapeo aquí es lo que permite que los servicios lancen errores
con significado de negocio y no códigos HTTP. Cada respuesta lleva además un
`code` estable, para que un cliente pueda reaccionar sin analizar el texto
del mensaje.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.domain.errors import (
    ArtifactNotFoundError,
    ConfigurationError,
    DocumentNotFoundError,
    InvalidDocumentError,
    JobNotFoundError,
    PipelineError,
    StageNotReadyError,
    StageNotSupportedError,
    UnknownAnalysisTypeError,
    UnknownGroupError,
)

logger = logging.getLogger(__name__)

# Cada error del dominio, con el código HTTP que le corresponde y un
# identificador estable para el cliente.
_MAPPING: tuple[tuple[type[PipelineError], int, str], ...] = (
    (DocumentNotFoundError, status.HTTP_404_NOT_FOUND, "document_not_found"),
    (ArtifactNotFoundError, status.HTTP_404_NOT_FOUND, "artifact_not_found"),
    (JobNotFoundError, status.HTTP_404_NOT_FOUND, "job_not_found"),
    (UnknownAnalysisTypeError, status.HTTP_422_UNPROCESSABLE_CONTENT, "unknown_analysis_type"),
    (UnknownGroupError, status.HTTP_422_UNPROCESSABLE_CONTENT, "unknown_group"),
    (InvalidDocumentError, status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_document"),
    (StageNotSupportedError, status.HTTP_422_UNPROCESSABLE_CONTENT, "stage_not_supported"),
    # 409: la petición es válida, pero falta un paso previo. El mensaje dice
    # cuál, porque es el error más común al encadenar endpoints a mano.
    (StageNotReadyError, status.HTTP_409_CONFLICT, "stage_not_ready"),
    # 500: el despliegue está mal configurado; no es culpa de quien llama.
    (ConfigurationError, status.HTTP_500_INTERNAL_SERVER_ERROR, "configuration_error"),
)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(PipelineError)
    async def handle_pipeline_error(_: Request, exc: PipelineError) -> JSONResponse:
        for error_type, http_status, code in _MAPPING:
            if isinstance(exc, error_type):
                if http_status >= 500:
                    logger.error("%s: %s", code, exc)
                return JSONResponse(
                    status_code=http_status, content={"detail": str(exc), "code": code}
                )
        logger.exception("Error de pipeline sin clasificar.")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc), "code": "pipeline_error"},
        )

    @app.exception_handler(ValueError)
    async def handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": str(exc), "code": "invalid_request"},
        )
