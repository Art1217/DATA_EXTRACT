"""Errores del dominio.

Cada uno expresa una condición de negocio concreta; la capa `api` los traduce
a códigos HTTP en un solo lugar (`app/api/errors.py`). El resto del código
lanza estos errores y nunca códigos HTTP, para que los servicios sigan siendo
usables fuera de un servidor web.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Raíz de todos los errores esperables del pipeline."""


class ConfigurationError(PipelineError):
    """El despliegue está mal configurado (prompt ausente, credenciales, etc.).

    Es culpa del servidor, no del cliente.
    """


class UnknownAnalysisTypeError(PipelineError):
    """Se pidió un `analysis_type` que no existe en el catálogo."""

    def __init__(self, key: str, *, valid: tuple[str, ...] = ()) -> None:
        self.key = key
        self.valid = valid
        detail = f"Tipo de análisis desconocido: '{key}'."
        if valid:
            detail += f" Válidos: {', '.join(valid)}."
        super().__init__(detail)


class UnknownGroupError(PipelineError):
    """Se pidió un grupo de contenido que no existe en el catálogo."""

    def __init__(self, value: str, *, valid: tuple[str, ...] = ()) -> None:
        self.value = value
        self.valid = valid
        detail = f"Grupo desconocido: '{value}'."
        if valid:
            detail += f" Válidas: {', '.join(valid)}."
        super().__init__(detail)


class DocumentNotFoundError(PipelineError):
    """No existe un documento con ese identificador."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"Documento no encontrado: {document_id}")


class ArtifactNotFoundError(PipelineError):
    """El artefacto pedido no está en el storage."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"Artefacto no encontrado: {key}")


class StageNotReadyError(PipelineError):
    """Falta correr una etapa previa de la que esta depende.

    El mensaje nombra explícitamente qué hay que ejecutar antes, porque es la
    causa más común de confusión al encadenar endpoints a mano.
    """

    def __init__(self, stage: str, *, requires: str, hint: str = "") -> None:
        self.stage = stage
        self.requires = requires
        self.hint = hint
        detail = f"La etapa '{stage}' requiere que '{requires}' se haya ejecutado antes."
        if hint:
            detail += f" {hint}"
        super().__init__(detail)


class StageNotSupportedError(PipelineError):
    """El análisis no contempla esa etapa.

    Por ejemplo: el estado de cambios en el patrimonio no tiene control de
    calidad, y el dictamen del auditor no tiene extracción. No es un fallo, es una característica del catálogo.
    """

    def __init__(self, stage: str, analysis_type: str) -> None:
        self.stage = stage
        self.analysis_type = analysis_type
        super().__init__(f"El análisis '{analysis_type}' no soporta la etapa '{stage}'.")


class JobNotFoundError(PipelineError):
    """No existe un job con ese identificador."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Job no encontrado: {job_id}")


class JobCancelledError(PipelineError):
    """El job fue cancelado mientras se ejecutaba.

    La lanza el propio trabajo al detectar la señal de cancelación entre
    bloques, de modo que la cancelación sea cooperativa y deje el storage
    en un estado consistente.
    """


class InvalidDocumentError(PipelineError):
    """El archivo subido no es un PDF legible."""
