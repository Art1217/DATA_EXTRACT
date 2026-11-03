"""Configuración central de la aplicación.

Única fuente de verdad para rutas, credenciales, modelos y parámetros de
procesamiento. Todo valor es override-able por variable de entorno o `.env`.

Importar este módulo NO tiene efectos secundarios: no crea directorios ni
exporta variables de entorno. Eso se hace de forma explícita llamando a
`ensure_directories()` y `configure_observability()`, para que el import no
falle en un filesystem de solo lectura.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/config/settings.py -> app/config -> app -> raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AIProvider(StrEnum):
    """De dónde salen las respuestas del modelo."""

    VERTEX = "vertex"
    """Gemini en Vertex AI, con ADC o service account. La vía de producción."""
    GEMINI_API = "gemini_api"
    """Gemini con una API key de Google AI Studio, sin proyecto de GCP."""
    CLAUDE_VERTEX = "claude_vertex"
    """Claude en Vertex AI Model Garden, con las mismas credenciales de GCP."""
    ANTHROPIC = "anthropic"
    """Claude con la API directa de Anthropic (`ANTHROPIC_API_KEY`)."""
    FAKE = "fake"
    """Respuestas deterministas sin red ni credenciales: modo demo y tests."""


class StorageBackend(StrEnum):
    LOCAL = "local"
    GCS = "gcs"


@dataclass(frozen=True, slots=True)
class PublishTarget:
    """A dónde se publican los entregables: bucket, dataset y credencial.

    Viaja como un solo objeto inmutable hasta los clientes de GCS y BigQuery,
    que se cachean por destino.
    """

    bucket: str | None
    prefix: str
    project: str | None
    """`None` significa "el del JSON de credenciales o el de las ADC"."""
    credentials_path: Path | None
    dataset: str
    table: str
    location: str
    enabled: bool

    @property
    def ready(self) -> bool:
        """Publicar exige estar activado y tener bucket al que subir."""
        return self.enabled and bool(self.bucket)

    @property
    def cache_key(self) -> tuple[str | None, str | None]:
        return (self.project, str(self.credentials_path) if self.credentials_path else None)

    def describe(self) -> dict:
        """Vista sin secretos, para `GET /v1/config`."""
        return {
            "bucket": self.bucket,
            "project": self.project,
            "bigquery_table": f"{self.dataset}.{self.table}",
            "credentials": self.credentials_path.name if self.credentials_path else "adc",
            "ready": self.ready,
        }


class Settings(BaseSettings):
    """Parámetros efectivos del backend.

    Los nombres de campo se mapean a variables de entorno en mayúsculas
    (`dpi` -> `DPI`, `job_workers` -> `JOB_WORKERS`), sin prefijo.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Rutas ------------------------------------------------------------
    base_dir: Path = Field(
        default=PROJECT_ROOT,
        description="Raíz del proyecto: de aquí cuelga el árbol `prompts/`.",
    )
    storage_dir: Path | None = Field(
        default=None,
        description="Raíz de artefactos en modo local. Por defecto <base_dir>/Storage.",
    )
    temp_dir: Path | None = Field(
        default=None,
        description="Raíz de workspaces temporales. Por defecto <base_dir>/Temp.",
    )

    # --- Proveedor de IA --------------------------------------------------
    ai_provider: AIProvider = Field(
        default=AIProvider.FAKE,
        description=(
            "vertex | gemini_api | claude_vertex | anthropic | fake. Por defecto `fake` "
            "para que corra sin credenciales."
        ),
    )
    ai_provider_extract: AIProvider | None = Field(
        default=None,
        description=(
            "Proveedor solo para extracción y calidad. Permite, por ejemplo, buscar con "
            "Gemini Flash y extraer con Claude. Si es None, se usa AI_PROVIDER en todo."
        ),
    )
    gemini_api_key: SecretStr | None = Field(
        default=None, description="API key de Google AI Studio (solo con `gemini_api`)."
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="API key de Anthropic (solo con `anthropic`). Sin ella, el SDK busca la suya.",
    )
    claude_effort_classify: str | None = Field(
        default="medium",
        description=(
            "Esfuerzo de Claude (low | medium | high | xhigh | max) en clasificación, "
            "identidad y búsqueda. Vacío: el valor por defecto del modelo."
        ),
    )
    claude_effort_extract: str | None = Field(
        default="high",
        description="Esfuerzo de Claude en extracción y calidad, donde cada cifra cuenta.",
    )
    claude_fallback_model: str | None = Field(
        default=None,
        description=(
            "Modelo Claude que reintenta si el principal declina por política. Debe "
            "estar habilitado en Model Garden. Vacío: sin reintento."
        ),
    )

    # --- Google Cloud -----------------------------------------------------
    gcp_project_id: str | None = Field(
        default=None,
        description="Proyecto de GCP. Si es None se deduce de las credenciales o de las ADC.",
    )
    gcp_location: str = Field(default="global", description="Región de Vertex AI.")
    credentials_path: Path | None = Field(
        default=None,
        description=(
            "JSON de service account. Si es None se usan las Application Default "
            "Credentials (`gcloud auth application-default login`), lo correcto en Cloud Run."
        ),
    )

    # --- Modelos ----------------------------------------------------------
    model_name_classify: str = Field(
        default="gemini-3-flash-preview",
        description="Modelo para clasificación, búsqueda e identidad (barato).",
    )
    model_name_extract: str = Field(
        default="gemini-3-pro-preview",
        description="Modelo para extracción y control de calidad (caro).",
    )

    # --- Document AI ------------------------------------------------------
    docai_processor_id: str | None = Field(
        default=None,
        description="Procesador OCR de Document AI. Sin él, los escaneados van directo a Gemini.",
    )
    docai_location: str = "us"

    # --- Parámetros de procesamiento --------------------------------------
    dpi: int = Field(default=300, ge=72, le=600, description="DPI de render para búsqueda.")
    clasif_dpi: int = Field(
        default=150,
        ge=72,
        le=600,
        description="DPI de render para clasificación: solo hay que reconocer el tipo de página.",
    )
    block_size: int = Field(default=30, ge=1, description="Páginas por bloque enviado al modelo.")
    busqueda_workers: int = Field(
        default=3, ge=1, description="Bloques en paralelo hacia la API dentro de un mismo job."
    )
    identity_pages: int = Field(
        default=10, ge=1, description="Primeras N páginas usadas para extraer la identidad."
    )

    chunk_size_digital: int = Field(default=25, ge=1)
    chunk_size_scanned: int = Field(default=25, ge=1)
    chunk_size_docai_json: int = Field(
        default=6,
        ge=1,
        description=(
            "Chunk cuando se prefiere el JSON de Document AI: cada página manda su "
            "JSON de tablas más una imagen de contexto, así que 25 satura la entrada."
        ),
    )
    scanned_img_dpi: int = Field(default=120, ge=72)
    chunk_overlap_pages: int = Field(
        default=1, ge=0, description="Páginas repetidas entre chunks para no cortar tablas."
    )
    chunk_anchor_rows: int = Field(
        default=3, ge=0, description="Últimas filas del chunk previo reenviadas como continuidad."
    )

    # --- Control de calidad -----------------------------------------------
    quality_tolerance: float = Field(
        default=1.0,
        ge=0,
        description=(
            "Diferencia absoluta admitida en las reglas contables: los estados vienen "
            "redondeados a miles y los totales suelen descuadrar en una unidad."
        ),
    )

    # --- Ejecución de jobs ------------------------------------------------
    job_workers: int = Field(
        default=2, ge=1, description="Documentos procesados en paralelo por el runner."
    )
    max_concurrent_llm_calls: int = Field(
        default=6,
        ge=1,
        description=(
            "Techo global de llamadas simultáneas al modelo. Necesario porque el "
            "paralelismo real es job_workers x busqueda_workers."
        ),
    )
    job_retention_days: int = Field(
        default=30, ge=1, description="Días que se conserva el registro de un job."
    )

    # --- Storage ----------------------------------------------------------
    storage_backend: StorageBackend = Field(
        default=StorageBackend.LOCAL,
        description="local: disco. gcs: los artefactos viven en `GCS_BUCKET`.",
    )
    gcs_bucket: str | None = Field(
        default=None, description="Bucket de artefactos (storage gcs) y de entregables."
    )
    gcs_prefix: str = Field(
        default="",
        description="Prefijo opcional dentro del bucket, por si se comparte con otro sistema.",
    )

    # --- Publicación de entregables (bucket + BigQuery) -------------------
    # Apagado por defecto: el backend funciona igual sin bucket ni dataset.
    publish_enabled: bool = Field(
        default=False,
        description="Sube los entregables al bucket y registra la fila en BigQuery.",
    )
    bigquery_dataset: str = "document_extraction"
    bigquery_table: str = "extractions"
    bigquery_location: str = Field(
        default="US", description="Región del dataset; debe coincidir con la ya creada."
    )

    # --- Observabilidad ---------------------------------------------------
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "document-extraction-ai"
    log_level: str = "INFO"

    # --- API --------------------------------------------------------------
    api_key: SecretStr | None = Field(
        default=None,
        description="Si se define, todo endpoint /v1 exige el header X-API-Key.",
    )
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_upload_mb: int = Field(default=200, ge=1)

    @field_validator(
        "ai_provider_extract",
        "credentials_path",
        "anthropic_api_key",
        "claude_effort_classify",
        "claude_effort_extract",
        "claude_fallback_model",
        mode="before",
    )
    @classmethod
    def _empty_as_none(cls, value):
        """`VARIABLE=` en el `.env` significa "sin definir", no un valor vacío."""
        return None if value == "" else value

    @model_validator(mode="after")
    def _resolve_default_paths(self) -> Settings:
        """Deriva las rutas que dependen de `base_dir` y no fueron fijadas."""
        if self.storage_dir is None:
            self.storage_dir = self.base_dir / "Storage"
        if self.temp_dir is None:
            self.temp_dir = self.base_dir / "Temp"
        if self.credentials_path is not None and not self.credentials_path.is_absolute():
            # `CREDENTIALS_PATH=credentials/sa.json` tiene que funcionar igual
            # desde cualquier directorio de trabajo.
            self.credentials_path = self.base_dir / self.credentials_path
        return self

    def publish_target(self) -> PublishTarget:
        """Destino de publicación: bucket y dataset del proyecto configurado."""
        return PublishTarget(
            bucket=self.gcs_bucket,
            prefix=self.gcs_prefix,
            project=self.gcp_project_id,
            credentials_path=self.credentials_path,
            dataset=self.bigquery_dataset,
            table=self.bigquery_table,
            location=self.bigquery_location,
            enabled=self.publish_enabled,
        )

    # --- Derivados --------------------------------------------------------
    @property
    def jobs_dir(self) -> Path:
        return self.storage_dir / "jobs"

    @property
    def documents_dir(self) -> Path:
        return self.storage_dir / "documents"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def prompt_path(self, relative: str) -> Path:
        """Resuelve la ruta de un prompt declarado en el catálogo.

        Los prompts se declaran con rutas relativas en formato POSIX
        (`prompts/search/situacion_financiera.prompt`) para que el catálogo
        sea legible y portable entre sistemas de archivos.
        """
        return self.base_dir.joinpath(*relative.split("/"))

    def ensure_directories(self) -> None:
        """Crea los directorios de trabajo. Explícito, nunca en el import."""
        for directory in (self.storage_dir, self.documents_dir, self.jobs_dir, self.temp_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def configure_observability(self) -> None:
        """Activa el trazado de LangSmith si hay API key configurada."""
        if not self.langsmith_api_key:
            return
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = self.langsmith_project


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instancia única de configuración (cacheada por proceso).

    Usar siempre esta función en vez de construir `Settings()` directamente,
    para que los tests puedan sustituirla con `get_settings.cache_clear()`.
    """
    settings = Settings()
    settings.configure_observability()
    return settings
