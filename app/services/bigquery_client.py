"""Cliente de BigQuery para la tabla de trazabilidad de extracciones.

Una fila por (documento, análisis) **publicado**, insertada siempre con
`INSERT` — nunca `MERGE`. Reextraer un documento agrega una fila nueva en vez
de pisar la anterior: la pregunta que esta tabla responde es "¿cuándo se
extrajo esto y con qué resultado?", y eso exige el histórico, no solo el
último estado. `extracted_at` y `job_id` distinguen cada corrida.

Las filas extraídas viajan en `rows_json` como texto. Su forma la define cada
`.prompt` y varía por análisis (ver `app/domain/models.py`), así que darles
columnas fijas obligaría a migrar el esquema cada vez que se ajusta un prompt.
Con `JSON_VALUE`/`JSON_QUERY_ARRAY` se consultan igual desde SQL.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from app.config import PublishTarget

logger = logging.getLogger(__name__)

# (nombre, tipo, modo). Se declara aquí y no solo en la consola de GCP para
# que `_ensure_table()` pueda crear la tabla en cualquier proyecto.
TABLE_SCHEMA: tuple[tuple[str, str, str], ...] = (
    ("batch", "STRING", "NULLABLE"),
    ("document_id", "STRING", "REQUIRED"),
    ("filename", "STRING", "NULLABLE"),
    ("sha256", "STRING", "NULLABLE"),
    ("company", "STRING", "NULLABLE"),
    ("fiscal_period", "STRING", "NULLABLE"),
    ("currency", "STRING", "NULLABLE"),
    ("analysis_type", "STRING", "REQUIRED"),
    ("analysis_group", "STRING", "NULLABLE"),
    ("row_count", "INTEGER", "NULLABLE"),
    ("quality_status", "STRING", "NULLABLE"),
    ("job_id", "STRING", "NULLABLE"),
    ("extracted_at", "TIMESTAMP", "NULLABLE"),
    ("published_at", "TIMESTAMP", "REQUIRED"),
    ("pdf_uri", "STRING", "NULLABLE"),
    ("excel_uri", "STRING", "NULLABLE"),
    ("search_json_uri", "STRING", "NULLABLE"),
    ("quality_json_uri", "STRING", "NULLABLE"),
    ("rows_json", "STRING", "NULLABLE"),
)

_clients: dict[tuple[str | None, str | None], Any] = {}
_ensured: set[str] = set()
_lock = threading.Lock()


def table_id(target: PublishTarget, project: str | None = None) -> str:
    """Identificador completo `proyecto.dataset.tabla`."""
    parts = [target.dataset, target.table]
    if project:
        parts.insert(0, project)
    return ".".join(parts)


def insert_extraction_row(target: PublishTarget, row: dict) -> None:
    """Inserta la fila. Lanza `RuntimeError` si BigQuery la rechaza."""
    client = _client(target)
    full_id = table_id(target, client.project)
    _ensure_table(client, full_id)
    errors = client.insert_rows_json(full_id, [row])
    if errors:
        raise RuntimeError(f"BigQuery rechazó la fila en {full_id}: {errors}")
    logger.info(
        "Fila registrada en %s (%s / %s)", full_id, row.get("document_id"), row.get("analysis_type")
    )


def _ensure_table(client: Any, full_id: str) -> None:
    """Crea la tabla si falta y le añade las columnas que le falten.

    Lo segundo importa tanto como lo primero: una tabla creada con el esquema
    de ayer rechaza una fila con una columna nueva (`no such field`). Solo se
    **añaden** columnas nullable, que es lo único que BigQuery permite sin
    reescribir la tabla — nunca se borra ni se cambia el tipo de una existente.

    El dataset **no** se crea: su región y sus permisos son una decisión de
    infraestructura, y crearlo al vuelo en la región equivocada es un error
    caro de deshacer. Si no existe, el fallo dirá exactamente eso.
    """
    with _lock:
        if full_id in _ensured:
            return

    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound

    declarado = [bigquery.SchemaField(name, kind, mode=mode) for name, kind, mode in TABLE_SCHEMA]
    try:
        tabla = client.get_table(full_id)
    except NotFound:
        tabla = bigquery.Table(full_id, schema=declarado)
        tabla.time_partitioning = bigquery.TimePartitioning(field="published_at")
        client.create_table(tabla, exists_ok=True)
        logger.info("Tabla de BigQuery creada: %s", full_id)
    else:
        existentes = {campo.name for campo in tabla.schema}
        faltantes = [campo for campo in declarado if campo.name not in existentes]
        if faltantes:
            tabla.schema = list(tabla.schema) + [
                bigquery.SchemaField(campo.name, campo.field_type, mode="NULLABLE")
                for campo in faltantes
            ]
            client.update_table(tabla, ["schema"])
            logger.info(
                "Columnas añadidas a %s: %s",
                full_id,
                ", ".join(campo.name for campo in faltantes),
            )

    with _lock:
        _ensured.add(full_id)


def _client(target: PublishTarget) -> Any:
    from google.cloud import bigquery

    from app.services.gcp_auth import credentials_and_project

    with _lock:
        client = _clients.get(target.cache_key)
        if client is not None:
            return client

    credentials, project = credentials_and_project(target.credentials_path, target.project)
    client = bigquery.Client(project=project, credentials=credentials, location=target.location)
    with _lock:
        return _clients.setdefault(target.cache_key, client)


def reset_clients() -> None:
    """Olvida los clientes cacheados. Solo para tests y recarga de settings."""
    with _lock:
        _clients.clear()
        _ensured.clear()
