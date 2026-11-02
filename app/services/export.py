"""Conversión de resultados a Excel.

La regla que aplica: cada fila extraída se completa con las columnas de
identidad del documento (empresa, período, moneda...), pero **el dato de la
fila manda**. La identidad solo rellena lo que la fila dejó vacío: una nota
puede venir en otra moneda o en otra unidad que la portada, y en ese caso la
fila es más específica.
"""

from __future__ import annotations

import io
import json
import logging
import re
import zipfile

import pandas as pd

logger = logging.getLogger(__name__)

# Fecha fija dentro del Excel, para que el archivo dependa solo de sus datos.
# Cuándo se generó ya está en el job y en la fila de BigQuery.
_CORE_XML = "docProps/core.xml"
_SELLO = b"2000-01-01T00:00:00Z"
_FECHA_ZIP = (2000, 1, 1, 0, 0, 0)
_FECHA_XML = re.compile(
    rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*(</dcterms:(?:created|modified)>)"
)

IDENTITY_COLUMNS: tuple[str, ...] = (
    "empresa",
    "ruc",
    "tipo_reporte",
    "fecha_cierre",
    "moneda",
    "unidad",
)

_EMPTY_VALUES = (None, "", "null")


def rows_with_identity(rows: list[dict], identity: dict | None) -> list[dict]:
    """Añade las columnas de identidad a cada fila, sin pisar lo que ya trae."""
    if not identity:
        return [row for row in rows if isinstance(row, dict)]

    defaults = {column: identity.get(column, "") for column in IDENTITY_COLUMNS}
    merged_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        merged = {
            column: row[column] if row.get(column) not in _EMPTY_VALUES else default
            for column, default in defaults.items()
        }
        merged.update({key: value for key, value in row.items() if key not in merged})
        merged_rows.append(merged)
    return merged_rows


def to_excel_bytes(rows: list[dict], identity: dict | None = None) -> bytes | None:
    """Genera un `.xlsx` en memoria. Devuelve `None` si no hay filas.

    Las mismas filas producen **siempre los mismos bytes**. Por defecto no es
    así: openpyxl estampa la fecha de creación en `docProps/core.xml`, de modo
    que reprocesar un documento daría un archivo distinto con exactamente los
    mismos datos, y la copia del bucket no se podría comparar con la local. Un
    artefacto que cambia sin que cambien sus datos no se puede verificar.
    """
    prepared = rows_with_identity(rows, identity)
    if not prepared:
        return None
    try:
        buffer = io.BytesIO()
        pd.DataFrame(_flat_cells(prepared)).to_excel(buffer, index=False, engine="openpyxl")
        return _reproducible(buffer.getvalue())
    except Exception as exc:
        logger.warning("No se pudo generar el Excel: %s", exc)
        return None


def _flat_cells(rows: list[dict]) -> list[dict]:
    """Pasa a texto JSON los valores anidados, que Excel no sabe guardar.

    Una sola celda con un objeto haría fallar el libro entero, y con él la
    descarga.
    """
    return [
        {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        for row in rows
    ]


def _reproducible(libro: bytes) -> bytes:
    """Reescribe el `.xlsx` con las fechas fijas.

    No basta con poner las propiedades del libro: `openpyxl.save_workbook`
    sobrescribe `dcterms:modified` con la hora actual justo antes de guardar,
    y además cada entrada del ZIP lleva su propia marca de tiempo. Así que se
    normaliza el archivo ya hecho, que es el único punto donde se controla
    todo lo que acabará en el disco.
    """
    origen = zipfile.ZipFile(io.BytesIO(libro))
    destino = io.BytesIO()
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as salida:
        for info in origen.infolist():
            datos = origen.read(info.filename)
            if info.filename == _CORE_XML:
                datos = _FECHA_XML.sub(rb"\g<1>" + _SELLO + rb"\g<2>", datos)
            entrada = zipfile.ZipInfo(info.filename, date_time=_FECHA_ZIP)
            entrada.compress_type = info.compress_type
            entrada.external_attr = info.external_attr
            salida.writestr(entrada, datos)
    return destino.getvalue()
