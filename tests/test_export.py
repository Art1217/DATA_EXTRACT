"""Exportación: la fila manda sobre la identidad del documento."""

from __future__ import annotations

import io
import time

import pandas as pd

from app.services import export

IDENTIDAD = {"empresa": "Minera Demo", "moneda": "USD", "unidad": "miles"}


def test_la_identidad_rellena_solo_lo_que_la_fila_deja_vacio():
    """Una nota puede venir en otra moneda: la fila es más específica."""
    filas = [
        {"moneda": "PEN", "valor": 1.2},
        {"moneda": None, "valor": 2.0},
        {"moneda": "", "valor": 3.0},
    ]

    resultado = export.rows_with_identity(filas, IDENTIDAD)

    assert [fila["moneda"] for fila in resultado] == ["PEN", "USD", "USD"]
    assert all(fila["empresa"] == "Minera Demo" for fila in resultado)


def test_la_cadena_null_se_trata_como_vacio():
    """El modelo devuelve a veces la cadena "null" en lugar de un nulo real."""
    resultado = export.rows_with_identity([{"unidad": "null"}], IDENTIDAD)
    assert resultado[0]["unidad"] == "miles"


def test_las_columnas_propias_de_la_fila_se_conservan():
    resultado = export.rows_with_identity([{"concepto": "Inventarios", "valor": 435}], IDENTIDAD)
    assert resultado[0]["concepto"] == "Inventarios"
    assert resultado[0]["valor"] == 435


def test_sin_identidad_las_filas_pasan_intactas():
    filas = [{"valor": 1.2}]
    assert export.rows_with_identity(filas, None) == filas


def test_el_excel_conserva_las_filas_y_las_columnas_de_identidad():
    workbook = export.to_excel_bytes([{"valor": 1.2}, {"valor": 2.4}], IDENTIDAD)
    assert workbook is not None

    frame = pd.read_excel(io.BytesIO(workbook))
    assert len(frame) == 2
    assert frame["empresa"].tolist() == ["Minera Demo", "Minera Demo"]
    assert "valor" in frame.columns


def test_los_valores_anidados_no_rompen_el_excel():
    workbook = export.to_excel_bytes([{"valor": 1, "detalle": {"pagina": 3}}])
    frame = pd.read_excel(io.BytesIO(workbook))
    assert frame["detalle"].tolist() == ['{"pagina": 3}']


def test_sin_filas_no_se_genera_excel():
    assert export.to_excel_bytes([], IDENTIDAD) is None


def test_las_mismas_filas_dan_siempre_los_mismos_bytes():
    """Sin esto no se puede comparar la copia del bucket con la local.

    openpyxl estampa la hora de guardado en `docProps/core.xml`, así que
    reprocesar un documento produciría un archivo distinto con los mismos
    datos.
    """
    filas = [{"valor": 1.2}, {"valor": 2.4}]

    primero = export.to_excel_bytes(filas, IDENTIDAD)
    time.sleep(1.1)  # el sello de openpyxl tiene resolución de segundos
    segundo = export.to_excel_bytes(filas, IDENTIDAD)

    assert primero == segundo


def test_datos_distintos_siguen_dando_archivos_distintos():
    """Fijar la fecha no puede volver el archivo ciego a sus propios datos."""
    uno = export.to_excel_bytes([{"valor": 1.2}], IDENTIDAD)
    otro = export.to_excel_bytes([{"valor": 9.9}], IDENTIDAD)

    assert uno != otro
