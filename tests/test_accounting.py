"""Reglas contables deterministas del control de calidad."""

from __future__ import annotations

import pytest

from app.adapters.fake_llm import FIXTURES
from app.domain.accounting import RULES, has_rules, parse_amount, run_checks


def fila(codigo, valor, periodo="2024"):
    return {"codigo": codigo, "valor": valor, "periodo": periodo}


@pytest.mark.parametrize("analisis", sorted(RULES))
def test_los_datos_de_demo_cuadran(analisis):
    """El modo demo enseña un control de calidad en verde, no por casualidad."""
    resultados = run_checks(analisis, FIXTURES[analisis])
    assert resultados, "cada regla debió evaluarse"
    assert {r.status for r in resultados} == {"ok"}


def test_un_balance_descuadrado_nombra_la_diferencia():
    resultados = run_checks(
        "situacion_financiera",
        [fila("TOTAL_ACTIVO", 1000), fila("TOTAL_PASIVO", 300), fila("TOTAL_PATRIMONIO", 650)],
    )
    ecuacion = next(r for r in resultados if r.rule == "ecuacion_contable")
    assert ecuacion.failed
    assert ecuacion.expected == 950 and ecuacion.actual == 1000
    assert ecuacion.difference == 50
    assert ecuacion.detail == "Descuadre de 50.00 en 2024"


def test_el_redondeo_a_miles_cabe_en_la_tolerancia():
    """Los estados vienen redondeados: un total suele diferir en una unidad."""
    filas = [fila("TOTAL_ACTIVO", 1001), fila("TOTAL_PASIVO", 400), fila("TOTAL_PATRIMONIO", 600)]
    assert not any(r.failed for r in run_checks("situacion_financiera", filas, tolerance=1))
    assert any(r.failed for r in run_checks("situacion_financiera", filas, tolerance=0))


def test_una_regla_sin_sus_terminos_se_omite_no_falla():
    """Un estado resumido sin subtotal de activo corriente sigue siendo válido."""
    resultados = run_checks("situacion_financiera", [fila("TOTAL_ACTIVO", 10)])
    assert {r.status for r in resultados} == {"omitida"}
    assert "TOTAL_PASIVO" in resultados[0].detail


def test_cada_periodo_se_valida_por_separado():
    filas = [
        fila("TOTAL_ACTIVO", 100, "2024"),
        fila("TOTAL_PASIVO_PATRIMONIO", 100, "2024"),
        fila("TOTAL_ACTIVO", 90, "2023"),
        fila("TOTAL_PASIVO_PATRIMONIO", 80, "2023"),
    ]
    resultados = [
        r
        for r in run_checks("situacion_financiera", filas)
        if r.rule == "total_pasivo_y_patrimonio"
    ]
    assert {r.period: r.status for r in resultados} == {"2023": "error", "2024": "ok"}


def test_el_efecto_cambiario_es_opcional():
    filas = [
        fila("EFECTIVO_INICIAL", 100),
        fila("AUMENTO_NETO_EFECTIVO", 20),
        fila("EFECTIVO_FINAL", 120),
    ]
    saldo = next(r for r in run_checks("flujo_efectivo", filas) if r.rule == "saldo_final")
    assert saldo.status == "ok"


def test_si_un_codigo_se_repite_manda_la_primera_aparicion():
    """Una tabla partida entre dos páginas que el modelo repitió."""
    filas = [
        fila("TOTAL_ACTIVO", 100),
        fila("TOTAL_ACTIVO", 999),
        fila("TOTAL_PASIVO_PATRIMONIO", 100),
    ]
    resultado = next(
        r
        for r in run_checks("situacion_financiera", filas)
        if r.rule == "total_pasivo_y_patrimonio"
    )
    assert resultado.status == "ok"


def test_las_filas_sin_codigo_o_sin_numero_se_ignoran():
    filas = [
        {"concepto": "Inventarios", "valor": 50, "periodo": "2024"},
        fila("TOTAL_ACTIVO", "ilegible"),
        "no es un dict",
    ]
    assert run_checks("situacion_financiera", filas) == [], "sin datos no hay período que validar"


def test_un_analisis_sin_reglas_no_produce_resultados():
    assert not has_rules("inventarios")
    assert run_checks("inventarios", [fila("TOTAL_ACTIVO", 1)]) == []


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        (1234.5, 1234.5),
        ("1,234.5", 1234.5),
        ("(1,234)", -1234.0),
        ("-", 0.0),
        ("US$ 2,000", 2000.0),
        ("", None),
        ("n/d", None),
        (None, None),
        (True, None),
        (float("nan"), None),
    ],
)
def test_parse_amount_entiende_el_formato_contable(texto, esperado):
    assert parse_amount(texto) == esperado
