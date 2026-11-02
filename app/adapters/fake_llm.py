"""Proveedor de modelo determinista para el modo demo (`AI_PROVIDER=fake`).

Responde sin red ni credenciales, con la misma forma JSON que piden los
prompts reales. Sirve para dos cosas:

* que cualquiera pueda clonar el repositorio, levantar la API y recorrer el
  pipeline completo —clasificación, búsqueda, extracción, calidad, Excel y
  ZIP— sin pagar un token;
* que el flujo HTTP se pueda probar de punta a punta en CI.

Los datos son de una empresa ficticia («Minera Andina Demo S.A.A.») y están
cuadrados a propósito: las reglas contables de `app/domain/accounting.py`
los aprueban, así que el demo enseña un control de calidad en verde. No
dependen del PDF que se suba: el modo fake no lee el documento.
"""

from __future__ import annotations

import json
import re

from app.adapters.llm import LLMResponse, Request

_PAGE_LABEL = re.compile(r"IMAGEN PÁGINA (\d+) START")

_GROUP_PROMPT = "grupo"
_GROUPS = ("Estados", "Notas")
_CODES = {
    "estados": ("SitFin", "Result", "Flujo", "Patrim"),
    "notas": ("PPE", "Invent", "Ventas"),
}

IDENTITY = {
    "empresa": "Minera Andina Demo S.A.A.",
    "ruc": "20000000001",
    "tipo_reporte": "Estados financieros separados auditados",
    "periodo": "2024",
    "fecha_cierre": "2024-12-31",
    "moneda": "USD",
    "unidad": "miles",
    "auditor": "Auditores Demo S.Civil de R.L.",
}


class FakeProvider:
    """Elige la respuesta por etapa (`task`) y por prompt (`prompt_name`)."""

    def generate(self, request: Request) -> LLMResponse:
        payload = _ANSWERS[request.task](request)
        return LLMResponse(text=json.dumps(payload, ensure_ascii=False), finish_reason="STOP")


# --------------------------------------------------------------------------
# Respuestas por etapa
# --------------------------------------------------------------------------


def _pages(request: Request) -> list[int]:
    """Páginas del bloque, leídas de las etiquetas que arma el adaptador."""
    return [
        int(match.group(1))
        for part in request.parts
        if part.text
        for match in [_PAGE_LABEL.search(part.text)]
        if match
    ]


def _classify(request: Request) -> dict:
    pages = _pages(request)
    if request.prompt_name == _GROUP_PROMPT:
        return {
            "clasificaciones": [
                {
                    "page_number": page,
                    "grupo": _GROUPS[(page - 1) % len(_GROUPS)],
                    "confianza": "ALTA",
                    "evidencia": "demo",
                }
                for page in pages
            ]
        }
    codes = _CODES.get(request.prompt_name, ())
    return {
        "clasificaciones": [
            {
                "page_number": page,
                "codigo": codes[index % len(codes)],
                "confianza": "ALTA",
                "evidencia": "demo",
            }
            for index, page in enumerate(pages)
            if codes
        ]
    }


def _search(request: Request) -> dict:
    pages = _pages(request)[:2]
    return {
        "relevant_sections": [
            {
                "page_number": page,
                "section_title": request.prompt_name,
                "confidence_score": 0.9,
                "reason": "demo",
            }
            for page in pages
        ],
        "alerts": [],
    }


def _identity(request: Request) -> dict:
    return dict(IDENTITY)


def _extract(request: Request) -> dict:
    return {"filas": [dict(row) for row in FIXTURES.get(request.prompt_name, ())]}


def _quality(request: Request) -> dict:
    return {"estado": "Aprobado", "observaciones": []}


_ANSWERS = {
    "classify": _classify,
    "search": _search,
    "identity": _identity,
    "extract": _extract,
    "quality": _quality,
}


# --------------------------------------------------------------------------
# Datos de demo (miles de USD, ejercicios 2024 y 2023)
# --------------------------------------------------------------------------


def _rows(filas: list[tuple]) -> list[dict]:
    """Expande `(sección, concepto, código, nota, tipo, v2024, v2023)` a formato largo."""
    rows = []
    for seccion, concepto, codigo, nota, tipo, actual, anterior in filas:
        for periodo, valor in (("2024", actual), ("2023", anterior)):
            rows.append(
                {
                    "seccion": seccion,
                    "concepto": concepto,
                    "codigo": codigo,
                    "nota": nota,
                    "periodo": periodo,
                    "valor": valor,
                    "tipo_fila": tipo,
                }
            )
    return rows


FIXTURES: dict[str, list[dict]] = {
    "situacion_financiera": _rows(
        [
            (
                "Activo corriente",
                "Efectivo y equivalentes de efectivo",
                None,
                "5",
                "partida",
                420_310,
                389_120,
            ),
            (
                "Activo corriente",
                "Cuentas por cobrar comerciales",
                None,
                "6",
                "partida",
                215_880,
                198_450,
            ),
            ("Activo corriente", "Inventarios", None, "7", "partida", 180_640, 172_300),
            (
                "Activo corriente",
                "Total activo corriente",
                "TOTAL_ACTIVO_CORRIENTE",
                None,
                "subtotal",
                816_830,
                759_870,
            ),
            (
                "Activo no corriente",
                "Propiedades, planta y equipo",
                None,
                "8",
                "partida",
                2_450_900,
                2_380_410,
            ),
            ("Activo no corriente", "Concesiones mineras", None, "9", "partida", 310_200, 315_600),
            (
                "Activo no corriente",
                "Total activo no corriente",
                "TOTAL_ACTIVO_NO_CORRIENTE",
                None,
                "subtotal",
                2_761_100,
                2_696_010,
            ),
            ("Activo", "Total activo", "TOTAL_ACTIVO", None, "total", 3_577_930, 3_455_880),
            ("Pasivo", "Cuentas por pagar comerciales", None, "10", "partida", 260_450, 240_300),
            ("Pasivo", "Obligaciones financieras", None, "11", "partida", 890_000, 920_000),
            ("Pasivo", "Provisión para cierre de minas", None, "12", "partida", 185_480, 176_580),
            ("Pasivo", "Total pasivo", "TOTAL_PASIVO", None, "total", 1_335_930, 1_336_880),
            ("Patrimonio", "Capital social", None, "13", "partida", 1_200_000, 1_200_000),
            ("Patrimonio", "Resultados acumulados", None, None, "partida", 1_042_000, 919_000),
            (
                "Patrimonio",
                "Total patrimonio",
                "TOTAL_PATRIMONIO",
                None,
                "total",
                2_242_000,
                2_119_000,
            ),
            (
                "Pasivo y patrimonio",
                "Total pasivo y patrimonio",
                "TOTAL_PASIVO_PATRIMONIO",
                None,
                "total",
                3_577_930,
                3_455_880,
            ),
        ]
    ),
    "resultados": _rows(
        [
            ("Resultados", "Ventas netas", "INGRESOS", "14", "partida", 1_850_400, 1_620_300),
            (
                "Resultados",
                "Costo de ventas",
                "COSTO_VENTAS",
                "15",
                "partida",
                -1_120_250,
                -1_010_800,
            ),
            ("Resultados", "Utilidad bruta", "UTILIDAD_BRUTA", None, "subtotal", 730_150, 609_500),
            ("Resultados", "Gastos de administración", None, None, "partida", -95_300, -88_200),
            ("Resultados", "Gastos de ventas", None, None, "partida", -40_150, -36_900),
            ("Resultados", "Gastos financieros", None, None, "partida", -62_700, -65_400),
            (
                "Resultados",
                "Utilidad antes de impuesto a la renta",
                "UTILIDAD_ANTES_IMPUESTOS",
                None,
                "subtotal",
                532_000,
                419_000,
            ),
            (
                "Resultados",
                "Impuesto a la renta",
                "IMPUESTO_RENTA",
                "16",
                "partida",
                -159_600,
                -125_700,
            ),
            (
                "Resultados",
                "Utilidad neta del ejercicio",
                "UTILIDAD_NETA",
                None,
                "total",
                372_400,
                293_300,
            ),
        ]
    ),
    "flujo_efectivo": _rows(
        [
            (
                "Operación",
                "Efectivo neto de actividades de operación",
                "FLUJO_OPERACION",
                None,
                "subtotal",
                610_500,
                540_200,
            ),
            (
                "Inversión",
                "Efectivo neto de actividades de inversión",
                "FLUJO_INVERSION",
                None,
                "subtotal",
                -420_700,
                -380_900,
            ),
            (
                "Financiamiento",
                "Efectivo neto de actividades de financiamiento",
                "FLUJO_FINANCIAMIENTO",
                None,
                "subtotal",
                -160_000,
                -145_000,
            ),
            (
                "Variación",
                "Aumento neto de efectivo",
                "AUMENTO_NETO_EFECTIVO",
                None,
                "subtotal",
                29_800,
                14_300,
            ),
            (
                "Variación",
                "Efectivo al inicio del ejercicio",
                "EFECTIVO_INICIAL",
                None,
                "partida",
                389_120,
                375_400,
            ),
            (
                "Variación",
                "Efecto de la variación del tipo de cambio",
                "EFECTO_TIPO_CAMBIO",
                None,
                "partida",
                1_390,
                -580,
            ),
            (
                "Variación",
                "Efectivo al final del ejercicio",
                "EFECTIVO_FINAL",
                None,
                "total",
                420_310,
                389_120,
            ),
        ]
    ),
    "cambios_patrimonio": _rows(
        [
            ("Resultados acumulados", "Saldo inicial", None, None, "partida", 919_000, 845_700),
            (
                "Resultados acumulados",
                "Utilidad neta del ejercicio",
                None,
                None,
                "partida",
                372_400,
                293_300,
            ),
            (
                "Resultados acumulados",
                "Dividendos declarados",
                None,
                "13",
                "partida",
                -249_400,
                -220_000,
            ),
            ("Resultados acumulados", "Saldo final", None, None, "total", 1_042_000, 919_000),
        ]
    ),
    "propiedades_planta_equipo": _rows(
        [
            ("Costo neto", "Terrenos", None, "8", "partida", 45_000, 45_000),
            ("Costo neto", "Edificios e instalaciones", None, "8", "partida", 380_500, 392_100),
            ("Costo neto", "Maquinaria y equipo", None, "8", "partida", 1_240_000, 1_198_310),
            ("Costo neto", "Costos de desarrollo de minas", None, "8", "partida", 610_400, 590_000),
            ("Costo neto", "Obras en curso", None, "8", "partida", 175_000, 155_000),
            (
                "Costo neto",
                "Total propiedades, planta y equipo",
                None,
                "8",
                "total",
                2_450_900,
                2_380_410,
            ),
        ]
    ),
    "inventarios": _rows(
        [
            ("Inventarios", "Concentrados de mineral", None, "7", "partida", 95_300, 90_100),
            ("Inventarios", "Mineral en cancha", None, "7", "partida", 45_200, 42_000),
            ("Inventarios", "Suministros y repuestos", None, "7", "partida", 40_140, 40_200),
            ("Inventarios", "Total inventarios", None, "7", "total", 180_640, 172_300),
        ]
    ),
    "ventas_por_producto": _rows(
        [
            ("Ventas por metal", "Cobre", None, "14", "partida", 1_020_000, 905_000),
            ("Ventas por metal", "Oro", None, "14", "partida", 540_400, 450_300),
            ("Ventas por metal", "Plata", None, "14", "partida", 190_000, 170_000),
            ("Ventas por metal", "Zinc", None, "14", "partida", 100_000, 95_000),
            ("Ventas por metal", "Total ventas netas", None, "14", "total", 1_850_400, 1_620_300),
        ]
    ),
}
