"""Reglas contables deterministas para el control de calidad.

Un modelo puede leer mal una cifra y aun así devolver un JSON impecable. Las
identidades contables no admiten esa duda: si el activo no es igual al pasivo
más el patrimonio, algo se extrajo mal, lo diga o no el modelo. Estas reglas
corren antes que la revisión con LLM, cuestan cero y no alucinan.

Trabajan sobre las filas en formato largo que devuelven los prompts de
extracción (una fila por concepto y período):

    {"concepto": "Total activo", "codigo": "TOTAL_ACTIVO",
     "periodo": "2024", "valor": 15234.0, ...}

Solo miran las filas con `codigo`: el prompt marca con un código canónico los
totales y subtotales que intervienen en alguna regla, y deja `null` el resto.
Una regla cuyos términos no aparecen se **omite**, no falla: un estado
resumido sin subtotal de activo corriente sigue siendo válido.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class Rule:
    """`total = suma(parts)`, período a período."""

    name: str
    description: str
    total: str
    parts: tuple[str, ...]
    optional: frozenset[str] = field(default_factory=frozenset)
    """Términos que valen cero si no aparecen (p. ej. el efecto del tipo de
    cambio en el flujo de efectivo, que muchas empresas no reportan)."""


@dataclass(frozen=True, slots=True)
class CheckResult:
    rule: str
    description: str
    period: str
    status: str
    """`ok`, `error` u `omitida`."""
    expected: float | None = None
    actual: float | None = None
    difference: float | None = None
    detail: str | None = None

    @property
    def failed(self) -> bool:
        return self.status == "error"

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "description": self.description,
            "period": self.period,
            "status": self.status,
            "expected": self.expected,
            "actual": self.actual,
            "difference": self.difference,
            "detail": self.detail,
        }


RULES: Mapping[str, tuple[Rule, ...]] = MappingProxyType(
    {
        "situacion_financiera": (
            Rule(
                name="ecuacion_contable",
                description="Total activo = total pasivo + total patrimonio",
                total="TOTAL_ACTIVO",
                parts=("TOTAL_PASIVO", "TOTAL_PATRIMONIO"),
            ),
            Rule(
                name="total_pasivo_y_patrimonio",
                description="Total activo = total pasivo y patrimonio",
                total="TOTAL_ACTIVO",
                parts=("TOTAL_PASIVO_PATRIMONIO",),
            ),
            Rule(
                name="activo_corriente_mas_no_corriente",
                description="Total activo = activo corriente + activo no corriente",
                total="TOTAL_ACTIVO",
                parts=("TOTAL_ACTIVO_CORRIENTE", "TOTAL_ACTIVO_NO_CORRIENTE"),
            ),
        ),
        "resultados": (
            Rule(
                name="utilidad_bruta",
                description="Utilidad bruta = ingresos + costo de ventas (negativo)",
                total="UTILIDAD_BRUTA",
                parts=("INGRESOS", "COSTO_VENTAS"),
            ),
            Rule(
                name="utilidad_neta",
                description="Utilidad neta = utilidad antes de impuestos + impuesto a la renta",
                total="UTILIDAD_NETA",
                parts=("UTILIDAD_ANTES_IMPUESTOS", "IMPUESTO_RENTA"),
            ),
        ),
        "flujo_efectivo": (
            Rule(
                name="variacion_neta",
                description="Variación neta = flujo de operación + inversión + financiamiento",
                total="AUMENTO_NETO_EFECTIVO",
                parts=("FLUJO_OPERACION", "FLUJO_INVERSION", "FLUJO_FINANCIAMIENTO"),
            ),
            Rule(
                name="saldo_final",
                description="Efectivo final = efectivo inicial + variación neta + efecto cambiario",
                total="EFECTIVO_FINAL",
                parts=("EFECTIVO_INICIAL", "AUMENTO_NETO_EFECTIVO", "EFECTO_TIPO_CAMBIO"),
                optional=frozenset({"EFECTO_TIPO_CAMBIO"}),
            ),
        ),
    }
)


def has_rules(analysis_key: str) -> bool:
    return analysis_key in RULES


def run_checks(
    analysis_key: str, rows: Iterable[dict], *, tolerance: float = 1.0
) -> list[CheckResult]:
    """Aplica las reglas del análisis a cada período presente en las filas."""
    rules = RULES.get(analysis_key, ())
    if not rules:
        return []

    by_period = _index_by_period(rows)
    results: list[CheckResult] = []
    for period in sorted(by_period):
        values = by_period[period]
        for rule in rules:
            results.append(_apply(rule, period, values, tolerance))
    return results


# --------------------------------------------------------------------------
# Internos
# --------------------------------------------------------------------------


def _apply(rule: Rule, period: str, values: dict[str, float], tolerance: float) -> CheckResult:
    needed = (rule.total, *(p for p in rule.parts if p not in rule.optional))
    missing = [code for code in needed if code not in values]
    if missing:
        return CheckResult(
            rule=rule.name,
            description=rule.description,
            period=period,
            status="omitida",
            detail=f"Faltan: {', '.join(missing)}",
        )

    expected = sum(values.get(code, 0.0) for code in rule.parts)
    actual = values[rule.total]
    difference = round(actual - expected, 2)
    passed = abs(difference) <= tolerance
    return CheckResult(
        rule=rule.name,
        description=rule.description,
        period=period,
        status="ok" if passed else "error",
        expected=round(expected, 2),
        actual=round(actual, 2),
        difference=difference,
        detail=None if passed else f"Descuadre de {difference:,.2f} en {period}",
    )


def _index_by_period(rows: Iterable[dict]) -> dict[str, dict[str, float]]:
    """`{periodo: {codigo: valor}}` con las filas que traen código y número.

    Si un código aparece dos veces en el mismo período (una tabla partida
    entre dos páginas que el modelo repitió), manda la primera aparición.
    """
    index: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = row.get("codigo")
        value = parse_amount(row.get("valor"))
        if not code or value is None:
            continue
        period = str(row.get("periodo") or "sin_periodo")
        index[period].setdefault(str(code).upper(), value)
    return index


_NUMBER = re.compile(r"[^0-9.\-]")


def parse_amount(value) -> float | None:
    """Convierte una cifra contable a número.

    Acepta lo que a veces devuelve el modelo aunque el prompt pida números:
    `"1,234.5"`, `"(1,234)"` para negativos y `"-"` para cero.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None

    text = str(value).strip()
    if text in {"-", "—", "–"}:
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    cleaned = _NUMBER.sub("", text)
    if cleaned in {"", "-", "."}:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return -abs(number) if negative else number
