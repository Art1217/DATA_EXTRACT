"""Catálogo declarativo de tipos de análisis.

Agregar un análisis nuevo es añadir una fila en este archivo y sus prompts en
`prompts/`. El endpoint `/v1/catalog` se sirve directamente de aquí, así que
el catálogo publicado nunca puede divergir del que usa el pipeline.

El dominio de ejemplo son los estados financieros auditados que las empresas
mineras listadas publican en la SMV. El motor no sabe nada de finanzas: todo
lo específico del dominio vive en este catálogo, en los prompts y en las
reglas de `app/domain/accounting.py`.

Las rutas de prompt se declaran relativas a la raíz del proyecto y en formato
POSIX; `Settings.prompt_path()` las resuelve.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from app.domain.errors import UnknownAnalysisTypeError, UnknownGroupError


class Group(StrEnum):
    """Grupo de contenido: el resultado de la Fase 1 de clasificación.

    Un mismo informe suele traer los dos: los estados principales al inicio y
    las notas explicativas después.
    """

    ESTADOS = "Estados"
    """Estados financieros principales."""
    NOTAS = "Notas"
    """Notas explicativas a los estados financieros."""


class Stage(StrEnum):
    """Etapas del pipeline, cada una expuesta como endpoint independiente."""

    CLASSIFICATION = "classification"
    IDENTITY = "identity"
    SEARCH = "search"
    EXTRACTION = "extraction"
    QUALITY = "quality"


@dataclass(frozen=True, slots=True)
class AnalysisType:
    """Un tipo de análisis y todo lo que el pipeline necesita saber de él."""

    key: str
    """Identificador estable usado en la API y en las rutas de storage."""

    label: str
    """Nombre legible para interfaces de usuario."""

    search_prompt: str
    extract_prompt: str | None = None
    quality_prompt: str | None = None
    """Revisión semántica con el modelo. Las reglas contables deterministas
    (`app/domain/accounting.py`) corren aparte y no dependen de que exista."""

    group: Group | None = None
    code: str | None = None
    """Etiqueta que devuelve la Fase 2 de clasificación para este análisis."""

    needs_identity: bool = True
    prefer_docai_json: bool = False
    """Tablas sin bordes o con encabezados complejos rinden mejor con el JSON
    de tablas de Document AI que con el PDF crudo, a costa de chunks más
    pequeños."""

    max_dpi: int | None = None
    max_block_size: int | None = None
    """Topes duros para documentos pesados (fotos, mapas) que reventarían el
    límite de tamaño de la petición."""

    rows_key: str | None = None
    """Clave bajo la que el prompt de extracción envuelve sus filas
    (`{"filas": [...]}`). `None` cuando responde con el array a secas."""

    @property
    def supports_extraction(self) -> bool:
        return self.extract_prompt is not None

    @property
    def supports_quality(self) -> bool:
        """Hay control de calidad si hay prompt o reglas deterministas."""
        from app.domain.accounting import has_rules

        return self.quality_prompt is not None or has_rules(self.key)

    @property
    def folder(self) -> str:
        """Cómo se agrupa este análisis en el ZIP y en el bucket.

        Por grupo cuando existe (`Estados/SitFin`); si no, la clave a secas.
        Vive aquí y no en quien empaqueta para que las dos salidas no puedan
        divergir.
        """
        if self.group and self.code:
            return f"{self.group.value}/{self.code}"
        return self.key

    def effective_dpi(self, requested: int) -> int:
        return min(requested, self.max_dpi) if self.max_dpi else requested

    def effective_block_size(self, requested: int) -> int:
        return min(requested, self.max_block_size) if self.max_block_size else requested


# --------------------------------------------------------------------------
# Constructor auxiliar: un análisis por fila, con sus tres prompts derivados
# de la clave.
# --------------------------------------------------------------------------


def _analysis(
    *,
    key: str,
    label: str,
    group: Group | None,
    code: str | None,
    extract: bool = True,
    quality: bool = False,
    needs_identity: bool = True,
) -> AnalysisType:
    return AnalysisType(
        key=key,
        label=label,
        group=group,
        code=code,
        search_prompt=f"prompts/search/{key}.prompt",
        extract_prompt=f"prompts/extraction/{key}.prompt" if extract else None,
        quality_prompt=f"prompts/quality/{key}.prompt" if quality else None,
        needs_identity=needs_identity,
        rows_key="filas" if extract else None,
    )


# --------------------------------------------------------------------------
# El catálogo
# --------------------------------------------------------------------------

_ANALYSIS_TYPES: tuple[AnalysisType, ...] = (
    # --- Estados principales -------------------------------------------
    _analysis(
        key="situacion_financiera",
        label="Estado de Situación Financiera",
        group=Group.ESTADOS,
        code="SitFin",
        quality=True,
    ),
    _analysis(
        key="resultados",
        label="Estado de Resultados",
        group=Group.ESTADOS,
        code="Result",
        quality=True,
    ),
    _analysis(
        key="flujo_efectivo",
        label="Estado de Flujos de Efectivo",
        group=Group.ESTADOS,
        code="Flujo",
    ),
    # Sin prompt de calidad ni reglas: un estado de cambios en el patrimonio
    # cruza columnas (capital, reservas, resultados acumulados) que no cuadran
    # fila a fila. La ausencia es deliberada.
    _analysis(
        key="cambios_patrimonio",
        label="Estado de Cambios en el Patrimonio",
        group=Group.ESTADOS,
        code="Patrim",
    ),
    # --- Notas ---------------------------------------------------------
    _analysis(
        key="propiedades_planta_equipo",
        label="Nota — Propiedades, Planta y Equipo",
        group=Group.NOTAS,
        code="PPE",
    ),
    _analysis(
        key="inventarios",
        label="Nota — Inventarios",
        group=Group.NOTAS,
        code="Invent",
    ),
    _analysis(
        key="ventas_por_producto",
        label="Nota — Ventas por metal o producto",
        group=Group.NOTAS,
        code="Ventas",
    ),
    # --- Solo búsqueda -------------------------------------------------
    # Localiza el dictamen del auditor para revisarlo a mano: no hay tabla
    # que extraer, así que el pipeline se detiene en el recorte.
    _analysis(
        key="dictamen_auditor",
        label="Dictamen del auditor independiente",
        group=None,
        code=None,
        extract=False,
        needs_identity=False,
    ),
)

ANALYSIS_TYPES: Mapping[str, AnalysisType] = MappingProxyType(
    {analysis.key: analysis for analysis in _ANALYSIS_TYPES}
)

# Prompts de clasificación, indexados por la etapa que los consume.
CLASSIFICATION_GROUP_PROMPT = "prompts/classification/grupo.prompt"
CLASSIFICATION_ANALYSIS_PROMPTS: Mapping[Group, str] = MappingProxyType(
    {
        Group.ESTADOS: "prompts/classification/estados.prompt",
        Group.NOTAS: "prompts/classification/notas.prompt",
    }
)

IDENTITY_PROMPT = "prompts/identity.prompt"


# --------------------------------------------------------------------------
# Consultas sobre el catálogo
# --------------------------------------------------------------------------


def get_analysis_type(key: str) -> AnalysisType:
    """Devuelve el análisis con esa clave o lanza `UnknownAnalysisTypeError`."""
    try:
        return ANALYSIS_TYPES[key]
    except KeyError:
        raise UnknownAnalysisTypeError(key, valid=tuple(ANALYSIS_TYPES)) from None


def parse_group(value: str) -> Group:
    """Convierte texto a `Group` tolerando mayúsculas/minúsculas."""
    for group in Group:
        if group.value.casefold() == value.casefold():
            return group
    raise UnknownGroupError(value, valid=tuple(g.value for g in Group))


def analyses_for_group(group: Group) -> tuple[AnalysisType, ...]:
    """Análisis pertenecientes a un grupo, en orden de catálogo."""
    return tuple(a for a in _ANALYSIS_TYPES if a.group is group)


def valid_codes(group: Group) -> frozenset[str]:
    """Etiquetas que la Fase 2 puede devolver para ese grupo.

    Sirve para filtrar alucinaciones del clasificador.
    """
    return frozenset(a.code for a in analyses_for_group(group) if a.code)


def find_by_group_and_code(group: Group, code: str) -> AnalysisType | None:
    """Traduce el resultado de la clasificación a un tipo de análisis."""
    for candidate in analyses_for_group(group):
        if candidate.code == code:
            return candidate
    return None


def missing_prompts(prompt_path_resolver) -> list[str]:
    """Rutas de prompt declaradas en el catálogo que no existen en disco.

    Se ejecuta al arrancar el servidor para fallar de inmediato en vez de a
    mitad de un job caro. `prompt_path_resolver` es `Settings.prompt_path`.
    """
    declared: list[str] = [
        IDENTITY_PROMPT,
        CLASSIFICATION_GROUP_PROMPT,
        *CLASSIFICATION_ANALYSIS_PROMPTS.values(),
    ]
    for analysis in _ANALYSIS_TYPES:
        declared.extend(
            p
            for p in (analysis.search_prompt, analysis.extract_prompt, analysis.quality_prompt)
            if p
        )
    return [relative for relative in declared if not prompt_path_resolver(relative).is_file()]
