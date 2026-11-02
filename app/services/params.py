"""Parámetros de recorrido de un documento.

Los valores por defecto viven en la configuración y algunos análisis imponen
topes más bajos (un informe con fotos y mapas revienta el
tamaño máximo de petición a 30 páginas y 300 DPI). `resolve()` concentra esa
resolución en un único sitio, en vez de repetir `min(...)` en cada etapa.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.domain.analysis import AnalysisType


@dataclass(frozen=True, slots=True)
class ScanParams:
    """Cómo recorrer el documento. `None` significa "usa la configuración"."""

    dpi: int | None = None
    block_size: int | None = None
    workers: int | None = None
    max_pages: int | None = None
    """Corta el recorrido a las primeras N páginas. Sirve para validar barato."""

    def resolve(
        self,
        settings: Settings,
        analysis: AnalysisType | None = None,
        *,
        classification: bool = False,
    ) -> ResolvedScanParams:
        """Aplica defaults de configuración y los topes del análisis.

        `classification=True` usa el DPI de clasificación, más bajo que el de
        búsqueda: para reconocer de qué tipo es una tabla no hace falta leer
        cada dígito.
        """
        base_dpi = settings.clasif_dpi if classification else settings.dpi
        dpi = self.dpi or base_dpi
        block_size = self.block_size or settings.block_size
        if analysis is not None:
            dpi = analysis.effective_dpi(dpi)
            block_size = analysis.effective_block_size(block_size)
        return ResolvedScanParams(
            dpi=dpi,
            block_size=block_size,
            workers=self.workers or settings.busqueda_workers,
            max_pages=self.max_pages,
        )


@dataclass(frozen=True, slots=True)
class ResolvedScanParams:
    """Parámetros ya concretos, listos para el recorrido por bloques."""

    dpi: int
    block_size: int
    workers: int
    max_pages: int | None = None
