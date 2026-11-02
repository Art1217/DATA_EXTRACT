"""El catálogo es la fuente de verdad del pipeline: se valida contra el disco."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.domain import analysis as catalog
from app.domain.errors import UnknownAnalysisTypeError, UnknownGroupError


def test_todos_los_prompts_declarados_existen_en_disco():
    """Un prompt ausente debe detectarse al arrancar, no a mitad de un job caro."""
    assert catalog.missing_prompts(get_settings().prompt_path) == []


def test_las_claves_del_catalogo_son_unicas_y_coinciden_con_su_entrada():
    for key, analysis in catalog.ANALYSIS_TYPES.items():
        assert analysis.key == key


@pytest.mark.parametrize(
    ("group", "esperados"),
    [(catalog.Group.ESTADOS, 4), (catalog.Group.NOTAS, 3)],
)
def test_cantidad_de_analisis_por_grupo(group, esperados):
    assert len(catalog.analyses_for_group(group)) == esperados


def test_los_codigos_son_unicos_dentro_de_cada_grupo():
    for group in catalog.Group:
        codes = [a.code for a in catalog.analyses_for_group(group)]
        assert len(codes) == len(set(codes))


def test_el_patrimonio_no_declara_control_de_calidad():
    """La ausencia es deliberada: sus columnas no cuadran fila a fila."""
    assert not catalog.get_analysis_type("cambios_patrimonio").supports_quality


def test_el_flujo_de_efectivo_tiene_calidad_sin_prompt():
    """Las reglas contables bastan para ofrecer control de calidad."""
    flujo = catalog.get_analysis_type("flujo_efectivo")
    assert flujo.quality_prompt is None
    assert flujo.supports_quality


def test_el_dictamen_solo_localiza_paginas():
    dictamen = catalog.get_analysis_type("dictamen_auditor")
    assert not dictamen.supports_extraction
    assert not dictamen.supports_quality
    assert not dictamen.needs_identity
    assert dictamen.folder == "dictamen_auditor"


def test_la_carpeta_del_zip_sale_del_grupo_y_el_codigo():
    assert catalog.get_analysis_type("inventarios").folder == "Notas/Invent"


def test_los_topes_de_render_solo_bajan_lo_pedido():
    analisis = catalog.AnalysisType(
        key="x", label="x", search_prompt="x", max_dpi=150, max_block_size=10
    )
    assert analisis.effective_dpi(300) == 150
    assert analisis.effective_dpi(100) == 100
    assert analisis.effective_block_size(30) == 10


def test_las_etiquetas_de_fase_2_se_traducen_a_claves_del_catalogo():
    encontrado = catalog.find_by_group_and_code(catalog.Group.ESTADOS, "SitFin")
    assert encontrado is not None and encontrado.key == "situacion_financiera"
    assert catalog.find_by_group_and_code(catalog.Group.ESTADOS, "PPE") is None


def test_parse_group_tolera_mayusculas_y_rechaza_lo_desconocido():
    assert catalog.parse_group("notas") is catalog.Group.NOTAS
    with pytest.raises(UnknownGroupError):
        catalog.parse_group("Memoria")


def test_un_analisis_desconocido_nombra_las_alternativas():
    with pytest.raises(UnknownAnalysisTypeError) as error:
        catalog.get_analysis_type("no_existe")
    assert "situacion_financiera" in str(error.value)
