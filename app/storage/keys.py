"""Construcción de claves de storage.

Todo artefacto se direcciona por una clave relativa con forma de ruta POSIX
(`documents/<id>/extraction/<analysis>/result.json`). Los servicios nunca
manipulan rutas absolutas: así el mismo código sirve para disco local hoy y
para un bucket de GCS mañana, donde estas claves son literalmente el nombre
del objeto.

Concentrar aquí el layout evita que los nombres de archivo se armen con
f-strings repartidas por todo el pipeline, y que dos corridas del mismo
documento se sobrescriban entre sí.
"""

from __future__ import annotations

DOCUMENTS = "documents"
JOBS = "jobs"


def document_root(document_id: str) -> str:
    return f"{DOCUMENTS}/{document_id}"


def source_pdf(document_id: str) -> str:
    return f"{document_root(document_id)}/source.pdf"


def document_meta(document_id: str) -> str:
    return f"{document_root(document_id)}/meta.json"


def identity(document_id: str) -> str:
    return f"{document_root(document_id)}/identity.json"


# --- Clasificación --------------------------------------------------------


def classification_group(document_id: str) -> str:
    """Fase 1: qué grupos de contenido (Estados/Notas) tiene el documento."""
    return f"{document_root(document_id)}/classification/grupos.json"


def classification_analysis(document_id: str, group: str) -> str:
    """Fase 2: qué análisis hay dentro de un grupo concreto."""
    return f"{document_root(document_id)}/classification/analisis_{group}.json"


def plan(document_id: str) -> str:
    return f"{document_root(document_id)}/classification/plan.json"


def plan_text(document_id: str) -> str:
    """Versión legible del plan (grupos y análisis) que acompaña al ZIP."""
    return f"{document_root(document_id)}/classification/plan.txt"


# --- Búsqueda -------------------------------------------------------------


def search_root(document_id: str, analysis_key: str) -> str:
    return f"{document_root(document_id)}/search/{analysis_key}"


def search_result(document_id: str, analysis_key: str) -> str:
    return f"{search_root(document_id, analysis_key)}/result.json"


def search_pdf(document_id: str, analysis_key: str) -> str:
    """PDF recortado a las páginas relevantes."""
    return f"{search_root(document_id, analysis_key)}/corte.pdf"


def search_pdf_searchable(document_id: str, analysis_key: str) -> str:
    """Versión del corte con capa de texto OCR.

    Es una clave distinta a propósito: sobrescribir el corte original con la
    versión digitalizada mutaría una entrada que otras etapas pueden estar
    leyendo.
    """
    return f"{search_root(document_id, analysis_key)}/corte_searchable.pdf"


# --- Extracción -----------------------------------------------------------


def extraction_root(document_id: str, analysis_key: str) -> str:
    return f"{document_root(document_id)}/extraction/{analysis_key}"


def extraction_result(document_id: str, analysis_key: str) -> str:
    return f"{extraction_root(document_id, analysis_key)}/result.json"


def extraction_docai(document_id: str, analysis_key: str) -> str:
    """Salida cruda de Document AI, para auditar de dónde salió cada dato."""
    return f"{extraction_root(document_id, analysis_key)}/docai.json"


def extraction_xlsx(document_id: str, analysis_key: str) -> str:
    return f"{extraction_root(document_id, analysis_key)}/result.xlsx"


# --- Calidad --------------------------------------------------------------


def quality_result(document_id: str, analysis_key: str) -> str:
    return f"{document_root(document_id)}/quality/{analysis_key}/result.json"


# --- Empaquetado y jobs ---------------------------------------------------


def bundle(document_id: str, bundle_id: str) -> str:
    return f"{document_root(document_id)}/bundles/{bundle_id}.zip"


BUNDLES = "bundles"


def batch_bundle(batch_id: str) -> str:
    """ZIP de una corrida entera: no cuelga de ningún documento porque los
    agrupa a todos."""
    return f"{BUNDLES}/{batch_id}.zip"


def job(job_id: str) -> str:
    return f"{JOBS}/{job_id}.json"
