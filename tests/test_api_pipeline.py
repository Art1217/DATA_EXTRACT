"""Recorrido completo por HTTP, etapa por etapa.

Con el modelo sustituido por el proveedor `fake`, pero ejercitando de verdad
los adaptadores, el encolado de jobs, el storage, el encadenado entre etapas y
las rutas.
"""

from __future__ import annotations

import io
import zipfile

from tests.conftest import wait_for_job


def subir(client, pdf_bytes, nombre: str = "eeff-2024.pdf") -> str:
    respuesta = client.post("/v1/documents", files={"file": (nombre, pdf_bytes, "application/pdf")})
    assert respuesta.status_code == 201, respuesta.text
    return respuesta.json()["document_id"]


def lanzar(client, ruta: str, cuerpo: dict | None = None) -> dict:
    respuesta = client.post(ruta, json=cuerpo or {})
    assert respuesta.status_code == 202, respuesta.text
    return wait_for_job(client, respuesta.json()["job_id"])


# --- Documentos -----------------------------------------------------------


def test_subir_el_mismo_pdf_dos_veces_no_lo_duplica(client, pdf_bytes):
    """El identificador es el hash del contenido: reingresar reutiliza todo."""
    primero = subir(client, pdf_bytes)
    segundo = subir(client, pdf_bytes)
    assert primero == segundo
    assert len(client.get("/v1/documents").json()) == 1


def test_la_ficha_reporta_paginas_y_si_es_escaneado(client, pdf_bytes):
    document_id = subir(client, pdf_bytes)
    ficha = client.get(f"/v1/documents/{document_id}").json()
    assert ficha["pages"] == 3
    assert ficha["is_scanned"] is False
    assert ficha["stages"]["identity"] is False


def test_un_documento_inexistente_da_404(client):
    assert client.get("/v1/documents/noexiste").status_code == 404
    assert client.get("/v1/documents/noexiste").json()["code"] == "document_not_found"


# --- Clasificación: Fase 1 y Fase 2 independientes ------------------------


def test_fase_1_detecta_los_grupos(client, pdf_bytes):
    document_id = subir(client, pdf_bytes)
    job = lanzar(client, f"/v1/documents/{document_id}/classification/groups")
    assert job["status"] == "succeeded"
    assert job["result"]["groups"] == ["Estados", "Notas"]
    assert job["result"]["pages_analyzed"] == 3


def test_fase_2_no_exige_haber_corrido_la_fase_1(client, pdf_bytes, fake_ai):
    """Quien ya sabe qué trae su documento se ahorra una pasada completa."""
    document_id = subir(client, pdf_bytes)

    job = lanzar(
        client,
        f"/v1/documents/{document_id}/classification/analysis",
        {"group": "Estados"},
    )

    assert job["status"] == "succeeded"
    assert job["result"]["analyses"] == ["SitFin", "Result", "Flujo"]
    assert job["result"]["analysis_types"] == [
        "situacion_financiera",
        "resultados",
        "flujo_efectivo",
    ]
    assert "classify:grupo" not in fake_ai, "no debió ejecutarse la Fase 1"


def test_fase_2_rechaza_un_grupo_inexistente(client, pdf_bytes):
    document_id = subir(client, pdf_bytes)
    respuesta = client.post(
        f"/v1/documents/{document_id}/classification/analysis", json={"group": "Memoria"}
    )
    assert respuesta.status_code == 422
    assert respuesta.json()["code"] == "unknown_group"


def test_la_clasificacion_descarta_etiquetas_inventadas(client, pdf_bytes, fake_ai):
    """Un código que no está en el catálogo no puede colarse en el plan."""
    fake_ai.overrides["classify"] = {
        "clasificaciones": [
            {"page_number": 1, "codigo": "SitFin"},
            {"page_number": 2, "codigo": "EBITDA"},
        ]
    }
    document_id = subir(client, pdf_bytes)
    job = lanzar(
        client, f"/v1/documents/{document_id}/classification/analysis", {"group": "Estados"}
    )
    assert job["result"]["analyses"] == ["SitFin"]


def test_el_plan_encadena_ambas_fases(client, pdf_bytes):
    document_id = subir(client, pdf_bytes)
    job = lanzar(client, f"/v1/documents/{document_id}/plan")
    assert job["result"]["groups"] == {
        "Estados": ["SitFin", "Result", "Flujo"],
        "Notas": ["PPE", "Invent", "Ventas"],
    }
    assert job["result"]["analysis_types"][0] == "situacion_financiera"


def test_el_plan_reutiliza_clasificaciones_ya_hechas(client, pdf_bytes, fake_ai):
    """Reclasificar cuesta una pasada entera: no se repite sin pedirlo."""
    document_id = subir(client, pdf_bytes)
    lanzar(client, f"/v1/documents/{document_id}/classification/groups")
    llamadas_fase1 = fake_ai["classify:grupo"]

    lanzar(client, f"/v1/documents/{document_id}/plan")

    assert fake_ai["classify:grupo"] == llamadas_fase1


# --- Etapas encadenadas ---------------------------------------------------


def test_recorrido_completo_etapa_por_etapa(client, pdf_bytes):
    document_id = subir(client, pdf_bytes)
    analisis = {"analysis_type": "situacion_financiera"}

    identidad = lanzar(client, f"/v1/documents/{document_id}/identity")
    assert identidad["result"]["empresa"] == "Minera Andina Demo S.A.A."

    busqueda = lanzar(client, f"/v1/documents/{document_id}/search", analisis)
    assert busqueda["result"]["relevant_pages"] == [1, 2]
    assert busqueda["result"]["cut_pdf_key"] is not None

    extraccion = lanzar(client, f"/v1/documents/{document_id}/extraction", analisis)
    assert extraccion["result"]["row_count"] == 32
    assert extraccion["result"]["identity"]["moneda"] == "USD"

    calidad = lanzar(client, f"/v1/documents/{document_id}/quality", analisis)
    assert calidad["result"]["status"] == "Aprobado"
    assert calidad["result"]["llm_review"] is True
    assert {c["status"] for c in calidad["result"]["checks"]} == {"ok"}

    inventario = client.get(f"/v1/documents/{document_id}").json()["stages"]
    assert inventario["search"] == ["situacion_financiera"]
    assert inventario["extraction"] == ["situacion_financiera"]
    assert inventario["quality"] == ["situacion_financiera"]


def test_un_balance_que_no_cuadra_se_rechaza_aunque_el_modelo_lo_apruebe(
    client, pdf_bytes, fake_ai
):
    """Las reglas contables mandan sobre la opinión del modelo."""
    fake_ai.overrides["extract"] = {
        "filas": [
            {"codigo": "TOTAL_ACTIVO", "periodo": "2024", "valor": 1000},
            {"codigo": "TOTAL_PASIVO", "periodo": "2024", "valor": 400},
            {"codigo": "TOTAL_PATRIMONIO", "periodo": "2024", "valor": 500},
        ]
    }
    document_id = subir(client, pdf_bytes)

    job = lanzar(
        client,
        f"/v1/documents/{document_id}/pipeline",
        {"analysis_type": "situacion_financiera"},
    )

    calidad = job["result"]["quality"]
    assert calidad["status"] == "Rechazado"
    assert "Descuadre de 100.00 en 2024" in calidad["errors"]
    assert fake_ai["quality"] == 1, "la revisión con el modelo también corrió"


def test_el_flujo_de_efectivo_se_valida_solo_con_reglas(client, pdf_bytes, fake_ai):
    """Sin prompt de calidad no se gasta en el modelo: bastan las reglas."""
    document_id = subir(client, pdf_bytes)
    job = lanzar(
        client, f"/v1/documents/{document_id}/pipeline", {"analysis_type": "flujo_efectivo"}
    )
    calidad = job["result"]["quality"]
    assert calidad["status"] == "Aprobado"
    assert calidad["llm_review"] is False
    assert "quality" not in fake_ai


def test_extraer_sin_busqueda_previa_indica_que_falta(client, pdf_bytes):
    """El error nombra la etapa pendiente: es el tropiezo más común."""
    document_id = subir(client, pdf_bytes)
    job = lanzar(client, f"/v1/documents/{document_id}/extraction", {"analysis_type": "resultados"})
    assert job["status"] == "failed"
    assert "search" in job["error"]


def test_calidad_sin_extraccion_previa_falla_con_el_motivo(client, pdf_bytes):
    document_id = subir(client, pdf_bytes)
    job = lanzar(client, f"/v1/documents/{document_id}/quality", {"analysis_type": "resultados"})
    assert job["status"] == "failed"
    assert "extraction" in job["error"]


def test_cambios_en_el_patrimonio_no_admite_control_de_calidad(client, pdf_bytes):
    document_id = subir(client, pdf_bytes)
    job = lanzar(
        client,
        f"/v1/documents/{document_id}/quality",
        {"analysis_type": "cambios_patrimonio"},
    )
    assert job["status"] == "failed"
    assert "no soporta" in job["error"]


def test_el_dictamen_solo_se_localiza(client, pdf_bytes, fake_ai):
    """Un análisis sin extracción se detiene en el recorte, sin error."""
    document_id = subir(client, pdf_bytes)
    job = lanzar(
        client, f"/v1/documents/{document_id}/pipeline", {"analysis_type": "dictamen_auditor"}
    )
    assert job["status"] == "succeeded"
    assert job["result"]["search"]["cut_pdf_key"] is not None
    assert "extract" not in fake_ai
    assert "identity" not in fake_ai


# --- Reentrada: el motivo de separar las etapas ---------------------------


def test_recortar_a_mano_permite_reextraer_sin_repetir_la_busqueda(client, pdf_bytes, fake_ai):
    document_id = subir(client, pdf_bytes)
    analisis = {"analysis_type": "resultados"}
    lanzar(client, f"/v1/documents/{document_id}/search", analisis)
    busquedas = fake_ai["search"]

    corte = lanzar(client, f"/v1/documents/{document_id}/cut", analisis | {"pages": [3]})
    assert corte["status"] == "succeeded"

    extraccion = lanzar(client, f"/v1/documents/{document_id}/extraction", analisis)

    assert extraccion["status"] == "succeeded"
    assert fake_ai["search"] == busquedas, "no debió repetirse la búsqueda"


def test_revalidar_filas_corregidas_no_requiere_documento(client, fake_ai):
    """Permite corregir un dato a mano y volver a validarlo sin reextraer."""
    respuesta = client.post(
        "/v1/quality/check",
        json={
            "analysis_type": "flujo_efectivo",
            "rows": [
                {"codigo": "EFECTIVO_INICIAL", "periodo": "2024", "valor": 100},
                {"codigo": "AUMENTO_NETO_EFECTIVO", "periodo": "2024", "valor": 20},
                {"codigo": "EFECTIVO_FINAL", "periodo": "2024", "valor": 120},
            ],
        },
    )
    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "Aprobado"
    assert respuesta.json()["rows_checked"] == 3
    assert "extract" not in fake_ai


def test_el_pipeline_reutiliza_lo_ya_calculado(client, pdf_bytes, fake_ai):
    document_id = subir(client, pdf_bytes)
    lanzar(client, f"/v1/documents/{document_id}/search", {"analysis_type": "resultados"})
    busquedas = fake_ai["search"]

    job = lanzar(client, f"/v1/documents/{document_id}/pipeline", {"analysis_type": "resultados"})

    assert job["status"] == "succeeded"
    assert fake_ai["search"] == busquedas
    assert any("reutiliz" in nota for nota in job["result"]["skipped"])


def test_el_pipeline_puede_limitarse_a_ciertas_etapas(client, pdf_bytes, fake_ai):
    document_id = subir(client, pdf_bytes)
    job = lanzar(
        client,
        f"/v1/documents/{document_id}/pipeline",
        {"analysis_type": "resultados", "stages": ["search"]},
    )
    assert job["status"] == "succeeded"
    assert "extract" not in fake_ai


# --- Flujo automático y paquete -------------------------------------------


def test_el_flujo_automatico_procesa_lo_que_encuentra(client, pdf_bytes):
    document_id = subir(client, pdf_bytes)
    job = lanzar(client, f"/v1/documents/{document_id}/auto")
    assert job["status"] == "succeeded"
    assert len(job["result"]["plan"]["analysis_types"]) == 6
    assert job["result"]["bundle_key"] is not None
    assert all(item["error"] is None for item in job["result"]["results"])


def test_plan_only_no_gasta_en_extraccion(client, pdf_bytes, fake_ai):
    """Sirve para revisar el plan antes de autorizar el gasto."""
    document_id = subir(client, pdf_bytes)
    job = lanzar(client, f"/v1/documents/{document_id}/auto", {"plan_only": True})
    assert job["result"]["plan_only"] is True
    assert "extract" not in fake_ai


def test_el_paquete_se_puede_descargar(client, pdf_bytes):
    document_id = subir(client, pdf_bytes)
    lanzar(
        client,
        f"/v1/documents/{document_id}/pipeline",
        {"analysis_type": "situacion_financiera"},
    )
    job = lanzar(client, f"/v1/documents/{document_id}/bundle")

    bundle_id = job["result"]["bundle_key"].rsplit("/", 1)[-1].removesuffix(".zip")
    descarga = client.get(f"/v1/documents/{document_id}/bundles/{bundle_id}")
    assert descarga.status_code == 200

    with zipfile.ZipFile(io.BytesIO(descarga.content)) as archivo:
        nombres = archivo.namelist()
    # Cada archivo dice de qué informe y de qué análisis salió: un ZIP se
    # descomprime, y ahí fuera el nombre es todo lo que queda.
    assert "eeff-2024_manifiesto.json" in nombres
    assert "Estados/SitFin/eeff-2024_situacion_financiera_extraccion.json" in nombres
    assert "Estados/SitFin/eeff-2024_situacion_financiera_extraccion.xlsx" in nombres


# --- Descargas ------------------------------------------------------------


def test_el_excel_se_genera_desde_la_extraccion_guardada(client, pdf_bytes):
    document_id = subir(client, pdf_bytes)
    lanzar(client, f"/v1/documents/{document_id}/pipeline", {"analysis_type": "resultados"})

    respuesta = client.get(f"/v1/documents/{document_id}/extraction/resultados/xlsx")

    assert respuesta.status_code == 200
    assert respuesta.headers["content-type"].startswith("application/vnd.openxml")


def test_el_pdf_recortado_se_puede_descargar(client, pdf_bytes):
    document_id = subir(client, pdf_bytes)
    lanzar(client, f"/v1/documents/{document_id}/search", {"analysis_type": "resultados"})

    respuesta = client.get(f"/v1/documents/{document_id}/search/resultados/pdf")

    assert respuesta.status_code == 200
    assert respuesta.content.startswith(b"%PDF")


# --- Catálogo, configuración y jobs ---------------------------------------


def test_el_catalogo_publica_lo_que_el_pipeline_sabe_hacer(client):
    catalogo = client.get("/v1/catalog/analysis-types").json()
    claves = {entrada["key"] for entrada in catalogo}
    assert {"situacion_financiera", "inventarios", "dictamen_auditor"} <= claves

    balance = next(e for e in catalogo if e["key"] == "situacion_financiera")
    assert balance["group"] == "Estados" and balance["code"] == "SitFin"
    assert balance["supports_extraction"] and balance["supports_quality"]
    assert "Total activo = total pasivo + total patrimonio" in balance["accounting_rules"]


def test_el_catalogo_de_grupos(client):
    grupos = {g["group"]: g for g in client.get("/v1/catalog/groups").json()}
    assert grupos["Notas"]["codes"] == ["PPE", "Invent", "Ventas"]


def test_la_configuracion_no_expone_secretos(client):
    config = client.get("/v1/config").json()
    assert config["ai_provider"] == "fake"
    assert config["storage"] == "local"
    assert "api_key" not in str(config).lower()


def test_los_jobs_se_pueden_filtrar_por_documento(client, pdf_bytes):
    document_id = subir(client, pdf_bytes)
    lanzar(client, f"/v1/documents/{document_id}/identity")

    jobs = client.get("/v1/jobs", params={"document_id": document_id}).json()

    assert len(jobs) == 1
    assert jobs[0]["type"] == "identity"
    assert jobs[0]["duration_seconds"] is not None


def test_los_eventos_sse_terminan_con_el_estado_final(client, pdf_bytes):
    document_id = subir(client, pdf_bytes)
    respuesta = client.post(f"/v1/documents/{document_id}/identity", json={})
    job_id = respuesta.json()["job_id"]
    wait_for_job(client, job_id)

    flujo = client.get(f"/v1/jobs/{job_id}/events")

    assert flujo.status_code == 200
    assert "succeeded" in flujo.text
