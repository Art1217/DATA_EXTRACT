"""Publicación al bucket y registro en BigQuery.

Ninguna prueba toca GCP: se sustituyen `upload_bytes` e `insert_extraction_row`
en el espacio de nombres de `publish`, que es donde quedaron enlazados por el
`import`. Lo que se ejercita es la orquestación real: qué claves se suben, qué
lleva la fila y qué pasa cuando GCP falla.
"""

from __future__ import annotations

import dataclasses
import io
import json
import zipfile

import pytest

from app.services import auto, pipeline
from app.storage import keys
from tests.conftest import build_pdf


@pytest.fixture
def gcp(monkeypatch):
    """Dobles de GCS y BigQuery que registran lo que se les pidió."""
    registro: dict[str, list] = {"objetos": [], "filas": []}

    def fake_upload(target, key, data):
        registro["objetos"].append((key, len(data)))
        return f"gs://{target.bucket}/{key}"

    def fake_insert(target, row):
        registro["filas"].append(row)

    monkeypatch.setattr("app.services.publish.upload_bytes", fake_upload)
    monkeypatch.setattr("app.services.publish.insert_extraction_row", fake_insert)
    return registro


@pytest.fixture
def publicando(settings):
    settings.publish_enabled = True
    settings.gcs_bucket = "bucket-de-prueba"
    return settings


def documento(context, pdf_bytes, nombre: str = "eeff-2024.pdf") -> str:
    from app.services.documents import register_document

    return register_document(context.storage, pdf_bytes, nombre).document_id


def corrida(context) -> str:
    from app.services.publish import batch_folder

    return batch_folder(context.batch)


# --- Análisis suelto ------------------------------------------------------


def test_publica_los_entregables_el_manifiesto_y_una_fila(
    context, pdf_bytes, fake_ai, publicando, gcp
):
    document_id = documento(context, pdf_bytes)

    outcome = pipeline.run_pipeline(context, document_id, "situacion_financiera")

    carpeta = f"{corrida(context)}/eeff-2024_{document_id[:8]}"
    base = f"{carpeta}/Estados/SitFin/eeff-2024_situacion_financiera"
    assert [key for key, _ in gcp["objetos"]] == [
        f"{base}_corte.pdf",
        f"{base}_extraccion.xlsx",
        f"{base}_busqueda.json",
        f"{base}_calidad.json",
        f"{carpeta}/eeff-2024_manifiesto.json",
    ]
    assert outcome.published["bigquery_row"] is True
    assert len(gcp["filas"]) == 1


def test_la_fila_lleva_identidad_uris_y_las_filas_extraidas(
    context, pdf_bytes, fake_ai, publicando, gcp
):
    document_id = documento(context, pdf_bytes)

    pipeline.run_pipeline(context, document_id, "situacion_financiera")

    fila = gcp["filas"][0]
    assert fila["document_id"] == document_id
    assert fila["filename"] == "eeff-2024.pdf"
    assert fila["company"] == "Minera Andina Demo S.A.A."
    assert fila["fiscal_period"] == "2024" and fila["currency"] == "USD"
    assert fila["analysis_type"] == "situacion_financiera"
    assert fila["analysis_group"] == "Estados"
    assert fila["row_count"] == 32
    assert fila["quality_status"] == "Aprobado"
    assert fila["pdf_uri"].startswith("gs://bucket-de-prueba/")
    assert fila["excel_uri"].endswith("eeff-2024_situacion_financiera_extraccion.xlsx")
    assert json.loads(fila["rows_json"])[0]["concepto"] == "Efectivo y equivalentes de efectivo"


def test_la_fila_tiene_exactamente_las_columnas_del_esquema(
    context, pdf_bytes, fake_ai, publicando, gcp
):
    """Una columna que la tabla no conoce haría que BigQuery rechace la fila."""
    from app.services.bigquery_client import TABLE_SCHEMA

    pipeline.run_pipeline(context, documento(context, pdf_bytes), "resultados")

    assert set(gcp["filas"][0]) == {nombre for nombre, _, _ in TABLE_SCHEMA}


def test_el_excel_queda_guardado_en_el_storage(context, pdf_bytes, fake_ai, publicando, gcp):
    """Se genera una vez al publicar, en vez de regenerarse en cada descarga."""
    document_id = documento(context, pdf_bytes)

    pipeline.run_pipeline(context, document_id, "resultados")

    assert context.storage.exists(keys.extraction_xlsx(document_id, "resultados"))


def test_apagado_por_defecto_no_llama_a_gcp(context, pdf_bytes, fake_ai, gcp):
    assert context.settings.publish_enabled is False
    document_id = documento(context, pdf_bytes)

    outcome = pipeline.run_pipeline(context, document_id, "resultados")

    assert gcp["objetos"] == [] and gcp["filas"] == []
    assert outcome.published is None


def test_sin_bucket_configurado_no_se_publica(context, pdf_bytes, fake_ai, settings, gcp):
    settings.publish_enabled = True
    settings.gcs_bucket = None
    document_id = documento(context, pdf_bytes)

    outcome = pipeline.run_pipeline(context, document_id, "resultados")

    assert gcp["objetos"] == [] and outcome.published is None


def test_un_documento_sin_hallazgos_deja_su_fila_en_cero(
    context, pdf_bytes, fake_ai, publicando, gcp
):
    """ "Se revisó y no había" es un resultado, distinto de "no se revisó"."""
    fake_ai.overrides["search"] = {"relevant_sections": [], "alerts": []}
    document_id = documento(context, pdf_bytes)

    outcome = pipeline.run_pipeline(context, document_id, "inventarios")

    assert outcome.extraction is None
    fila = gcp["filas"][0]
    assert fila["row_count"] == 0
    assert fila["pdf_uri"] is None
    assert "pdf_uri" in outcome.published["missing"]


def test_una_corrida_solo_de_busqueda_no_publica(context, pdf_bytes, fake_ai, publicando, gcp):
    from app.domain.analysis import Stage

    pipeline.run_pipeline(
        context, documento(context, pdf_bytes), "resultados", stages=frozenset({Stage.SEARCH})
    )

    assert gcp["filas"] == []


def test_un_fallo_de_gcs_no_hace_fallar_el_pipeline(
    context, pdf_bytes, fake_ai, publicando, gcp, monkeypatch
):
    def revienta(target, key, data):
        raise PermissionError("403 storage.objects.create")

    monkeypatch.setattr("app.services.publish.upload_bytes", revienta)

    outcome = pipeline.run_pipeline(context, documento(context, pdf_bytes), "resultados")

    assert outcome.extraction is not None
    assert "403" in outcome.published["error"]
    assert outcome.published["bigquery_row"] is True, "la fila se registra igual"


def test_un_fallo_de_bigquery_no_hace_fallar_el_pipeline(
    context, pdf_bytes, fake_ai, publicando, gcp, monkeypatch
):
    def revienta(target, row):
        raise RuntimeError("dataset no existe")

    monkeypatch.setattr("app.services.publish.insert_extraction_row", revienta)

    outcome = pipeline.run_pipeline(context, documento(context, pdf_bytes), "resultados")

    assert outcome.published["bigquery_row"] is False
    assert "dataset no existe" in outcome.published["error"]


# --- Modo automático ------------------------------------------------------


def test_el_modo_automatico_registra_una_fila_por_analisis_y_sube_el_plan(
    context, pdf_bytes, fake_ai, publicando, gcp
):
    document_id = documento(context, pdf_bytes)

    resultado = auto.run_auto(context, document_id)

    assert len(gcp["filas"]) == len(resultado.plan.analysis_types)
    assert resultado.published["plan_txt_uri"].endswith("eeff-2024_plan.txt")


def test_el_plan_txt_declara_grupo_analisis_y_resultado(context, pdf_bytes, fake_ai):
    document_id = documento(context, pdf_bytes)

    auto.run_auto(context, document_id)

    texto = context.storage.read_bytes(keys.plan_text(document_id)).decode("utf-8")
    assert "GRUPO: Estados" in texto
    assert "[situacion_financiera]" in texto
    assert "calidad: Aprobado" in texto


def test_el_zip_incluye_el_plan_en_texto(context, pdf_bytes, fake_ai):
    from app.services import bundle

    document_id = documento(context, pdf_bytes)
    resultado = auto.run_auto(context, document_id)

    with zipfile.ZipFile(io.BytesIO(context.storage.read_bytes(resultado.bundle_key))) as zipf:
        assert "eeff-2024_plan.txt" in zipf.namelist()
    assert bundle is not None


# --- Nombres en el bucket: es para personas -------------------------------


def test_una_corrida_agrupa_todos_sus_documentos(context, pdf_bytes, fake_ai, publicando, gcp):
    context = dataclasses.replace(context, batch="20260811-1215")
    uno = documento(context, pdf_bytes, "uno.pdf")
    dos = documento(context, build_pdf(["otro " + "x" * 300]), "dos.pdf")

    pipeline.run_pipeline(context, uno, "resultados")
    pipeline.run_pipeline(context, dos, "resultados")

    carpetas = {key.split("/")[0] for key, _ in gcp["objetos"]}
    assert carpetas == {"run_20260811-1215"}
    assert {fila["batch"] for fila in gcp["filas"]} == {"run_20260811-1215"}


def test_sin_lote_la_corrida_lleva_la_fecha_del_momento():
    from app.services.publish import batch_folder

    assert batch_folder(None).startswith("run_20")


def test_el_lote_se_limpia_antes_de_ser_una_ruta():
    from app.services.publish import batch_folder

    assert batch_folder("../../etc/passwd") == "run_etc-passwd"


def test_dos_informes_con_el_mismo_nombre_no_se_pisan():
    from app.services.publish import document_folder

    assert document_folder("a" * 64, "eeff.pdf") != document_folder("b" * 64, "eeff.pdf")


def test_el_nombre_del_archivo_se_limpia_para_el_bucket():
    from app.services.publish import document_folder

    assert document_folder("abcdef1234", "Estados Financieros 2024 (auditados).pdf") == (
        "Estados-Financieros-2024-auditados_abcdef12"
    )


def test_un_nombre_con_ruta_se_queda_con_el_archivo():
    from app.services.publish import document_slug

    assert document_slug("C:\\Users\\yo\\Descargas\\eeff.pdf") == "eeff"


def test_sin_nombre_utilizable_se_cae_al_id():
    from app.services.publish import document_folder

    assert document_folder("abcdef1234", "¿¿??.pdf") == "abcdef1234"


def test_el_prefijo_configurado_antepone_sin_duplicar_barras(publicando):
    from app.services import gcs_client

    publicando.gcs_prefix = "/entregables/"
    destino = publicando.publish_target()
    assert gcs_client.object_name(destino, "documents/x/y.pdf") == "entregables/documents/x/y.pdf"


def test_cada_entregable_declara_su_content_type():
    from app.services.gcs_client import content_type_for

    assert content_type_for("a/b.pdf") == "application/pdf"
    assert content_type_for("a/b.xlsx").startswith("application/vnd.openxml")
    assert content_type_for("a/b.json") == "application/json"
    assert content_type_for("a/sin_extension") == "application/octet-stream"


# --- BigQuery -------------------------------------------------------------


def test_una_tabla_vieja_recibe_las_columnas_nuevas(monkeypatch, publicando):
    """Una tabla creada con el esquema de ayer rechazaría la fila de hoy."""
    from google.cloud import bigquery

    from app.services import bigquery_client

    bigquery_client.reset_clients()
    destino = publicando.publish_target()

    class TablaVieja:
        schema = [
            bigquery.SchemaField(nombre, tipo, mode=modo)
            for nombre, tipo, modo in bigquery_client.TABLE_SCHEMA
            if nombre != "currency"
        ]

    actualizada: dict = {}

    class ClienteFalso:
        project = "mi-proyecto"

        def get_table(self, full_id):
            actualizada["tabla"] = full_id
            return TablaVieja()

        def update_table(self, tabla, campos):
            actualizada["columnas"] = [c.name for c in tabla.schema]
            actualizada["campos"] = campos

        def insert_rows_json(self, full_id, filas):
            return []

    monkeypatch.setattr(bigquery_client, "_client", lambda target: ClienteFalso())
    bigquery_client.insert_extraction_row(destino, {"document_id": "x"})

    assert actualizada["tabla"] == "mi-proyecto.document_extraction.extractions"
    assert "currency" in actualizada["columnas"]
    assert actualizada["campos"] == ["schema"]
    bigquery_client.reset_clients()


def test_una_fila_rechazada_por_bigquery_es_un_error(monkeypatch, publicando):
    from app.services import bigquery_client

    bigquery_client.reset_clients()

    class ClienteFalso:
        project = "p"

        def get_table(self, full_id):
            class Tabla:
                schema = []

            return Tabla()

        def update_table(self, tabla, campos):
            pass

        def insert_rows_json(self, full_id, filas):
            return [{"errors": ["no such field"]}]

    monkeypatch.setattr(bigquery_client, "_client", lambda target: ClienteFalso())
    with pytest.raises(RuntimeError, match="no such field"):
        bigquery_client.insert_extraction_row(publicando.publish_target(), {"document_id": "x"})
    bigquery_client.reset_clients()
