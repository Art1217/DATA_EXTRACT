"""Corridas reconstruidas a partir de los jobs.

No hay tabla de lotes: el identificador viaja en `payload.batch` de cada job y
la corrida se recompone leyendo eso. Estas pruebas fijan esa reconstrucción,
que es lo que permite volver sobre una corrida de hace días.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.jobs.models import Job, JobStatus, JobType
from app.jobs.store import FileJobStore
from app.services import batches


@pytest.fixture
def store(storage) -> FileJobStore:
    return FileJobStore(storage)


def _job(
    store,
    batch,
    *,
    document_id,
    status=JobStatus.SUCCEEDED,
    tipo=JobType.PIPELINE,
    intento=1,
    **extra,
):
    """Un job del lote. `intento` distingue reejecuciones del mismo documento."""
    job = Job(
        job_id=f"{batch}-{document_id}-{tipo.value}-{intento}",
        type=tipo,
        status=status,
        document_id=document_id,
        analysis_type=extra.pop("analysis_type", "resultados"),
        payload={"batch": batch},
        **extra,
    )
    store.save(job)
    return job


def _momento(texto: str) -> datetime:
    """`20260811-1521` como fecha real: el orden de las corridas sale de aquí,
    no del orden en que se hayan escrito los jobs."""
    return datetime.strptime(texto, "%Y%m%d-%H%M").replace(tzinfo=UTC)


def test_agrupa_los_jobs_por_su_etiqueta_de_lote(store, storage):
    _job(store, "20260811-1521", document_id="doc1", created_at=_momento("20260811-1521"))
    _job(store, "20260811-1521", document_id="doc2", created_at=_momento("20260811-1521"))
    _job(store, "20260810-0900", document_id="doc3", created_at=_momento("20260810-0900"))

    corridas = batches.list_batches(store, storage)

    assert [c["batch"] for c in corridas] == ["20260811-1521", "20260810-0900"], (
        "la más nueva primero"
    )
    assert corridas[0]["documents"] == 2


def test_cuenta_completados_y_fallidos(store, storage):
    _job(store, "L1", document_id="doc1")
    _job(store, "L1", document_id="doc2", status=JobStatus.FAILED, error="reventó")

    corrida = batches.list_batches(store, storage)[0]

    assert corrida["succeeded"] == 1
    assert corrida["failed"] == 1


def test_un_documento_relanzado_cuenta_una_vez_por_su_ultimo_intento(store, storage):
    """Una corrida se cuenta por documentos, no por intentos.

    Contando jobs, un lote de 18 documentos del que se relanzaron 9 diría
    "18 completados y 9 con error": 27 desenlaces para 18 documentos, y una
    corrida entera correcta pintada de rojo.
    """
    _job(
        store,
        "L1",
        document_id="doc1",
        status=JobStatus.FAILED,
        error="se cortó",
        created_at=_momento("20260812-1009"),
    )
    _job(
        store,
        "L1",
        document_id="doc1",
        status=JobStatus.SUCCEEDED,
        intento=2,
        created_at=_momento("20260812-1836"),
    )

    corrida = batches.list_batches(store, storage)[0]

    assert corrida["documents"] == 1
    assert corrida["succeeded"] == 1
    assert corrida["failed"] == 0, "el intento fallido sigue en el historial, no en el recuento"


def test_el_detalle_muestra_el_ultimo_intento(store, storage):
    _job(
        store,
        "L1",
        document_id="doc1",
        status=JobStatus.FAILED,
        error="se cortó",
        created_at=_momento("20260812-1009"),
    )
    _job(
        store,
        "L1",
        document_id="doc1",
        status=JobStatus.SUCCEEDED,
        intento=2,
        created_at=_momento("20260812-1836"),
    )

    items = batches.batch_detail(store, storage, "L1")["items"]

    assert len(items) == 1
    assert items[0]["status"] == "succeeded"
    assert items[0]["error"] is None


def test_la_fecha_de_la_corrida_es_la_del_primer_intento(store, storage):
    """Relanzar no cambia cuándo se lanzó la corrida."""
    _job(
        store,
        "L1",
        document_id="doc1",
        status=JobStatus.FAILED,
        created_at=_momento("20260812-1009"),
    )
    _job(store, "L1", document_id="doc1", intento=2, created_at=_momento("20260812-1836"))

    corrida = batches.list_batches(store, storage)[0]

    assert corrida["created_at"].startswith("2026-08-12T10:09")


def test_los_jobs_auxiliares_no_cuentan_como_documentos(store, storage):
    """Empaquetar lleva la etiqueta del lote pero no es un documento."""
    _job(store, "L1", document_id="doc1")
    _job(store, "L1", document_id=None, tipo=JobType.BUNDLE_BATCH, analysis_type=None)

    corrida = batches.list_batches(store, storage)[0]

    assert corrida["documents"] == 1


def test_un_job_sin_lote_no_inventa_una_corrida(store, storage):
    job = Job(job_id="suelto", type=JobType.PIPELINE, document_id="doc1", payload={})
    store.save(job)

    assert batches.list_batches(store, storage) == []


def test_el_detalle_trae_una_fila_por_documento(store, storage, pdf_bytes):
    from app.services.documents import register_document

    meta = register_document(storage, pdf_bytes, "eeff-2024.pdf")
    _job(
        store,
        "L1",
        document_id=meta.document_id,
        result={"extraction": {"row_count": 7}, "quality": {"status": "Aprobado"}},
    )

    detalle = batches.batch_detail(store, storage, "L1")

    fila = detalle["items"][0]
    assert fila["filename"] == "eeff-2024.pdf"
    assert fila["row_count"] == 7
    assert fila["quality_status"] == "Aprobado"


def test_el_zip_se_reporta_solo_si_existe(store, storage):
    from app.storage import keys

    _job(store, "L1", document_id="doc1")
    assert batches.list_batches(store, storage)[0]["bundle_ready"] is False

    storage.write_bytes(keys.batch_bundle("L1"), b"PK")

    assert batches.list_batches(store, storage)[0]["bundle_ready"] is True


def test_los_analisis_de_una_corrida_salen_de_sus_jobs(store, storage):
    """Es lo que acota el ZIP a la corrida y no a todo el historial del documento."""
    _job(store, "L1", document_id="doc1", analysis_type="resultados")
    _job(store, "L2", document_id="doc1", analysis_type="inventarios")

    assert batches.batch_analyses(store, "L1") == {"doc1": ["resultados"]}
    assert batches.batch_analyses(store, "L2") == {"doc1": ["inventarios"]}


def test_el_modo_automatico_declara_todos_sus_analisis(store, storage):
    """Un job de `auto` cubre varios análisis del mismo documento."""
    _job(
        store,
        "L1",
        document_id="doc1",
        tipo=JobType.AUTO,
        analysis_type=None,
        result={"results": [{"analysis_type": "resultados"}, {"analysis_type": "inventarios"}]},
    )

    assert batches.batch_analyses(store, "L1") == {"doc1": ["inventarios", "resultados"]}


def test_una_corrida_desconocida_devuelve_vacio_sin_error(store, storage):
    detalle = batches.batch_detail(store, storage, "no-existe")

    assert detalle["items"] == []
    assert detalle["documents"] == 0
