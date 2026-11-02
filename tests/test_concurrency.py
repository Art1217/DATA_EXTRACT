"""Aislamiento entre jobs simultáneos.

El pipeline anterior escribía las páginas renderizadas en `Temp/page_<n>.jpg`,
con la clave formada solo por el número de página, y al terminar borraba todos
los JPG del directorio compartido. Con un documento a la vez no se notaba; con
dos jobs concurrentes cada uno sobrescribía y borraba las imágenes del otro, de
modo que el modelo podía recibir páginas del documento equivocado.

Estas pruebas fijan el invariante que lo impide: cada ejecución renderiza en su
propio directorio y solo limpia el suyo.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf

from app.adapters.preprocessing import preprocess_pdf_page
from app.storage.workspace import purge_stale, workspace
from tests.conftest import build_pdf, wait_for_job


def test_preprocess_escribe_en_el_directorio_indicado(tmp_path, settings):
    """Sin `out_dir`, dos documentos colisionarían en el mismo `page_0.jpg`."""
    propio = tmp_path / "mio"
    propio.mkdir()

    with pymupdf.open(stream=build_pdf(["hola"]), filetype="pdf") as documento:
        ruta = Path(preprocess_pdf_page(documento, 0, dpi=72, out_dir=propio))

    assert ruta.parent == propio
    assert not list(settings.temp_dir.glob("page_*.jpg"))


def test_dos_workspaces_simultaneos_no_comparten_directorio(settings):
    with (
        workspace(settings.temp_dir, prefix="a") as primero,
        workspace(settings.temp_dir, prefix="b") as segundo,
    ):
        assert primero.root != segundo.root

        (primero.root / "page_0.jpg").write_bytes(b"A")
        (segundo.root / "page_0.jpg").write_bytes(b"B")

        assert (primero.root / "page_0.jpg").read_bytes() == b"A"
        assert (segundo.root / "page_0.jpg").read_bytes() == b"B"


def test_al_salir_solo_se_borra_el_workspace_propio(settings):
    with workspace(settings.temp_dir, prefix="vive") as superviviente:
        marca = superviviente.root / "page_0.jpg"
        marca.write_bytes(b"A")

        with workspace(settings.temp_dir, prefix="efimero") as pasajero:
            (pasajero.root / "page_0.jpg").write_bytes(b"B")
            efimero = pasajero.root

        assert not efimero.exists()
        assert marca.exists(), "un workspace no debe borrar el de otro job"


def test_purge_stale_respeta_los_workspaces_recientes(settings):
    with workspace(settings.temp_dir, prefix="reciente") as espacio:
        assert purge_stale(settings.temp_dir, older_than_hours=1.0) == 0
        assert espacio.root.exists()


def test_dos_busquedas_concurrentes_usan_directorios_distintos(settings, monkeypatch):
    """Dos documentos procesados a la vez no comparten imágenes intermedias.

    Una barrera obliga a que ambas búsquedas estén dentro de la etapa al mismo
    tiempo: sin ella la prueba pasaría aunque los jobs se ejecutaran en serie.
    """
    from fastapi.testclient import TestClient

    from app.config import get_settings

    monkeypatch.setenv("JOB_WORKERS", "2")
    get_settings.cache_clear()
    get_settings().ensure_directories()

    directorios_por_llamada: list[set[str]] = []
    cerrojo = threading.Lock()
    ambos_dentro = threading.Barrier(2, timeout=30)

    def fake_search(image_data, prompt_path=None, entity_context=None):
        directorios = {str(Path(ruta).parent) for _, ruta in image_data}
        with cerrojo:
            directorios_por_llamada.append(directorios)
        ambos_dentro.wait()
        return {
            "relevant_sections": [
                {"page_number": page, "section_title": "Resultados"} for page, _ in image_data
            ]
        }

    monkeypatch.setattr("app.services.search.search_pages_block", fake_search)

    from app.main import app

    with TestClient(app) as client:
        documentos = [
            client.post(
                "/v1/documents",
                files={
                    "file": (
                        f"doc{indice}.pdf",
                        build_pdf([f"documento {indice} " + "x" * 300]),
                        "application/pdf",
                    )
                },
            ).json()["document_id"]
            for indice in range(2)
        ]

        with ThreadPoolExecutor(max_workers=2) as ejecutor:
            peticiones = [
                ejecutor.submit(
                    client.post,
                    f"/v1/documents/{document_id}/search",
                    json={"analysis_type": "resultados", "use_identity": False},
                )
                for document_id in documentos
            ]
            jobs = [peticion.result().json()["job_id"] for peticion in peticiones]

        resultados = [wait_for_job(client, job_id, timeout=60) for job_id in jobs]

    get_settings.cache_clear()

    assert all(resultado["status"] == "succeeded" for resultado in resultados)
    assert len(directorios_por_llamada) == 2, "ambos jobs debieron entrar a la etapa"

    primero, segundo = directorios_por_llamada
    assert primero.isdisjoint(segundo), (
        "dos jobs concurrentes renderizaron en el mismo directorio: sus imágenes se pisarían"
    )
