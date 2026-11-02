"""Storage: aislamiento de claves, atomicidad y limpieza."""

from __future__ import annotations

import pytest

from app.domain.errors import ArtifactNotFoundError
from app.storage import keys
from app.storage.base import validate_key


@pytest.mark.parametrize(
    "clave",
    ["../secretos", "documentos/../../fuera", "/absoluta", "con\\backslash", "C:/unidad", ""],
)
def test_se_rechazan_claves_que_escapan_de_la_raiz(clave):
    """Parte de la clave viene de fuera: no puede convertirse en un salto de ruta."""
    with pytest.raises(ValueError):
        validate_key(clave)


def test_lectura_de_una_clave_ausente_es_un_error_de_dominio(storage):
    with pytest.raises(ArtifactNotFoundError):
        storage.read_bytes("documents/inexistente/meta.json")


def test_read_json_or_none_no_falla_ante_lo_que_no_existe(storage):
    assert storage.read_json_or_none("documents/inexistente/meta.json") is None


def test_la_escritura_no_deja_temporales(storage, settings):
    storage.write_json("documents/abc/meta.json", {"a": 1})
    assert storage.read_json("documents/abc/meta.json") == {"a": 1}
    sobrantes = list((settings.storage_dir / "documents" / "abc").glob("*.tmp"))
    assert sobrantes == []


def test_sobrescribir_reemplaza_por_completo(storage):
    storage.write_json("documents/abc/meta.json", {"largo": "x" * 500})
    storage.write_json("documents/abc/meta.json", {"corto": 1})
    assert storage.read_json("documents/abc/meta.json") == {"corto": 1}


def test_list_devuelve_claves_relativas_ordenadas(storage):
    storage.write_bytes("documents/abc/b.json", b"{}")
    storage.write_bytes("documents/abc/a.json", b"{}")
    storage.write_bytes("documents/otro/c.json", b"{}")
    assert list(storage.list("documents/abc")) == [
        "documents/abc/a.json",
        "documents/abc/b.json",
    ]


def test_delete_prefix_borra_todo_el_documento(storage):
    storage.write_bytes("documents/abc/source.pdf", b"%PDF")
    storage.write_bytes("documents/abc/extraction/resultados/result.json", b"{}")
    assert storage.delete_prefix("documents/abc") == 2
    assert not storage.exists("documents/abc/source.pdf")


def test_materialize_entrega_una_ruta_real(storage, tmp_path):
    storage.write_bytes("documents/abc/source.pdf", b"%PDF-1.7")
    with storage.materialize("documents/abc/source.pdf") as ruta:
        assert ruta.is_file()
        assert ruta.read_bytes() == b"%PDF-1.7"


def test_import_file_con_move_no_deja_el_original(storage, tmp_path):
    origen = tmp_path / "producido.pdf"
    origen.write_bytes(b"%PDF")
    storage.import_file(origen, "documents/abc/search/resultados/corte.pdf", move=True)
    assert not origen.exists()
    assert storage.exists("documents/abc/search/resultados/corte.pdf")


def test_las_claves_separan_el_corte_original_del_ocr():
    """El flujo anterior sobrescribía el recorte con la versión digitalizada."""
    assert keys.search_pdf("d", "resultados") != keys.search_pdf_searchable("d", "resultados")


def test_cada_analisis_tiene_su_propio_espacio_de_claves():
    assert keys.extraction_result("d", "resultados") != keys.extraction_result("d", "inventarios")


def test_dos_documentos_no_comparten_claves():
    assert keys.search_pdf("doc1", "resultados") != keys.search_pdf("doc2", "resultados")


def test_la_escritura_reintenta_si_el_reemplazo_falla_transitoriamente(storage, monkeypatch):
    """En Windows, `os.replace` puede fallar si otro hilo tiene el destino
    abierto en ese instante — se vio ocurrir de verdad bajo carga real de
    cientos de jobs escribiendo su progreso a la vez. Sin reintento, esa
    escritura se pierde en silencio."""
    import os as os_module

    real_replace = os_module.replace
    attempts = {"count": 0}

    def flaky_replace(src, dst):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError("simulando un handle abierto por otro hilo")
        return real_replace(src, dst)

    monkeypatch.setattr(os_module, "replace", flaky_replace)

    storage.write_json("documents/abc/meta.json", {"ok": True})

    assert attempts["count"] == 3
    assert storage.read_json("documents/abc/meta.json") == {"ok": True}


def test_la_escritura_no_reintenta_indefinidamente(storage, monkeypatch):
    """Si el fallo no es transitorio, debe propagarse — no colgarse en un
    bucle silencioso — y no dejar el archivo temporal huérfano."""
    import os as os_module

    def always_fails(src, dst):
        raise PermissionError("bloqueado de verdad")

    monkeypatch.setattr(os_module, "replace", always_fails)

    with pytest.raises(PermissionError):
        storage.write_json("documents/abc/meta.json", {"ok": True})

    sobrantes = list((storage.root / "documents" / "abc").glob("*.tmp"))
    assert sobrantes == []
