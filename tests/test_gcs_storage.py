"""`GCSStorage` sobre un bucket en memoria.

El doble imita lo poco del SDK de `google-cloud-storage` que el adaptador usa:
`blob()`, `get_blob()`, `list_blobs()` y los métodos de subida, bajada y
borrado de cada blob.
"""

from __future__ import annotations

import pytest

from app.domain.errors import ArtifactNotFoundError
from app.storage.gcs import GCSStorage


class _Blob:
    def __init__(self, bucket: _Bucket, name: str) -> None:
        self._bucket = bucket
        self.name = name

    @property
    def size(self) -> int:
        return len(self._bucket.objects[self.name][0])

    def exists(self) -> bool:
        return self.name in self._bucket.objects

    def upload_from_string(self, data, content_type=None):
        self._bucket.objects[self.name] = (bytes(data), content_type)

    def upload_from_filename(self, filename, content_type=None):
        with open(filename, "rb") as handle:
            self.upload_from_string(handle.read(), content_type)

    def download_as_bytes(self) -> bytes:
        return self._bucket.objects[self.name][0]

    def delete(self):
        self._bucket.objects.pop(self.name)


class _Bucket:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str | None]] = {}

    def blob(self, name):
        return _Blob(self, name)

    def get_blob(self, name):
        return _Blob(self, name) if name in self.objects else None

    def list_blobs(self, prefix=""):
        return [_Blob(self, name) for name in sorted(self.objects) if name.startswith(prefix)]


@pytest.fixture
def bucket():
    return _Bucket()


@pytest.fixture
def gcs(bucket):
    return GCSStorage(bucket, prefix="artefactos")


def test_escribe_y_lee_bajo_el_prefijo(gcs, bucket):
    gcs.write_json("documents/abc/meta.json", {"pages": 3})

    assert "artefactos/documents/abc/meta.json" in bucket.objects
    assert bucket.objects["artefactos/documents/abc/meta.json"][1] == "application/json"
    assert gcs.read_json("documents/abc/meta.json") == {"pages": 3}
    assert gcs.size("documents/abc/meta.json") > 0


def test_una_clave_ausente_es_un_error_de_dominio(gcs):
    with pytest.raises(ArtifactNotFoundError):
        gcs.read_bytes("documents/no/existe.json")
    assert gcs.read_json_or_none("documents/no/existe.json") is None
    assert gcs.exists("documents/no/existe.json") is False


def test_list_devuelve_claves_sin_el_prefijo_del_bucket(gcs):
    gcs.write_bytes("documents/abc/search/x/result.json", b"{}")
    gcs.write_bytes("documents/abc/meta.json", b"{}")
    gcs.write_bytes("documents/abcd/meta.json", b"{}")

    assert list(gcs.list("documents/abc")) == [
        "documents/abc/meta.json",
        "documents/abc/search/x/result.json",
    ]


def test_delete_prefix_borra_solo_lo_suyo(gcs):
    gcs.write_bytes("documents/abc/a.json", b"1")
    gcs.write_bytes("documents/abc/b/c.json", b"2")
    gcs.write_bytes("documents/abcd/a.json", b"3")

    assert gcs.delete_prefix("documents/abc") == 2
    assert gcs.exists("documents/abcd/a.json")


def test_import_file_con_move_borra_el_temporal(gcs, tmp_path):
    origen = tmp_path / "corte.pdf"
    origen.write_bytes(b"%PDF-1.7")

    gcs.import_file(origen, "documents/abc/search/x/corte.pdf", move=True)

    assert not origen.exists()
    assert gcs.read_bytes("documents/abc/search/x/corte.pdf") == b"%PDF-1.7"


def test_materialize_baja_a_un_temporal_y_lo_limpia(gcs):
    gcs.write_bytes("documents/abc/source.pdf", b"%PDF-1.7 contenido")

    with gcs.materialize("documents/abc/source.pdf") as ruta:
        assert ruta.read_bytes() == b"%PDF-1.7 contenido"
        assert ruta.suffix == ".pdf"

    assert not ruta.exists()


def test_rechaza_claves_que_escapan_del_prefijo(gcs):
    with pytest.raises(ValueError):
        gcs.write_bytes("../fuera.json", b"x")


def test_el_pipeline_entero_corre_sobre_el_bucket(settings, fake_ai, bucket, pdf_bytes, tmp_path):
    """El disco del servidor queda como espacio de trabajo: todo vive en el bucket."""
    from app.services import pipeline
    from app.services.context import Progress, StageContext
    from app.services.documents import register_document
    from app.storage.workspace import Workspace

    storage = GCSStorage(bucket)
    ctx = StageContext(
        settings=settings,
        storage=storage,
        workspace=Workspace(tmp_path / "ws"),
        progress=Progress.none(),
    )
    document_id = register_document(storage, pdf_bytes, "eeff-2024.pdf").document_id

    outcome = pipeline.run_pipeline(ctx, document_id, "situacion_financiera")

    assert outcome.quality.status == "Aprobado"
    assert f"documents/{document_id}/extraction/situacion_financiera/result.json" in bucket.objects
    assert f"documents/{document_id}/search/situacion_financiera/corte.pdf" in bucket.objects
