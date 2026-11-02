"""Ingesta y consulta de documentos.

El identificador de un documento es el hash de su contenido. Eso da
deduplicación sin mantener un índice aparte: volver a subir el mismo PDF
devuelve el mismo `document_id` y, con él, todos los artefactos ya calculados.
Como efecto secundario, quien vuelva a subir un archivo idéntico con otro
nombre reutiliza el registro existente y conserva el nombre original; el
contenido es lo que identifica al documento, no cómo se llame el archivo.
"""

from __future__ import annotations

import hashlib
import logging

import pymupdf

from app.domain.errors import DocumentNotFoundError, InvalidDocumentError
from app.domain.models import DocumentMeta, StageInventory
from app.storage import keys
from app.storage.base import Storage

logger = logging.getLogger(__name__)

_ID_LENGTH = 32
_SCAN_SAMPLE_PAGES = 5
_MIN_CHARS_PER_PAGE = 50


def compute_document_id(content: bytes) -> tuple[str, str]:
    """Devuelve `(document_id, sha256)` para el contenido dado."""
    digest = hashlib.sha256(content).hexdigest()
    return digest[:_ID_LENGTH], digest


def register_document(storage: Storage, content: bytes, filename: str) -> DocumentMeta:
    """Guarda el PDF y su ficha. Si ya existía, devuelve la ficha existente."""
    document_id, sha256 = compute_document_id(content)

    existing = storage.read_json_or_none(keys.document_meta(document_id))
    if existing is not None:
        _restore_source(storage, document_id, content)
        logger.info("Documento ya registrado, se reutiliza: %s", document_id)
        return DocumentMeta.model_validate(existing)

    pages, is_scanned = _inspect_pdf(content)
    meta = DocumentMeta(
        document_id=document_id,
        filename=filename,
        sha256=sha256,
        pages=pages,
        size_bytes=len(content),
        is_scanned=is_scanned,
    )
    storage.write_bytes(keys.source_pdf(document_id), content)
    storage.write_json(keys.document_meta(document_id), meta.model_dump(mode="json"))
    logger.info("Documento registrado: %s (%s, %d páginas)", document_id, filename, pages)
    return meta


def _restore_source(storage: Storage, document_id: str, content: bytes) -> None:
    """Repone el PDF original si la liberación de disco se lo llevó.

    Un documento puede estar registrado y **sin su PDF**: al liberar espacio
    se borra `source.pdf` y se conserva la ficha, que es lo que mantiene
    reconocible al documento. Dedupicar solo por la ficha daba entonces una
    subida que no hacía nada, y el error aparecía mucho después —«artefacto no
    encontrado: source.pdf»— al intentar procesarlo, con el usuario mirando el
    archivo que acababa de subir.

    Como el identificador es el hash del contenido, lo que llega aquí es
    necesariamente el mismo PDF: reponerlo no puede mezclar dos documentos.
    """
    if storage.exists(keys.source_pdf(document_id)):
        return
    storage.write_bytes(keys.source_pdf(document_id), content)
    logger.info("PDF original repuesto para %s: vuelve a estar completo.", document_id)


def get_document(storage: Storage, document_id: str) -> DocumentMeta:
    payload = storage.read_json_or_none(keys.document_meta(document_id))
    if payload is None:
        raise DocumentNotFoundError(document_id)
    return DocumentMeta.model_validate(payload)


def list_documents(storage: Storage) -> list[DocumentMeta]:
    """Documentos registrados, del más reciente al más antiguo."""
    documents: list[DocumentMeta] = []
    for key in storage.list(keys.DOCUMENTS):
        if not key.endswith("/meta.json"):
            continue
        payload = storage.read_json_or_none(key)
        if payload is None:
            continue
        try:
            documents.append(DocumentMeta.model_validate(payload))
        except ValueError as exc:
            logger.warning("Ficha de documento ilegible en %s: %s", key, exc)
    documents.sort(key=lambda doc: doc.created_at, reverse=True)
    return documents


def delete_document(storage: Storage, document_id: str) -> int:
    """Borra el documento y todos sus artefactos. Devuelve cuántos archivos."""
    if not storage.exists(keys.document_meta(document_id)):
        raise DocumentNotFoundError(document_id)
    return storage.delete_prefix(keys.document_root(document_id))


def manifest(storage: Storage, document_id: str, analysis_types: list[str]) -> str:
    """Ficha completa del documento en JSON: qué es, de quién y qué se le hizo.

    Acompaña al documento en el ZIP y en el bucket. Es lo único que lleva la
    identidad —compañía, cuenca, formación— fuera de las filas del Excel, y lo
    que permite saber qué informe produjo una carpeta sin abrir sus archivos.
    """
    import json

    return json.dumps(
        {
            "document": storage.read_json_or_none(keys.document_meta(document_id)),
            "identity": storage.read_json_or_none(keys.identity(document_id)),
            "plan": storage.read_json_or_none(keys.plan(document_id)),
            "analysis_types": analysis_types,
        },
        indent=2,
        ensure_ascii=False,
    )


def inventory(storage: Storage, document_id: str) -> StageInventory:
    """Qué etapas ya se ejecutaron sobre el documento.

    Se deduce de las claves presentes en el storage, no de un registro
    aparte: el artefacto es la prueba de que la etapa corrió, así que no
    pueden desincronizarse.
    """
    root = keys.document_root(document_id)
    present = set(storage.list(root))

    def stage_keys(segment: str) -> list[str]:
        """Análisis que tienen artefacto bajo `<root>/<segment>/<clave>/`."""
        prefix = f"{root}/{segment}/"
        found = {
            key[len(prefix) :].split("/", 1)[0]
            for key in present
            if key.startswith(prefix) and "/" in key[len(prefix) :]
        }
        return sorted(found)

    classified_groups = sorted(
        key.rsplit("/", 1)[-1].removeprefix("analisis_").removesuffix(".json")
        for key in present
        if "/classification/analisis_" in key
    )

    return StageInventory(
        identity=keys.identity(document_id) in present,
        classification_group=keys.classification_group(document_id) in present,
        classification_analysis=classified_groups,
        plan=keys.plan(document_id) in present,
        search=stage_keys("search"),
        extraction=stage_keys("extraction"),
        quality=stage_keys("quality"),
    )


def _inspect_pdf(content: bytes) -> tuple[int, bool]:
    """Cuenta páginas y decide si el PDF es escaneado.

    Escaneado = las primeras páginas casi no tienen texto extraíble, así que
    hay que pasarlo por OCR antes de extraer.
    """
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            pages = document.page_count
            if pages == 0:
                raise InvalidDocumentError("El PDF no tiene páginas.")
            sampled = min(_SCAN_SAMPLE_PAGES, pages)
            characters = sum(len(document[i].get_text().strip()) for i in range(sampled))
            is_scanned = (characters / sampled) < _MIN_CHARS_PER_PAGE
            return pages, is_scanned
    except InvalidDocumentError:
        raise
    except Exception as exc:
        raise InvalidDocumentError(f"No se pudo leer el PDF: {exc}") from exc
