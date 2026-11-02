"""Alta y consulta de documentos."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.deps import SettingsDep, StorageDep
from app.api.schemas import DocumentDetailResponse, DocumentResponse
from app.domain.errors import InvalidDocumentError
from app.services import documents
from app.storage import keys

router = APIRouter(prefix="/v1/documents", tags=["documentos"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    storage: StorageDep, settings: SettingsDep, file: UploadFile = File(...)
) -> DocumentResponse:
    """Sube un PDF y devuelve su identificador.

    El identificador es el hash del contenido: volver a subir el mismo archivo
    devuelve el mismo documento junto con todo lo ya procesado, en vez de
    duplicar el trabajo.
    """
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"El archivo supera el máximo de {settings.max_upload_mb} MB.",
        )
    if not content:
        raise InvalidDocumentError("El archivo está vacío.")

    meta = documents.register_document(storage, content, file.filename or "documento.pdf")
    return DocumentResponse.model_validate(meta, from_attributes=True)


@router.get("", response_model=list[DocumentResponse])
def list_documents(storage: StorageDep) -> list[DocumentResponse]:
    return [
        DocumentResponse.model_validate(meta, from_attributes=True)
        for meta in documents.list_documents(storage)
    ]


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(document_id: str, storage: StorageDep) -> DocumentDetailResponse:
    """Ficha del documento junto con el inventario de etapas ya ejecutadas."""
    meta = documents.get_document(storage, document_id)
    return DocumentDetailResponse(
        **meta.model_dump(),
        stages=documents.inventory(storage, document_id).model_dump(),
    )


@router.get("/{document_id}/file")
def download_document(document_id: str, storage: StorageDep) -> StreamingResponse:
    meta = documents.get_document(storage, document_id)
    return StreamingResponse(
        storage.open(keys.source_pdf(document_id)),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{meta.filename}"'},
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, storage: StorageDep) -> None:
    """Borra el documento y todos sus artefactos derivados."""
    documents.delete_document(storage, document_id)
