"""OCR con Document AI y reconstrucción de un PDF con capa de texto.

Se usa para los PDFs escaneados y para los análisis que prefieren las tablas
ya reconstruidas por Document AI. El import de `google.cloud.documentai` es
perezoso: el backend arranca sin la librería mientras no se configure un
procesador (`DOCAI_PROCESSOR_ID`).
"""

import os
import tempfile

import pymupdf

from app.config import get_settings

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _get_client():
    """Cliente de Document AI con la service account configurada o las ADC."""
    import google.auth
    from google.cloud import documentai

    settings = get_settings()
    if settings.credentials_path:
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            str(settings.credentials_path), scopes=_SCOPES
        )
        project = settings.gcp_project_id or creds.project_id
    else:
        creds, project = google.auth.default(scopes=_SCOPES)
        project = settings.gcp_project_id or project
    opts = {"api_endpoint": f"{settings.docai_location}-documentai.googleapis.com"}
    return documentai.DocumentProcessorServiceClient(
        credentials=creds, client_options=opts
    ), project


def process_with_document_ai(pdf_path, processor_id=None):
    """Envía un PDF al procesador Document OCR y devuelve el objeto Document.

    Usa `imageless_mode` para extender el límite síncrono de 15 a 30 páginas.
    """
    from google.cloud import documentai

    settings = get_settings()
    client, project_id = _get_client()
    processor_name = client.processor_path(
        project_id, settings.docai_location, processor_id or settings.docai_processor_id
    )

    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    raw_document = documentai.RawDocument(content=pdf_data, mime_type="application/pdf")
    request = documentai.ProcessRequest(
        name=processor_name, raw_document=raw_document, imageless_mode=True
    )
    result = client.process_document(request=request)
    return result.document


def _cell_text(cell, full_text):
    text = ""
    for seg in cell.layout.text_anchor.text_segments:
        start = int(seg.start_index)
        end = int(seg.end_index)
        text += full_text[start:end]
    return text.strip()


def _layout_box(layout, page_width, page_height):
    """
    Convierte el bounding_poly de un Layout (normalizado o absoluto) a un
    rect (x0, y0, x1, y1) en puntos PDF, con origen arriba-izquierda.
    Devuelve None si no hay bounding box utilizable.
    """
    poly = layout.bounding_poly
    if poly.normalized_vertices:
        xs = [v.x * page_width for v in poly.normalized_vertices]
        ys = [v.y * page_height for v in poly.normalized_vertices]
    elif poly.vertices:
        xs = [v.x for v in poly.vertices]
        ys = [v.y for v in poly.vertices]
    else:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def build_searchable_pdf(document, pdf_path):
    """
    Genera una copia del PDF con una capa de texto invisible alineada a los
    tokens detectados por Document AI, sobre la imagen escaneada original
    (no se re-renderiza nada). El resultado se comporta como un PDF digital:
    el texto es seleccionable/extraíble y la página visual no cambia.

    Devuelve la ruta del PDF temporal generado (debe eliminarse tras usarse).
    Lanza la excepción si falla — el llamador decide el fallback.
    """
    full_text = document.text
    doc = pymupdf.open(pdf_path)

    for page_data in document.pages:
        page_index = page_data.page_number - 1
        if page_index < 0 or page_index >= doc.page_count:
            continue
        page = doc[page_index]
        page_width, page_height = page.rect.width, page.rect.height

        for token in page_data.tokens:
            text = _cell_text(token, full_text)
            if not text:
                continue
            box = _layout_box(token.layout, page_width, page_height)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            fontsize = max((y1 - y0) * 0.85, 1)
            try:
                page.insert_text(
                    (x0, y1 - (y1 - y0) * 0.15),
                    text,
                    fontsize=fontsize,
                    render_mode=3,  # invisible
                )
            except Exception:
                continue

    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    doc.save(tmp_path, garbage=4, deflate=True, clean=True)
    doc.close()
    return tmp_path


def clean_doc_ai_result(document):
    """
    Extrae solo texto y tablas del Document de Document AI.
    Descarta bounding boxes, tokens, confianzas y bytes de imagen.
    Devuelve un dict slim apto para enviar como texto a Gemini.
    """
    full_text = document.text
    pages_data = []

    for page in document.pages:
        # Tablas detectadas
        tables = []
        for table in page.tables:
            header_rows = [
                [_cell_text(c, full_text) for c in row.cells] for row in table.header_rows
            ]
            body_rows = [[_cell_text(c, full_text) for c in row.cells] for row in table.body_rows]
            if header_rows or body_rows:
                tables.append({"header_rows": header_rows, "body_rows": body_rows})

        # Bloques de texto libre (sin bounding boxes)
        text_blocks = []
        for block in page.blocks:
            block_text = ""
            for seg in block.layout.text_anchor.text_segments:
                start = int(seg.start_index)
                end = int(seg.end_index)
                block_text += full_text[start:end]
            if block_text.strip():
                text_blocks.append(block_text.strip())

        pages_data.append(
            {
                "page_number": page.page_number,
                "tables": tables,
                "text_blocks": text_blocks,
            }
        )

    return {"pages": pages_data}
