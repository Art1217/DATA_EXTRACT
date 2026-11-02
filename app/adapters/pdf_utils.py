import logging
import os
import tempfile

import pymupdf

from app.config import get_settings

logger = logging.getLogger(__name__)


def get_page_count(pdf_path):
    """Retorna el número total de páginas en un PDF."""
    try:
        doc = pymupdf.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count
    except Exception as e:
        logger.warning(f"Error abriendo PDF: {e}")
        return 0


def extract_pages(input_pdf, pages_list, output_filename=None, output_dir=None):
    """
    Extrae las páginas indicadas del PDF original y las guarda en un nuevo archivo.
    """
    if not output_dir:
        # Solo por si alguien llama sin destino: el backend siempre pasa el
        # workspace del job, que es lo que evita que dos documentos a la vez
        # se pisen los archivos intermedios.
        output_dir = str(get_settings().temp_dir)

    if not pages_list:
        logger.info("No hay páginas para extraer.")
        return None

    # Asegurar que las páginas estén ordenadas y sean únicas (1-indexed a 0-indexed)
    pages_to_extract = sorted(list(set([p - 1 for p in pages_list if p is not None and p > 0])))

    try:
        doc = pymupdf.open(input_pdf)
        new_doc = pymupdf.open()  # Documento destino

        # Copiar las páginas clasificadas una por una para asegurar compatibilidad
        for p in pages_to_extract:
            new_doc.insert_pdf(doc, from_page=p, to_page=p, annots=True)

        if not output_filename:
            base_name = os.path.basename(input_pdf).replace(".pdf", "")
            output_filename = f"corte_{base_name}.pdf"

        output_path = os.path.join(output_dir, output_filename)
        # Limpieza agresiva de basura (garbage=4) y compresión (deflate=True)
        # Esto asegura que el peso del archivo sea Proporcional a las páginas cortadas
        new_doc.save(output_path, garbage=4, deflate=True, clean=True)
        new_doc.close()
        doc.close()

        logger.info(f"Archivo generado exitosamente en: {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"Error extrayendo páginas: {e}")
        return None


def is_scanned_pdf(pdf_path, sample_pages=5, min_chars_per_page=50):
    """
    Devuelve True si el PDF es escaneado (sin texto extraíble).
    Muestrea las primeras sample_pages páginas y calcula el promedio de caracteres.
    """
    try:
        doc = pymupdf.open(pdf_path)
        pages_to_check = min(sample_pages, doc.page_count)
        total_chars = sum(len(doc[i].get_text().strip()) for i in range(pages_to_check))
        doc.close()
        avg_chars = total_chars / pages_to_check if pages_to_check > 0 else 0
        return avg_chars < min_chars_per_page
    except Exception as e:
        logger.warning(f"Error detectando tipo PDF: {e}")
        return False


def render_pages_as_images(pdf_path, dpi=120):
    """
    Renderiza cada página del PDF como JPEG bytes.
    Devuelve lista de bytes, una entrada por página.
    """
    images = []
    try:
        doc = pymupdf.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            images.append(pix.tobytes("jpeg", jpg_quality=85))
        doc.close()
    except Exception as e:
        logger.warning(f"Error renderizando imágenes de página: {e}")
    return images


def split_pdf_into_chunks(pdf_path, chunk_size=20, overlap=0):
    """
    Divide un PDF en sub-PDFs temporales de chunk_size páginas cada uno.
    Si overlap > 0, cada chunk (salvo el primero) repite las últimas
    `overlap` páginas del chunk anterior, para que una tabla cortada
    justo en el borde quede completa en al menos un chunk.
    Devuelve lista de rutas temporales (deben eliminarse después de usarse).
    """
    overlap = max(0, min(overlap, chunk_size - 1))
    chunk_paths = []
    try:
        doc = pymupdf.open(pdf_path)
        total = doc.page_count
        start = 0
        while start < total:
            end = min(start + chunk_size - 1, total - 1)
            chunk_doc = pymupdf.open()
            chunk_doc.insert_pdf(doc, from_page=start, to_page=end)
            fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)  # liberar el handle para que PyMuPDF pueda escribir en Windows
            chunk_doc.save(tmp_path, garbage=4, deflate=True, clean=True)
            chunk_doc.close()
            chunk_paths.append(tmp_path)
            if end >= total - 1:
                break
            start = end + 1 - overlap
        doc.close()
    except Exception as e:
        logger.warning(f"Error dividiendo PDF en chunks: {e}")
        for p in chunk_paths:
            if os.path.exists(p):
                os.remove(p)
        return []
    return chunk_paths


def save_digitized_pdf(original_pdf_path, digitized_paths, overlap=0):
    """
    Fusiona los PDFs digitalizados (uno por chunk, con capa de texto OCR invisible)
    y sobrescribe original_pdf_path, para que el archivo final descargable también
    tenga texto seleccionable — no solo la copia temporal usada para la extracción.

    Si falta la versión digitalizada de algún chunk (None), no toca el archivo
    original para evitar dejar un PDF incompleto o con páginas fuera de orden.
    Limpia los temporales recibidos en digitized_paths en cualquier caso.
    """
    valid = [p for p in digitized_paths if p]
    try:
        if not digitized_paths or len(valid) != len(digitized_paths):
            return
        merged = pymupdf.open()
        for i, p in enumerate(valid):
            with pymupdf.open(p) as d:
                from_page = min(overlap, d.page_count - 1) if i > 0 else 0
                merged.insert_pdf(d, from_page=from_page)
        merged.save(original_pdf_path, garbage=4, deflate=True, clean=True)
        merged.close()
    except Exception as e:
        logger.warning(f"No se pudo guardar el PDF digitalizado final: {e}")
    finally:
        for p in valid:
            if os.path.exists(p):
                os.remove(p)
