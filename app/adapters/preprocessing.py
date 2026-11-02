"""Render de páginas de PDF a imagen para enviarlas al modelo."""

import os

import cv2
import numpy as np

from app.config import get_settings


def preprocess_pdf_page(doc, page_num, dpi=None, out_dir=None):
    """Renderiza una página del PDF y la guarda como JPG.

    JPG en vez de PNG porque reduce el peso alrededor de un 80%, y la subida
    al modelo es el cuello de botella real.

    `out_dir` es el directorio donde se escribe la imagen. Cada job debe pasar
    el suyo: el nombre de archivo solo lleva el número de página, así que dos
    documentos que compartan directorio se sobrescriben mutuamente. Si se
    omite se usa el temporal global, únicamente por compatibilidad con los
    scripts antiguos de un solo documento.
    """
    settings = get_settings()
    render_dpi = dpi if dpi else settings.dpi
    target_dir = out_dir if out_dir else settings.temp_dir

    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=render_dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    os.makedirs(target_dir, exist_ok=True)
    temp_path = os.path.join(str(target_dir), f"page_{page_num}.jpg")
    cv2.imwrite(temp_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return temp_path
