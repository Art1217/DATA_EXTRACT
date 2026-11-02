"""Extracción de tablas a filas estructuradas y revisión de calidad con LLM.

Es la etapa cara: usa el modelo grande y procesa cada página del recorte. Por
eso gran parte de este módulo existe para no perder una respuesta ya pagada:

* los PDFs largos se parten en chunks con una página de solape, y entre chunks
  se reenvían las últimas filas extraídas para que una tabla partida no se
  duplique ni pierda filas en el borde;
* si el modelo responde vacío (agotó su presupuesto razonando), se reintenta
  página por página;
* si responde con JSON dañado, se repara antes de darlo por perdido;
* los escaneados pasan por Document AI, y el corte queda con capa de texto.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

from langsmith import traceable
from tenacity import retry, stop_after_attempt, wait_exponential

from app.adapters import llm
from app.adapters.pdf_utils import (
    get_page_count,
    is_scanned_pdf,
    render_pages_as_images,
    save_digitized_pdf,
    split_pdf_into_chunks,
)
from app.config import get_settings

logger = logging.getLogger(__name__)


def _build_continuity_anchor(rows, max_rows=None):
    """
    Construye una nota corta con las últimas filas extraídas del chunk anterior,
    para que el siguiente chunk no las repita y mantenga la secuencia de la tabla
    (concepto, período, numeración, etc.) en el borde entre chunks.
    Costo acotado: como máximo `max_rows` filas, no crece con el número de chunks.
    """
    if not rows:
        return None
    if max_rows is None:
        max_rows = get_settings().chunk_anchor_rows
    snippet = json.dumps(rows[-max_rows:], ensure_ascii=False)
    return (
        "\n\nCONTEXTO DEL CHUNK ANTERIOR (continuidad):\n"
        "Estas fueron las últimas filas extraídas del bloque de páginas inmediatamente anterior "
        "(puede haber 1 página de solape entre chunks):\n"
        f"{snippet}\n"
        "No repitas estas filas en tu respuesta. Si la tabla continúa desde aquí, "
        "mantén la secuencia de forma consistente con estas filas."
    )


EMPTY_RESPONSE_MARKER = "EMPTY_RESPONSE"

# Algunos prompts reservan un hueco para pegar el texto del informe. Aquí el
# informe nunca va incrustado: viaja como PDF (o como JSON de OCR + imágenes)
# en la misma petición, así que el hueco se sustituye por una nota que lo dice.
_HUECO_INFORME = "{{TEXTO_DEL_INFORME}}"
_NOTA_INFORME = "(El informe va adjunto a este mensaje.)"


def _read_prompt(prompt_path, context_note=None):
    with open(prompt_path, encoding="utf-8") as f:
        prompt_content = f.read().replace(_HUECO_INFORME, _NOTA_INFORME)
    if context_note:
        prompt_content += context_note
    return prompt_content


def _unwrap_rows(result, rows_key):
    """
    Saca las filas de la clave que las envuelve (`{"filas": [...]}`).
    Lo que no trae esa clave —una respuesta con error, o filas ya sueltas—
    se devuelve intacto, para que el llamador lo trate como siempre.
    """
    if not rows_key:
        return result
    if isinstance(result, dict):
        inner = result.get(rows_key)
        return inner if isinstance(inner, list) else result
    if isinstance(result, list):
        rows = []
        for item in result:
            inner = item.get(rows_key) if isinstance(item, dict) else None
            if isinstance(inner, list):
                rows.extend(inner)
            else:
                rows.append(item)
        return rows
    return result


def _parse_model_json(response, context_label=""):
    """
    Parsea la respuesta JSON de un modelo Gemini.
    Si response.text es None, el modelo devolvió la respuesta vacía — típicamente
    porque agotó su presupuesto de salida razonando internamente sobre un documento
    denso, sin llegar a escribir el JSON final. Es un caso distinto a un JSON
    truncado (que sí tiene texto, solo que incompleto): aquí no hay nada que
    repair_truncated_json pueda reparar, así que se marca con EMPTY_RESPONSE_MARKER
    para que el llamador decida (p. ej. reintentar con un documento más chico).
    """
    label = f" {context_label}" if context_label else ""
    text = response.text
    if text is None:
        logger.warning(
            f"Respuesta vacía del modelo{label} (finish_reason={response.finish_reason})."
        )
        return {"error": EMPTY_RESPONSE_MARKER, "finish_reason": str(response.finish_reason)}
    try:
        return json.loads(text)
    except Exception as e:
        logger.warning(
            f"Error parseando extracción JSON{label}: {e} — Intentando reparar JSON truncado..."
        )
        repaired = repair_truncated_json(text)
        if repaired is not None:
            logger.info("JSON reparado exitosamente.")
            return repaired
        logger.warning("No se pudo reparar el JSON.")
        return {"error": str(e), "raw": text}


def _find_last_complete_element_end(text):
    """
    Escanea un array JSON (texto que empieza en '[') y devuelve el índice
    (exclusivo) justo después del último elemento de nivel superior que
    cerró completamente — es decir, donde la profundidad de llaves/corchetes
    vuelve a 1 (justo dentro del array). Ignora llaves dentro de strings.
    Esto recupera filas completas tanto si el texto quedó truncado a media
    fila, como si el modelo agregó una llave de cierre sobrante al final
    (que llevaría la profundidad a 0, no a 1, y por eso se ignora sola).
    Devuelve None si no encuentra ningún elemento completo.
    """
    depth = 0
    in_string = False
    escape = False
    last_end = None
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 1:
                last_end = i + 1
    return last_end


# Nombre de campo al que el modelo le comió la comilla de apertura (o las dos):
#   "periodo": "2024",
#   valor": 744.03,      <-- esto
# Se ancla a principio de línea porque un JSON válido no puede tener un salto de
# línea crudo dentro de un string: lo que empieza una línea nunca es texto de un
# valor, así que no hay riesgo de tocar datos.
_CLAVE_SIN_COMILLA = re.compile(r'(?m)^(\s*)([A-Za-z_][A-Za-z0-9_]*)"?(\s*:)')


def _repair_property_names(text: str) -> str:
    """Devuelve el texto con las comillas de los nombres de campo repuestas."""
    return _CLAVE_SIN_COMILLA.sub(r'\1"\2"\3', text)


def _salvage_array_elements(text: str):
    """
    Último recurso para un array dañado por el medio: lo recorre elemento a
    elemento y se queda con los que parsean, saltándose los rotos.

    Existe porque el modelo estropea una fila suelta en medio de cientos
    buenas, y perder la extracción entera por una fila es mucho peor que
    entregarla sin esa fila. Devuelve None si no salva nada,
    para no hacer pasar por resultado lo que fue un fracaso completo.
    """
    inicio = text.find("[")
    if inicio == -1:
        return None

    decoder = json.JSONDecoder()
    filas = []
    descartadas = 0
    i = inicio + 1
    while i < len(text):
        while i < len(text) and text[i] in ", \t\r\n":
            i += 1
        if i >= len(text) or text[i] == "]":
            break
        try:
            obj, fin = decoder.raw_decode(text, i)
            filas.append(obj)
            i = fin
        except json.JSONDecodeError:
            siguiente = _end_of_element(text, i)
            if siguiente is None:
                break
            descartadas += 1
            i = siguiente

    if not filas:
        return None
    logger.warning(f"JSON dañado: se salvaron {len(filas)} filas y se descartaron {descartadas}.")
    return filas


def _end_of_element(text: str, start: int):
    """Índice justo después del elemento que empieza en `start`, aunque no parsee.

    Cuenta llaves y corchetes ignorando los que van dentro de strings, igual
    que `_find_last_complete_element_end`. Sirve para saltar por encima de un
    elemento roto y seguir con el siguiente.
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth <= 0:
                return i + 1
    return None


def repair_truncated_json(raw_text: str):
    """
    Intenta reparar un JSON (array u objeto) que llegó dañado del modelo.
    Estrategia, de menos a más destructiva:
      1. Repone las comillas de los nombres de campo mal escritos.
      2. Elimina lo que sobra después del último elemento de nivel superior
         que cerró completamente (ignora filas a medias y llaves sobrantes).
      3. Cierra correctamente el array o el objeto raíz.
      4. Como último recurso, salva las filas que parseen y descarta las rotas.
      5. Devuelve el objeto Python parseado, o None si aún falla.

    Los pasos 1 y 4 existen porque el modelo a veces escribe `valor":` en vez
    de `"valor":` en medio de una respuesta larga. Reparar solo la cola
    (truncamiento) dejaría ese defecto intacto y se perdería la extracción
    entera.
    """
    text = raw_text.strip()

    # Quitar bloque de código markdown si está presente
    text = re.sub(r"^```[\w]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    # Determinar si la raíz es array [ o dict {
    is_array = text.startswith("[")

    # Intentar parsear tal cual primero (puede que solo falte espacio)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Comillas de los nombres de campo. Se hace antes que nada porque no
    # descarta ni un dato: si era solo eso, se recupera la respuesta entera.
    con_claves = _repair_property_names(text)
    if con_claves != text:
        try:
            parsed = json.loads(con_claves)
            logger.warning("JSON reparado: nombres de campo sin comillas.")
            return parsed
        except json.JSONDecodeError:
            text = con_claves

    if is_array:
        end = _find_last_complete_element_end(text)
        if end is not None:
            truncated = text[:end].rstrip().rstrip(",")
            candidate = truncated + "\n]"
            try:
                parsed = json.loads(candidate)
                logger.warning(f"JSON truncado reparado: se recuperaron {len(parsed)} filas.")
                return parsed
            except json.JSONDecodeError:
                pass
        salvadas = _salvage_array_elements(text)
        if salvadas is not None:
            return salvadas
    else:
        # Para objetos raíz, intentar cerrar el objeto
        last_value_end = max(text.rfind('"'), text.rfind("}"), text.rfind("]"))
        if last_value_end != -1:
            candidate = text[: last_value_end + 1].rstrip().rstrip(",") + "\n}"
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    return None


def _generate_extraction(parts, prompt_path):
    settings = get_settings()
    return llm.generate(
        task="extract",
        prompt_name=Path(prompt_path).stem,
        model=settings.model_name_extract,
        parts=parts,
        temperature=0.1,
        max_output_tokens=65536,
    )


@traceable(run_type="llm", name="Extracción PDF")
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=15))
def extract_structured_data(pdf_path, prompt_path, context_note=None):
    """Envía el PDF (vectorial o digitalizado) para extraer sus tablas.

    context_note: texto corto opcional con el contexto de continuidad del chunk anterior.
    """
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    parts = [
        llm.blob(pdf_data, "application/pdf"),
        llm.text(_read_prompt(prompt_path, context_note)),
    ]
    return _parse_model_json(_generate_extraction(parts, prompt_path))


@traceable(run_type="llm", name="Extracción DocAI")
def extract_structured_data_from_text(
    cleaned_doc, prompt_path, page_images=None, context_note=None
):
    """Extrae enviando el contenido de Document AI (texto + tablas).

    Opcionalmente adjunta la imagen de cada página como contexto visual para
    tablas sin bordes, donde el JSON de OCR puede desalinear columnas.
    """
    prompt_content = _read_prompt(prompt_path, context_note)
    doc_text = json.dumps(cleaned_doc, ensure_ascii=False, indent=2)

    visual_note = ""
    if page_images:
        visual_note = (
            "\n\nNOTA — CONTEXTO VISUAL ACTIVO:\n"
            "Se adjuntan las imágenes de cada página del documento (en orden) junto con el JSON de OCR.\n"
            "El JSON puede tener errores de alineación en tablas sin líneas divisorias: "
            "encabezados y datos pueden estar desplazados entre columnas.\n"
            "Usa las imágenes como fuente de verdad para verificar qué valor corresponde "
            "a cada columna. Cuando haya discrepancia entre el JSON y la imagen, prioriza la imagen."
        )
    full_prompt = (
        f"{prompt_content}{visual_note}\n\n"
        f"DOCUMENTO A PROCESAR (texto y tablas extraídas por OCR):\n{doc_text}"
    )

    parts = [llm.blob(image, "image/jpeg") for image in page_images or []]
    parts.append(llm.text(full_prompt))
    return _parse_model_json(_generate_extraction(parts, prompt_path), context_label="(DocAI)")


def _extract_with_adaptive_split(pdf_path, prompt_path, context_note=None):
    """Extrae y, si la respuesta vino vacía, reintenta página por página.

    Una respuesta vacía significa que el modelo agotó su presupuesto de salida
    razonando sobre un documento denso. No se limita cuánto puede razonar: se
    reduce cuánto contenido procesa de una vez, y se mantiene la continuidad
    entre páginas con el mismo anchor que se usa entre chunks.
    """
    result = extract_structured_data(pdf_path, prompt_path, context_note=context_note)
    is_empty = isinstance(result, dict) and result.get("error") == EMPTY_RESPONSE_MARKER
    total_pages = get_page_count(pdf_path)

    if not is_empty or total_pages <= 1:
        return result

    logger.info(
        f"Respuesta vacía. Procesando {os.path.basename(pdf_path)} ({total_pages} pág.) página por página..."
    )
    page_paths = split_pdf_into_chunks(pdf_path, chunk_size=1, overlap=0)
    if len(page_paths) < 2:
        _remove_all(page_paths)
        return result  # no se pudo dividir más; devolver el fallo original

    all_rows = []
    anchor = context_note
    try:
        for p in page_paths:
            sub_result = extract_structured_data(p, prompt_path, context_note=anchor)
            sub_rows = _as_rows(sub_result)
            all_rows.extend(sub_rows)
            anchor = _build_continuity_anchor(sub_rows) or anchor
    finally:
        _remove_all(page_paths)
    return all_rows


def _as_rows(result):
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and "error" not in result:
        return [result]
    return []


def _remove_all(paths):
    for p in paths:
        if p and os.path.exists(p):
            os.remove(p)


def _docai_document(pdf_path, processor_id):
    """OCR de Document AI. Devuelve `(document, cleaned)` o `None` si falla."""
    try:
        from app.adapters.doc_ai import clean_doc_ai_result, process_with_document_ai

        logger.info(f"Document AI OCR: {os.path.basename(pdf_path)}")
        document = process_with_document_ai(pdf_path, processor_id=processor_id)
        return document, clean_doc_ai_result(document)
    except Exception as e:
        logger.warning(f"Document AI falló ({e}). Enviando PDF directo al modelo.")
        return None


def _extract_single(pdf_path, prompt_path, is_scanned, context_note=None, prefer_docai_json=False):
    """Extrae un único PDF (chunk o completo).

    Tres caminos, según el PDF y el análisis:

    * **Digital**: el PDF va directo al modelo.
    * **Escaneado**: Document AI hace el OCR, se genera un PDF con capa de
      texto invisible alineada a la imagen y ese PDF va al modelo, igual que
      uno digital.
    * **`prefer_docai_json`**: el modelo recibe las tablas ya reconstruidas
      por Document AI (que resuelve la columna de cada valor por su posición
      geométrica) más la imagen de cada página. Rinde mejor en tablas sin
      bordes verticales o con encabezados complejos, sea el PDF digital o no.

    Sin `DOCAI_PROCESSOR_ID`, o si Document AI falla, todo cae al camino digital.

    Devuelve `(resultado, cleaned_doc | None, searchable_pdf_path | None)`. El
    llamador es responsable de conservar o eliminar el PDF digitalizado.
    """
    settings = get_settings()
    processor_id = settings.docai_processor_id
    needs_docai = prefer_docai_json or is_scanned

    if not needs_docai:
        return _extract_with_adaptive_split(pdf_path, prompt_path, context_note), None, None
    if not processor_id:
        logger.warning("DOCAI_PROCESSOR_ID no configurado. Enviando PDF directo al modelo.")
        return _extract_with_adaptive_split(pdf_path, prompt_path, context_note), None, None

    ocr = _docai_document(pdf_path, processor_id)
    if ocr is None:
        return _extract_with_adaptive_split(pdf_path, prompt_path, context_note), None, None
    document, cleaned = ocr

    if prefer_docai_json:
        searchable_pdf_path = None
        if is_scanned:
            try:
                from app.adapters.doc_ai import build_searchable_pdf

                searchable_pdf_path = build_searchable_pdf(document, pdf_path)
            except Exception as e:
                logger.warning(
                    f"No se pudo generar el PDF digitalizado ({e}) (no afecta la extracción)."
                )
        page_images = render_pages_as_images(pdf_path, dpi=settings.scanned_img_dpi)
        result = extract_structured_data_from_text(
            cleaned, prompt_path, page_images=page_images, context_note=context_note
        )
        return result, cleaned, searchable_pdf_path

    # Escaneado: un solo archivo con visual + texto exacto, sin duplicar el
    # contenido en un canal JSON aparte.
    try:
        from app.adapters.doc_ai import build_searchable_pdf

        searchable_pdf_path = build_searchable_pdf(document, pdf_path)
        logger.info("PDF digitalizado generado (capa de texto OCR).")
        result = _extract_with_adaptive_split(searchable_pdf_path, prompt_path, context_note)
        return result, cleaned, searchable_pdf_path
    except Exception as e:
        logger.warning(
            f"No se pudo usar el PDF digitalizado ({e}). Usando JSON + imágenes de contexto."
        )
        page_images = render_pages_as_images(pdf_path, dpi=settings.scanned_img_dpi)
        return (
            extract_structured_data_from_text(
                cleaned, prompt_path, page_images=page_images, context_note=context_note
            ),
            cleaned,
            None,
        )


@traceable(run_type="chain", name="Extracción por chunks")
def extract_structured_data_chunked(pdf_path, prompt_path, prefer_docai_json=False, rows_key=None):
    """Extrae las filas de un PDF, partiéndolo en chunks si hace falta.

    - Escaneado → Document AI + modelo (CHUNK_SIZE_SCANNED páginas/chunk).
    - Digital → PDF directo al modelo (CHUNK_SIZE_DIGITAL páginas/chunk).
    - Entre chunks se reenvía un anchor acotado (últimas filas del chunk
      anterior) para evitar filas duplicadas o perdidas en tablas que cruzan
      el borde.
    - Si el PDF era escaneado, se sobrescribe `pdf_path` con la versión
      digitalizada, para que el archivo descargable también sea seleccionable.

    rows_key: clave bajo la que el prompt envuelve sus filas (ver `_unwrap_rows`).
    Devuelve `(filas, merged_cleaned_doc | None)`.
    """
    settings = get_settings()
    total_pages = get_page_count(pdf_path)
    is_scanned = is_scanned_pdf(pdf_path)
    if prefer_docai_json:
        # Cada página manda su JSON de tablas + imagen de contexto: con chunks
        # de 25 páginas la entrada se satura y la respuesta se degrada.
        chunk_size = settings.chunk_size_docai_json
    else:
        chunk_size = settings.chunk_size_scanned if is_scanned else settings.chunk_size_digital
    tipo = "escaneado" if is_scanned else "digital"
    logger.info(f"PDF {tipo} — {total_pages} páginas (chunks de {chunk_size}).")

    chunk_paths = []
    if total_pages > chunk_size:
        logger.info(
            f"Dividiendo en chunks de {chunk_size} páginas "
            f"(solape de {settings.chunk_overlap_pages} página/s)..."
        )
        chunk_paths = split_pdf_into_chunks(
            pdf_path, chunk_size=chunk_size, overlap=settings.chunk_overlap_pages
        )
        if not chunk_paths:
            logger.warning("No se pudo dividir el PDF. Extrayendo sin chunks.")

    if not chunk_paths:
        result, cleaned, searchable_path = _extract_single(
            pdf_path, prompt_path, is_scanned, prefer_docai_json=prefer_docai_json
        )
        if is_scanned:
            save_digitized_pdf(pdf_path, [searchable_path], overlap=0)
        return _unwrap_rows(result, rows_key), cleaned

    all_rows = []
    all_pages = []
    digitized_paths = []
    anchor_note = None
    try:
        for i, chunk_path in enumerate(chunk_paths):
            logger.info(f"Chunk {i + 1}/{len(chunk_paths)}")
            result, cleaned, searchable_path = _extract_single(
                chunk_path,
                prompt_path,
                is_scanned,
                context_note=anchor_note,
                prefer_docai_json=prefer_docai_json,
            )
            digitized_paths.append(searchable_path)
            chunk_rows = _as_rows(_unwrap_rows(result, rows_key))
            all_rows.extend(chunk_rows)
            if cleaned:
                all_pages.extend(cleaned.get("pages", []))
            anchor_note = _build_continuity_anchor(chunk_rows)
            if i < len(chunk_paths) - 1 and not _is_fake():
                # Respiro entre chunks para no chocar con el rate limit.
                time.sleep(3)
    finally:
        _remove_all(chunk_paths)

    if is_scanned:
        save_digitized_pdf(pdf_path, digitized_paths, overlap=settings.chunk_overlap_pages)

    merged_cleaned = {"pages": all_pages} if all_pages else None
    logger.info(f"Chunks completados — {len(all_rows)} filas totales extraídas.")
    return all_rows, merged_cleaned


def _is_fake() -> bool:
    from app.config.settings import AIProvider

    return get_settings().ai_provider is AIProvider.FAKE


QA_STATUS_KEY = "estado"
QA_ERRORS_KEY = "observaciones"


@traceable(run_type="llm", name="Revisión de calidad")
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=3, max=10))
def run_quality_review(rows, prompt_path, identity=None):
    """Pide al modelo una revisión semántica de las filas extraídas.

    Complementa a las reglas contables deterministas: revisa lo que una suma
    no puede ver (signos invertidos, períodos cruzados, conceptos mal
    clasificados).
    """
    settings = get_settings()
    context = (
        f"\n\nFICHA DEL DOCUMENTO:\n{json.dumps(identity, ensure_ascii=False)}" if identity else ""
    )
    full_prompt = (
        f"{_read_prompt(prompt_path)}{context}\n\n"
        f"DATOS A EVALUAR:\n{json.dumps(rows, indent=2, ensure_ascii=False)}"
    )
    response = llm.generate(
        task="quality",
        prompt_name=Path(prompt_path).stem,
        model=settings.model_name_extract,
        parts=[llm.text(full_prompt)],
        temperature=0.1,
    )

    try:
        parsed = json.loads(response.text or "")
        if isinstance(parsed, list):
            if len(parsed) == 1 and isinstance(parsed[0], dict):
                return parsed[0]
            return {
                QA_STATUS_KEY: "Error",
                QA_ERRORS_KEY: [f"Respuesta inesperada (lista de {len(parsed)} elementos)"],
            }
        return parsed
    except Exception as e:
        logger.warning(f"Error parseando el JSON de calidad: {e}")
        return {QA_STATUS_KEY: "Error", QA_ERRORS_KEY: [str(e)]}
