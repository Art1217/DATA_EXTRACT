# Imagen de la API (incluye el runner de jobs).
#
# `--workers 1` no es negociable: el runner de app/jobs/runner.py vive con
# hilos dentro del proceso de la API (una cola en memoria por proceso). Un
# segundo worker de uvicorn sería un segundo runner compitiendo por el mismo
# storage sin saber del otro. El paralelismo se ajusta con JOB_WORKERS.
FROM python:3.13-slim

# libglib2.0-0 y libgomp1: los piden opencv-python-headless y PyMuPDF en
# tiempo de ejecución aunque el wheel headless no traiga GUI.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY pyproject.toml README.md ./
COPY app/ ./app/
RUN pip install --no-cache-dir ".[gcp,claude]"

# Los prompts son datos de arranque: el lifespan de app/main.py falla de
# inmediato si falta uno declarado en el catálogo. Van dentro de la imagen
# para que sea reproducible por sí sola.
COPY prompts/ ./prompts/

RUN useradd --create-home --uid 1000 app && mkdir -p Storage Temp && chown -R app /srv
USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8000

EXPOSE 8000

# `PORT` lo inyecta Cloud Run; en local vale 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
