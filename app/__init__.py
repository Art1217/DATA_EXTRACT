"""Backend del Pipeline Inteligente de Documentos.

Arquitectura por capas, con las dependencias apuntando siempre hacia adentro:

    api/       HTTP: routers, schemas, códigos de estado
      ↓
    jobs/      ejecución asíncrona, estado y progreso
      ↓
    services/  lógica de cada etapa del pipeline
      ↓
    domain/    catálogo de análisis, contratos y errores
    storage/   persistencia de artefactos (local hoy, GCS mañana)

Regla: `services/` no importa nada de `api/` ni de `jobs/`. Recibe rutas y un
callback de progreso, y devuelve datos. Eso mantiene cada etapa ejecutable
desde un script, un test o un worker remoto sin cambiarle una línea.
"""
