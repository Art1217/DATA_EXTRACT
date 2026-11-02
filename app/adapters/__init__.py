"""Adaptadores a servicios externos.

Todo lo que habla con algo de fuera del proceso vive aquí: el modelo de
lenguaje (`llm`, con sus proveedores Vertex AI, Gemini API y `fake`), los
adaptadores de etapa que arman cada petición (`ai_classifier`,
`ai_searcher`, `ai_extractor`), Document AI (`doc_ai`) y PyMuPDF/OpenCV
(`pdf_utils`, `preprocessing`).

Es la capa más externa: `services/` los llama, ellos no saben nada de
servicios, jobs ni HTTP. Cada función recibe rutas y datos, y devuelve datos.
"""
