"""Lógica de cada etapa del pipeline.

Estos módulos no conocen HTTP ni la cola de jobs: reciben un `StageContext`
(configuración, storage, workspace, progreso, cancelación) y devuelven datos.
Por eso siguen siendo ejecutables desde un script, un test o un worker remoto.
"""
