"""Ejecución asíncrona de trabajos largos.

Un documento tarda minutos u horas: no cabe en el ciclo de una petición HTTP.
Cada operación cara devuelve un `job_id` y el cliente sigue el avance por
sondeo o por SSE.
"""
