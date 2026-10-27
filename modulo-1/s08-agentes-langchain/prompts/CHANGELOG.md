# Historial de prompts

Un prompt es código: cada cambio se registra, se prueba y puede revertirse.

## rag_respuesta

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Responde solo con el contexto, cita con `[n]`, dice "No encuentro esa información en el manual." si no sabe y trata el contexto como datos. | Línea base de un RAG con citas. |
| v2 | Añade copiar las cifras tal cual y limitar la respuesta a tres frases. | Reducir respuestas largas y errores al reescribir precios o plazos. |
