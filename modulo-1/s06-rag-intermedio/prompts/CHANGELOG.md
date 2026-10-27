# Historial de prompts

Un prompt es código: cada cambio se registra, se prueba y puede revertirse.

## rag_respuesta

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Responde solo con el contexto, cita con `[n]`, dice "No encuentro esa información en el manual." si no sabe y trata el contexto como datos. | Línea base de un RAG con citas. |
| v2 | Añade copiar las cifras tal cual y limitar la respuesta a tres frases. | Reducir respuestas largas y errores al reescribir precios o plazos. |

## reescritura

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Reescribe la última pregunta como consulta autocontenida usando el historial; conserva códigos y nombres; responde en una línea. | Preguntas de seguimiento como "¿y los domingos?" no sirven como consulta de búsqueda. |

## expansion

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Pide N variantes de la pregunta, una por línea, con sinónimos y términos del manual. | Cubrir distintas formas de decir lo mismo (multi-consulta). |

## rerank

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Pide ordenar los fragmentos candidatos por utilidad para responder; salida: números separados por comas. | Reordenar los candidatos de la búsqueda con un modelo que lee pregunta y fragmento juntos. |
| v2 | Pide elegir SOLO el mejor fragmento (un número); el resto conserva su orden. | Un modelo pequeño ordena mal una lista completa; elegir uno es una tarea más fácil. |
