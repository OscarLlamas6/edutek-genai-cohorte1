# Historial de prompts

Un prompt es código: cada cambio se registra, se prueba y puede revertirse.

## rag_respuesta

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Responde solo con el contexto, cita con `[n]`, dice "No encuentro esa información en el manual." si no sabe y trata el contexto como datos. | Línea base de un RAG con citas. |
| v2 | Añade copiar las cifras tal cual y limitar la respuesta a tres frases. | Reducir respuestas largas y errores al reescribir precios o plazos. |
| v3 | Prohíbe estimar o deducir cifras, extiende el "no sé" a los temas parecidos pero ausentes, pide cubrir cada interpretación de una pregunta ambigua y pide comprobar las cifras antes de responder. | Reducir alucinaciones medidas en la Sesión 7: respuestas a preguntas fuera de alcance parecidas y cifras sin respaldo. |

## juez_fidelidad

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Pide decidir si la respuesta es fiel al contexto, sin rúbrica. | Línea base ingenua de un juez. |
| v2 | Rúbrica estricta: dividir en afirmaciones, buscar respaldo explícito de cada una y declarar "no_fiel" si falta una sola. | Un juez ingenuo tiende a aprobar todo: se midió contra etiquetas humanas. |

## juez_correccion

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Compara la respuesta con la respuesta esperada con reglas explícitas, incluida la abstención. | Línea base. |

## descomponer

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Pide separar la pregunta en 1 a 3 subpreguntas de un solo tema, con un ejemplo. | Recuperar por separado los datos de cada tema en preguntas multi-sección. |
| v2 | Quita el ejemplo resuelto y exige usar las palabras y los datos de la pregunta. | Con v1, el modelo local copiaba el ejemplo en la salida (subpreguntas sobre paquetes de mensajería en 3 de 9 preguntas probadas). Los ejemplos resueltos se filtran en modelos pequeños. |

## hyde

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Pide escribir un párrafo hipotético con estilo de manual que responda la pregunta. | Buscar con un texto que se parezca más a los fragmentos que a la pregunta (HyDE). |

## extraer_tripletas

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Extrae tripletas con tres relaciones permitidas; los nombres salen del texto. | Línea base de extracción de un grafo con un LLM. |
| v2 | Añade la lista de entidades conocidas: nombres exactos y solo esos. | Los nombres libres no se pueden unir en un grafo (el mismo servicio con tres escrituras). |
