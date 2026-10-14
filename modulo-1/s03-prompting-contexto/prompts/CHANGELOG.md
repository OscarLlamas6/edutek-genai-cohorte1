# Historial de prompts

Un prompt es código: cada cambio se registra, se prueba y puede revertirse.

## clasificador_tickets

| Versión | Qué cambió | Por qué |
| --- | --- | --- |
| v1 | Instrucción mínima (zero-shot). | Línea base para comparar. |
| v2 | Rol, categorías y prioridades permitidas, reglas, delimitadores `<ticket>` y formato de salida estricto. | v1 devolvía formatos y categorías inconsistentes. |
| v3 | Se añaden 3 ejemplos (few-shot) tomados de `ejemplos.json`. | Fijar el criterio de prioridad y el estilo del resumen. |

Regla del equipo: **ningún cambio de prompt se acepta sin correr los casos de `datos/casos_tickets.json` y comparar contra la versión anterior.**
