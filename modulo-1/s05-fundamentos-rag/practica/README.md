# Prácticas S5 · RAG

Dos programas comentados, del más simple al más completo:

| Programa | Para qué |
| --- | --- |
| [`rag_basico.py`](rag_basico.py) | **Empieza aquí.** El flujo completo en 7 fases (cargar, fragmentar, vectorizar, guardar, recuperar, aumentar, generar), mostrando lo que produce cada una. Todo en memoria, sin opciones |
| [`rag_simple.py`](rag_simple.py) | El mismo flujo con Neon/Pinecone, citas verificadas, umbral, evaluación y comparación de proveedores |

## Práctica 1 · RAG básico, fase por fase

```bash
uv run python practica/rag_basico.py "¿Cuánto cuesta el plan de 300 Mbps?"
uv run python practica/rag_basico.py --sin-llm "¿Cuánto cuesta el plan de 300 Mbps?"   # se detiene antes del LLM: gratis
uv run python practica/rag_basico.py --proveedor local "¿Ofrecen televisión por cable?"
```

**Terminado cuando…** puedes decir, sin mirar el código, qué entra y qué sale de cada fase; cambiaste `ESTRATEGIA` y `K` en el archivo y explicas qué cambió en los fragmentos, en los puntajes y en la respuesta.

## Práctica 2 · RAG simple con citas

**Pregunta que responde:** ¿puedo hacer que un LLM responda sobre mis documentos, citando de dónde sale cada dato y diciendo "no sé" cuando el documento no lo contiene?

## Ejecutar (desde la carpeta de la sesión)

```bash
uv run python practica/rag_simple.py --almacen neon --ingestar
uv run python practica/rag_simple.py --almacen neon "¿Cuánto cuesta el plan de 300 Mbps?"
uv run python practica/rag_simple.py --almacen neon "¿Ofrecen televisión por cable?"       # fuera de alcance
uv run python practica/rag_simple.py --almacen neon --evaluar --estrategia oraciones --max-palabras 40
uv run python practica/rag_simple.py --almacen memoria --proveedor todos "¿Qué pasa si no pago a tiempo?"   # compara Anthropic, OpenAI y local
uv run python practica/rag_simple.py --almacen memoria --proveedor local "¿A qué hora atienden los domingos?"
```

Con `--almacen memoria` el documento se ingesta automáticamente en cada ejecución.

## Terminado cuando…

- Haces al menos tres preguntas y compruebas que la respuesta cita la sección correcta.
- Una pregunta fuera de alcance recibe "No encuentro esa información en el manual."
- Probaste dos configuraciones de fragmentado con `--evaluar` y explicas cuál fue mejor.

## Retos

1. Usa tu **propio documento** (`--documento ruta.md`, con secciones `##`) y escribe 5 preguntas de prueba.
2. Crea `prompts/rag_respuesta/v3.j2`, anótalo en `prompts/CHANGELOG.md` y compara con `--version v3`.
3. Calibra `--umbral`: ¿qué valor descarta temas ajenos sin perder preguntas válidas?
4. Encuentra una pregunta que tu RAG responda **mal** y explica por qué.
