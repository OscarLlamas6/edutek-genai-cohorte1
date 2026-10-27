# Práctica S7 · Medidor de calidad RAG

Programa único y comentado: [`medidor_calidad_rag.py`](medidor_calidad_rag.py).

**Pregunta que responde:** ¿mi RAG es bueno, y cómo lo sé? Si cambio el prompt, el tamaño de los fragmentos o la técnica de recuperación, ¿mejora, empeora o da igual? ¿Puedo desplegar el cambio?

Evalúa tu RAG con un **golden set** (`datos/golden_set.json`: 24 preguntas de 6 tipos, 10 de ellas marcadas como "rápidas") y entrega un informe con **Recall@K, Precision@K y MRR** (recuperación), **dato correcto, citas válidas, "no sé" y cifras con respaldo** (generación), el **diagnóstico por capa** de cada pregunta y si **cumple el SLO**.

## Ejecutar (desde la carpeta de la sesión)

```bash
uv run python practica/medidor_calidad_rag.py --sin-generacion                       # solo recuperación: gratis y en segundos
uv run python practica/medidor_calidad_rag.py --rapido --version v3                  # las 10 preguntas rápidas
uv run python practica/medidor_calidad_rag.py --version v3 --nombre base             # las 24; guarda informes/base.json y .md
uv run python practica/medidor_calidad_rag.py --tecnica small-to-big -k 2 --version v3 --comparar-con informes/base.json
uv run python practica/medidor_calidad_rag.py --tecnica descomposicion --version v3
uv run python practica/medidor_calidad_rag.py --version v3 --juez fidelidad          # añade el juez LLM
uv run python practica/medidor_calidad_rag.py --version v3 --exigir-slo              # código de salida 1 si no cumple el SLO
```

Opciones principales: `--tecnica {basica,small-to-big,ventana,descomposicion,hyde}`, `--version {v1,v2,v3}` (prompt), `-k`, `--umbral`, `--proveedor {anthropic,openai,local}`, `--golden` y `--documento` (para usar tus propios datos).

Los informes se guardan en `informes/` (ignorada por Git).

## Terminado cuando…

- Corriste el medidor con **al menos dos configuraciones distintas** (por ejemplo, prompt v1 y v3) y guardaste los dos informes.
- Con `--comparar-con` identificaste **una mejora y una regresión**, y puedes explicar en qué **capa** (recuperación o generación) ocurrió cada una.
- Explicas por qué una diferencia de una o dos preguntas es ruido.
- Probaste `--exigir-slo` y viste el código de salida (`echo $?`).

## Retos

1. **Tu propio golden set:** 10 preguntas sobre un documento tuyo, con `secciones` y `claves`. Pásalo con `--golden` y `--documento`.
2. **Una técnica, una falla:** elige una falla del informe, aplica una técnica y vuelve a medir.
3. **Prompt v4:** crea `prompts/rag_respuesta/v4.j2`, anótalo en `prompts/CHANGELOG.md` y compáralo con `--comparar-con`.
4. **Tu SLO:** edita `SLO_POR_DEFECTO` en `evaluacion.py` con umbrales que *tú* defenderías ante tu equipo, y justifícalos.
5. **Sin generación vs con generación:** ¿qué problemas detecta `--sin-generacion` y cuáles se te escapan?
