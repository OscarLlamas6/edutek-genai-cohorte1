# Práctica S6 · Laboratorio de chunking

Programa único y comentado: [`laboratorio_chunking.py`](laboratorio_chunking.py).

**Pregunta que responde:** con **mis** documentos y **mis** preguntas, ¿qué estrategia de fragmentado (y qué tamaño) recupera mejor, y cuánto contexto (dinero) cuesta cada una?

No llama al LLM: solo usa el modelo de embeddings, así que es **gratis** con el modelo local y se puede repetir las veces que quieras.

## Ejecutar (desde la carpeta de la sesión)

```bash
uv run python practica/laboratorio_chunking.py
uv run python practica/laboratorio_chunking.py --modo hibrido --detalle
uv run python practica/laboratorio_chunking.py --estrategias fijo,oraciones,recursivo --tamanos 40,70
uv run python practica/laboratorio_chunking.py --estrategias padre-hijo,seccion --tipos semantica,lexica
uv run python practica/laboratorio_chunking.py --tipos lexica --modo bm25
uv run python practica/laboratorio_chunking.py --estrategias oraciones,docling-hibrido --tokens 64,128,256   # requiere Docling
uv run python practica/laboratorio_chunking.py --documentos mi_carpeta/ --preguntas mis_preguntas.json --csv resultados.csv
```

## La configuración `docling-hibrido` (opcional)

Si instalaste Docling (`uv sync --group docling`), el laboratorio añade solo la configuración **`docling-hibrido`**: lee **todos** los documentos con Docling (un solo parser, con OCR para el acta escaneada) y los parte con su `HybridChunker` (jerarquía de encabezados + límite de **tokens**, medidos con el tokenizador del modelo de embeddings). Su tamaño se da en tokens con `--tokens` (por defecto 128). Los metadatos son los mismos que los de las demás estrategias (`fuente`, `seccion`, `tipo`, `publico`), más `paginas` y `tokens`.

> ⚠️ **No es la búsqueda híbrida.** `--modo hibrido` combina BM25 y vectores **al buscar**; `docling-hibrido` decide **dónde cortar** los documentos. Se pueden combinar: `--estrategias docling-hibrido --modo hibrido`.
>
> Sin Docling, el laboratorio funciona igual y avisa de que omitió esta configuración.

## Cómo leer la tabla

| Columna | Qué significa |
| --- | --- |
| `hit@1` / `hit@k` | En cuántas preguntas el primer fragmento (o alguno de los k primeros) contiene la respuesta |
| `MRR` | 1.0 = la respuesta siempre es el primer resultado; 0 = nunca aparece |
| `ctx` | Palabras promedio que se enviarían al LLM: es lo que pagarías en tokens |
| `tam.` | Palabras por fragmento (o **tokens**, con la etiqueta `tok`, en `docling-hibrido`) |

Con pocas preguntas, 1 o 2 aciertos de diferencia son **ruido**. Mira también `ctx`: una sección entera casi siempre "gana" porque es grande.

## Formato de tus preguntas

```json
[
  {"id": "X01", "pregunta": "¿Cuánto cuesta el plan Familiar?", "clave": "39 dólares", "tipo": "semantica"}
]
```

`clave` es el texto que debe aparecer en un fragmento para que ese fragmento responda la pregunta. Tipos usados por el programa: `semantica`, `lexica`, `tabla`, `filtro` (con un campo `filtro`) y `conversacional` (con `historial`).

## Terminado cuando…

- Ejecutaste el laboratorio con los tres modos de búsqueda (`vectorial`, `bm25`, `hibrido`) y explicas qué cambió.
- Probaste al menos dos tamaños para la misma estrategia y explicas cuál elegirías **y por qué** (calidad y contexto).
- Revisaste con `--detalle` una pregunta que falló y explicas la causa.

## Retos

1. Añade **tu propio documento** a una carpeta nueva, escribe 8 preguntas con su `clave` y compara con el corpus de la sesión.
2. Prueba `--tamanos 20,40,70,120,200`. ¿Dónde empieza a empeorar la recuperación?
3. Escribe una pregunta que **ninguna** configuración responda bien y explica por qué.
4. Con `EMBEDDINGS_PROVIDER=openai`, ¿cambia la mejor estrategia? Anota cuánto costó.
5. Con Docling instalado, prueba `--estrategias oraciones,docling-hibrido --tokens 64,128,256`. ¿Gana `docling-hibrido`? Con 22 preguntas, 1 o 2 aciertos son ruido: mira también el contexto y piensa qué te aporta Docling aparte de la nota (tablas, páginas para citar, el acta escaneada).
6. Cambia las tablas a su formato por defecto en `parsing_docling.py` (`tablas_markdown=False`) y mira qué pregunta pierde `docling-hibrido`. ¿Por qué? (pista: la `clave` de P16.)
