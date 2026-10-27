# Sesión 7 · RAG avanzado: técnicas, GraphRAG y evaluación

**Módulo 1 · Semana 3 · AI Engineer I** · Hito **H3** (RAG con evaluación)

## Objetivos

1. Explicar por qué falla un RAG básico en preguntas multi-sección y multi-salto, y **medirlo**.
2. Aplicar técnicas avanzadas de recuperación: *small-to-big*, encabezados contextuales, descomposición de preguntas y HyDE.
3. Explicar **GraphRAG** (grafo de conocimiento, búsqueda local y global), cuándo aporta y cuándo no, y compararlo con números frente a la búsqueda vectorial.
4. Evaluar un RAG **por capas** (Recall@K, Precision@K, MRR; dato correcto, citas, "no sé", cifras con respaldo) y calibrar un **juez LLM** contra etiquetas humanas.
5. Reconocer las causas de las alucinaciones, mitigarlas y comprobar la mejora con el ciclo *medir → cambiar → volver a medir*.

> La reescritura de la consulta, la búsqueda híbrida y el reranking son de la Sesión 6 y no se repiten aquí.
> Esto **no es un agente**: son flujos fijos. El *RAG agéntico* se toca solo como adelanto conceptual.

## Contenido de esta carpeta

| Ruta | Qué es |
| --- | --- |
| [`s07_rag_avanzado.ipynb`](s07_rag_avanzado.ipynb) | **Notebook de la sesión** (partes 0 a 11) |
| [`MARCO_TEORICO.md`](MARCO_TEORICO.md) | Definiciones y conceptos de la sesión (sin código) |
| [`evaluacion.py`](evaluacion.py) | Métricas de recuperación, comprobaciones de generación, juez LLM, diagnóstico por capa, SLO e informes |
| [`avanzado.py`](avanzado.py) | *Small-to-big*, encabezados contextuales, descomposición, fusión RRF y HyDE |
| [`grafo.py`](grafo.py) | GraphRAG a escala de juguete con `networkx`: grafo, recorridos, recuperador de grafo, extracción de tripletas con un LLM y evaluación multi-salto |
| [`rag.py`](rag.py), [`embeddings.py`](embeddings.py), [`almacenes.py`](almacenes.py), [`cliente_llm.py`](cliente_llm.py), [`plantillas.py`](plantillas.py) | Copias de las sesiones anteriores (el RAG que se evalúa) |
| `prompts/` | Plantillas `rag_respuesta` (v1 a v3), `juez_fidelidad`, `juez_correccion`, `descomponer`, `hyde`, `extraer_tripletas`; `prompts/CHANGELOG.md` |
| `practica/` | **Medidor de calidad RAG**: programa completo y comentado |
| `datos/` | Manual de "Andes Fibra"; `golden_set.json` (24 preguntas); `etiquetas_juez.json` (21 respuestas etiquetadas por una persona); catálogo de servicios de TI, su grafo curado y 8 preguntas multi-salto |
| `pyproject.toml`, `uv.lock`, `.python-version` | Entorno de la sesión (**Python 3.12**) |

## Requisitos

| | |
| --- | --- |
| **Python** | **3.12** (uv lo descarga si no lo tienes) |
| **Programas** | `git` y [`uv`](../../docs/INSTALACION.md) (recomendado; ver [alternativas](../../docs/INSTALACION.md#5-alternativas-a-uv)) y LM Studio ([guía](../../docs/LMSTUDIO.md)) |
| **Embeddings** | `nomic-embed-text-v2-moe` en LM Studio (o `EMBEDDINGS_PROVIDER=openai`) |
| **LLM** | API key de **Anthropic u OpenAI**, o un modelo local en LM Studio (`RAG_PROVIDER=local`). **El modelo local alcanza para toda la sesión** |
| **Cuentas** | Ninguna: esta sesión usa solo el almacén en memoria |
| **Equipo** | 16 GB de RAM, sin GPU |

### Dependencias de Python

| Paquete | Para qué |
| --- | --- |
| `anthropic`, `openai` | Clientes del LLM (y de LM Studio / embeddings) |
| `jinja2`, `pydantic` | Plantillas de prompt y validación de la salida del juez y de las tripletas |
| `numpy` | Vectores |
| `networkx` | El grafo de conocimiento (Parte 5 y 6) |
| `matplotlib` | Gráficos del notebook |
| `psycopg[binary]`, `pgvector`, `pinecone` | Los almacenes de la Sesión 4 (en esta sesión no se usan, pero `almacenes.py` los importa) |
| `python-dotenv` | Leer credenciales del `.env` |
| `jupyterlab`, `ipykernel`, `ipywidgets` | Ejecutar notebooks (grupo `dev`) |

## Instalación

> **¿No usas uv?** Esta carpeta trae `requirements.txt` para `pip`. Donde veas `uv run <comando>`, ejecuta `<comando>` con tu entorno activado.

```bash
cd modulo-1/s07-rag-avanzado
uv sync
```

Variables del `.env` (además de las de sesiones anteriores): `RAG_PROVIDER` (`anthropic`, `openai` o `local`), y, opcionales, `JUEZ_PROVIDER` y `JUEZ_MODEL` (quién actúa de juez; por defecto, el mismo proveedor que genera) y `JUEZ_MAX_TOKENS` (tope de salida del juez; por defecto 1500).

## Ejecutar el notebook

```bash
uv run jupyter lab
```

O en VS Code: abre **esta carpeta**, abre el `.ipynb` y elige el kernel `.venv`. Debe ejecutarse **desde esta carpeta**.

## Ejecutar la práctica

```bash
uv run python practica/medidor_calidad_rag.py --sin-generacion
uv run python practica/medidor_calidad_rag.py --rapido --version v3
uv run python practica/medidor_calidad_rag.py --tecnica small-to-big -k 2 --version v3 --comparar-con informes/base.json
uv run python practica/medidor_calidad_rag.py --version v3 --exigir-slo
```

Detalles y retos: [`practica/README.md`](practica/README.md).

## 💰 Qué consume dinero en esta sesión

| Dónde | Qué hace | ¿Cuesta? | Estimación |
| --- | --- | --- | --- |
| Embeddings locales, métricas de recuperación, *small-to-big*, grafo y recorridos | Todo lo que no llama al LLM | **No** | — |
| **Parte 4** | Descomposición y HyDE: 1 llamada corta por pregunta (≈ 48 llamadas) | **Sí**, si usas Anthropic/OpenAI | ≈ US$ 0.011 |
| **Parte 6** | Preguntas multi-salto (5 recuperadores × 8) y extracción de tripletas (2 × 16 secciones) | **Sí** | ≈ US$ 0.016 |
| **Parte 7** | Evaluar 24 preguntas con el RAG de la Sesión 5 | **Sí** | ≈ US$ 0.012 |
| **Parte 8** | Demostraciones de alucinación y 4 configuraciones nuevas × 24 preguntas | **Sí** | ≈ US$ 0.055 |
| **Parte 9** | Calibrar 3 jueces con 21 respuestas y cruzar juez y código en 10 preguntas | **Sí** | ≈ US$ 0.04 |
| **Parte 10** y práctica | 10 preguntas del set rápido (la práctica completa: 24) | **Sí** | ≈ US$ 0.005 (≈ US$ 0.012 la práctica completa) |
| Con `RAG_PROVIDER=local` (y juez local) | Todo lo anterior | **No** | 0 |

Total aproximado con un modelo de proveedor: **menos de US$ 0.15** (cota superior; el gasto real suele ser menor). Detalle en [`docs/COSTOS.md`](../../docs/COSTOS.md).

> 🧪 **El modelo local de 3B parámetros sirve para ver el flujo y el método, no para sacar buenas notas**: cita peor, se abstiene de más y es un **juez poco fiable**. Es parte de la lección. Si tienes un modelo de proveedor, compara: es la mejor forma de ver cuánto pesa el modelo frente a la técnica.
>
> ⚙️ Un servidor local puede dar respuestas ligeramente distintas entre corridas, incluso con temperatura 0. Una diferencia de una o dos preguntas entre corridas es ruido.
