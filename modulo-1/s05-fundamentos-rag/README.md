# Sesión 5 · Fundamentos de RAG

**Módulo 1 · Semana 2 · AI Engineer I** · cierre de la Semana 2 (**Hito H2**)

## Objetivos

1. Explicar qué es RAG, qué problema resuelve y cuándo conviene.
2. Construir un RAG simple de punta a punta: fragmentar, vectorizar, guardar, recuperar y generar una respuesta **con citas** (y con "no sé").
3. Medir la recuperación por separado de la generación y reconocer limitaciones y riesgos (alucinaciones, citas falsas, inyección indirecta).

> Esto **no es un agente**: es un flujo fijo de pasos. Los agentes vienen después.

## Contenido de esta carpeta

| Ruta | Qué es |
| --- | --- |
| [`s05_fundamentos_rag.ipynb`](s05_fundamentos_rag.ipynb) | **Notebook de la sesión** (10 partes) |
| [`rag.py`](rag.py) | El RAG: fragmentar, ingestar, recuperar, generar con citas, verificar y evaluar |
| [`embeddings.py`](embeddings.py), [`almacenes.py`](almacenes.py) | Copia de la Sesión 4 (embeddings y almacenes memoria/Neon/Pinecone) |
| [`cliente_llm.py`](cliente_llm.py), [`plantillas.py`](plantillas.py) | Copias de las Sesiones 2 y 3 (LLM y plantillas de prompt) |
| `prompts/rag_respuesta/` | Plantillas `v1` y `v2` del prompt del RAG; `prompts/CHANGELOG.md` |
| `practica/` | **`rag_basico.py`** (las 7 fases del RAG, paso a paso) y **`rag_simple.py`** (RAG con citas, Neon/Pinecone y evaluación) |
| `datos/` | Manual ficticio de "Andes Fibra" y 14 preguntas de evaluación (2 fuera de alcance) |
| `pyproject.toml`, `uv.lock`, `.python-version` | Entorno de la sesión (**Python 3.12**) |

## Requisitos

| | |
| --- | --- |
| **Python** | **3.12** (uv lo descarga si no lo tienes) |
| **Programas** | `git` y [`uv`](../../docs/INSTALACION.md) (recomendado; ver [alternativas](../../docs/INSTALACION.md#5-alternativas-a-uv)) y LM Studio ([guía](../../docs/LMSTUDIO.md)) |
| **Embeddings** | `nomic-embed-text-v2-moe` en LM Studio (o `EMBEDDINGS_PROVIDER=openai`) |
| **LLM** | API key de **Anthropic u OpenAI**, o un modelo local en LM Studio (`RAG_PROVIDER=local`) |
| **Cuentas (opcionales)** | Neon y/o Pinecone ([`docs/CUENTA_NEON.md`](../../docs/CUENTA_NEON.md), [`docs/CUENTA_PINECONE.md`](../../docs/CUENTA_PINECONE.md)). Sin ellas, el notebook usa la memoria |
| **Equipo** | 16 GB de RAM, sin GPU |

### Dependencias de Python

| Paquete | Para qué |
| --- | --- |
| `anthropic`, `openai` | Clientes del LLM (y de LM Studio / embeddings) |
| `jinja2`, `pydantic` | Plantillas de prompt y validación |
| `numpy` | Vectores |
| `psycopg[binary]`, `pgvector`, `pinecone` | Neon y Pinecone |
| `python-dotenv` | Leer credenciales del `.env` |
| `jupyterlab`, `ipykernel`, `ipywidgets` | Ejecutar notebooks (grupo `dev`) |

## Instalación

> **¿No usas uv?** Esta carpeta trae `requirements.txt` para `pip`. Donde veas `uv run <comando>`, ejecuta `<comando>` con tu entorno activado.

```bash
cd modulo-1/s05-fundamentos-rag
uv sync
```

Variables del `.env` (además de las de sesiones anteriores): `RAG_PROVIDER` (`anthropic`, `openai` o `local`; si no existe se usa `LLM_PROVIDER`), `NEON_DATABASE_URL`, `PINECONE_API_KEY`.

## Ejecutar el notebook

```bash
uv run jupyter lab
```

O en VS Code: abre **esta carpeta**, abre el `.ipynb` y elige el kernel `.venv`. Debe ejecutarse **desde esta carpeta**.

## Ejecutar la práctica

```bash
uv run python practica/rag_simple.py --almacen memoria "¿Cuánto cuesta el plan de 300 Mbps?"
uv run python practica/rag_simple.py --almacen neon --ingestar
uv run python practica/rag_simple.py --almacen neon "¿Ofrecen televisión por cable?"
uv run python practica/rag_simple.py --almacen neon --evaluar
```

Detalles y retos: [`practica/README.md`](practica/README.md).

## 💰 Qué consume dinero en esta sesión

| Dónde | Qué hace | ¿Cuesta? | Estimación |
| --- | --- | --- | --- |
| Embeddings locales / Neon / Pinecone (planes gratuitos) | Ingesta y recuperación | **No** | — |
| **Parte 6** | 4 preguntas con el LLM | **Sí**, si usas Anthropic/OpenAI | ≈ US$ 0.002 |
| **Parte 7** | Evaluar 14 preguntas | **Sí** | ≈ US$ 0.007 |
| **Parte 8** | Demostración de inyección (2 llamadas) | **Sí** | ≈ US$ 0.001 |
| **Parte 7** (comparación) | Mismas 14 preguntas con cada proveedor disponible | **Sí**, ≈ US$ 0.007 por proveedor de pago | local: 0 |
| Con `RAG_PROVIDER=local` | Todo lo anterior | **No** | 0 |

Total aproximado con un modelo de proveedor: **menos de US$ 0.02**. Detalle en [`docs/COSTOS.md`](../../docs/COSTOS.md).

> Un modelo local pequeño (3B parámetros) sirve para ver el flujo, pero suele **citar peor y equivocarse más** que un modelo de proveedor. Es parte de la lección: compara ambos.
