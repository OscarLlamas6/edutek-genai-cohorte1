# Sesión 9 · LangChain: características útiles y RAG agéntico

**Módulo 1 · Semana 3 · AI Engineer I**

## Objetivos

1. Recordar las piezas de un agente en LangChain 1.x y leer su traza.
2. Usar las características que hacen utilizable a un agente: salida estructurada, streaming, middleware, memoria de corto plazo, contexto de ejecución y manejo de errores en herramientas.
3. Saber cuándo bajar de `create_agent` a un grafo propio con LangGraph.
4. Explicar el **RAG agéntico**: qué decisiones pasan al modelo, qué patrones existen, qué cuesta y qué riesgos trae.

La sesión es **teoría puntual con código mínimo**; la práctica completa vive en el repositorio `ai-agents-playground`, carpeta `langchain-agentic-rag`.

## Contenido de esta carpeta

| Ruta | Qué es |
| --- | --- |
| [`s09_langchain_rag_agentico.ipynb`](s09_langchain_rag_agentico.ipynb) | **Notebook teórico** (12 partes, con un modelo de juguete que no necesita red) |
| [`MARCO_TEORICO.md`](MARCO_TEORICO.md) | Definiciones y conceptos de la sesión (sin código) |
| `pyproject.toml`, `uv.lock`, `.python-version`, `requirements.txt` | Entorno de la sesión (**Python 3.12**) |

La práctica: **`ai-agents-playground/langchain-agentic-rag/`** (repositorio aparte: <https://github.com/OscarLlamas6/ai-agents-playground>).

## Requisitos

| | |
| --- | --- |
| **Python** | **3.12** (uv lo descarga si no lo tienes) |
| **Programas** | `git` y [`uv`](../../docs/INSTALACION.md) (recomendado; ver [alternativas](../../docs/INSTALACION.md#5-alternativas-a-uv)) |
| **Notebook** | No necesita llaves ni servidor: usa un modelo de juguete. Una celda opcional prueba un modelo real |
| **Práctica** | LM Studio con `text-embedding-nomic-embed-text-v2-moe` y un modelo de ~12B o más (por ejemplo `google/gemma-4-12b-qat`), o un proveedor (Anthropic u OpenAI) |
| **Equipo** | 16 GB de RAM, sin GPU |

### Dependencias

| Paquete | Para qué |
| --- | --- |
| `langchain` | `create_agent`, `@tool`, mensajes, middleware, `init_chat_model` |
| `langgraph` | El grafo sobre el que corre el agente; checkpointers |
| `langchain-anthropic`, `langchain-openai` | Modelos de Anthropic y OpenAI (y LM Studio, vía el protocolo de OpenAI) |
| `pydantic` | Esquemas de salida estructurada |
| `python-dotenv` | Leer credenciales del `.env` |
| `jupyterlab`, `ipykernel`, `ipywidgets`, `nbconvert` | Notebooks (grupo `dev`) |

## Instalación

> **¿No usas uv?** Esta carpeta trae `requirements.txt` para `pip`. Donde veas `uv run <comando>`, ejecuta `<comando>` con tu entorno activado.

```bash
cd modulo-1/s09-langchain-rag-agentico
uv sync
uv run jupyter lab
```

O en VS Code: abre **esta carpeta**, abre el `.ipynb` y elige el kernel `.venv`.

## 💰 Qué consume dinero

| Dónde | ¿Cuesta? |
| --- | --- |
| Todo el notebook con el modelo de juguete | **No** |
| Celda opcional de la Parte 1 (`USAR_MODELO_REAL`) | No con LM Studio; con Anthropic/OpenAI, ≈ US$ 0.001 (requiere activar `EJECUTAR_LLAMADAS_DE_PAGO`) |
| Práctica `langchain-agentic-rag` | No con LM Studio; con un proveedor, menos de un centavo por pregunta (2 a 6 llamadas) |

Detalle general en [`docs/COSTOS.md`](../../docs/COSTOS.md).

## Versiones probadas

`langchain 1.4.9`, `langchain-core 1.7.6`, `langgraph 1.2.20`, `langchain-anthropic 1.7.10`, `langchain-openai 1.7.4`, `pydantic 2.14.3` (noviembre de 2026). El protocolo de streaming de eventos `v3` aparece como experimental en esta versión.
