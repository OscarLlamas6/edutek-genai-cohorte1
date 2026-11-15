# Sesión 11 · LangGraph: agentes como grafos

**Módulo 1 · Semana 3 · AI Engineer I**

## Objetivos

1. Explicar por qué un grafo (control explícito frente a `create_agent`) y cuándo **no** usar LangGraph.
2. Construir un grafo paso a paso: estado y *reducers*, nodos, aristas y aristas condicionales, `START`/`END`, compilación, `invoke` y `stream`.
3. Darle memoria y control humano: checkpointers y `thread_id` (persistencia y *time travel*), `interrupt()` + `Command(resume=...)` y el Store (memoria de largo plazo).
4. Hacerlo robusto: `recursion_limit`, presupuesto de pasos, falta de progreso, errores y reintentos, *streaming* y observabilidad.
5. Estructurar y probar un proyecto LangGraph (`langgraph.json`, `langgraph dev`, pruebas sin red).
6. Servir un agente: API FastAPI con `session_id` = `thread_id`, aprobación por HTTP, sesiones que sobreviven a un reinicio, y cuándo FastAPI propio y cuándo LangGraph Server.
7. Leer un agente completo hecho con `StateGraph`: operaciones sobre **Trello** con lecturas, escrituras **aprobadas por una persona**, RAG, memoria y tope de pasos.

La sesión es **100 % LangGraph**: **teoría puntual (≈ 30 min) y un notebook paso a paso**; la práctica completa vive en el repositorio `ai-agents-playground`, carpeta `langgraph-trello-ops-agent`.

## Contenido de esta carpeta

| Ruta | Qué es |
| --- | --- |
| [`s11_langgraph_agentes.ipynb`](s11_langgraph_agentes.ipynb) | **Notebook teórico** (15 partes, de la 0 a la 14, con un modelo y un Trello de juguete que no necesitan red; la API se prueba en proceso) |
| [`MARCO_TEORICO.md`](MARCO_TEORICO.md) | Definiciones y conceptos de la sesión (sin código) |
| `pyproject.toml`, `uv.lock`, `.python-version`, `requirements.txt` | Entorno de la sesión (**Python 3.12**) |

La práctica: **`ai-agents-playground/langgraph-trello-ops-agent/`** (repositorio aparte: <https://github.com/OscarLlamas6/ai-agents-playground>).

> **Esta carpeta es autocontenida.** El notebook trae una versión mínima del agente (Parte 11), con su propio Trello simulado. La última celda de la Parte 13 ejecuta además el agente completo **si** encuentra `ai-agents-playground` junto a este repositorio; si no, lo omite.

## Requisitos

| | |
| --- | --- |
| **Python** | **3.12** (uv lo descarga si no lo tienes) |
| **Programas** | `git` y [`uv`](../../docs/INSTALACION.md) (recomendado; ver [alternativas](../../docs/INSTALACION.md#5-alternativas-a-uv)) |
| **Notebook** | No necesita llaves, cuenta de Trello ni servidor: usa un modelo y un Trello de juguete. Una celda opcional prueba un modelo real |
| **Práctica** | Modo simulado sin nada más. La API, la interfaz Streamlit y la colección Bruno (esta última necesita Node.js para su CLI, o la aplicación de Bruno) forman parte del proyecto. Opcional: un proveedor (Anthropic u OpenAI) o LM Studio con un modelo de ~12B o más, y una cuenta gratuita de Trello (de **prueba**) |
| **Equipo** | 16 GB de RAM, sin GPU |

### Dependencias

| Paquete | Para qué |
| --- | --- |
| `langgraph` | `StateGraph`, checkpointers, `interrupt`, Store |
| `fastapi`, `httpx` | La mini API de la Parte 12 y su `TestClient` (en proceso, sin abrir puertos) |
| `langgraph-checkpoint-sqlite` | `SqliteSaver` (persistencia entre ejecuciones) |
| `langchain` | `create_agent` (para compararlo), `@tool`, mensajes, `init_chat_model` |
| `langchain-anthropic`, `langchain-openai` | Modelos de Anthropic y OpenAI (y LM Studio, vía el protocolo de OpenAI) |
| `pydantic`, `python-dotenv` | Estado con Pydantic; leer credenciales del `.env` |
| `numpy`, `requests` | Solo para la celda opcional que importa el agente completo del playground |
| `jupyterlab`, `ipykernel`, `ipywidgets`, `nbconvert` | Notebooks (grupo `dev`) |

## Instalación

> **¿No usas uv?** Esta carpeta trae `requirements.txt` para `pip`. Donde veas `uv run <comando>`, ejecuta `<comando>` con tu entorno activado.

```bash
cd modulo-1/s11-langgraph-agentes
uv sync
uv run jupyter lab
```

O en VS Code: abre **esta carpeta**, abre el `.ipynb` y elige el kernel `.venv`.

## 💰 Qué consume dinero

| Dónde | ¿Cuesta? |
| --- | --- |
| Todo el notebook con el modelo y el Trello de juguete | **No** |
| Celda opcional de la Parte 11 (`USAR_MODELO_REAL`) | No con LM Studio; con Anthropic/OpenAI, ≈ US$ 0.003 por ejecución (requiere activar `EJECUTAR_LLAMADAS_DE_PAGO`) |
| Práctica `langgraph-trello-ops-agent` | No en modo simulado; con un proveedor, centavos por petición (de 4 a 8 llamadas). La API de Trello es gratuita |

Detalle general en [`docs/COSTOS.md`](../../docs/COSTOS.md).

## Versiones probadas

`langgraph 1.2.22`, `langgraph-checkpoint 4.3.2`, `langgraph-checkpoint-sqlite 3.1.3`, `langchain 1.5.0`, `langchain-core 1.8.0`, `fastapi 0.163.0`, `httpx 0.28.1`, `pydantic 2.14.3` (noviembre de 2026). El notebook se ejecutó de punta a punta en modo simulado, sin errores.
