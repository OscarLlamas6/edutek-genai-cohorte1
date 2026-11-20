# Sesión 10 · MCP: herramientas conectadas

**Módulo 1 · AI Engineer I**

## Objetivos

1. Explicar **qué es MCP** y por qué surge (problema N×M), su arquitectura *host / cliente / servidor*, transportes y ciclo de vida.
2. Usar las **tres primitivas** (tools, resources, prompts) con ejemplos ejecutables.
3. **Explorar un servidor existente** (el servidor oficial de referencia de hora) con un cliente mínimo y a mano, en JSON-RPC.
4. **Construir un servidor MCP propio** con `FastMCP` que **lee y escribe** en un CRM (HubSpot en la práctica; simulado en el notebook).
5. **Conectar un agente de LangChain** con `langchain-mcp-adapters`: a tu servidor, con aprobación humana antes de escribir y a **dos servidores a la vez**.
6. **Registrar tu servidor en un cliente real** (Claude Code, Claude Desktop) y **depurarlo**.
7. **Servir tu agente como una API** (FastAPI) con **sesiones persistentes** en SQLite (`session_id` = `thread_id`), y conocer la interfaz Streamlit y la colección Bruno de la práctica.

La sesión es **teoría y demostración con código mínimo**; el proyecto completo vive en el repositorio `ai-agents-playground`, carpeta `langchain-mcp-advanced-tooling`. La seguridad de MCP se verá en el Módulo 2.

## Contenido de esta carpeta

| Ruta | Qué es |
| --- | --- |
| [`s10_mcp_herramientas.ipynb`](s10_mcp_herramientas.ipynb) | **Notebook teórico** (13 partes, con un modelo de juguete y un CRM simulado: sin llaves ni red) |
| [`MARCO_TEORICO.md`](MARCO_TEORICO.md) | Definiciones y conceptos de la sesión (sin código) |
| `pyproject.toml`, `uv.lock`, `.python-version`, `requirements.txt` | Entorno de la sesión (**Python 3.12**) |

La práctica: **`ai-agents-playground/langchain-mcp-advanced-tooling/`** (repositorio aparte: <https://github.com/OscarLlamas6/ai-agents-playground>).

## Requisitos

| | |
| --- | --- |
| **Python** | **3.12** (uv lo descarga si no lo tienes) |
| **Programas** | `git` y [`uv`](../../docs/INSTALACION.md) (recomendado; ver [alternativas](../../docs/INSTALACION.md#5-alternativas-a-uv)) |
| **Notebook** | No necesita llaves ni servidor. Las partes 4 y 7 descargan un paquete pequeño con `uvx` (necesita internet; sin `uvx`, la Parte 4 se omite y la 7 usa un reloj local) |
| **Opcionales** | Node 22.23+ para el MCP Inspector; Claude Code o Claude Desktop para la Parte 8; Node 18+ para el CLI de Bruno de la práctica |
| **Práctica** | Modo simulado sin cuenta. Opcional: cuenta gratuita de HubSpot con una Service Key (guía: [`docs/CUENTA_HUBSPOT.md`](../../docs/CUENTA_HUBSPOT.md)); y un modelo (LM Studio, Anthropic u OpenAI) |
| **Equipo** | 16 GB de RAM, sin GPU |

### Dependencias

| Paquete | Para qué |
| --- | --- |
| `mcp` (`1.30.0`) | SDK oficial de MCP: `FastMCP` (servidor) y `ClientSession` (cliente) |
| `langchain-mcp-adapters` (`0.3.2`) | Convierte herramientas MCP en `@tool` de LangChain; multi-servidor |
| `langchain`, `langgraph` | `create_agent`, middleware (aprobación humana), checkpointers |
| `langchain-anthropic`, `langchain-openai` | Modelos reales (opcional) |
| `pydantic` | Tipos con restricciones (el esquema de las herramientas) |
| `fastapi`, `httpx` | La API de la Parte 10 y su `TestClient` (sin servidor) |
| `langgraph-checkpoint-sqlite`, `aiosqlite` | Checkpointer persistente en SQLite (sesiones que sobreviven a un reinicio) |
| `python-dotenv` | Leer credenciales del `.env` |
| `jupyterlab`, `ipykernel`, `ipywidgets`, `nbconvert` | Notebooks (grupo `dev`) |

## Instalación

> **¿No usas uv?** Esta carpeta trae `requirements.txt` para `pip`. Donde veas `uv run <comando>`, ejecuta `<comando>` con tu entorno activado.

```bash
cd modulo-1/s10-mcp-herramientas
uv sync
uv run jupyter lab
```

O en VS Code: abre **esta carpeta**, abre el `.ipynb` y elige el kernel `.venv`.

## 💰 Qué consume dinero

| Dónde | ¿Cuesta? |
| --- | --- |
| Todo el notebook con el modelo de juguete y el CRM simulado | **No** |
| Celda opcional de la Parte 6 (`USAR_MODELO_REAL`) | No con LM Studio; con Anthropic/OpenAI, ≈ US$ 0.002 (requiere activar `EJECUTAR_LLAMADAS_DE_PAGO`) |
| Práctica `langchain-mcp-advanced-tooling` en modo simulado | **No** |
| Práctica con HubSpot real | El plan gratuito de HubSpot; el modelo, menos de un centavo por pregunta con uno económico |

Detalle general en [`docs/COSTOS.md`](../../docs/COSTOS.md).

> ⚠️ Todos los datos del curso son **ficticios**. Con HubSpot real usa una **cuenta de pruebas** y guarda el token **solo** en el `.env` (que Git ignora).
