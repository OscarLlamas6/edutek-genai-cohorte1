# Sesión 12 · Automatización con n8n y cierre del Módulo 1

**Módulo 1 · Semana 5 · AI Engineer I**

## Objetivos

1. Decidir cuándo conviene un workflow de bajo código y cuándo código propio.
2. Leer un workflow de n8n: triggers, nodos, items, expresiones, credenciales, ejecuciones y depuración.
3. Armar un agente en n8n (modelo, memoria, herramientas), con salida estructurada y sub-workflows.
4. Usar MCP en n8n (cliente y servidor).
5. Diseñar un flujo con aprobación humana, reintentos y *error workflow*.
6. Conocer los límites reales: versionado, pruebas, licencia *fair-code*, operación en *self-hosting* y seguridad.
7. Repasar el Módulo 1 y saber qué trae el Módulo 2.

La sesión es **teoría con demostración**: la práctica completa vive en el repositorio `ai-agents-playground`, carpeta `n8n-workflows`.

## Contenido de esta carpeta

| Ruta | Qué es |
| --- | --- |
| [`s12_n8n_automatizacion.ipynb`](s12_n8n_automatizacion.ipynb) | **Notebook teórico** (12 partes; un mini-motor de *items*, un simulador de reintentos, un revisor de workflows y celdas **opcionales** que llaman al n8n local) |
| [`MARCO_TEORICO.md`](MARCO_TEORICO.md) | Definiciones y conceptos de la sesión (sin código) |
| `pyproject.toml`, `uv.lock`, `.python-version`, `requirements.txt` | Entorno de la sesión (**Python 3.12**) |

La práctica: **`ai-agents-playground/n8n-workflows/`** (repositorio aparte: <https://github.com/OscarLlamas6/ai-agents-playground>). Allí están el `docker-compose.yml` con n8n, los siete workflows importables y los tickets de ejemplo.

## Requisitos

| | |
| --- | --- |
| **Python** | **3.12** (uv lo descarga si no lo tienes) |
| **Programas** | `git` y [`uv`](../../docs/INSTALACION.md) (recomendado; ver [alternativas](../../docs/INSTALACION.md#5-alternativas-a-uv)) |
| **Notebook** | No necesita llaves, ni servidor, ni n8n. Las celdas de la Parte 9 son opcionales y necesitan el n8n del playground |
| **Práctica** | Docker con `docker compose`. Funciona **sin llaves** (modo simulado). Con un modelo real: LM Studio (≈ 12B o más) o una API key de OpenAI o Anthropic |
| **Equipo** | 16 GB de RAM, sin GPU |

### Dependencias

| Paquete | Para qué |
| --- | --- |
| `requests` | Llamar a los webhooks del n8n local (Parte 9, opcional) |
| `python-dotenv` | Leer el token del `.env` del playground (Parte 9, opcional) |
| `jupyterlab`, `ipykernel`, `ipywidgets`, `nbconvert` | Notebooks (grupo `dev`) |

## Instalación

> **¿No usas uv?** Esta carpeta trae `requirements.txt` para `pip`. Donde veas `uv run <comando>`, ejecuta `<comando>` con tu entorno activado.

```bash
cd modulo-1/s12-n8n-automatizacion
uv sync
uv run jupyter lab
```

O en VS Code: abre **esta carpeta**, abre el `.ipynb` y elige el kernel `.venv`.

El notebook busca los workflows exportados en `../../../ai-agents-playground/n8n-workflows/workflows` (clonando los dos repositorios uno junto al otro). Si los tienes en otro lugar, define la variable de entorno `N8N_WORKFLOWS_DIR`.

## 💰 Qué consume dinero

| Dónde | ¿Cuesta? |
| --- | --- |
| Todo el notebook | **No** |
| Celdas opcionales de la Parte 9 (llaman al n8n local) | No en modo `simulado`; con modelo real en n8n, centavos o menos por ticket |
| Práctica `n8n-workflows` | **US$ 0** en modo `simulado`. Con `llm`: de 1 a ≈ 5 llamadas al modelo por ticket (clasificación y, si es alto o crítico, un agente) y de 2 a 4 por mensaje del chat; gratis con LM Studio |

Detalle general en [`docs/COSTOS.md`](../../docs/COSTOS.md).

## Versiones probadas

n8n **2.47.7** (imagen `n8nio/n8n:2.47.7`, versión estable a mediados de noviembre de 2026). Notebook probado con Python 3.12, `requests` y `python-dotenv` (ver `uv.lock`). El notebook se ejecutó completo sin n8n y, aparte, con n8n local en modo simulado.
