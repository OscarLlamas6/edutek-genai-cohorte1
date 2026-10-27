# Sesión 8 · Fundamentos de un agente de IA con LangChain

**Módulo 1 · Semana 3 · AI Engineer I** · primer vistazo a los agentes (la semana siguiente va de lleno a ellos)

## Objetivos

1. Explicar qué es un agente (modelo + herramientas + bucle) y en qué se diferencia de un chatbot y de un RAG.
2. Entender el *tool calling*: el modelo **pide** una herramienta y **tu código la ejecuta**. Escribir el bucle a mano y compararlo con LangChain (`create_agent`).
3. Diseñar herramientas que un modelo use bien, **medir** si las elige bien y reconocer los riesgos (acciones no pedidas, inyección, bucles).

> Un agente es un flujo **no fijo**: el modelo decide el camino. Eso lo hace flexible, pero más caro y menos predecible que un RAG.

## Contenido de esta carpeta

| Ruta | Qué es |
| --- | --- |
| [`s08_agentes_langchain.ipynb`](s08_agentes_langchain.ipynb) | **Notebook de la sesión** (10 partes) |
| [`MARCO_TEORICO.md`](MARCO_TEORICO.md) | Definiciones y conceptos de la sesión (sin código) |
| [`servicios.py`](servicios.py) | El "sistema de monitoreo" simulado de Andes Fibra (estado de servicios y tickets) |
| [`herramientas.py`](herramientas.py) | Las herramientas del agente (`@tool`): estado, listado, manual (el RAG de la Sesión 5) y reporte de fallas |
| [`agentes.py`](agentes.py) | Crear el modelo y el agente de LangChain, correr, ver la traza, medir tokens y evaluar la elección de herramientas |
| [`agente_manual.py`](agente_manual.py) | El **mismo bucle hecho a mano** con el SDK de Anthropic u OpenAI, sin LangChain |
| [`rag.py`](rag.py), [`embeddings.py`](embeddings.py), [`almacenes.py`](almacenes.py) | Copia de las Sesiones 4 y 5 (el manual se consulta como herramienta) |
| [`cliente_llm.py`](cliente_llm.py), [`plantillas.py`](plantillas.py) | Copias de las Sesiones 2 y 3 |
| `practica/` | **Bot de estado de servicios**: programa completo y comentado |
| `datos/` | `servicios.json` (estado simulado), `manual_soporte.md` (de la Sesión 5) y `preguntas_agente.json` (7 preguntas para evaluar la elección de herramientas) |
| `pyproject.toml`, `uv.lock`, `.python-version` | Entorno de la sesión (**Python 3.12**) |

## Requisitos

| | |
| --- | --- |
| **Python** | **3.12** (uv lo descarga si no lo tienes) |
| **Programas** | `git` y [`uv`](../../docs/INSTALACION.md) (recomendado; ver [alternativas](../../docs/INSTALACION.md#5-alternativas-a-uv)) y LM Studio ([guía](../../docs/LMSTUDIO.md)) |
| **Embeddings** | `nomic-embed-text-v2-moe` en LM Studio (para la herramienta `consultar_manual`) |
| **Modelo del agente** | API key de **Anthropic u OpenAI**, **o** un modelo local **de ~12B o más** en LM Studio (ver aviso abajo) |
| **Equipo** | 16 GB de RAM, sin GPU |

> ⚠️ **Modelo local y herramientas.** Usar herramientas es más difícil que conversar. Con `llama-3.2-3b-instruct` (3B) el agente acertó **2 de 6** preguntas de la prueba de elección de herramientas; con `google/gemma-4-12b-qat` (≈ 7 GB), **6 de 6**. Si no tienes una llave de proveedor, descarga un modelo de ~12B en LM Studio y pon `LOCAL_LLM_MODEL=google/gemma-4-12b-qat` (y `AGENT_PROVIDER=local`) en tu `.env`. Con 16 GB de RAM funciona si cierras otras aplicaciones.

### Dependencias de Python

| Paquete | Para qué |
| --- | --- |
| `langchain` | **`create_agent`** y el middleware |
| `langchain-anthropic`, `langchain-openai` | Los modelos de LangChain para cada proveedor (y para LM Studio, con `ChatOpenAI`) |
| `langgraph` | El motor del agente (se instala solo con `langchain`) |
| `anthropic`, `openai` | Los SDK, para el bucle a mano y el cliente de las sesiones anteriores |
| `jinja2`, `pydantic` | Plantillas de prompt y validación (salida estructurada) |
| `numpy` | Vectores (para `consultar_manual`) |
| `python-dotenv` | Leer credenciales del `.env` |
| `jupyterlab`, `ipykernel`, `ipywidgets` | Ejecutar notebooks (grupo `dev`) |

Versiones con las que se probó esta sesión (octubre de 2026): `langchain` 1.4.7 · `langchain-core` 1.7.0 · `langgraph` 1.2.18 · `langchain-anthropic` 1.7.6 · `langchain-openai` 1.7.3 · `anthropic` 1.17.0 · `openai` 3.39.0. El `uv.lock` las fija.

## Instalación

> **¿No usas uv?** Esta carpeta trae `requirements.txt` para `pip`. Donde veas `uv run <comando>`, ejecuta `<comando>` con tu entorno activado.

```bash
cd modulo-1/s08-agentes-langchain
uv sync
```

Variables del `.env` (además de las de sesiones anteriores): `AGENT_PROVIDER` (`anthropic`, `openai` o `local`; si no existe se usa `LLM_PROVIDER`), `LOCAL_LLM_URL` y `LOCAL_LLM_MODEL` si usas un modelo local. LangSmith (trazas en la nube) **no** hace falta.

## Ejecutar el notebook

```bash
uv run jupyter lab
```

O en VS Code: abre **esta carpeta**, abre el `.ipynb` y elige el kernel `.venv`. Debe ejecutarse **desde esta carpeta**. Si editas un archivo `.py` (por ejemplo `herramientas.py`), **reinicia el kernel**.

## Ejecutar la práctica

```bash
uv run python practica/bot_estado_servicios.py "¿Cómo está la app móvil?"
uv run python practica/bot_estado_servicios.py --a-mano "¿Cómo está la app móvil?"
uv run python practica/bot_estado_servicios.py --evaluar
uv run python practica/bot_estado_servicios.py --chat
```

Detalles y retos: [`practica/README.md`](practica/README.md).

## 💰 Qué consume dinero en esta sesión

Un agente hace **varias llamadas al modelo por pregunta** (decidir, leer cada resultado, responder) y reenvía el historial en cada una.

| Dónde | Qué hace | ¿Cuesta? | Estimación |
| --- | --- | --- | --- |
| Embeddings locales (`consultar_manual`) y herramientas | Buscar en el manual y consultar el estado | **No** | — |
| **Parte 2** | Una vuelta cruda + 2 preguntas con el bucle a mano | **Sí**, si usas Anthropic/OpenAI | ≈ US$ 0.002 |
| **Parte 4** | Modelo, `bind_tools` y un agente | **Sí** | ≈ US$ 0.001 |
| **Parte 6** | 4 preguntas + evaluación de 6 | **Sí** | ≈ US$ 0.005 |
| **Parte 7** | Salida estructurada, streaming y memoria | **Sí** | ≈ US$ 0.003 |
| **Parte 8** | Demostración de inyección (2 preguntas) | **Sí** | ≈ US$ 0.002 |
| **Práctica** | Una pregunta ≈ US$ 0.0005 a 0.002; `--evaluar` ≈ US$ 0.005 | **Sí** | — |
| Con `AGENT_PROVIDER=local` | Todo lo anterior | **No** | 0 |
| Partes 1, 3, 5 | Sin modelo (el modelo de juguete es un guion) | **No** | 0 |

Total aproximado con un modelo de proveedor: **menos de US$ 0.03**. Detalle en [`docs/COSTOS.md`](../../docs/COSTOS.md).

> Los precios son los del modelo económico de cada proveedor (octubre de 2026): confírmalos en la página oficial antes de presupuestar. Si un agente entra en un bucle, el gasto crece: por eso el programa corta con un **límite de pasos**.

## Solución de problemas

| Síntoma | Causa y solución |
| --- | --- |
| `⚠️ El servidor rechazó la salida del modelo (HTTP 400)` | El modelo local pequeño escribió una petición de herramienta con formato inválido. Usa un modelo de ~12B o un proveedor |
| El agente llama herramientas sin necesidad o repite llamadas | Modelo demasiado pequeño. Compáralo con `--evaluar` |
| `consultar_manual` devuelve "el manual no está disponible" | No responde el servidor de embeddings: activa el servidor local de LM Studio y carga `nomic-embed-text-v2-moe` |
| `Failed to load model` en LM Studio | El modelo no cabe o falló al cargar: ciérralo y cárgalo de nuevo desde LM Studio, o elige uno más pequeño |
| Un tutorial de LangChain no funciona | Probablemente usa la API anterior a la 1.0 (`AgentExecutor`, `initialize_agent`). Usa `create_agent` |
| Cambié un `.py` y no se nota | Reinicia el kernel del notebook |
