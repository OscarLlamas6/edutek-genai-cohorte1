# Sesión 2 · Ecosistema de modelos y consumo vía API

**Módulo 1 · Semana 1 · AI Engineer I**

## Objetivos

1. Elegir un modelo con criterio (calidad, costo, privacidad, latencia) y decidir entre comprar una API o construir tu propio serving.
2. Llamar a un modelo de forma robusta: streaming, medición de latencia (TTFT, tokens/s) y reintentos para errores transitorios.
3. Obtener salida estructurada: pedir JSON, validarlo con un esquema y reaccionar si falla.

## Contenido de esta carpeta

| Ruta | Qué es |
| --- | --- |
| [`s02_ecosistema_api.ipynb`](s02_ecosistema_api.ipynb) | **Notebook de la sesión** (9 partes) |
| [`cliente_llm.py`](cliente_llm.py) | **Cliente reutilizable** (streaming, reintentos, costo, Anthropic u OpenAI con una variable de entorno, modelo local con LM Studio, JSON validado). Pasa al repositorio del agente |
| `practica/` | **Comparador de modelos**: programa completo y comentado |
| `datos/tickets.json` | Tickets de ejemplo para la salida estructurada |
| `pyproject.toml`, `uv.lock`, `.python-version` | Entorno de la sesión (**Python 3.12**) |

## Requisitos

| | |
| --- | --- |
| **Python** | **3.12** (uv lo descarga si no lo tienes) |
| **Programas** | `git` y [`uv`](../../docs/INSTALACION.md) (recomendado; puedes usar pyenv, venv, conda u otro: ver [alternativas](../../docs/INSTALACION.md#5-alternativas-a-uv)) |
| **Programa adicional (recomendado)** | **[LM Studio](https://lmstudio.ai)** con un modelo pequeño tipo Llama 3.2 3B instruct (≈ 2 GB de descarga). Alternativas: Ollama, llama-server |
| **Cuenta** | API key de **Anthropic u OpenAI** (a tu elección, con `LLM_PROVIDER` en el `.env`; ver [`docs/PROVEEDORES.md`](../../docs/PROVEEDORES.md)) |
| **Equipo** | 16 GB de RAM, sin GPU (alcanza para un modelo de ~3B parámetros; será lento y es normal) |

### Dependencias de Python

| Paquete | Para qué |
| --- | --- |
| `anthropic` | Cliente oficial de la API de Claude (proveedor Anthropic) |
| `openai` | Cliente oficial de OpenAI (proveedor OpenAI y servidor local de LM Studio) |
| `openai` | Cliente de Python para el servidor local de LM Studio (API compatible con OpenAI) |
| `pydantic` | Esquemas y validación de la salida estructurada |
| `python-dotenv` | Leer la API key desde `.env` |
| `jupyterlab`, `ipykernel`, `ipywidgets` | Ejecutar notebooks (grupo `dev`) |

## Instalación

> **¿No usas uv?** Es lo recomendado para seguir el repositorio tal cual, pero no es obligatorio. Esta carpeta trae un `requirements.txt` para usar `pip` con venv, conda, pyenv, etc. (ver [alternativas a uv](../../docs/INSTALACION.md#5-alternativas-a-uv)). Donde veas `uv run <comando>`, ejecuta `<comando>` con tu entorno activado.

```bash
# desde la raíz del repo (una sola vez): cp .env.example .env  y pega tu llave
cd modulo-1/s02-ecosistema-api
uv sync
```

### Instalar LM Studio y el modelo open-weight

1. Instala **[LM Studio](https://lmstudio.ai)** (aplicación de escritorio para Windows, macOS y Linux).
2. Descarga un modelo pequeño (≈ 2 GB; por ejemplo, un Llama 3.2 de 3B *instruct*).
3. Pantalla *Local Model API* → activa **Local API server** (la Base URL suele ser `http://localhost:1234/v1`, pero **puede ser otro puerto**: usa el que muestre tu pantalla).
4. Copia el identificador del modelo tal como lo muestra LM Studio y, si no es `llama-3.2-3b-instruct`, ponlo en tu `.env`: `LOCAL_LLM_MODEL=...`
5. Verifica: `curl http://localhost:1234/v1/models` (con tu puerto) y pon `LOCAL_LLM_URL` en tu `.env` con esa misma URL.

Guía completa de LM Studio (interfaz, terminal `lms` y solución de problemas): [`docs/LMSTUDIO.md`](../../docs/LMSTUDIO.md). Resumen de modelos locales y alternativas: [`docs/INSTALACION.md`](../../docs/INSTALACION.md#8-modelos-open-weight-locales-sesión-2).

> **¿Prefieres Ollama, llama-server u otro programa local?** Son equivalentes para aprender. Todos exponen una API compatible con OpenAI, así que solo cambias `LOCAL_LLM_URL` y `LOCAL_LLM_MODEL` en tu `.env` (por ejemplo, para Ollama: `http://localhost:11434/v1` y `llama3.2:3b`).

> **Sin modelo local:** el notebook lo detecta y omite esa parte; la práctica puede comparar dos modelos por API (por ejemplo `--a anthropic:claude-haiku-5-5 --b openai:gpt-6-luna`, si tienes las dos llaves).

## Ejecutar el notebook

```bash
uv run jupyter lab         # JupyterLab
```

O en VS Code: abre **esta carpeta**, abre el `.ipynb` y elige el kernel `.venv`.

## Ejecutar la práctica

```bash
uv run python practica/comparador_modelos.py "Explica qué es un balanceador de carga en 3 frases"
uv run python practica/comparador_modelos.py "..." --a anthropic:claude-haiku-5-5 --b openai:gpt-6-luna   # sin modelo local (requiere las dos llaves)
```

Instrucciones y criterios de "terminado": [`practica/README.md`](practica/README.md).

## 💰 Qué consume dinero en esta sesión

| Dónde | Qué hace | ¿Cuesta? | Estimación |
| --- | --- | --- | --- |
| Parte 1 | Lista de modelos disponibles | **No** | US$ 0 |
| **Parte 2** | Dos llamadas con distinto `system` | **Sí** | ≈ US$ 0.0001 |
| **Parte 3** | Streaming vs no streaming (2 llamadas) | **Sí** | ≈ US$ 0.0003 |
| Parte 4 | Reintentos **simulados**; errores 401/404 reales | **No** (los errores se rechazan antes de generar) | US$ 0 |
| **Parte 5** | Misma pregunta al modelo por API (1 llamada). La parte con el **modelo local es gratis por token** | **Sí** (solo el modelo por API) | ≈ US$ 0.0002 |
| Partes 6 y 7 | Matriz de decisión y cálculo de Build vs Buy | **No** | US$ 0 |
| **Parte 8** | Salida estructurada, 3 a 7 llamadas | **Sí** | ≈ US$ 0.002 |
| Práctica | 1 llamada al modelo por API por ejecución (el modelo local no cobra por token) | **Sí** | ≈ US$ 0.0003 |
| **Total del notebook** | | | **< US$ 0.01** |

El modelo local no cobra por token, pero **sí consume** CPU, RAM, tiempo y electricidad de tu equipo.

Las celdas de pago están marcadas con 💰 **FYI** en su primera línea. Apaga todo el gasto con `EJECUTAR_LLAMADAS_DE_PAGO = False`.

## Problemas frecuentes

| Síntoma | Solución |
| --- | --- |
| `No hay servidor local en http://localhost:1234/v1` | Abre LM Studio → *Developer* → **Start Server**; revisa `LOCAL_LLM_URL` |
| `El modelo '...' no aparece en el servidor local` | Pon en `LOCAL_LLM_MODEL` el identificador exacto que muestra LM Studio |
| El modelo local muy lento | Normal sin GPU; prueba un modelo más pequeño o reduce `max_tokens` |
| `Falta ANTHROPIC_API_KEY` / `Falta OPENAI_API_KEY` | `cp .env.example .env` (raíz del repo), elige `LLM_PROVIDER`, pega la llave de ese proveedor y reinicia el kernel |
| Los tiempos (TTFT, tokens/s) cambian entre ejecuciones | Es normal: red y carga del servicio; mide varias veces |

Más soluciones: [`docs/INSTALACION.md`](../../docs/INSTALACION.md#11-problemas-frecuentes).
