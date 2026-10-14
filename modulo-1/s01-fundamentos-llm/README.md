# Sesión 1 · Fundamentos del LLM y ventanas de contexto

**Módulo 1 · Semana 1 · AI Engineer I**

## Objetivos

1. Ubicar a los LLM en la historia de la IA (de la IA clásica a hoy) y explicar qué hace un LLM: predecir el siguiente token.
2. Explicar qué es un token y una ventana de contexto, por qué ambos cuestan dinero y calcular el costo de una llamada.
3. Explicar por qué un LLM no se prueba como software tradicional (es probabilístico) y qué hacen los parámetros de generación.

## Contenido de esta carpeta

| Ruta | Qué es |
| --- | --- |
| [`s01_fundamentos_llm.ipynb`](s01_fundamentos_llm.ipynb) | **Notebook de la sesión** (prólogo de historia + 8 partes, con explicaciones y ejercicios) |
| `practica/` | **Calculadora de tokens y costo**: programa completo y comentado |
| [`cliente_llm.py`](cliente_llm.py) | Cliente del curso para Anthropic u OpenAI (se estudia a fondo en la Sesión 2; aquí lo usa el notebook) |
| `datos/` | Textos de ejemplo (ticket en español e inglés, un bloque de código) |
| `pyproject.toml`, `uv.lock` | Dependencias y versiones exactas |
| `.python-version` | **Python 3.12** |

## Requisitos

| | |
| --- | --- |
| **Python** | **3.12** (uv lo descarga si no lo tienes) |
| **Programas** | `git` y [`uv`](../../docs/INSTALACION.md) (recomendado; puedes usar pyenv, venv, conda u otro: ver [alternativas](../../docs/INSTALACION.md#5-alternativas-a-uv)) |
| **Cuenta** | API key de **Anthropic u OpenAI** (a tu elección, con `LLM_PROVIDER` en el `.env`; ver [`docs/PROVEEDORES.md`](../../docs/PROVEEDORES.md)) *(opcional hasta la Parte 5; ver costos)* |
| **Internet** | Sí: descarga tablas de tokenizadores la primera vez (~10 MB) y llama a la API |
| **Equipo** | 16 GB de RAM, sin GPU |

### Dependencias de Python

| Paquete | Para qué |
| --- | --- |
| `anthropic` | Cliente oficial de la API de Claude (proveedor Anthropic) |
| `openai` | Cliente oficial de OpenAI (proveedor OpenAI y servidor local de LM Studio) |
| `tiktoken` | Tokenizador local (familias `o200k_base` y `cl100k_base`) |
| `tokenizers` | Tokenizador de un modelo open-weight (Hugging Face) |
| `pydantic` | Validación de datos (la usa `cliente_llm.py`) |
| `python-dotenv` | Leer la API key desde `.env` |
| `jupyterlab`, `ipykernel`, `ipywidgets` | Ejecutar notebooks (grupo `dev`) |

## Instalación

> **¿No usas uv?** Es lo recomendado para seguir el repositorio tal cual, pero no es obligatorio. Esta carpeta trae un `requirements.txt` para usar `pip` con venv, conda, pyenv, etc. (ver [alternativas a uv](../../docs/INSTALACION.md#5-alternativas-a-uv)). Donde veas `uv run <comando>`, ejecuta `<comando>` con tu entorno activado.

Desde la raíz del repositorio (ver [`docs/INSTALACION.md`](../../docs/INSTALACION.md) para instalar uv y configurar tu llave):

```bash
cp .env.example .env              # una sola vez; elige LLM_PROVIDER y pega la llave de ese proveedor
cd modulo-1/s01-fundamentos-llm
uv sync                           # descarga Python 3.12 si hace falta, crea .venv e instala todo
```

### Cómo se fijó la versión de Python (para que sepas reproducirlo)

```bash
uv python pin 3.12                # escribe .python-version
# y en pyproject.toml:  requires-python = ">=3.12,<3.13"
```

## Ejecutar el notebook

**JupyterLab:**

```bash
uv run jupyter lab
```

**VS Code:** abre **esta carpeta**, abre el `.ipynb` y elige el kernel `.venv` (Python 3.12).

Ejecuta las celdas **en orden** con `Shift + Enter`.

## Ejecutar la práctica

```bash
uv run python practica/calculadora_tokens.py datos/ticket_es.txt
uv run python practica/calculadora_tokens.py datos/ticket_en.txt --modelo claude-opus-5-5 --salida 500   # o --modelo gpt-6-astra
uv run python practica/calculadora_tokens.py datos/ticket_es.txt --tokenizadores o200k cl100k qwen
uv run python practica/calculadora_tokens.py datos/ticket_es.txt --exacto   # conteo exacto de tu proveedor (requiere key)
```

Instrucciones y criterios de "terminado": [`practica/README.md`](practica/README.md).

## 💰 Qué consume dinero en esta sesión

| Dónde | Qué hace | ¿Cuesta? | Estimación |
| --- | --- | --- | --- |
| Prólogo y Partes 0 a 4 del notebook | Historia, n-gramas, tokenizadores, simulaciones y cálculos locales | **No** | US$ 0 |
| Parte 2 (celda opcional) | Conteo de tokens del proveedor (no genera texto) | **No** en Anthropic (`count_tokens`, gratis con límite de peticiones). En OpenAI, confirma en su página de precios | US$ 0 |
| **Parte 5** | Primera llamada y demostración de que la API no tiene estado (4 llamadas, con el modelo económico: `claude-haiku-5-5` o `gpt-6-luna`) | **Sí** | < US$ 0.001 |
| **Parte 6** | Parámetros de generación (≈ 17 llamadas muy cortas: con Anthropic `claude-haiku-4-5`, con OpenAI `gpt-6-luna` con el razonamiento apagado; una llamada rechazada con error 400 no se cobra) | **Sí** | ≈ US$ 0.005 |
| Práctica | Calculadora local | **No** (`--exacto` no genera texto) | US$ 0 |
| **Total de la sesión** | | | **< US$ 0.01** |

Las celdas de pago están marcadas con 💰 **FYI** en su primera línea. Puedes apagar todo el gasto poniendo `EJECUTAR_LLAMADAS_DE_PAGO = False` en la Parte 0 del notebook.

> Antes de empezar, configura un **límite de gasto mensual** en la consola de tu proveedor. Detalles en [`docs/COSTOS.md`](../../docs/COSTOS.md).

## Problemas frecuentes

| Síntoma | Solución |
| --- | --- |
| "Esta sesión está fijada a Python 3.12" | Selecciona el kernel `.venv` y reinicia el kernel |
| Aviso "unauthenticated requests to the HF Hub" | Es inofensivo, ignóralo |
| La Parte 5 omite las llamadas | No hay API key en `.env` (o `EJECUTAR_LLAMADAS_DE_PAGO = False`) |
| Error 400 en la Parte 6 | **Es esperado**: esa celda demuestra que el modelo rechaza `temperature` (`claude-haiku-5-5`, o `gpt-6-luna` con razonamiento activo) |

Más soluciones: [`docs/INSTALACION.md`](../../docs/INSTALACION.md#11-problemas-frecuentes).
