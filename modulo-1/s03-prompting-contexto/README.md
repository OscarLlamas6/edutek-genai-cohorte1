# Sesión 3 · Prompting y context engineering

**Módulo 1 · Semana 1 · AI Engineer I** · cierre de la Semana 1 (**Hito H1**)

## Objetivos

1. Estructurar un prompt (rol, contexto, restricciones, ejemplos, formato) y elegir entre zero-shot, few-shot y chain-of-thought.
2. Decidir qué información entra a la ventana de contexto y en qué orden (context engineering), incluyendo compresión y caché.
3. Dejar el resultado como una plantilla versionada con salida validada, y tratar todo texto externo como no confiable.

## Contenido de esta carpeta

| Ruta | Qué es |
| --- | --- |
| [`s03_prompting_contexto.ipynb`](s03_prompting_contexto.ipynb) | **Notebook de la sesión** (10 partes) |
| [`plantillas.py`](plantillas.py) | Carga y renderiza **plantillas de prompt versionadas** (Jinja2). Pasa al repositorio del agente |
| [`validacion.py`](validacion.py) | Valida la salida del modelo contra un esquema (Pydantic). Pasa al repositorio del agente |
| `prompts/clasificador_tickets/` | Plantillas `v1`, `v2`, `v3` + ejemplos; `prompts/CHANGELOG.md` registra los cambios |
| [`cliente_llm.py`](cliente_llm.py) | Copia del cliente de la Sesión 2, para que esta sesión funcione sola |
| `practica/` | **Clasificador de tickets**: programa completo y comentado |
| `datos/` | Incidente ficticio, 10 casos de prueba y un ticket con instrucción incrustada |
| `pyproject.toml`, `uv.lock`, `.python-version` | Entorno de la sesión (**Python 3.12**) |

## Requisitos

| | |
| --- | --- |
| **Python** | **3.12** (uv lo descarga si no lo tienes) |
| **Programas** | `git` y [`uv`](../../docs/INSTALACION.md) (recomendado; puedes usar pyenv, venv, conda u otro: ver [alternativas](../../docs/INSTALACION.md#5-alternativas-a-uv)) |
| **Cuenta** | API key de **Anthropic u OpenAI** (a tu elección, con `LLM_PROVIDER` en el `.env`; ver [`docs/PROVEEDORES.md`](../../docs/PROVEEDORES.md)) |
| **Equipo** | 16 GB de RAM, sin GPU |

### Dependencias de Python

| Paquete | Para qué |
| --- | --- |
| `anthropic` | Cliente oficial de la API de Claude (proveedor Anthropic) |
| `openai` | Cliente oficial de OpenAI (proveedor OpenAI y servidor local de LM Studio) |
| `jinja2` | Plantillas de prompt con variables |
| `pydantic` | Esquemas y validación de la salida |
| `tiktoken` | Conteo local aproximado de tokens (gratis) |
| `python-dotenv` | Leer la API key desde `.env` |
| `jupyterlab`, `ipykernel`, `ipywidgets` | Ejecutar notebooks (grupo `dev`) |

## Instalación

> **¿No usas uv?** Es lo recomendado para seguir el repositorio tal cual, pero no es obligatorio. Esta carpeta trae un `requirements.txt` para usar `pip` con venv, conda, pyenv, etc. (ver [alternativas a uv](../../docs/INSTALACION.md#5-alternativas-a-uv)). Donde veas `uv run <comando>`, ejecuta `<comando>` con tu entorno activado.

```bash
# desde la raíz del repo (una sola vez): cp .env.example .env  y pega tu llave
cd modulo-1/s03-prompting-contexto
uv sync
```

## Ejecutar el notebook

```bash
uv run jupyter lab
```

O en VS Code: abre **esta carpeta**, abre el `.ipynb` y elige el kernel `.venv`. El notebook debe ejecutarse **desde esta carpeta** porque lee `datos/` y `prompts/` con rutas relativas.

## Ejecutar la práctica

```bash
uv run python practica/clasificador_tickets.py "El switch del piso 3 reinicia sin parar"
uv run python practica/clasificador_tickets.py "El switch del piso 3 reinicia sin parar" --version v2
uv run python practica/clasificador_tickets.py --evaluar --version v3     # 10 casos
```

Instrucciones y criterios de "terminado": [`practica/README.md`](practica/README.md).

## 💰 Qué consume dinero en esta sesión

| Dónde | Qué hace | ¿Cuesta? | Estimación |
| --- | --- | --- | --- |
| **Parte 1** | Zero-shot, few-shot y CoT (8 llamadas, modelo sin razonamiento) | **Sí** | ≈ US$ 0.002 |
| **Parte 2** | Iterar un prompt (5 llamadas, modelo económico) + tu propio prompt | **Sí** | ≈ US$ 0.001 |
| **Parte 3** | Esfuerzo `low` vs `high` (2 llamadas) | **Sí** | ≈ US$ 0.001 |
| **Parte 4** | Jerarquía de instrucciones (2 llamadas) | **Sí** | < US$ 0.0005 |
| Parte 5 | Medir el presupuesto del contexto con `tiktoken` | **No** | US$ 0 |
| **Parte 6** | Caché de prompt con un prefijo largo (2 llamadas, **opcional**) | **Sí** | ≈ US$ 0.002 |
| **Parte 7** | Evaluar 3 versiones × 10 casos (**30 llamadas**) | **Sí** | ≈ US$ 0.003 |
| Parte 8 | Validación con respuestas simuladas | **No** | US$ 0 |
| **Parte 9** | Prompt injection de ejemplo (2 llamadas) | **Sí** | < US$ 0.0005 |
| Práctica | 1 llamada por ticket; `--evaluar` = 10 llamadas | **Sí** | ≈ US$ 0.0001 por ticket |
| **Total del notebook** | ~50 llamadas | | **< US$ 0.01** |

Las celdas de pago están marcadas con 💰 **FYI** en su primera línea. Apaga todo el gasto con `EJECUTAR_LLAMADAS_DE_PAGO = False`.

## Problemas frecuentes

| Síntoma | Solución |
| --- | --- |
| `FileNotFoundError: datos/...` | Ejecuta el notebook con la carpeta de la sesión como directorio de trabajo |
| `UndefinedError: 'categorias' is undefined` | Falta pasar una variable a la plantilla (es el comportamiento esperado de `StrictUndefined`) |
| La Parte 7 da 10/10 en todas las versiones | Tus casos son fáciles: añade casos difíciles a `datos/casos_tickets.json` |
| La Parte 6 muestra `0` tokens leídos de caché | La caché tiene condiciones (tamaño mínimo del prefijo, vigencia): también es parte de la lección |
| `Falta ANTHROPIC_API_KEY` / `Falta OPENAI_API_KEY` | `cp .env.example .env` (raíz del repo), elige `LLM_PROVIDER`, pega la llave de ese proveedor y reinicia el kernel |

Más soluciones: [`docs/INSTALACION.md`](../../docs/INSTALACION.md#11-problemas-frecuentes).
