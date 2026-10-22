# Sesión 4 · Embeddings y bases de datos vectoriales

**Módulo 1 · Semana 2 · AI Engineer I**

## Objetivos

1. Explicar qué es un embedding y medir similitud con el coseno.
2. Construir y **evaluar** un buscador semántico (hit@k).
3. Entender qué es una base de datos vectorial (ANN, HNSW) y usar dos: **Neon** (PostgreSQL + pgvector) y **Pinecone**, con sus planes, costos y límites.

## Contenido de esta carpeta

| Ruta | Qué es |
| --- | --- |
| [`s04_embeddings_bases_vectoriales.ipynb`](s04_embeddings_bases_vectoriales.ipynb) | **Notebook de la sesión** (10 partes) |
| [`embeddings.py`](embeddings.py) | Convierte texto en vectores (modelo local con LM Studio u OpenAI) y calcula similitud |
| [`almacenes.py`](almacenes.py) | Una misma interfaz para tres almacenes: memoria, **Neon** y **Pinecone** |
| `practica/` | **Buscador de documentos**: programa completo y comentado |
| `datos/` | 24 FAQs de una mesa de ayuda, 10 consultas de prueba y 8 notas en Markdown |
| `pyproject.toml`, `uv.lock`, `.python-version` | Entorno de la sesión (**Python 3.12**) |

## Requisitos

| | |
| --- | --- |
| **Python** | **3.12** (uv lo descarga si no lo tienes) |
| **Programas** | `git` y [`uv`](../../docs/INSTALACION.md) (recomendado; ver [alternativas](../../docs/INSTALACION.md#5-alternativas-a-uv)) y **LM Studio** ([guía](../../docs/LMSTUDIO.md)) |
| **Modelo de embeddings** | `nomic-embed-text-v2-moe` en LM Studio (≈ 344 MB, multilingüe). Alternativa: embeddings de OpenAI |
| **Cuentas (gratuitas)** | **Neon** → [`docs/CUENTA_NEON.md`](../../docs/CUENTA_NEON.md) · **Pinecone** → [`docs/CUENTA_PINECONE.md`](../../docs/CUENTA_PINECONE.md). Sin cuenta, el notebook funciona igual y omite esas partes |
| **Equipo** | 16 GB de RAM, sin GPU |

### Dependencias de Python

| Paquete | Para qué |
| --- | --- |
| `openai` | Cliente que habla con LM Studio (embeddings locales) u OpenAI |
| `numpy`, `matplotlib` | Cálculo con vectores y gráficos |
| `psycopg[binary]`, `pgvector` | Conectar con PostgreSQL / Neon y usar vectores |
| `pinecone` | SDK oficial de Pinecone |
| `python-dotenv` | Leer credenciales del `.env` |
| `jupyterlab`, `ipykernel`, `ipywidgets` | Ejecutar notebooks (grupo `dev`) |

## Instalación

> **¿No usas uv?** Esta carpeta trae `requirements.txt` para `pip` (venv, conda, pyenv…). Donde veas `uv run <comando>`, ejecuta `<comando>` con tu entorno activado.

```bash
# desde la raíz del repo (una sola vez): cp .env.example .env  y completa tus valores
cd modulo-1/s04-embeddings-bases-vectoriales
uv sync
```

### Variables del `.env` de esta sesión

```
# Embeddings (por defecto local, gratis)
EMBEDDINGS_PROVIDER=local          # o "openai"
# EMBEDDINGS_MODEL=text-embedding-nomic-embed-text-v2-moe

# Neon (cadena de conexión de tu proyecto)
NEON_DATABASE_URL=postgresql://usuario:clave@ep-xxxx-pooler.region.aws.neon.tech/neondb?sslmode=require

# Pinecone
PINECONE_API_KEY=pega-aqui-tu-llave
```

## Ejecutar el notebook

```bash
uv run jupyter lab
```

O en VS Code: abre **esta carpeta**, abre el `.ipynb` y elige el kernel `.venv`. Debe ejecutarse **desde esta carpeta** (usa rutas relativas).

## Ejecutar la práctica

```bash
uv run python practica/buscador_documentos.py --almacen memoria --indexar --buscar "se venció el certificado"
uv run python practica/buscador_documentos.py --almacen neon --indexar
uv run python practica/buscador_documentos.py --comparar "¿cuánto se conservan las copias de seguridad?"
```

Detalles y retos: [`practica/README.md`](practica/README.md).

## 💰 Qué consume dinero en esta sesión

| Dónde | Qué hace | ¿Cuesta? |
| --- | --- | --- |
| Embeddings **locales** (LM Studio) | Vectorizar textos | **No** |
| Embeddings de **OpenAI** (opcional) | Vectorizar textos | Fracciones de centavo (≈ US$ 0.02 por millón de tokens) |
| **Neon** plan Free | Base PostgreSQL + pgvector | **No**; consume cuota (1 GB, 100 CU-horas por proyecto) |
| **Pinecone** plan Starter | Índice serverless | **No**; consume cuota (2 GB, 1 M lecturas y 2 M escrituras al mes) |

Detalle completo en [`docs/COSTOS.md`](../../docs/COSTOS.md).

## Proveedores de embeddings (local y/o de pago)

Todo funciona con **embeddings locales** (LM Studio, gratis) **o con los de OpenAI** (`EMBEDDINGS_PROVIDER=openai` y tu `OPENAI_API_KEY`; ≈ US$ 0.02 por millón de tokens). **Anthropic no tiene modelo de embeddings**: si tu proveedor de chat es Anthropic, usas el local o el de OpenAI. El notebook (Parte 4) **compara ambos** con tus propios datos; lo ideal es tener los dos para ver la diferencia. Al cambiar de proveedor, reindexa: los vectores de modelos distintos no se pueden mezclar.

## Problemas frecuentes

| Síntoma | Causa y solución |
| --- | --- |
| `No hay servidor local de embeddings` | Abre LM Studio, activa el servidor local y descarga `nomic-embed-text-v2-moe` |
| Resultados malos en español | Estás usando un modelo centrado en inglés; usa el multilingüe |
| `dimension` distinta al guardado | Cambiaste de modelo: vacía/elimina la tabla o el índice y reindexa |
| Neon tarda en responder la primera vez | La base estaba suspendida; despierta en ~1 s |
| Pinecone tarda en crear el índice | La primera vez tarda ~1 min |
