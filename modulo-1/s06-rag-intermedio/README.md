# Sesión 6 · RAG intermedio: chunking y recuperación

**Módulo 1 · AI Engineer I** · de un RAG simple a uno que funciona con documentos reales

## Objetivos

1. Ingerir documentos reales (Markdown, HTML, DOCX y PDF), conservar su estructura y saber cuándo hace falta OCR. Conocer un parser moderno con análisis de *layout* (**Docling**): tablas, orden de lectura y OCR, con sus errores.
2. Comparar estrategias de fragmentado (tamaño fijo, oraciones, recursivo, semántico, por sección, padre-hijo y el **chunking híbrido de Docling**) **midiendo** con hit@k y MRR, sin olvidar el costo del contexto.
3. Combinar búsqueda por palabras (BM25) y semántica con fusión RRF, filtrar por metadatos, reordenar candidatos (reranking) y reescribir la consulta.

> Esto **sigue sin ser un agente**: es un flujo fijo de pasos. Lo que mejora es la calidad de lo que llega al prompt.

## Contenido de esta carpeta

| Ruta | Qué es |
| --- | --- |
| [`s06_rag_intermedio.ipynb`](s06_rag_intermedio.ipynb) | **Notebook de la sesión** (12 partes, de la 0 a la 11; con las secciones 2.5 *Docling*, 3.4 *chunking híbrido* y 4.2 *`docling-hibrido` en el laboratorio*) |
| [`MARCO_TEORICO.md`](MARCO_TEORICO.md) | Definiciones y conceptos de la sesión, sin código |
| [`ingesta.py`](ingesta.py) | Lectores de Markdown, HTML, DOCX y PDF (con aviso de páginas sin texto) |
| [`parsing_docling.py`](parsing_docling.py) | **Opcional (Docling):** un solo convertidor para todos los formatos, OCR, `HybridChunker` y `precargar()` de los modelos |
| [`fragmentado.py`](fragmentado.py) | Estrategias `recursivo` y `semantico`, patrón padre-hijo y metadatos (`publico`, `tipo`) |
| [`recuperacion.py`](recuperacion.py) | BM25, fusión RRF, búsqueda híbrida, filtros, reranking, reescritura de la consulta, `responder_avanzado` y métricas |
| [`rag.py`](rag.py) | El RAG de la Sesión 5 (copia): fragmentar, recuperar, generar con citas |
| [`embeddings.py`](embeddings.py), [`almacenes.py`](almacenes.py) | Copia de la Sesión 4 |
| [`cliente_llm.py`](cliente_llm.py), [`plantillas.py`](plantillas.py) | Copias de las Sesiones 2 y 3 |
| `prompts/` | Plantillas versionadas: `rag_respuesta`, `reescritura`, `expansion` y `rerank`; `CHANGELOG.md` |
| `practica/` | **Laboratorio de chunking**: programa completo y comentado (acepta `docling-hibrido` si Docling está instalado) |
| `datos/` | Cuatro documentos ficticios de "Andes Fibra" en cuatro formatos, un PDF escaneado y 30 preguntas de prueba |
| `pyproject.toml`, `uv.lock`, `requirements.txt`, `.python-version` | Entorno de la sesión (**Python 3.12**) |

### Los datos

| Archivo | Formato | Para qué sirve |
| --- | --- | --- |
| `manual_soporte.md` | Markdown | El manual de la Sesión 5 |
| `guia_tecnica.html` | HTML | Quitar menús y scripts; códigos de error y un catálogo de alarmas `OPT-xxxx` (pone a prueba la búsqueda por palabras) |
| `condiciones_servicio.docx` | DOCX | Estilos de título, una tabla y secciones por tipo de cliente (filtros por metadatos) |
| `boletin_mantenimiento.pdf` | PDF con texto | Una tabla que se aplana al extraerla (y que Docling reconstruye) |
| `acta_escaneada.pdf` | PDF escaneado | No tiene texto: hace falta OCR (Docling lo hace en 2.5) |
| `preguntas_recuperacion.json` | JSON | 30 preguntas, cada una con su `clave`, su tipo y, si aplica, un filtro o un historial |
| `generar_documentos.py` | Python | Regenera el DOCX y los PDF (usa `python-docx` y `reportlab`) |

## Requisitos

| | |
| --- | --- |
| **Python** | **3.12** (uv lo descarga si no lo tienes) |
| **Programas** | `git` y [`uv`](../../docs/INSTALACION.md) (recomendado; ver [alternativas](../../docs/INSTALACION.md#5-alternativas-a-uv)) y LM Studio ([guía](../../docs/LMSTUDIO.md)) |
| **Embeddings** | `nomic-embed-text-v2-moe` en LM Studio (o `EMBEDDINGS_PROVIDER=openai`) |
| **LLM** | API key de **Anthropic u OpenAI**, o un modelo local en LM Studio (`RAG_PROVIDER=local`). Solo hace falta para las partes 8, 9 y 10 |
| **Cuentas** | Ninguna: esta sesión usa un índice en memoria |
| **Equipo** | 16 GB de RAM, sin GPU |

### Dependencias de Python

| Paquete | Para qué |
| --- | --- |
| `anthropic`, `openai` | Clientes del LLM (y de LM Studio / embeddings) |
| `beautifulsoup4` | Leer HTML y descartar menús y scripts |
| `python-docx` | Leer DOCX (estilos y tablas) |
| `pypdf` | Leer PDF con texto |
| `jinja2`, `pydantic` | Plantillas de prompt y validación |
| `numpy` | Vectores |
| `psycopg[binary]`, `pgvector`, `pinecone` | Neon y Pinecone (los usa `almacenes.py`; en esta sesión no se necesitan cuentas) |
| `python-dotenv` | Leer credenciales del `.env` |
| `jupyterlab`, `ipykernel`, `ipywidgets`, `reportlab` | Notebooks y generación de los PDF de ejemplo (grupo `dev`) |
| `docling`, `torch`, `torchvision` | **Opcional** (grupo `docling`): parsing con *layout*, tablas, OCR y chunking híbrido. Ver abajo |

## Instalación

> **¿No usas uv?** Esta carpeta trae `requirements.txt` para `pip`. Donde veas `uv run <comando>`, ejecuta `<comando>` con tu entorno activado.

```bash
cd modulo-1/s06-rag-intermedio
uv sync
```

### Parsing moderno con Docling (opcional: instálalo ANTES de clase)

Las secciones 2.5, 3.4 y 4.2 del notebook (y la estrategia `docling-hibrido` del laboratorio) usan [Docling](https://github.com/docling-project/docling). **No viene en el entorno base**: es pesado, y así el resto de la sesión funciona aunque no lo instales. Si falta, esas celdas muestran un aviso con el comando exacto y se omiten.

```bash
cd modulo-1/s06-rag-intermedio
uv sync --group docling                       # añade Docling y PyTorch (solo CPU)
uv run python -c "from parsing_docling import precargar; precargar()"   # descarga los modelos YA, no en clase
```

| Qué | Cuánto (medido) |
| --- | --- |
| Entorno de la sesión | ≈ 390 MB sin Docling; ≈ 1.8 GB con el grupo `docling` (PyTorch CPU ocupa ≈ 780 MB) |
| Modelos de layout y tablas (Hugging Face) | ≈ 500 MB; más ≈ 60 MB de OCR y ≈ 20 MB del tokenizador |
| `precargar()` con el caché vacío | ≈ 33 s en total en la máquina de pruebas (depende de tu conexión); con los modelos ya descargados, ≈ 12 s |
| Primer PDF en el notebook | Unos segundos (carga los modelos); después, ≈ 1 a 2 s por PDF pequeño. Markdown, HTML y DOCX: centésimas de segundo |

> **Hazlo antes de clase y con buena conexión.** Los modelos se descargan **una sola vez** y quedan en el caché del equipo (`~/.cache/huggingface`); no necesitan GPU. Si la descarga se corta, repite el comando.

**Nota sobre PyTorch (solo CPU).** Por defecto, `pip` y `uv` instalan la versión de PyTorch con CUDA, que arrastra varios GB que aquí no sirven. Por eso `pyproject.toml` fija PyTorch al índice **solo CPU** (`https://download.pytorch.org/whl/cpu`) en Linux y Windows (en macOS la versión normal ya es solo CPU). Si usas `pip` en lugar de `uv`: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` y después `pip install docling`.

Variables del `.env` (las de sesiones anteriores): `EMBEDDINGS_PROVIDER`, `EMBEDDINGS_MODEL`, `LOCAL_LLM_URL`, `LOCAL_LLM_MODEL` y `RAG_PROVIDER` (`anthropic`, `openai` o `local`; si no existe se usa `LLM_PROVIDER`).

## Ejecutar el notebook

```bash
uv run jupyter lab
```

O en VS Code: abre **esta carpeta**, abre el `.ipynb` y elige el kernel `.venv`. Debe ejecutarse **desde esta carpeta**.

## Ejecutar la práctica

```bash
uv run python practica/laboratorio_chunking.py
uv run python practica/laboratorio_chunking.py --modo hibrido --detalle
uv run python practica/laboratorio_chunking.py --estrategias padre-hijo,seccion,oraciones --tamanos 35,70
```

Detalles y retos: [`practica/README.md`](practica/README.md).

## 💰 Qué consume dinero en esta sesión

| Dónde | Qué hace | ¿Cuesta? | Estimación |
| --- | --- | --- | --- |
| Embeddings locales, BM25, RRF, filtros, **Partes 1 a 7** y la práctica | Ingesta y búsqueda | **No** | — |
| **Parte 8** | Reranking: 2 versiones × 22 preguntas (≈ 44 llamadas) | **Sí**, si usas Anthropic/OpenAI | ≈ US$ 0.006 |
| **Parte 9** | Reescritura (4 llamadas) y multi-consulta (9 llamadas) | **Sí** | ≈ US$ 0.0015 |
| **Parte 10** | Pipeline completo con 3 preguntas | **Sí** | ≈ US$ 0.003 |
| Con `RAG_PROVIDER=local` | Todo lo anterior | **No** | 0 |
| Embeddings de OpenAI (opcional) | Vectorizar el corpus por cada configuración del laboratorio | **Sí**, fracciones de centavo | < US$ 0.01 |

Total aproximado con un modelo de proveedor: **menos de US$ 0.02**. Detalle en [`docs/COSTOS.md`](../../docs/COSTOS.md). El interruptor `EJECUTAR_LLAMADAS_DE_PAGO` del notebook apaga todo el gasto.

> Un modelo local pequeño (3B parámetros) sirve para ver el flujo, pero hace **peor** las tareas de las partes 8 y 9 (reranking, reescritura, multi-consulta). Es parte de la lección: cada paso con LLM se justifica midiendo.

## Solución de problemas

| Síntoma | Causa probable | Qué hacer |
| --- | --- | --- |
| `❌ Embeddings: No hay servidor local…` | LM Studio apagado o `LOCAL_LLM_URL` con otro puerto | Activa el servidor local y revisa el puerto en `.env` |
| `ModuleNotFoundError: bs4 / docx / pypdf` | Entorno sin sincronizar | `uv sync` dentro de esta carpeta |
| El PDF escaneado "no tiene secciones" | Es una imagen: no hay texto que extraer | Es lo esperado; hace falta OCR (Parte 2) |
| Los números difieren 1 o 2 aciertos de la guía | Otro modelo de embeddings, o un LLM no determinista | Normal con 22 preguntas; compara diferencias grandes |
| Las celdas de Docling dicen `Docling no está instalado` | No instalaste el grupo opcional | `uv sync --group docling` y reinicia el kernel |
| Docling tarda mucho la primera vez o falla al descargar | Se están descargando los modelos (≈ 500 MB) | Espera, o corre `precargar()` con buena conexión; repite si se cortó |
| `ModuleNotFoundError: docling_core` o errores de `torch` | Entorno mezclado o PyTorch distinto del de CPU | `uv sync --group docling` dentro de esta carpeta; no mezcles con otro entorno |
| `RapidOCR returned empty result!` | El OCR no encontró texto (imagen casi vacía o de muy baja resolución) | No es un fallo del programa: es la lección de 2.5.3. Revisa la fuente |
