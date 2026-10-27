# Edutek · AI Engineer · Cohorte 1

Código, notebooks y prácticas del programa **AI Engineer** de Edutek (Guatemala), en tres módulos.

| Módulo | Curso | Estado |
| --- | --- | --- |
| [`modulo-1/`](modulo-1/) | AI Engineer I · Fundamentos de IA Generativa | En curso: Semanas 1 a 3 disponibles |
| [`modulo-2/`](modulo-2/) | AI Engineer II · Arquitectura y Seguridad de Sistemas Agénticos | Próximamente |
| [`modulo-3/`](modulo-3/) | AI Engineer III · LLMOps en Producción | Próximamente |

Hilo conductor del programa: **LLM → RAG → agente con herramientas (MCP) → sistema multi-agente seguro → sistema en producción.**

---

## Cómo está organizado

```
edutek-genai-cohorte1/
├── README.md                ← este archivo
├── .env.example             ← plantilla de variables de entorno (copiar a .env)
├── docs/
│   ├── INSTALACION.md       ← instalar uv (o usar tu gestor), Git, LM Studio y tu API key
│   ├── LMSTUDIO.md          ← configurar LM Studio (modelo local) paso a paso
│   ├── COSTOS.md            ← qué consume dinero, cuánto y cómo protegerte
│   ├── CUENTA_NEON.md       ← crear tu cuenta de Neon (PostgreSQL + pgvector), paso a paso
│   ├── CUENTA_PINECONE.md   ← crear tu cuenta de Pinecone (base vectorial), paso a paso
│   └── PROVEEDORES.md       ← elegir Anthropic u OpenAI (y adaptar otros)
└── modulo-1/
    ├── README.md            ← índice del módulo y estado de las sesiones
    └── sNN-nombre/          ← una carpeta por sesión
        ├── README.md        ← requisitos, instalación y cómo ejecutar
        ├── pyproject.toml   ← dependencias de la sesión
        ├── .python-version  ← versión de Python de la sesión
        ├── uv.lock          ← versiones exactas (reproducible)
        ├── *.ipynb          ← el notebook didáctico de la sesión
        └── practica/        ← programa de práctica completo y comentado
```

**Cada sesión es un proyecto independiente**, con su propio entorno virtual y su propia versión de Python. No necesitas haber terminado la sesión anterior para hacer la siguiente.

## Inicio rápido (5 pasos)

```bash
# 1. Clona el repositorio
git clone <URL-del-repositorio> edutek-genai-cohorte1
cd edutek-genai-cohorte1

# 2. Instala uv (una sola vez; recomendado, pero puedes usar tu propio gestor: ver docs/INSTALACION.md)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Crea tu archivo de variables de entorno, elige el proveedor (anthropic u openai) y pega tu API key
cp .env.example .env

# 4. Entra a la sesión y crea su entorno (uv descarga la versión de Python correcta)
cd modulo-1/s01-fundamentos-llm
uv sync

# 5. Abre el notebook
uv run jupyter lab
```

Guía completa y solución de problemas: [`docs/INSTALACION.md`](docs/INSTALACION.md). **uv es lo recomendado, no obligatorio**: si ya usas pyenv, venv, virtualenv, conda u otro, ahí encontrarás cómo hacerlo con tu herramienta.

## 💰 Transparencia de costos

Este curso usa una **API key de Anthropic o de OpenAI** (a tu elección: el repositorio funciona de forma nativa con ambos y se cambia con la variable `LLM_PROVIDER`; ver [`docs/PROVEEDORES.md`](docs/PROVEEDORES.md)), un servicio de **pago por uso**. Queremos que sepas **siempre** qué consume dinero y cuánto:

- Cada README de sesión incluye una **tabla de costos** con lo que se gasta y una estimación.
- Cada celda de un notebook que **llama a la API de pago** está marcada con 💰 **FYI** en su primera línea (con una estimación de costo).
- Cada notebook tiene un **interruptor** (`EJECUTAR_LLAMADAS_DE_PAGO`) para apagar todo el gasto.
- Las sesiones 1 a 3 suman, en conjunto, **menos de US$ 0.05** de consumo. Las sesiones 4 y 5 (embeddings, bases vectoriales y RAG) usan embeddings locales y los **planes gratuitos** de Neon y Pinecone (sin tarjeta o sin cobro), y menos de US$ 0.02 de LLM si usas un proveedor. Las sesiones 6 a 8 (RAG intermedio y avanzado, y un primer agente con LangChain) suman menos de US$ 0.20 con un proveedor (US$ 0 con modelo local). La estimación del resto del curso se publicará sesión a sesión.

Detalles y recomendaciones en [`docs/COSTOS.md`](docs/COSTOS.md); te sugerimos configurar un límite de gasto mensual en tu cuenta.

## Reglas del repositorio

- 🔑 **Las llaves de API van en variables de entorno** (`.env`), nunca en el código ni en los notebooks. `.env` está en el `.gitignore`.
- 🧪 **Datos de ejemplo del mundo de TI** (tickets, logs, manuales). Nunca datos reales ni confidenciales.
- 💻 Todo se ejecuta en un equipo de **16 GB de RAM sin GPU**, consumiendo modelos por API.
