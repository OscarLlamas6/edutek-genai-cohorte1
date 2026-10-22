# Módulo 1 · AI Engineer I: Fundamentos de IA Generativa

**30 h · 15 sesiones · 5 semanas.** De cómo funciona un LLM al desarrollo de un agente con RAG y herramientas MCP.

Cada sesión es un **proyecto independiente** (su propio entorno, su propia versión de Python) con un **notebook didáctico**, una **práctica** (programa completo y comentado) y un **README** con instrucciones, dependencias y costos.

> 💰 El curso usa una API key de Anthropic Platform (pago por uso). Antes de empezar lee [`../docs/COSTOS.md`](../docs/COSTOS.md) e [`../docs/INSTALACION.md`](../docs/INSTALACION.md).

## Sesiones

| Semana | Sesión | Tema | Python | Notebook | Estado |
| --- | --- | --- | --- | --- | --- |
| 1 | [S1](s01-fundamentos-llm/) | Fundamentos del LLM y ventanas de contexto | 3.12 | ✅ | Disponible |
| 1 | [S2](s02-ecosistema-api/) | Ecosistema de modelos y consumo vía API | 3.12 | ✅ | Disponible |
| 1 | [S3](s03-prompting-contexto/) | Prompting y context engineering | 3.12 | ✅ | Disponible |
| 2 | [S4](s04-embeddings-bases-vectoriales/) | Embeddings y bases de datos vectoriales (Neon y Pinecone) | 3.12 | ✅ | Disponible |
| 2 | [S5](s05-fundamentos-rag/) | Fundamentos de RAG y un RAG simple con citas | 3.12 | ✅ | Disponible |
| 3 | S7 | RAG en profundidad: chunking y recuperación | — | — | Próximamente |
| 3 | S8 | RAG: evaluación y alucinaciones | — | — | Próximamente |
| 3 | S9 | Tool calling / function calling | — | — | Próximamente |
| 4 | S10 | Introducción a agentes | — | — | Próximamente |
| 4 | S11 | Memoria y estado en agentes | — | — | Próximamente |
| 4 | S12 | MCP: fundamentos | — | — | Próximamente |
| 5 | S13 | Construyendo con MCP | — | — | Próximamente |
| 5 | S14 | Automatización con n8n | — | — | Próximamente |
| 5 | S15 | Agente con RAG + MCP | — | — | Próximamente |

## Cómo trabajar una sesión

```bash
cd modulo-1/s01-fundamentos-llm   # entra a la carpeta de la sesión
uv sync                           # crea el entorno con la versión de Python de la sesión
uv run jupyter lab                # abre el notebook
```

Cada README de sesión detalla requisitos, instalación, ejecución del notebook, ejecución de la práctica, tabla de costos y solución de problemas.

## Flujo de una sesión (2 h)

1. **Teoría (≈ 1 h).** Conceptos con apoyo de láminas.
2. **Práctica (≈ 1 h).** El notebook guía los experimentos y la práctica los consolida.
3. **Puesta en común (≈ 10 min).** Qué falló, qué sorprendió.
