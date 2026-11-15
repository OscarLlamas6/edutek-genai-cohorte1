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
| 3 | [S6](s06-rag-intermedio/) | RAG intermedio: ingesta, chunking, búsqueda híbrida, filtros, reranking y reescritura | 3.12 | ✅ | Disponible |
| 3 | [S7](s07-rag-avanzado/) | RAG avanzado y GraphRAG (notebook + implementación en `rag-playground/rag-avanzado`) | 3.12 | ✅ | Disponible |
| 3 | [S8](s08-agentes-langchain/) | Fundamentos de agentes de IA con LangChain (notebook + `ai-agents-playground/langchain-basic`) | 3.12 | ✅ | Disponible |
| 4 | [S9](s09-langchain-rag-agentico/) | LangChain: características útiles, ingeniería del bucle, memoria de corto y largo plazo, RAG agéntico y el agente servido como API (notebook + `ai-agents-playground/langchain-agentic-rag`) | 3.12 | ✅ | Disponible |
| 4 | [S10](s10-mcp-herramientas/) | MCP: arquitectura, primitivas, uso y servidor propio (HubSpot) (notebook + `ai-agents-playground/langchain-mcp-advanced-tooling`) | 3.12 | ✅ | Disponible · jue 19 nov |
| 5 | [S11](s11-langgraph-agentes/) | LangGraph y un agente con RAG, memoria y aprobación humana (notebook + `ai-agents-playground/langgraph-trello-ops-agent`) | 3.12 | ✅ | Disponible · mar 24 nov |
| 5 | [S12](s12-n8n-automatizacion/) | Automatización con n8n y cierre del módulo (notebook + `ai-agents-playground/n8n-workflows`) | 3.12 | ✅ | Disponible · jue 26 nov |

> **Numeración.** La S6 a la S8 de las carpetas corresponden a la S7 a la S9 del programa (las sesiones de embeddings y RAG simple se fusionaron en la Semana 2). La **S9 del repositorio es una sesión adicional** (RAG agéntico con LangChain) y desde la **S10** las carpetas **ya no siguen el programa uno a uno**: cada clase de 2,5 h agrupa temas (ver el cambio de calendario al final).
>
> **Dónde vive la práctica.** Desde la S7 hay un solo notebook por sesión y las implementaciones de referencia están en dos repositorios aparte: [`rag-playground`](https://github.com/OscarLlamas6/rag-playground) (RAG simple, intermedio y avanzado) y [`ai-agents-playground`](https://github.com/OscarLlamas6/ai-agents-playground) (un agente por carpeta). Desde la S7 el proveedor de LLM por defecto es **OpenAI**, con soporte para Anthropic y LM Studio (`LLM_PROVIDER`).

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

> **Cambio de calendario (noviembre de 2026).** Desde el 17 de noviembre las clases son martes y jueves de 2,5 h. La clase del **17 de noviembre** (S9) cubre también los temas de las sesiones S10 (agentes: bucle y límites) y S11 (memoria) del programa; MCP (S12 y S13 del programa) se da el 19, el agente con LangGraph (S15 del programa) el 24 y n8n (S14 del programa) el 26. Desde la S10 de las carpetas, este repositorio solo incluye notebook y marco teórico: el código de práctica está en `ai-agents-playground`.
