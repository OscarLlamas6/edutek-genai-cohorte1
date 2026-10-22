# Proveedor de LLM: Anthropic u OpenAI (y otros)

El repositorio funciona **de forma nativa con dos proveedores**: **Anthropic** (modelos Claude) y **OpenAI**. Se elige con **una variable de entorno** en tu archivo `.env`; el resto del código (notebooks, prácticas, `cliente_llm.py`) no cambia.

> Los dos caminos se prueban de la misma forma en clase, pero con una advertencia honesta: el instructor apoya con ambos, y las demostraciones sobre comportamientos específicos (por ejemplo, qué parámetros rechaza un modelo) se explican para los dos.

## Elegir el proveedor

En el `.env` de la raíz del repositorio (copia de `.env.example`):

```
# Opción A · Anthropic (por defecto)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Opción B · OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

Después de editar el `.env`, **reinicia el kernel** del notebook (las variables se leen al iniciar). Solo necesitas la llave del proveedor que uses.

| Variable | Para qué | Por defecto |
| --- | --- | --- |
| `LLM_PROVIDER` | Proveedor: `anthropic` u `openai` | `anthropic` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | La llave del proveedor elegido | (obligatoria) |
| `LLM_MODEL` | Modelo económico a usar | `claude-haiku-5-5` (Anthropic) · `gpt-6-luna` (OpenAI) |
| `LLM_MODEL_SAMPLING` | Modelo para experimentar con `temperature`/`top_p`/`top_k` (Sesión 1) | `claude-haiku-4-5` (Anthropic) · `gpt-6-luna` (OpenAI) |
| `OPENAI_BASE_URL` | URL de otro servicio compatible con OpenAI (Groq, OpenRouter…) | la de OpenAI |

Los nombres de modelo cambian cada pocos meses: si uno deja de existir, define `LLM_MODEL` (y actualiza la tabla `PRECIOS` de `cliente_llm.py`).

## Cómo funciona por dentro

`cliente_llm.py` ofrece una función común, `llamar_llm()`, que llama al proveedor activo y devuelve siempre el mismo objeto (`ResultadoLLM`: texto, tokens, tiempos, costo). Debajo hay una función por proveedor:

| Función | Proveedor | API que usa |
| --- | --- | --- |
| `llamar_claude()` | Anthropic | Messages API |
| `llamar_openai()` | OpenAI | Responses API |
| `llamar_local()` | Modelo local (LM Studio, Ollama, llama-server) | Chat Completions compatible con OpenAI |

## Diferencias que verás en las sesiones

| Qué | Anthropic | OpenAI |
| --- | --- | --- |
| **Modelo económico del curso** | `claude-haiku-5-5` (US$ 0.10 / 0.50 por millón de tokens) | `gpt-6-luna` (US$ 0.10 / 0.50) |
| **Límite de salida** | `max_tokens` | `max_output_tokens` (mínimo 16); incluye el razonamiento interno |
| **Razonamiento** | `output_config.effort` | `reasoning.effort` (incluye el nivel `none`) |
| **`temperature` / `top_p`** | `claude-haiku-5-5` los rechaza; `claude-haiku-4-5` los acepta | `gpt-6-luna` los acepta **solo con el razonamiento apagado** |
| **`top_k`** | Existe | **No existe** (nuestro cliente lo ignora) |
| **Conteo de tokens** | `count_tokens` (gratuito); `tiktoken` solo aproxima | `tiktoken` (oficial) y `responses.input_tokens.count` |
| **Caché de prompt** | Hay que pedirla (`cache_control`) | Automática (prefijos ≥ 1 024 tokens) |
| **Salida estructurada nativa** | `messages.parse(output_format=...)` | `responses.parse(text_format=...)` |
| **Instrucciones del sistema** | `system` | `instructions` |

En nuestro cliente el razonamiento se controla con `esfuerzo` = `"minimo"`, `"bajo"`, `"medio"` o `"alto"` (en OpenAI, `"minimo"` apaga el razonamiento; en Anthropic equivale a `effort: low`).

## Costos

Los dos proveedores son de **pago por uso**. Los precios de referencia (octubre de 2026) están en `docs/COSTOS.md` y en la tabla `PRECIOS` de `cliente_llm.py`. Los dos modelos económicos cuestan lo mismo, así que las estimaciones 💰 FYI de los notebooks valen para ambos. **Configura un límite de gasto** en la consola de tu proveedor.

## Embeddings (Sesiones 4 y 5)

Los embeddings se eligen **aparte** del LLM, con `EMBEDDINGS_PROVIDER`:

| Valor | Qué usa | Costo |
| --- | --- | --- |
| `local` (por defecto) | `nomic-embed-text-v2-moe` en LM Studio (multilingüe) | Gratis |
| `openai` | `text-embedding-3-small` | ≈ US$ 0.02 por millón de tokens |

**Anthropic no ofrece un modelo de embeddings propio**: con `LLM_PROVIDER=anthropic` usas embeddings locales u OpenAI. Los vectores de modelos distintos no se pueden mezclar: si cambias de modelo, reindexa todo. En la Sesión 5, `RAG_PROVIDER` (`anthropic`, `openai` o `local`) elige quién redacta las respuestas.

## Otros proveedores: Gemini, Groq, OpenRouter, Mistral…

Muchos servicios aceptan el **formato de la API de OpenAI**. Con el proveedor `openai` normalmente basta cambiar la URL base, la llave y el modelo:

```
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://<url-del-proveedor>/v1     # la indica la documentación de tu proveedor
OPENAI_API_KEY=<la llave de ese proveedor>
LLM_MODEL=<modelo-de-ese-proveedor>
```

Esto **no está probado por el curso** y no hay garantía de que cada ejemplo se comporte igual: muchos servicios compatibles solo implementan Chat Completions y no la Responses API que usa `llamar_openai()`. Si es tu caso, adapta esa función (o usa `llamar_local()`, que habla Chat Completions). Revisa también la tabla `PRECIOS`: un modelo que no esté en ella cuenta como costo 0.

Con Gemini, Mistral u otro proveedor sin modo de compatibilidad, escribe una función equivalente en `cliente_llm.py` que devuelva un `ResultadoLLM`.

## Modelos locales (sin costo por token)

Con **LM Studio** (el del repositorio), Ollama o llama-server puedes ejecutar modelos *open-weight* pequeños en tu equipo. Ver `docs/LMSTUDIO.md` y `docs/INSTALACION.md`, sección 8.
