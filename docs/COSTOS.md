# Costos del curso: qué consume dinero y qué no

Queremos ser **100 % transparentes** contigo. Este documento dice qué cuesta dinero, cuánto, y cómo protegerte.

> El repositorio funciona con **Anthropic u OpenAI** (variable `LLM_PROVIDER`; ver [`PROVEEDORES.md`](PROVEEDORES.md)). Los dos modelos económicos del curso cuestan lo mismo, así que las estimaciones valen para ambos.

## Qué usamos y cómo se cobra

| Servicio | Cómo se cobra | Se usa en |
| --- | --- | --- |
| **API del proveedor elegido** (Anthropic Platform u OpenAI Platform) | **Pago por uso**: por token de entrada y de salida | Todo el curso |
| LM Studio, Ollama o llama-server (modelos open-weight locales) | **No hay cobro por token.** Usa tu CPU, RAM y electricidad | Sesión 2 |
| Tokenizadores locales (`tiktoken`, Hugging Face `tokenizers`) | Gratis (descarga única de tablas) | Sesión 1 |
| Conteo de tokens del proveedor | Anthropic (`count_tokens`): **gratis**, con límite de peticiones. OpenAI (`responses.input_tokens.count`): no genera texto; confirma en su página de precios si se factura | Sesión 1 |
| Consulta de la lista de modelos del proveedor | Gratis (no genera tokens) | Sesión 2 |
| Peticiones **rechazadas** con error (400, 401, 404) | No se cobran: no generan tokens | Sesión 2 |
| **Embeddings locales** (LM Studio, `nomic-embed-text-v2-moe`) | Sin cobro por token | Sesiones 4 y 5 |
| **Embeddings de OpenAI** (opcional, `text-embedding-3-small`) | Pago por uso: ≈ US$ 0.02 por millón de tokens (fracciones de centavo en el curso) | Sesiones 4 y 5 |
| **Neon** (PostgreSQL + pgvector), plan **Free** | US$ 0, **sin tarjeta**; cuota de 1 GB y 100 CU-horas por proyecto | Sesiones 4 y 5 |
| **Pinecone**, plan **Starter** | US$ 0; cuota de 2 GB, 2 M unidades de escritura y 1 M de lectura al mes, 5 índices | Sesiones 4 y 5 |

Las sesiones 6 a 8 no añaden servicios nuevos: usan los mismos embeddings locales y el mismo proveedor de LLM. LangChain (S8) es una biblioteca gratuita; su servicio de trazas (LangSmith) **no se usa** y queda apagado.

Más adelante pueden aparecer otros servicios. **Se avisará de forma explícita en el README de cada sesión antes de usarlos.**

### Neon y Pinecone: cuotas, no dinero

Con los planes gratuitos **no se te cobra**: si te pasas de la cuota, el servicio se limita o se detiene hasta el mes siguiente (o hasta que decidas subir de plan). Los planes de pago (Neon Launch/Scale, Pinecone Standard con mínimo mensual) **solo se activan si tú los contratas**. Los precios cambian: confirma en [neon.com/pricing](https://neon.com/pricing) y [pinecone.io/pricing](https://www.pinecone.io/pricing/). Guías de cuenta: [`CUENTA_NEON.md`](CUENTA_NEON.md) y [`CUENTA_PINECONE.md`](CUENTA_PINECONE.md).

## Cómo se calcula el costo de una llamada

```
costo = (tokens de entrada × precio de entrada + tokens de salida × precio de salida) / 1 000 000
```

Los precios se expresan en **USD por millón de tokens**. Los tokens de salida cuestan varias veces más que los de entrada, y los tokens de **razonamiento interno** de los modelos que "piensan" se cobran **como salida**.

Precios de referencia usados en el curso (**octubre de 2026**; confirma siempre en la página oficial de precios de tu proveedor):

| Modelo | Entrada (USD / M tokens) | Salida (USD / M tokens) | Uso en el curso |
| --- | --- | --- | --- |
| `claude-haiku-5-5` (Anthropic) | 0.10 | 0.50 | Modelo principal para practicar con Anthropic |
| `gpt-6-luna` (OpenAI) | 0.10 | 0.50 | Modelo principal para practicar con OpenAI |
| `claude-haiku-4-5` (Anthropic) | 1.00 | 5.00 | Con Anthropic, cuando necesitamos ajustar `temperature` / `top_p` / `top_k` o evitar el razonamiento interno (con OpenAI se usa `gpt-6-luna` con el razonamiento apagado) |
| `claude-sonnet-5-5` | 2.00 | 10.00 | Solo como referencia de precios |
| `claude-opus-5-5` | 4.00 | 20.00 | Solo como referencia de precios |
| `claude-fable-5-1` | 10.00 | 50.00 | Solo como referencia de precios |
| `gpt-6-sol`, `gpt-6-astra` (OpenAI) | 2.00 · 10.00 | 10.00 · 50.00 | Solo como referencia de precios |

## Consumo estimado por sesión

Estimaciones calculadas con las llamadas del notebook y de las prácticas. Tu gasto real puede variar un poco (por ejemplo, si repites celdas).

| Sesión | Llamadas de pago (aprox.) | Costo estimado del notebook | Práctica |
| --- | --- | --- | --- |
| S1 · Fundamentos del LLM | ~20 | < US$ 0.01 | US$ 0 (todo local; `--exacto` no genera texto) |
| S2 · Ecosistema y API | ~8 a 12 | < US$ 0.01 | ~US$ 0.0003 por ejecución |
| S3 · Prompting y context engineering | ~50 | < US$ 0.01 | ~US$ 0.001 por evaluación de 10 casos |
| S4 · Embeddings y bases vectoriales | 0 con embeddings locales | US$ 0 (con embeddings de OpenAI: < US$ 0.001) | US$ 0 (con OpenAI: fracciones de centavo) |
| S5 · Fundamentos de RAG | ~25 al LLM (con modelo de proveedor); +14 por cada proveedor extra que compares | < US$ 0.02 (US$ 0 con `RAG_PROVIDER=local`) | ≈ US$ 0.0005 por pregunta |
| S6 · RAG intermedio | ~30 al LLM en las partes 8 a 10 (con modelo de proveedor) | < US$ 0.02 (US$ 0 con modelo local) | US$ 0 (el laboratorio de chunking no llama al LLM) |
| S7 · RAG avanzado y evaluación | descomposición, HyDE, generación y juez (con modelo de proveedor) | < US$ 0.15, cota superior (US$ 0 con modelo local) | ≈ US$ 0.0005 por pregunta evaluada; el juez suma otra llamada por pregunta |
| S8 · Agentes con LangChain | un agente hace varias llamadas por pregunta | < US$ 0.03 (US$ 0 con modelo local) | ≈ US$ 0.0005 a 0.002 por pregunta |
| S9 a S14 | _Se publicará en cada sesión_ | | |

## Cómo vas a reconocer lo que cuesta dinero

1. **README de la sesión:** tabla "💰 Qué consume dinero".
2. **Portada del notebook:** tabla de costos y mapa de partes con la columna "¿Consume dinero?".
3. **Cada celda de pago** empieza con un comentario `# 💰 FYI: esta celda consume dinero (≈ US$ …)`.
4. **Contador de gasto** (`gasto`) en los notebooks: suma el costo real de cada llamada a partir del campo `usage` que devuelve la API.

## Cómo protegerte (haz esto antes de la primera clase)

1. **Configura un límite de gasto mensual** en la consola de tu proveedor (Anthropic Platform u OpenAI Platform). Es la protección más importante: aunque algo falle, no pasarás del tope.
2. **Carga un saldo pequeño** (unos pocos dólares) en lugar de dejar una tarjeta sin tope.
3. **Usa `max_tokens` bajo** al practicar: es el control más directo del costo de salida.
4. **Prefiere modelos pequeños** (`claude-haiku-5-5`) para aprender.
5. **Usa el interruptor** `EJECUTAR_LLAMADAS_DE_PAGO = False` de los notebooks si solo quieres leer o no tienes llave.
6. **No compartas tu API key.** Quien la tenga gasta tu saldo. Si la expones por error, **revócala de inmediato** en la consola y crea otra.
7. **Revisa tu consumo** en la consola al terminar cada sesión.

## Preguntas frecuentes

**¿Puedo hacer el curso sin gastar nada?**
Puedes seguir la parte conceptual y todo lo que corre en local (tokenizadores, simulaciones, validación, plantillas). Para las llamadas a modelos necesitas una llave; en clase se puede trabajar en pareja o siguiendo la pantalla del instructor.

**¿Qué pasa si me quedo sin saldo?**
La API responde con un error y no se cobra más. Recarga y vuelve a ejecutar la celda.

**¿Los datos que envío se guardan?**
Depende de los términos del proveedor. Por eso en el curso usamos **solo datos de ejemplo**, nunca datos reales ni confidenciales.
