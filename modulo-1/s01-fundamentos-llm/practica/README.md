# Práctica S1 · Calculadora de tokens y costo

**Pregunta que responde:** ¿cuántos tokens tiene este texto y cuánto costaría enviarlo a un modelo?

El programa está completo y comentado línea a línea en [`calculadora_tokens.py`](calculadora_tokens.py). La práctica consiste en **ejecutarlo, leerlo y modificarlo**.

## Cómo ejecutarla

```bash
# desde la carpeta s01-fundamentos-llm/
uv run python practica/calculadora_tokens.py datos/ticket_es.txt
uv run python practica/calculadora_tokens.py datos/ticket_en.txt --modelo claude-opus-5-5 --salida 500   # o --modelo gpt-6-astra
uv run python practica/calculadora_tokens.py datos/ticket_es.txt --tokenizadores o200k cl100k qwen
uv run python practica/calculadora_tokens.py datos/ticket_es.txt --exacto
```

> 💰 **Costo: US$ 0.** Todo se calcula en tu computadora. La opción `--exacto` usa el servicio de conteo de tokens de tu proveedor, que **no genera texto**: en Anthropic es **gratuito** (con límite de peticiones); en OpenAI, confirma en su página de precios. Necesita tu API key.

## Qué observar

1. ¿Cuántos tokens usa el ticket en español frente al de inglés? ¿Cuánto más cuesta con un modelo caro?
2. ¿Los tokenizadores coinciden entre sí? ¿Y con el conteo exacto de tu proveedor (`--exacto`)?
3. ¿Qué pasa con el bloque de código (`datos/log_codigo.txt`)?

## Retos

1. Prueba con **tu propio texto** (un log, un correo; datos de ejemplo, nunca reales).
2. Añade una columna que muestre el costo de **10 000 llamadas**.
3. Añade un cuarto tokenizador de tu elección.

## Criterios de "terminado"

- [ ] Ejecutaste la calculadora con el texto en español, el de inglés y el código.
- [ ] Puedes explicar por qué el español gasta más tokens que su equivalente en inglés.
- [ ] Puedes explicar por qué el conteo de `tiktoken` no es exacto para Claude (y por qué para modelos de OpenAI sí es la herramienta oficial).

## Para la puesta en común

¿Qué te sorprendió del tokenizador? ¿Cuánto cambió la respuesta con temperatura alta?
