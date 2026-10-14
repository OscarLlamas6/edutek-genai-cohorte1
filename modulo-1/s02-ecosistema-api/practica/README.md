# Práctica S2 · Comparador de modelos

**Pregunta que responde:** ¿qué diferencia hay, en respuesta, tokens, tiempo y costo, entre un modelo propietario y uno open-weight con el mismo prompt?

El programa está completo y comentado en [`comparador_modelos.py`](comparador_modelos.py). Usa [`../cliente_llm.py`](../cliente_llm.py), que ya implementa el streaming, los reintentos y las mediciones: **léelo**, es el módulo que usaremos todo el curso.

## Cómo ejecutarla

```bash
# desde la carpeta s02-ecosistema-api/
uv run python practica/comparador_modelos.py "Explica qué es un balanceador de carga en 3 frases"
uv run python practica/comparador_modelos.py "..." --a api --b local                                  # modelo económico de tu proveedor vs. modelo local
uv run python practica/comparador_modelos.py "..." --a api --b local:<identificador>                   # un modelo local concreto
uv run python practica/comparador_modelos.py "..." --a anthropic:claude-haiku-5-5 --b openai:gpt-6-luna   # sin modelo local (requiere las dos llaves)
```

> 💰 **Costo:** la llamada por API cuesta una fracción de centavo (≈ US$ 0.0003 con el modelo económico y `--max-tokens 400`). El modelo local (LM Studio) corre en tu computadora y **no cobra por token**, pero usa tu CPU, RAM y electricidad.

## Qué observar

1. ¿Qué modelo respondió mejor? ¿Y más rápido? ¿Cuál tuvo menor TTFT?
2. ¿Cuántos tokens usó cada uno para la misma respuesta?
3. ¿Qué cambia si bajas `--max-tokens` a 50?

## Retos

1. Prueba cuatro tipos de prompt: un cálculo, una redacción, código y una pregunta en inglés. ¿Cambia el ganador?
2. Ejecuta el mismo comando 5 veces. ¿Cuánto varían el TTFT y los tokens por segundo?
3. Añade al comparador una opción `--json` que pida la respuesta en JSON y la valide con `pedir_json()` de `cliente_llm.py`.

## Criterios de "terminado"

- [ ] Ves las dos respuestas aparecer en vivo (streaming).
- [ ] La tabla final muestra tokens, TTFT, tiempo total, tokens/segundo y costo por modelo.
- [ ] Puedes explicar cuál elegirías para un chatbot interno y por qué.

## Para la puesta en común

¿Qué diferencias notaron entre los dos modelos? ¿Alguien vio un error y cómo lo manejó el reintento?
