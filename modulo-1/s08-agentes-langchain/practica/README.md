# Práctica S8 · Bot de estado de servicios

Programa único y comentado: [`bot_estado_servicios.py`](bot_estado_servicios.py).

**Pregunta que responde:** ¿puedo dar a un modelo herramientas (consultar el estado de un servicio, buscar en un manual) y dejar que **él** decida cuándo usarlas, en vez de programar cada camino?

## Ejecutar (desde la carpeta de la sesión)

```bash
uv run python practica/bot_estado_servicios.py "¿Cómo está la app móvil?"
uv run python practica/bot_estado_servicios.py "¿Hay algún servicio caído?"
uv run python practica/bot_estado_servicios.py "No puedo pagar con tarjeta, ¿qué hago?"
uv run python practica/bot_estado_servicios.py --a-mano "¿Cómo está la app móvil?"          # sin LangChain
uv run python practica/bot_estado_servicios.py --herramientas todas "Abre un ticket: los pagos con tarjeta fallan desde las 5 pm"
uv run python practica/bot_estado_servicios.py --chat                                       # con memoria
uv run python practica/bot_estado_servicios.py --evaluar                                    # mide la elección de herramientas
uv run python practica/bot_estado_servicios.py --ver-esquemas                               # lo que el modelo recibe de cada herramienta
uv run python practica/bot_estado_servicios.py --proveedor local "¿Cuánto es el 15 % de 39?"
```

Opciones: `--proveedor` (`anthropic`, `openai` o `local`; por defecto `AGENT_PROVIDER` o `LLM_PROVIDER`) y `--herramientas` (`estado`, `lectura` —por defecto— o `todas`, que añade `reportar_falla`, la única que **escribe**).

El programa muestra la **traza** (qué pidió el modelo y qué devolvió cada herramienta), las llamadas al modelo, los tokens y el costo.

> **Modelo local:** un modelo de 3B parámetros (`llama-3.2-3b-instruct`) elige mal las herramientas. Para esta práctica usa uno de ~12B (por ejemplo `google/gemma-4-12b-qat`, con `LOCAL_LLM_MODEL` en tu `.env`) o un proveedor.

## Terminado cuando…

- Haces al menos cuatro preguntas distintas y, mirando la traza, explicas **por qué** el modelo eligió cada herramienta (o ninguna).
- Una pregunta que no necesita herramientas (un saludo, una cuenta) se responde **sin llamarlas**.
- Ejecutaste `--evaluar` y puedes decir cuántas de las 6 o 7 preguntas eligieron bien las herramientas, y cuál falló.
- Comparaste `--a-mano` con la versión de LangChain y puedes señalar qué hace LangChain por ti.

## Retos

1. **Tu herramienta:** añade `servicios_con_problemas()` a [`../herramientas.py`](../herramientas.py) (solo los servicios que no estén `operativo`) y mide con `--evaluar` si mejora o empeora la elección.
2. **El agente lee el mundo, no lo recuerda:** edita [`../datos/servicios.json`](../datos/servicios.json), pon el portal en `"caido"` y vuelve a preguntar.
3. **Descripción vaga:** empeora a propósito el docstring de una herramienta y repite `--evaluar`. ¿Cuánto cae el acierto?
4. **Modelos:** compara un modelo de 3B, uno de ~12B y un proveedor con `--evaluar`.
5. **Mínimo privilegio:** compara `--herramientas lectura` y `--herramientas todas` con "¿cómo está la app móvil?". ¿Alguno creó un ticket sin que se lo pidieran?
6. **Un límite:** limita `reportar_falla` a una llamada por ejecución con `ToolCallLimitMiddleware` y pide dos tickets seguidos.
