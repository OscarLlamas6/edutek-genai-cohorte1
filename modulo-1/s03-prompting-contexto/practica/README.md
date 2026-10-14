# Práctica S3 · Clasificador de tickets

**Pregunta que responde:** ¿puedo convertir texto libre de un usuario en datos confiables usando un prompt versionado y una validación?

El programa está completo y comentado en [`clasificador_tickets.py`](clasificador_tickets.py). Junta los tres módulos de la sesión: `cliente_llm.py` (llamar), `plantillas.py` (prompt versionado) y `validacion.py` (salida validada).

## Cómo ejecutarla

```bash
# desde la carpeta s03-prompting-contexto/
uv run python practica/clasificador_tickets.py "El switch del piso 3 reinicia sin parar"
uv run python practica/clasificador_tickets.py "No puedo entrar a la VPN" --version v2
uv run python practica/clasificador_tickets.py --evaluar --version v3        # 10 casos de prueba
```

> 💰 **Costo:** una llamada al modelo económico de tu proveedor por ticket (≈ US$ 0.0001). `--evaluar` corre 10 casos: ≈ US$ 0.001. Evaluar las tres versiones: ≈ US$ 0.003.

## Qué observar

1. Mira las tres plantillas en [`../prompts/clasificador_tickets/`](../prompts/clasificador_tickets/) y el [`CHANGELOG.md`](../prompts/CHANGELOG.md): ¿qué agrega cada versión?
2. Corre `--evaluar` con `v1`, `v2` y `v3`. ¿Cambia el JSON válido? ¿Y la categoría correcta?
3. ¿Qué casos fallan siempre? ¿Son los ambiguos (C-07 y C-09)?

## Retos

1. Crea `prompts/clasificador_tickets/v4.j2` cambiando **una sola cosa**, anótalo en el `CHANGELOG.md` y compara con `v3`.
2. Añade 5 casos difíciles a `datos/casos_tickets.json` y reevalúa.
3. Prueba con tickets de **tu** trabajo (inventados o anonimizados, nunca datos reales).

## Criterios de "terminado"

- [ ] Una plantilla en archivo, con variables y delimitadores, guardada en Git.
- [ ] La salida se valida contra un esquema; una respuesta inválida produce un error claro.
- [ ] Probaste al menos cinco entradas distintas y comparaste dos versiones del prompt.

## Para la puesta en común

¿Qué cambio de prompt mejoró más el resultado? ¿Algún caso falló siempre, sin importar la versión?
