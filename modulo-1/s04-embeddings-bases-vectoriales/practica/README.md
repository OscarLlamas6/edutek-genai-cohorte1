# Práctica S4 · Buscador de documentos

Programa único y comentado: [`buscador_documentos.py`](buscador_documentos.py).

**Pregunta que responde:** ¿puedo encontrar el párrafo que responde a una pregunta aunque no use las mismas palabras?

## Ejecutar (desde la carpeta de la sesión)

```bash
uv run python practica/buscador_documentos.py --almacen memoria --indexar --buscar "se venció el certificado"
uv run python practica/buscador_documentos.py --almacen neon --indexar
uv run python practica/buscador_documentos.py --almacen neon --buscar "no puedo entrar a mi cuenta"
uv run python practica/buscador_documentos.py --almacen pinecone --indexar --buscar "¿cada cuánto cambian la clave?"
uv run python practica/buscador_documentos.py --comparar "¿cuánto se conservan las copias de seguridad?"
uv run python practica/buscador_documentos.py --almacen neon --vaciar     # borra los datos
```

La memoria no persiste: úsala siempre con `--indexar` y `--buscar` juntos.

## Terminado cuando…

- Indexaste las notas en al menos un almacén persistente (Neon o Pinecone) y buscaste con palabras distintas a las del texto.
- Puedes explicar de dónde viene cada resultado (fuente y puntaje).
- Hiciste al menos uno de los retos.

## Retos

1. Busca con sinónimos y preguntas largas. ¿Dónde falla?
2. Añade un `.md` a `datos/notas/`, reindexa y comprueba que lo encuentra.
3. Compara Neon y Pinecone con `--comparar`.
4. Cambia la forma de fragmentar (frases, 2 párrafos) y mide el efecto.
