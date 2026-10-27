# LM Studio: guía de configuración

LM Studio es el programa que usamos en el curso para ejecutar **modelos open-weight en tu propio equipo**. Lo usamos primero en la **Sesión 2** (comparar un modelo local con Claude) y nos servirá más adelante para modelos de *embeddings* (Semana 2). Todo corre en tu computadora y **no hay cobro por token**.

> Si prefieres otro programa (Ollama, llama-server…), también sirve: ver [INSTALACION.md, sección 8](INSTALACION.md#8-modelos-open-weight-locales-sesión-2). Esta guía cubre solo LM Studio.

## 1. Instalar

Descarga LM Studio desde su página oficial (lmstudio.ai). Hay versiones para Windows, macOS y Linux. Es una aplicación de escritorio con interfaz gráfica; al instalarla también deja disponible la herramienta de línea de comandos **`lms`**.

## 2. Descargar un modelo de chat

Necesitas un modelo **de chat (instruct)**, pequeño, que quepa en tu equipo. Para el curso: **Llama 3.2 3B Instruct** (≈ 2 GB; cabe en 16 GB de RAM sin GPU).

**Desde la interfaz gráfica:** pestaña de búsqueda → busca "Llama 3.2 3B Instruct" → descarga la versión cuantizada recomendada (Q4_K_M o similar).

**Desde la terminal:**

```bash
lms get llama-3.2-3b-instruct      # busca y descarga (te pide elegir si hay varias opciones)
lms ls                             # lista los modelos descargados
```

Debes ver algo así:

```
LLM                      PARAMS    ARCH     SIZE       DEVICE
llama-3.2-3b-instruct    3B        Llama    2.02 GB    Local
```

> No confundas **modelos de chat** (LLM) con **modelos de embeddings**: estos últimos convierten texto en vectores y **no sirven para conversar**. Si `lms ls` solo muestra modelos de la sección `EMBEDDING`, aún te falta descargar uno de chat.

> **Bionic y la nube:** LM Studio Bionic tiene un asistente propio que usa un modelo alojado en la nube (se reconoce por el ícono de nube junto al nombre del modelo). **No es un modelo local**: no aparece en el servidor local ni sirve para el curso.

## 2b. Descargar un modelo de embeddings (Sesiones 4 y 5)

Las sesiones de embeddings y RAG necesitan, **además del modelo de chat**, un **modelo de embeddings**. Usamos **`nomic-embed-text-v2-moe`** (≈ 344 MB, 768 dimensiones, **multilingüe**). Los modelos centrados en inglés (como `nomic-embed-text-v1.5`) recuperan mucho peor en español.

**Desde la interfaz gráfica:** busca "nomic embed text v2 moe" y descarga la versión `Q4_K_M`.

**Desde la terminal:**

```bash
lms get https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF
lms ls        # debe aparecer en la sección EMBEDDING como text-embedding-nomic-embed-text-v2-moe
```

No hace falta "cargarlo" a mano: LM Studio lo carga cuando el código lo pide, siempre que el servidor local esté activo. El nombre exacto que muestra `lms ls` es el que va en `EMBEDDINGS_MODEL` si lo quieres fijar.

> Si no puedes descargarlo, usa los embeddings de OpenAI (`EMBEDDINGS_PROVIDER=openai` en tu `.env`): cuestan fracciones de centavo.

## 2c. Un modelo más grande para agentes (Sesión 8)

El modelo de 3B de la sección 2 alcanza para ver el flujo de un RAG, pero **no para un agente**: en la Sesión 8, con las mismas herramientas y preguntas, el 3B eligió bien la herramienta en 2 de 6 casos y un modelo de unos 12 000 millones de parámetros en 6 de 6. Si no quieres usar un proveedor de pago, descarga uno de ~12B que admita herramientas (por ejemplo `google/gemma-4-12b-qat`, ≈ 7 GB; necesita unos 10 GB de RAM libres):

```bash
lms get google/gemma-4-12b-qat
lms load google/gemma-4-12b-qat    # cárgalo a mano: la carga automática puede fallar la primera vez
```

Y en tu `.env`: `LOCAL_LLM_MODEL=google/gemma-4-12b-qat`. Si tu equipo no lo aguanta, usa un proveedor de pago para esa sesión (cuesta centavos).

## 3. Activar el servidor local

LM Studio expone los modelos mediante una API compatible con el formato de OpenAI.

**Desde la interfaz:** pantalla **Local Model API** (en *Developer*):

1. Activa **Local API server** (debe decir *Running*).
2. Copia la **Base URL** que aparece. Por defecto suele ser `http://localhost:1234/v1`, pero **puede ser otro puerto** (por ejemplo `1235`). Usa siempre el que muestra tu pantalla.
3. Deja activado **Just-in-time model loading**: así el modelo se carga solo cuando llega la primera petición, sin cargarlo a mano.

**Desde la terminal:**

```bash
lms server status                  # ¿está corriendo?
lms server start                   # inícialo (consulta `lms server --help` si cambia la sintaxis)
```

## 4. Verificar

Con el servidor activo (cambia el puerto por el tuyo):

```bash
curl http://localhost:1234/v1/models
```

Debe devolver una lista que incluya `llama-3.2-3b-instruct`. Prueba también una conversación:

```bash
curl http://localhost:1234/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "llama-3.2-3b-instruct",
  "messages": [{"role": "user", "content": "Hola, ¿qué eres?"}],
  "max_tokens": 80
}'
```

> **La primera petición tarda más**: con *Just-in-time loading*, LM Studio carga el modelo en memoria en ese momento. Las siguientes son mucho más rápidas. Esta diferencia se ve en el **TTFT** de la práctica.

## 5. Conectar el repositorio

En el archivo `.env` de la raíz del repositorio (copia de `.env.example`) define:

```
LOCAL_LLM_URL=http://localhost:1234/v1        # la Base URL de tu pantalla Local Model API
LOCAL_LLM_MODEL=llama-3.2-3b-instruct         # el identificador que muestra `lms ls`
```

Prueba desde `modulo-1/s02-ecosistema-api/`:

```bash
uv run python practica/comparador_modelos.py "Explica qué es un balanceador de carga en 3 frases" --a local --b local
```

Ves la respuesta en vivo y una tabla con tokens, TTFT, tiempo y costo (US$ 0 por token). El notebook de la Sesión 2 usa la misma configuración (Parte 5).

## 6. Comandos útiles de `lms`

| Qué quieres hacer | Comando |
| --- | --- |
| Ver los modelos descargados | `lms ls` |
| Ver los modelos cargados en memoria | `lms ps` |
| Buscar y descargar un modelo | `lms get <nombre>` |
| Cargar / descargar de memoria un modelo | `lms load <modelo>` / `lms unload <modelo>` |
| Chatear en la terminal | `lms chat <modelo>` |
| Estado del servidor local | `lms server status` |
| Ver las peticiones que llegan al servidor | `lms log stream` (o la sección *Server logs* de la pantalla Local Model API) |
| Ayuda de cualquier comando | `lms <comando> --help` |

## 7. Solución de problemas

| Síntoma | Causa probable | Solución |
| --- | --- | --- |
| `No hay servidor local en http://localhost:1234/v1` | El servidor no está activo o el puerto es otro | Activa *Local API server* y copia la Base URL exacta a `LOCAL_LLM_URL` |
| `El modelo '...' no aparece en el servidor local` | El identificador no coincide | Copia el identificador exacto de `lms ls` o de `/v1/models` a `LOCAL_LLM_MODEL` |
| `/v1/models` solo muestra modelos de embeddings | No has descargado un modelo de chat | Descarga uno (sección 2) |
| `lms ls` falla con **"Invalid passkey for lms CLI client"** | Quedó un proceso `llmster` viejo ejecutándose junto a la aplicación y `lms` se conecta a ese | Cierra ese proceso (`pkill llmster`) y vuelve a ejecutar `lms ls`: se levanta uno nuevo. Si reaparece, cierra la aplicación o el servicio que no uses |
| La primera respuesta tarda mucho | El modelo se está cargando en memoria (*Just-in-time loading*) | Es normal; las siguientes serán rápidas. Puedes precargarlo con `lms load` |
| Muy lento en general | Equipo sin GPU o modelo grande | Usa un modelo más pequeño o reduce `max_tokens` |
| Las columnas de tokens salen en 0 en el comparador | Tu versión del servidor no informa el uso de tokens al hacer streaming | Es solo informativo; el resto de la comparación funciona |
| Usas WSL2 y LM Studio en Windows | El servidor corre en Windows y `localhost` de WSL2 no lo alcanza | Prueba la IP del host de Windows, la red en modo espejo de WSL2 (*mirrored*), o ejecuta el notebook en Windows. Ajusta `LOCAL_LLM_URL` |

## 8. Recomendaciones para la clase

- **Descarga el modelo antes de la clase**: son ≈ 2 GB y la red del aula se satura.
- **Anota tu puerto y tu identificador de modelo** y compruébalos con `curl` antes de empezar.
- **No te asustes si el modelo local se equivoca**: un modelo de 3B puede confundir conceptos o inventar. Es parte de la lección sobre elegir un modelo con criterio (compáralo con Claude).
- Los **tiempos** (TTFT, tokens/s) dependen de tu hardware: no los compares con los de un compañero que tenga otra máquina.
