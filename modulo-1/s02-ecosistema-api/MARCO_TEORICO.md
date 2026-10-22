# Marco teórico · Sesión 2: Ecosistema de modelos y consumo vía API

> Documento de apoyo al notebook `s02_ecosistema_api.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: el código y los experimentos están en el notebook.

## Índice

1. [El ecosistema de modelos](#1-el-ecosistema-de-modelos)
2. [Modelos propietarios vs. open-weight](#2-modelos-propietarios-vs-open-weight)
3. [Formas de acceder a un modelo](#3-formas-de-acceder-a-un-modelo)
4. [Anatomía de una llamada a la API](#4-anatomía-de-una-llamada-a-la-api)
5. [Streaming y métricas de latencia](#5-streaming-y-métricas-de-latencia)
6. [Errores y reintentos](#6-errores-y-reintentos)
7. [Criterios para elegir un modelo](#7-criterios-para-elegir-un-modelo)
8. [Build vs. Buy](#8-build-vs-buy)
9. [Salida estructurada](#9-salida-estructurada)
10. [Glosario rápido](#10-glosario-rápido)

---

## 1. El ecosistema de modelos

- **Modelo fundacional (*foundation model*):** modelo grande entrenado con datos masivos y de propósito general, que se adapta a muchas tareas (texto, imagen, código…).
- **Familia de modelos:** conjunto de modelos de un mismo proveedor en distintos tamaños y precios (p. ej. uno pequeño y rápido, uno mediano, uno grande y potente).
- **Modelo *instruct* / de chat:** versión ajustada para seguir instrucciones y conversar, en oposición al modelo "base" que solo continúa texto.
- **Modelo de razonamiento:** modelo que dedica tokens extra a "pensar" antes de responder; suele ser más preciso en problemas complejos pero más lento y costoso.
- **Modelo multimodal:** acepta o produce más de un tipo de dato (texto, imágenes, audio).
- **Versión / alias de modelo:** los proveedores publican nombres que cambian cada pocos meses. **Los nombres cambian; los criterios de elección no.**

## 2. Modelos propietarios vs. open-weight

| Aspecto | Propietario (cerrado) | Open-weight (pesos abiertos) |
| --- | --- | --- |
| Acceso | Solo vía API del proveedor | Pesos descargables; puedes alojarlo tú |
| Control y privacidad | El dato viaja al proveedor | El dato puede quedarse en tu infraestructura |
| Calidad de punta | Normalmente la mayor | Cada vez más cercana, según tamaño |
| Costo | Pago por uso (tokens) | Costo de hardware/operación, sin pago por token |
| Personalización | Limitada (según el proveedor) | Amplia (ajuste fino, cuantización, etc.) |
| Operación | La resuelve el proveedor | La resuelves tú |

Definiciones:

- **Modelo propietario:** pesos y arquitectura no públicos; solo se usa mediante el servicio del proveedor.
- **Open-weight:** el proveedor publica los **pesos** del modelo para descargarlos y ejecutarlos.
- **Open source (código abierto):** además de los pesos, implica una **licencia** que permite uso, modificación y redistribución con pocas restricciones. **Abierto ≠ open source:** muchos modelos "abiertos" tienen licencias con restricciones; siempre se debe leer la licencia.
- **Licencia:** términos legales que definen qué puedes hacer con el modelo (uso comercial, redistribución, límites).
- **Cuantización:** técnica que reduce la precisión numérica de los pesos (p. ej. de 16 a 4 bits) para que el modelo ocupe menos memoria y corra en hardware modesto, a costa de algo de calidad.
- **Parámetros (7B, 70B…):** el tamaño del modelo; más parámetros suele implicar más calidad pero más memoria y cómputo.

## 3. Formas de acceder a un modelo

1. **API directa del proveedor:** el camino más simple; pagas por token.
2. **A través de una nube (Bedrock, Azure, Vertex AI):** el mismo modelo propietario, consumido por una plataforma cloud, con controles de red, identidad, facturación y cumplimiento de la empresa.
3. **Ejecución local:** herramientas como **LM Studio**, **Ollama** o **llama-server** (llama.cpp) permiten correr modelos open-weight en tu propio equipo y exponerlos con una API local.

Términos asociados:

- **Servidor de inferencia / serving:** software que carga un modelo y atiende solicitudes (por ejemplo, por HTTP).
- **API compatible con OpenAI:** convención de endpoints y formato de mensajes que muchos servidores locales imitan, de modo que el mismo cliente sirve para varios backends.
- **Base URL / endpoint:** dirección donde escucha la API (en local, normalmente `http://localhost:<puerto>/v1`).
- **Hosting gestionado:** terceros que alojan modelos abiertos y los venden por API.
- **GPU / VRAM:** el hardware (y su memoria de video) que limita qué modelos puedes ejecutar localmente.

## 4. Anatomía de una llamada a la API

Toda llamada a un LLM tiene tres partes:

1. **Qué modelo** usar y con qué **parámetros** (temperatura, `max_tokens`…).
2. **Los mensajes** (instrucciones + conversación).
3. **La respuesta** (texto generado + metadatos de uso).

**Roles de los mensajes:**

- **`system`:** instrucciones de alto nivel que fijan el comportamiento, tono y reglas del asistente.
- **`user`:** lo que pide la persona (o tu aplicación).
- **`assistant`:** lo que respondió el modelo; se reenvía en turnos siguientes para dar continuidad.

Conceptos clave:

- **Sin estado (*stateless*):** la API no recuerda llamadas previas; tú reenvías el historial cada vez.
- **Objeto de uso (`usage`):** informa tokens de entrada y salida; es la base para calcular el costo.
- **Prompt de sistema:** el mensaje `system`; cambiarlo cambia el comportamiento sin cambiar el modelo.
- **Cliente (SDK):** librería que envuelve las llamadas HTTP a la API del proveedor.
- **Capa de abstracción:** un módulo propio que unifica proveedores distintos tras la misma interfaz, para no acoplar el código a uno solo.

## 5. Streaming y métricas de latencia

- **Latencia:** tiempo entre la solicitud y la respuesta.
- **Streaming:** el servidor envía la respuesta **por partes, a medida que se genera**, en lugar de esperar a que termine. Mejora la percepción de velocidad.
- **TTFT (*Time To First Token*):** tiempo hasta que llega el **primer token**. Define cuánto "tarda en empezar a responder".
- **Tokens por segundo (TPS):** velocidad de generación una vez iniciada la respuesta. Define cuán fluida se siente la lectura.
- **Latencia total:** TTFT + tiempo de generar el resto.
- **Evento / *chunk*:** cada fragmento que llega durante un streaming.

En un asistente conversacional importan **ambas métricas**: TTFT (cuánto tarda en empezar) y TPS (qué tan rápido fluye).

## 6. Errores y reintentos

- **Error transitorio:** falla temporal que puede resolverse reintentando.
  - **429 (*Too Many Requests*):** límite de tasa (*rate limit*) excedido.
  - **5xx:** fallas del servidor (500, 502, 503, 529 de sobrecarga…).
  - **Errores de red / *timeouts*.**
- **Error permanente:** reintentar no sirve; hay que corregir la solicitud.
  - **400:** solicitud mal formada (parámetros inválidos, contexto excesivo).
  - **401/403:** credenciales inválidas o sin permiso.
  - **404:** modelo o recurso inexistente.
- **Límite de tasa (*rate limit*):** tope de solicitudes o tokens por minuto que impone el proveedor.
- **Reintento con espera exponencial (*exponential backoff*):** tras cada fallo se espera más tiempo (1 s, 2 s, 4 s, 8 s…), para no saturar un servicio ya sobrecargado.
- **Jitter:** componente **aleatorio** añadido a la espera, para que muchos clientes no reintenten todos al mismo instante (*thundering herd*, estampida de reintentos).
- **Idempotencia:** propiedad de una operación que puede repetirse sin efectos secundarios adicionales; clave para decidir si es seguro reintentar.
- **Tiempo de espera (*timeout*):** máximo que se aguarda una respuesta antes de darla por fallida.

Regla: **reintenta solo errores transitorios**; un error 400 se corrige, no se repite.

## 7. Criterios para elegir un modelo

Cuatro criterios principales:

1. **Calidad:** qué tan bien resuelve *tu* tarea (no un ranking genérico).
2. **Costo:** precio por token y costo total por tarea completada.
3. **Privacidad y cumplimiento:** dónde viajan y se almacenan los datos; regulaciones aplicables.
4. **Latencia:** TTFT y tokens por segundo requeridos.

Aclaraciones y conceptos:

- **Benchmark:** prueba estandarizada para comparar modelos. Útil como orientación, pero **no sustituye medir con tus propios casos**.
- **Regla de oro:** empieza con el modelo más simple/barato que cumpla la tarea y sube de nivel solo si los datos lo justifican; decide con **evaluaciones propias**, no por moda.
- **Matriz de decisión:** tabla que puntúa cada opción según criterios ponderados por su importancia.
- **Dependencia de proveedor (*vendor lock-in*):** costo de migrar si el sistema depende de funciones exclusivas de un proveedor.

## 8. Build vs. Buy

- **Buy (comprar):** consumir un modelo por API pagando por uso. Ventajas: rapidez, cero operación de infraestructura, acceso a modelos de punta. Desventajas: costo variable, dependencia, datos que salen de tu entorno.
- **Build (construir):** alojar tu propio modelo (open-weight). Ventajas: control, privacidad, costo predecible a gran escala. Desventajas: hardware, operación, mantenimiento, personal especializado.
- **Punto de equilibrio (*break-even*):** volumen de uso a partir del cual el costo fijo de operar tu propio serving es menor que pagar por token.
- **Costo total de propiedad (TCO):** suma de hardware, energía, operación, personal y oportunidad, no solo el precio de la GPU.
- **Utilización:** una GPU cara y poco usada encarece cada token; la decisión depende del volumen y de que el uso sea sostenido.

La decisión es **de números y de operación**. El punto de partida típico es **comprar** y evaluar construir cuando el volumen o la privacidad lo exijan.

## 9. Salida estructurada

Cuando un programa (y no una persona) consume la respuesta, el texto libre no basta: se necesita un **formato predecible**.

- **Salida estructurada:** respuesta del modelo en un formato definido (normalmente **JSON**) que sigue un esquema.
- **JSON (*JavaScript Object Notation*):** formato de texto para datos con pares clave–valor, listas y tipos básicos.
- **Esquema (*schema*):** descripción formal de los campos, tipos y valores permitidos de un dato.
- **Pydantic:** librería de Python para definir esquemas como clases y **validar** datos contra ellos.
- **Validación:** comprobar que la salida cumple el esquema (campos obligatorios presentes, tipos correctos, valores permitidos).
- **Los tres pasos:** **pedir** el JSON → **validar** al recibirlo → si falla, **reintentar** (informando el error al modelo) o **fallar con un mensaje claro**.
- **Salida estructurada nativa:** funcionalidad del proveedor que restringe la generación para que el resultado cumpla el esquema.
- **Decodificación restringida:** técnica que limita los tokens permitidos en cada paso para garantizar un formato válido.
- **Degradación elegante (*graceful degradation*):** si todo falla, caer a una alternativa segura (por ejemplo, derivar a una persona) en vez de romper el sistema.

Enfoques: **(1)** pedir JSON en el prompt, validar y reintentar (funciona con casi cualquier modelo); **(2)** usar la salida estructurada nativa (más fiable, depende del proveedor).

## 10. Glosario rápido

| Término | Definición breve |
| --- | --- |
| **Propietario** | Modelo accesible solo vía el servicio del proveedor |
| **Open-weight** | Pesos descargables; no siempre open source |
| **Cuantización** | Reducir la precisión de los pesos para ahorrar memoria |
| **Servidor de inferencia** | Software que sirve un modelo mediante una API |
| **Role (`system`/`user`/`assistant`)** | Quién emite cada mensaje |
| **Stateless** | La API no guarda memoria entre llamadas |
| **Streaming** | Respuesta entregada por partes mientras se genera |
| **TTFT** | Tiempo hasta el primer token |
| **TPS** | Tokens por segundo |
| **Rate limit** | Tope de solicitudes o tokens por periodo |
| **Backoff exponencial** | Esperas crecientes entre reintentos |
| **Jitter** | Aleatoriedad en la espera para evitar sincronización |
| **Esquema** | Definición formal de la estructura de un dato |
| **Pydantic** | Librería para definir y validar esquemas en Python |
| **Build vs. Buy** | Alojar tu propio modelo vs. consumirlo por API |
| **TCO** | Costo total de propiedad |
