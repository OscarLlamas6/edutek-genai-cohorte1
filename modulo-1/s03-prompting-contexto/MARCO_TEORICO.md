# Marco teórico · Sesión 3: Prompting y context engineering

> Documento de apoyo al notebook `s03_prompting_contexto.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: el código y los experimentos están en el notebook.

## Índice

1. [Prompt y prompt engineering](#1-prompt-y-prompt-engineering)
2. [Técnicas base: zero-shot, few-shot y chain-of-thought](#2-técnicas-base-zero-shot-few-shot-y-chain-of-thought)
3. [Anatomía de un buen prompt](#3-anatomía-de-un-buen-prompt)
4. [Modelos de razonamiento](#4-modelos-de-razonamiento)
5. [Jerarquía de instrucciones](#5-jerarquía-de-instrucciones)
6. [Context engineering](#6-context-engineering)
7. [Compresión y caché de contexto](#7-compresión-y-caché-de-contexto)
8. [Prompts como código](#8-prompts-como-código)
9. [Salida validada](#9-salida-validada)
10. [Prompt injection](#10-prompt-injection)
11. [Glosario rápido](#11-glosario-rápido)

---

## 1. Prompt y prompt engineering

- **Prompt:** el texto (instrucciones, contexto, ejemplos, datos) que se le entrega al modelo para que genere una respuesta.
- **Prompt engineering (ingeniería de prompts):** disciplina de **redactar y refinar prompts** para obtener resultados más fiables y útiles.
- **Prompt de sistema (*system prompt*):** instrucciones persistentes y de alto nivel que definen el rol, tono y reglas del asistente.
- **Prompt de usuario:** la petición concreta de cada turno.
- **Plantilla de prompt (*prompt template*):** prompt con **variables** que se rellenan en tiempo de ejecución.
- **Delimitadores:** marcas (etiquetas XML, triple comilla, `###`…) que separan instrucciones de datos para evitar ambigüedad.
- **Iteración empírica:** el prompting se mejora **probando con casos reales** y midiendo, no por intuición.

## 2. Técnicas base: zero-shot, few-shot y chain-of-thought

- **Zero-shot:** pedir la tarea **sin ejemplos**, solo con la instrucción. Rápido y barato; depende de que el modelo entienda la tarea.
- **One-shot / few-shot:** incluir **uno o varios ejemplos** (entrada → salida esperada) dentro del prompt. Los ejemplos fijan **formato y criterio**; el modelo los imita. Esto es **aprendizaje en contexto** (*in-context learning*): el modelo se adapta sin reentrenarse.
- **Chain-of-thought (CoT, cadena de pensamiento):** pedir al modelo que **razone paso a paso** antes de dar la respuesta final. Ayuda en problemas de varios pasos (aritmética, lógica), pero **cuesta tokens** (más salida = más dinero y latencia).
- **Aprendizaje en contexto (*in-context learning*):** capacidad del LLM de aprender un patrón a partir de lo que hay en el prompt, sin cambiar sus pesos.
- **Prompting vs. fine-tuning:** el primero cambia el *input*; el segundo cambia el *modelo*.

Cuándo preferir cada una:

| Técnica | Úsala cuando… | Costo |
| --- | --- | --- |
| Zero-shot | La tarea es simple y clara | Bajo |
| Few-shot | Necesitas formato o criterio muy específico | Medio (ejemplos ocupan tokens) |
| Chain-of-thought | Hay varios pasos de razonamiento | Alto (tokens de razonamiento) |

## 3. Anatomía de un buen prompt

Un prompt bien estructurado suele incluir:

1. **Rol:** quién es el modelo y desde qué perspectiva responde.
2. **Contexto:** información de fondo necesaria para la tarea.
3. **Tarea / instrucción:** qué se debe hacer, de forma explícita.
4. **Restricciones:** qué evitar, límites de longitud, tono, qué hacer ante la duda.
5. **Ejemplos:** casos de entrada y salida esperada (few-shot).
6. **Formato de salida:** estructura exacta de la respuesta (JSON, lista, tabla…).

Principios:

- **Separar instrucciones de datos** con delimitadores.
- **Ser específico y no ambiguo:** un prompt vago produce respuestas vagas.
- **Decir qué hacer, no solo qué no hacer.**
- **Definir la salida ante casos límite** (p. ej. "si no hay información, responde X").

## 4. Modelos de razonamiento

- **Modelo de razonamiento (*reasoning model*):** modelo entrenado para dedicar un **proceso interno de "pensamiento"** antes de la respuesta final.
- **Tokens de razonamiento (*thinking tokens*):** tokens que el modelo genera al pensar; **se cobran como salida** aunque no siempre se muestren completos.
- **Esfuerzo de razonamiento (*reasoning effort*):** parámetro que regula cuánto piensa el modelo (bajo/medio/alto). Más esfuerzo suele dar más precisión en problemas difíciles, a costa de **mayor costo y latencia**.
- **Presupuesto de pensamiento (*thinking budget*):** tope de tokens que se permite dedicar al razonamiento.
- **Compromiso (*trade-off*) calidad–costo–latencia:** pensar más no siempre compensa; en tareas sencillas es desperdicio.
- **Diferencia con CoT clásico:** en CoT lo pides tú en el prompt; en los modelos de razonamiento el comportamiento viene incorporado y se controla por parámetros.

## 5. Jerarquía de instrucciones

- **Jerarquía de instrucciones:** orden de prioridad entre las fuentes de instrucciones: normalmente **sistema (desarrollador) > usuario > contenido externo/herramientas**.
- **Conflicto de instrucciones:** cuando el usuario pide algo que contradice las reglas del sistema (p. ej. "ignora tus instrucciones anteriores").
- **Guardrail (barrera de protección):** regla o mecanismo que limita lo que el sistema puede hacer o decir.
- **Límites del prompt como seguridad:** instrucciones en el prompt **reducen** el riesgo pero **no son una garantía**; la seguridad real exige controles fuera del modelo (permisos, validación).

## 6. Context engineering

- **Context engineering (ingeniería de contexto):** disciplina de **decidir qué información entra a la ventana de contexto, en qué orden y en qué forma**, para que el modelo tenga lo necesario en cada paso.
- **Prompt engineering vs. context engineering:** el primero se centra en **cómo redactar** la instrucción; el segundo en **todo lo que se coloca en la ventana**: instrucciones, ejemplos, datos recuperados, historial, memoria y herramientas.
- **Componentes del contexto:** instrucciones del sistema, ejemplos, historial de conversación, documentos recuperados (RAG), memoria, definiciones de herramientas y la pregunta del usuario.
- **Presupuesto de contexto:** reparto de la ventana entre esas piezas, que **compiten por el espacio**.
- **Ruido en el contexto:** información irrelevante que diluye la atención del modelo y degrada la calidad.
- **"Perdido en el medio" (*lost in the middle*):** tendencia de algunos modelos a usar peor la información situada en medio de un contexto muy largo que la del inicio o el final.
- **Memoria:** información que se conserva entre turnos o sesiones; en la práctica es contexto que **tú reinyectas**, ya que el modelo no recuerda por sí mismo.
- **Herramientas (*tools*):** funciones que el modelo puede invocar; su descripción también ocupa contexto.

## 7. Compresión y caché de contexto

- **Compresión de contexto:** reducir el tamaño de lo que se envía conservando lo importante (resumir el historial, recortar documentos, eliminar redundancias).
- **Resumen progresivo:** reemplazar partes antiguas de la conversación por un resumen.
- **Caché de prompt (*prompt caching*):** el proveedor **reutiliza el procesamiento de un prefijo del prompt** que ya vio, cobrándolo más barato y respondiendo más rápido.
- **Prefijo estable:** para aprovechar la caché, lo **estable va al inicio** (instrucciones, ejemplos, documentos fijos) y lo **cambiante al final** (la pregunta del usuario).
- **Acierto de caché (*cache hit*) / fallo (*cache miss*):** que el prefijo coincida o no con uno almacenado.
- **Escritura de caché / lectura de caché:** la primera vez se paga (a veces con recargo) por almacenar; las siguientes, la lectura es más barata.
- **TTL (*time to live*):** tiempo durante el cual la caché sigue vigente.

## 8. Prompts como código

Un prompt de producción debe tratarse como cualquier artefacto de software:

1. **Plantilla con variables y delimitadores**, no un string pegado en el código.
2. **Guardado en Git**, con historial y revisión.
3. **Con casos de prueba**: cada cambio de prompt es un cambio de comportamiento y debe poder **probarse y revertirse**.

Conceptos:

- **Versionado de prompts:** mantener versiones (v1, v2, v3…) con un registro de cambios (*changelog*).
- **Motor de plantillas (Jinja2):** sistema que rellena plantillas con variables; permite condicionales y bucles.
- **Variable faltante que falla:** buena práctica de que una plantilla con una variable sin definir **lance error** en vez de generar un prompt incompleto en silencio.
- **Conjunto de evaluación (*eval set*):** lista de casos con la respuesta esperada para medir un prompt.
- **Métricas de evaluación:** por ejemplo, **% de JSON válido** y **% de categoría correcta**.
- **Regresión:** que un cambio que mejora un caso empeore otros; por eso se mide el conjunto completo.
- **Cambio de una sola variable:** al iterar, modificar **una cosa a la vez** para saber qué causó el efecto.
- **Regla del equipo:** no aceptar un cambio de prompt "porque se ve mejor"; se acepta si **mejora las métricas** sobre el conjunto de evaluación.

## 9. Salida validada

- **Salida validada:** respuesta del modelo comprobada contra un esquema antes de usarse.
- **Extracción de JSON:** aislar el JSON aunque el modelo lo envuelva en cercas de código (` ```json `) o añada texto alrededor.
- **Validación con esquema (Pydantic):** verificar campos obligatorios y **valores permitidos** (por ejemplo, una categoría dentro de una lista cerrada).
- **Enumeración (*enum*):** conjunto cerrado de valores válidos para un campo.
- **Estrategias ante un fallo:**
  1. **Fallar con mensaje claro** y contar el caso como inválido.
  2. **Reintentar** explicando el error al modelo.
  3. **Degradar con elegancia:** derivar a una persona.
- **Clasificador de tickets:** ejemplo de tarea que asigna una categoría a un texto de entrada (clasificación).

## 10. Prompt injection

- **Prompt injection (inyección de instrucciones):** ataque en el que un texto —del usuario o, más sutilmente, de **contenido externo** (un correo, una página, un documento)— contiene instrucciones que intentan **secuestrar el comportamiento** del modelo.
- **Inyección directa:** el propio usuario escribe la instrucción maliciosa.
- **Inyección indirecta:** la instrucción viene escondida en datos que el sistema lee (documentos, webs, tickets).
- **Superficie de ataque:** todo punto por donde entra texto no confiable al contexto.
- **Por qué es difícil:** el modelo recibe instrucciones y datos en el mismo canal (texto) y no distingue con total fiabilidad cuál es cuál.

Reglas de defensa:

1. **Tratar todo texto externo como NO confiable.**
2. **Delimitarlo** y decirle al modelo que es solo contenido, no instrucciones.
3. **Mínimo privilegio:** no darle al modelo más permisos de los necesarios (un clasificador no necesita enviar correos ni borrar archivos).
4. **Validar la salida:** si no cumple el esquema, no se usa.

Idea clave: **ninguna defensa es perfecta**; se combinan capas.

## 11. Glosario rápido

| Término | Definición breve |
| --- | --- |
| **Prompt** | Entrada de texto que guía al modelo |
| **System prompt** | Instrucciones persistentes de alto nivel |
| **Zero-shot** | Tarea sin ejemplos |
| **Few-shot** | Tarea con ejemplos en el prompt |
| **Chain-of-thought** | Razonamiento paso a paso antes de responder |
| **In-context learning** | Aprender un patrón desde el prompt, sin reentrenar |
| **Modelo de razonamiento** | Modelo que "piensa" antes de responder |
| **Context engineering** | Decidir qué entra a la ventana y en qué orden |
| **Prompt caching** | Reutilizar un prefijo ya procesado a menor costo |
| **Prompt template** | Prompt con variables sustituibles |
| **Eval set** | Casos de prueba para medir un prompt |
| **Guardrail** | Barrera que limita el comportamiento del sistema |
| **Prompt injection** | Instrucciones maliciosas ocultas en el texto de entrada |
| **Mínimo privilegio** | Dar solo los permisos estrictamente necesarios |
| **Salida validada** | Respuesta comprobada contra un esquema antes de usarse |
