# Marco teórico · Sesión 8: Fundamentos de agentes de IA (con LangChain)

> Documento de apoyo al notebook `s08_agentes_langchain.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: el código y los experimentos están en el notebook y en la carpeta `langchain-basic` del repositorio `ai-agents-playground`.

## Índice

1. [Qué es un agente](#1-qué-es-un-agente)
2. [El espectro de autonomía](#2-el-espectro-de-autonomía)
3. [El modelo](#3-el-modelo)
4. [Los mensajes y el estado](#4-los-mensajes-y-el-estado)
5. [Herramientas y *tool calling*](#5-herramientas-y-tool-calling)
6. [Diseño de buenas herramientas](#6-diseño-de-buenas-herramientas)
7. [El prompt de sistema](#7-el-prompt-de-sistema)
8. [El bucle](#8-el-bucle)
9. [Límites y condiciones de parada](#9-límites-y-condiciones-de-parada)
10. [LangChain: qué es y qué hace por ti](#10-langchain-qué-es-y-qué-hace-por-ti)
11. [Cuándo no usar un framework](#11-cuándo-no-usar-un-framework)
12. [Riesgos y seguridad](#12-riesgos-y-seguridad)
13. [Cómo se evalúa un agente](#13-cómo-se-evalúa-un-agente)
14. [Glosario rápido](#14-glosario-rápido)

---

## 1. Qué es un agente

- **Agente de IA:** sistema en el que un **modelo de lenguaje decide**, en un **bucle**, qué hacer a continuación —normalmente, **pedir que se ejecute una herramienta**— y usa el resultado para decidir el paso siguiente, hasta cumplir un objetivo.

Sus **tres piezas**:

| Pieza | Qué es |
| --- | --- |
| **Modelo** | El LLM: razona y decide el siguiente paso |
| **Herramientas** | Funciones del código que el modelo puede pedir que se ejecuten (consultar una API, hacer un cálculo, crear un ticket) |
| **Bucle** | *modelo → herramienta → modelo → …* hasta que el modelo responde con texto |

Alrededor de ellas hay más elementos que un agente necesita para funcionar bien: **mensajes** (el estado), **prompt de sistema**, **límites** del bucle y una **traza** para entender qué pasó.

Ideas relacionadas:

- **Autonomía:** grado en que el sistema decide por sí mismo el camino.
- **Observación:** el resultado de una herramienta, que el modelo "observa" antes de decidir.
- **Paso o vuelta:** una iteración del bucle. Cada vuelta incluye al menos una llamada al modelo, y por eso **cuesta**.
- **Harness (arnés):** todo lo que rodea al bucle del modelo: el prompt, las herramientas y las piezas que modifican su comportamiento (límites, reintentos, aprobación humana).

## 2. El espectro de autonomía

No es blanco o negro: es una escala de **quién decide el camino**.

| | Chatbot | Flujo fijo | Workflow con ramas | Agente |
| --- | --- | --- | --- | --- |
| **Quién decide los pasos** | Nadie: una respuesta | El código, siempre los mismos | El código, con condiciones | **El modelo** |
| **Llamadas al modelo por pregunta** | Una | Pocas y fijas | Pocas y previsibles | **Varias e impredecibles** |
| **Predecible y fácil de probar** | Mucho | Mucho | Bastante | **Poco** |
| **Resuelve lo imprevisto** | No | No | Algo | **Sí** |

- **Workflow:** sistema en el que el **código** orquesta los pasos, aunque use modelos en varios de ellos.
- **Agente:** sistema en el que el **modelo** decide los pasos.

Ideas: más autonomía permite resolver lo que no se puede prever, pero **cuesta más, tarda más y es más difícil de controlar y de probar**. La regla práctica es empezar por lo más simple que cumpla el objetivo y añadir autonomía solo donde haga falta.

## 3. El modelo

- **Modelo de chat:** modelo de lenguaje que recibe una lista de mensajes y devuelve un mensaje.
- **Interfaz común:** en LangChain todos los proveedores exponen las mismas operaciones —responder completo (`invoke`), responder por trozos (`stream`) y describirle herramientas (`bind_tools`)—, por eso el resto del agente no cambia al cambiar de proveedor.

Formas de obtener un modelo:

| Forma | Cuándo |
| --- | --- |
| Texto `proveedor:modelo` | Lo más corto; configuración por defecto |
| Función de inicialización de modelos | Cuando se necesita fijar parámetros |
| Clase del proveedor | Cuando se necesitan opciones propias del proveedor |

Parámetros estándar: modelo, llave de API (normalmente de una variable de entorno), temperatura, máximo de tokens, tiempo de espera y número de reintentos (por defecto 6, con espera creciente ante errores de red, de límite de peticiones y del servidor).

- **Servidor local compatible con OpenAI:** servidores como LM Studio hablan el formato de OpenAI; se usan con la misma clase de modelo cambiando la dirección base.
- **Tamaño del modelo:** usar herramientas es una tarea más exigente que redactar. Un modelo pequeño (≈ 3B de parámetros) suele elegir mal la herramienta, repetir llamadas o ignorar el resultado; uno de ≈ 12B o más lo hace mucho mejor. Se prueba antes de decidir.

## 4. Los mensajes y el estado

La conversación completa es una **lista de mensajes**. Es el **estado** de un agente durante una ejecución.

| Tipo | Rol | Quién lo escribe | Contiene |
| --- | --- | --- | --- |
| Mensaje de sistema | `system` | El desarrollador | Instrucciones fijas |
| Mensaje humano | `user` | La persona | La pregunta |
| Mensaje de IA | `assistant` | El **modelo** | Texto, o una petición de herramientas |
| Mensaje de herramienta | `tool` | El **código** | El resultado de una herramienta, enlazado a su petición por un identificador |

Un **mensaje de IA** tiene: el texto, las **peticiones de herramientas** (cada una con nombre, argumentos e identificador; vacías si solo responde), los **bloques de contenido** en un formato estándar entre proveedores, y los **metadatos de uso** (tokens de entrada y de salida).

- **Petición de herramienta (*tool call*):** estructura que devuelve el modelo en lugar de texto: el nombre de la herramienta y sus argumentos.
- **Estado append-only:** los mensajes solo se agregan; nunca se borran. El último mensaje es la respuesta final.
- Los mensajes pueden escribirse como objetos o como **diccionarios** con rol y contenido; el framework los convierte.

## 5. Herramientas y *tool calling*

- **Herramienta (*tool*):** función de código que el modelo puede pedir que se ejecute.
- ***Tool calling* (llamada a herramientas):** mecanismo por el que el modelo, en lugar de responder con texto, devuelve una **petición estructurada** de ejecutar una herramienta.

**El modelo nunca ejecuta nada: solo pide.** El mecanismo tiene cuatro pasos:

| Paso | Quién | Qué ocurre |
| --- | --- | --- |
| 1 | El desarrollador | Describe las herramientas al modelo (un **esquema**: nombre, descripción, parámetros) |
| 2 | El modelo | Si necesita una, responde con una petición |
| 3 | El código | Ejecuta la función real |
| 4 | El desarrollador | Devuelve el resultado al modelo, que decide si pide otra o responde |

Lo único que el modelo sabe de una herramienta es:

| Parte | Origen en LangChain | Para qué la usa el modelo |
| --- | --- | --- |
| **Nombre** | Nombre de la función | Elegir cuál usar |
| **Descripción** | *Docstring* de la función | Saber qué hace y cuándo usarla |
| **Parámetros** | *Type hints* de la firma (obligatorios) | Saber qué argumentos pedir y de qué tipo |

Por eso una herramienta es, en la práctica, **un prompt disfrazado de función**. El **esquema** que se envía es un JSON; su forma cambia entre proveedores, pero el contenido es el mismo.

- **Llamadas en paralelo:** un modelo puede pedir varias herramientas en un mismo mensaje.
- **Tipos de resultado de una herramienta:** texto, datos estructurados, bloques de contenido (por ejemplo, imágenes) o una actualización del estado del agente.

## 6. Diseño de buenas herramientas

| Criterio | Regla |
| --- | --- |
| **Nombre** | Claro y específico, en `snake_case` (letras, números, guion bajo y guion para ser compatible con los proveedores) |
| **Descripción** | Qué hace, cuándo usarla y cuándo no; breve e informativa |
| **Parámetros** | Pocos, tipados, con valores válidos nombrados |
| **Resultado** | Corto: todo lo que devuelve entra al contexto y cuesta tokens |
| **Errores** | Un texto legible que diga qué corregir, en lugar de una excepción cruda; así el modelo puede reintentar |
| **Efectos** | Separar las que solo leen de las que modifican algo; estas últimas requieren más control |
| **Cantidad** | Pocas y bien diseñadas: demasiadas herramientas parecidas confunden al modelo y aumentan los errores |

- **Idempotencia:** ejecutar una herramienta dos veces produce el mismo efecto que una. Importa porque los agentes repiten llamadas.
- **Solo lectura primero:** las herramientas que consultan son más seguras que las que modifican.

## 7. El prompt de sistema

- **Prompt de sistema:** instrucciones fijas que acompañan a toda la conversación.

Qué conviene incluir: el **rol**, **cuándo usar herramientas**, **qué no hacer** (por ejemplo, no inventar datos), el **formato y tono** de la respuesta y los **límites** (por ejemplo, nunca modificar algo sin confirmación).

- En un agente importa más que en un chatbot, porque además de responder el modelo **decide acciones**.
- **No es una barrera de seguridad:** orienta al modelo pero no garantiza nada. Un texto malicioso que llegue por una herramienta puede intentar contradecirlo. Los límites que importan se imponen **en el código**.

## 8. El bucle

El bucle es el corazón del agente:

1. Se llama al modelo con los mensajes.
2. Si responde **con texto** (sin peticiones), **terminó**.
3. Si **pide herramientas**, el código las ejecuta y agrega cada resultado a los mensajes.
4. Se vuelve al paso 1.

- **ReAct (*Reason + Act*):** patrón en el que el modelo alterna **razonar** qué hacer, **actuar** (pedir una herramienta) y **observar** el resultado.
- **Costo del bucle:** cada vuelta es una llamada al modelo. Un agente hace varias por pregunta: el costo, la latencia y el riesgo se multiplican respecto a un flujo fijo.
- **Traza:** la lista de mensajes de una ejecución, en orden. Es la forma de entender **por qué** el agente respondió lo que respondió.

## 9. Límites y condiciones de parada

Un bucle necesita saber cuándo terminar.

| Cómo termina | Significado |
| --- | --- |
| **Respuesta con texto** | Terminó bien: el modelo ya no pidió herramientas |
| **Límite de pasos** | Se alcanzó el máximo de pasos del grafo interno sin terminar; el framework lanza un error de recursión |
| **Freno propio** | Un límite de llamadas a herramientas, un presupuesto de tokens o de dinero, un tiempo máximo, o la detección de falta de progreso (repetir la misma llamada) |

- **Límite de pasos:** cuenta pasos del grafo interno; cada llamada al modelo y cada ejecución de herramientas suman uno. Con un límite de 8 caben unas 4 vueltas del modelo.
- Sin límites, un agente confundido puede repetir sin fin, y **cada vuelta se cobra**.
- **Criterio de salida:** definir qué significa "terminado" es parte del diseño del agente; no se confía solo en que el modelo lo declare.

Otras piezas del agente, que se estudian más adelante en el curso:

| Pieza | Qué es |
| --- | --- |
| **Memoria de corto plazo** | Que el agente recuerde entre turnos: requiere guardar el estado (*checkpointer*) y un identificador de conversación; el de memoria sirve para aprender, en producción se usa uno persistente |
| **Salida estructurada** | Que la respuesta final sea un dato validado (un modelo de datos) en lugar de texto libre |
| ***Streaming*** | Ver la ejecución mientras ocurre |
| **Middleware** | Piezas que modifican el bucle sin reescribirlo: reintentos, enmascarar datos personales, aprobación humana, resumir la conversación |

## 10. LangChain: qué es y qué hace por ti

- **LangChain:** framework de código abierto para construir aplicaciones con modelos de lenguaje. Aporta una **interfaz común de modelos**, el decorador de herramientas, un **formato estándar de mensajes**, la función **`create_agent`** y el **middleware**.
- **`create_agent`:** recibe un modelo, herramientas y un prompt, y devuelve un agente con el bucle ya hecho.

Qué hace `create_agent` por ti: describe las herramientas al modelo, ejecuta el bucle, valida los argumentos, aplica un límite de pasos, mantiene los mensajes como estado y traduce entre formatos de proveedores.

| Parámetro de `create_agent` | Para qué sirve |
| --- | --- |
| Modelo | Texto `proveedor:modelo` u objeto de modelo |
| Herramientas | La lista de herramientas |
| Prompt de sistema | Texto o mensaje de sistema |
| Formato de respuesta | Salida estructurada |
| Checkpointer | Memoria entre turnos |
| Middleware | Modificar el bucle |
| Esquemas de estado y de contexto | Estado propio y datos por ejecución |
| Nombre | Identificador, útil si el agente es parte de otro |

Paquetes del ecosistema:

| Paquete | Qué es |
| --- | --- |
| `langchain` | API de alto nivel: agentes, modelos, herramientas, mensajes, middleware |
| `langchain-core` | Abstracciones base |
| `langgraph` | **Por debajo**: ejecuta el agente como un **grafo con estado** (nodos, aristas y ciclos); se usa directamente para flujos a medida |
| `langchain-anthropic`, `langchain-openai`, … | Un paquete por proveedor |
| `langsmith` | Servicio de trazas y evaluación (opcional) |
| `langchain-classic` | Interfaces antiguas conservadas por compatibilidad |

**Cambios de la versión 1.0.** Muchos tutoriales anteriores usan interfaces reemplazadas:

| Antes de la 1.0 | Hoy |
| --- | --- |
| `AgentExecutor`, `initialize_agent` | `create_agent` |
| `create_react_agent` de `langgraph.prebuilt` | `create_agent` |
| `LLMChain` | Componer modelos directamente |
| Mensajes desde `langchain.schema` | Mensajes desde `langchain.messages` |

## 11. Cuándo no usar un framework

| Situación | Alternativa |
| --- | --- |
| Una sola llamada al modelo | El SDK del proveedor basta |
| Se quiere control total del bucle | Escribirlo a mano son unas veinte líneas |
| Se necesita estabilidad máxima | Un framework que evoluciona rápido es una dependencia que se vigila: **fijar las versiones** |

## 12. Riesgos y seguridad

Un agente **ejecuta**, y por eso el riesgo es distinto al de un chatbot.

| Riesgo | Qué es | Defensa |
| --- | --- | --- |
| **Acción no pedida** | El modelo ejecuta algo que nadie le pidió | Confirmar antes de actuar (aprobación humana) |
| **Acción repetida** | Reintenta y duplica un efecto | Herramientas idempotentes |
| **Inyección indirecta** | Un texto que llega por una herramienta contiene instrucciones y el modelo las obedece | Tratar todo resultado como **datos, no órdenes**; limitar lo que el agente puede hacer |
| **Bucle sin fin** | Repite sin avanzar | Límite de pasos y de presupuesto |
| **Argumentos peligrosos** | El modelo inventa o manipula un parámetro | Validar en el código; nunca confiar |
| **Falla inesperada** | Una herramienta falla | Errores legibles y reintentos controlados |

- **Mínimo privilegio:** el agente solo debe tener las herramientas que necesita; las que modifican algo, con aprobación humana. La seguridad no puede depender de que el modelo "se porte bien".

## 13. Cómo se evalúa un agente

Se mide, y por capas:

| Capa | Pregunta |
| --- | --- |
| **Elección de herramienta** | ¿Llamó a la correcta? ¿A ninguna cuando no hacía falta? |
| **Argumentos** | ¿Eran correctos y válidos? |
| **Trayectoria** | ¿Cuántos pasos dio? ¿Repitió llamadas? ¿Se desvió? |
| **Resultado** | ¿La respuesta final es correcta y está respaldada por lo que devolvieron las herramientas? |
| **Costo y tiempo** | ¿Cuántas llamadas al modelo y cuántos tokens por pregunta? |

- La **traza** es la materia prima de la evaluación.
- El **tamaño del modelo** pesa mucho: con las mismas herramientas y preguntas, un modelo pequeño elige peor.

## 14. Glosario rápido

| Término | Definición |
| --- | --- |
| **Agente** | Sistema donde un modelo decide, en un bucle, qué herramientas usar |
| **Workflow** | Sistema donde el código orquesta los pasos |
| **Herramienta (*tool*)** | Función de código que el modelo puede pedir que se ejecute |
| ***Tool calling*** | Mecanismo por el que el modelo devuelve una petición estructurada de herramienta |
| **Petición de herramienta** | Nombre y argumentos que el modelo pide ejecutar |
| **Esquema de herramienta** | Descripción (nombre, descripción, parámetros) que recibe el modelo |
| **Mensaje** | Unidad de la conversación: sistema, humano, IA o herramienta |
| **Estado** | Los mensajes acumulados de una ejecución |
| **Prompt de sistema** | Instrucciones fijas de toda la conversación |
| **Bucle / ReAct** | Razonar, actuar y observar, repetido hasta responder |
| **Paso** | Una vuelta del bucle |
| **Límite de pasos** | Máximo de pasos antes de cortar la ejecución |
| **`create_agent`** | Función de LangChain que arma el agente completo |
| **LangGraph** | Biblioteca de grafos con estado sobre la que corre el agente |
| **Middleware** | Pieza que modifica el bucle sin reescribirlo |
| **Checkpointer** | Componente que guarda el estado para dar memoria entre turnos |
| **Traza** | La lista de mensajes de una ejecución, en orden |
| **Idempotente** | Que ejecutarlo varias veces equivale a ejecutarlo una |
| **Inyección indirecta** | Instrucciones ocultas en un texto que llega por una herramienta |
| **Mínimo privilegio** | Dar al agente solo las herramientas imprescindibles |
