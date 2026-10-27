# Marco teórico · Sesión 8: Fundamentos de un agente de IA con LangChain

> Documento de apoyo al notebook `s08_agentes_langchain.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: el código y los experimentos están en el notebook.

## Índice

1. [Qué es un agente](#1-qué-es-un-agente)
2. [Agente frente a chatbot, RAG y flujo fijo](#2-agente-frente-a-chatbot-rag-y-flujo-fijo)
3. [Tool calling (llamada a herramientas)](#3-tool-calling-llamada-a-herramientas)
4. [Diseño de buenas herramientas](#4-diseño-de-buenas-herramientas)
5. [LangChain: qué es y qué aporta](#5-langchain-qué-es-y-qué-aporta)
6. [El agente de LangChain por dentro](#6-el-agente-de-langchain-por-dentro)
7. [Salida estructurada, streaming y memoria](#7-salida-estructurada-streaming-y-memoria)
8. [Evaluar a un agente](#8-evaluar-a-un-agente)
9. [Riesgos y seguridad](#9-riesgos-y-seguridad)
10. [Glosario rápido](#10-glosario-rápido)

---

## 1. Qué es un agente

- **Agente:** sistema en el que un **modelo de lenguaje decide**, en un **bucle**, qué hacer a continuación —normalmente, **pedir que se ejecute una herramienta**— y usa el resultado para decidir el siguiente paso, hasta que puede responder.

Sus **tres piezas**:

1. **Modelo:** razona y decide el siguiente paso.
2. **Herramientas:** funciones de tu código que el modelo puede pedir que se ejecuten (consultar una API, buscar en documentos, crear un ticket).
3. **Bucle:** *modelo → herramienta → modelo → …* hasta que el modelo responde con texto.

Ideas relacionadas:

- **Autonomía:** grado en que el sistema decide por sí mismo el camino. Un agente es más autónomo que un flujo fijo, y por eso es **más flexible y menos predecible**.
- **Observación:** el resultado de una herramienta, que el modelo "observa" antes de decidir el siguiente paso.
- **Paso (*turno* del bucle):** una vuelta de modelo + herramientas. Cada paso es una llamada al modelo y **cuesta**.
- **Condición de parada:** el modelo responde con texto sin pedir más herramientas (o se alcanza un límite).
- **Harness (arnés):** todo lo que rodea al bucle del modelo: el prompt, las herramientas y las piezas que modifican su comportamiento (límites, reintentos, aprobación humana).

## 2. Agente frente a chatbot, RAG y flujo fijo

| | Chatbot | RAG (Sesión 5) | Agente |
| --- | --- | --- | --- |
| **¿Quién decide el camino?** | Nadie: una sola llamada | El programador (flujo fijo) | El modelo, paso a paso |
| **Datos externos** | No | Sí, siempre los mismos (búsqueda en documentos) | Los que **elija**, o ninguno |
| **¿Puede actuar en el mundo?** | No | No | Sí, si tiene herramientas que escriben |
| **Llamadas al modelo por pregunta** | 1 | 1 | Varias, impredecibles |
| **Costo y latencia** | Bajos y predecibles | Bajos y predecibles | Más altos y variables |
| **Riesgo principal** | Alucinación | Alucinación y documentos envenenados | Lo anterior + acciones equivocadas |

- **Flujo de trabajo (*workflow*):** pasos fijos definidos por el programador, aunque algunos usen un LLM. Un RAG es un flujo de trabajo.
- **Cuándo usar un agente:** cuando **el camino depende de la pregunta** y no se puede programar de antemano.
- **Cuándo no usarlo:** si el camino es conocido y siempre el mismo, el código normal o el RAG es **más barato, más rápido y más fácil de auditar**. Regla general: **usa lo más simple que resuelva el problema.**
- **Un RAG puede ser una herramienta del agente:** el agente decide **cuándo** consultar los documentos, y cuándo no.

## 3. Tool calling (llamada a herramientas)

- **Tool calling / function calling:** capacidad del modelo de responder, en vez de con texto, con una **petición estructurada** (nombre de la herramienta + argumentos) para que se ejecute una función.

Las **cuatro piezas**:

| # | Pieza | Quién la hace |
| --- | --- | --- |
| 1 | **Describir** las herramientas con un esquema (nombre, descripción, parámetros) | El desarrollador |
| 2 | **Pedir** una herramienta: el modelo responde con nombre y argumentos | El modelo |
| 3 | **Ejecutar** la función y obtener el resultado | El código del desarrollador |
| 4 | **Devolver** el resultado al modelo y repetir | El desarrollador (el bucle) |

**Idea central: el modelo nunca ejecuta nada.** Solo escribe una petición con la forma correcta. Si el código no la ejecuta, no pasa nada. Esa separación es **la base de la seguridad** de los agentes: quien controla el código que ejecuta, controla lo que el agente puede hacer.

Conceptos:

- **Esquema de la herramienta (*tool schema*):** descripción en formato JSON Schema de lo que la herramienta hace y de los argumentos que acepta. Es **todo** lo que el modelo sabe de ella.
- **Petición de herramienta (*tool call*):** la respuesta del modelo que pide ejecutar una herramienta, con un identificador.
- **Resultado de herramienta (*tool result*):** lo que el código devuelve al modelo, enlazado con el identificador de la petición.
- **Llamadas en paralelo:** el modelo puede pedir **varias herramientas a la vez** cuando las consultas son independientes.
- **El historial crece:** el modelo no tiene memoria propia; en cada vuelta se **reenvía** la conversación completa (pregunta, peticiones y resultados). Por eso **los tokens de entrada de un agente se acumulan** y por eso cuesta más que un chatbot.
- **Formatos por proveedor:** cada proveedor expresa estas piezas de forma distinta (por ejemplo, los argumentos llegan como texto JSON en un formato y como objeto en otro; los resultados vuelven como un tipo de mensaje distinto). Un framework puede **unificarlos**.
- **Formato estricto:** la petición debe respetar un formato; los modelos pequeños fallan a veces en esto, y un servidor puede **rechazar** la salida del modelo.

## 4. Diseño de buenas herramientas

> **Una herramienta es un prompt disfrazado de función.** El modelo decide usarla leyendo solo su **nombre, descripción y parámetros**.

| Aspecto | Buena práctica | Mal ejemplo |
| --- | --- | --- |
| **Nombre** | Verbo + objeto: `estado_servicio` | `api2`, `datos` |
| **Descripción** | Qué hace, **cuándo usarla y cuándo no** | "Hace cosas" |
| **Parámetros** | Tipados, con descripción y valores válidos | Sin tipo ni descripción |
| **Resultado** | Corto y legible | Volcar miles de líneas |
| **Errores** | Texto que explica **cómo corregir** | Un *traceback* o un valor vacío |
| **Efectos** | Separar **leer** de **escribir**; la que escribe es **idempotente** | Una herramienta que lee y modifica |
| **Cantidad** | Pocas y distintas entre sí | Muchas, parecidas |

Conceptos:

- **Herramienta de lectura:** consulta sin modificar nada. Bajo riesgo.
- **Herramienta de escritura (con efectos):** modifica el mundo (crea un ticket, envía un correo). Requiere más cuidado.
- **Idempotencia:** repetir la acción con los mismos datos **no cambia el resultado** (no se duplican tickets). Importa porque los modelos **reintentan y se repiten**.
- **Errores esperados vs. inesperados:** los esperados (un servicio que no existe) se devuelven como **texto legible** para que el modelo se corrija; los inesperados (se cayó la base de datos) se manejan con reintentos o mensajes seguros.
- **Validación de argumentos:** se valida contra el esquema antes de ejecutar; los argumentos inválidos se devuelven al modelo como error. La validación **de negocio** (valores permitidos, longitudes) debe hacerse en el código, no confiarse al prompt.
- **Exceso de herramientas:** cada herramienta añade tokens al prompt y más opciones para confundirse.

## 5. LangChain: qué es y qué aporta

- **LangChain:** *framework* de código abierto para construir aplicaciones con LLM.

Lo que aporta:

| Aporta | En palabras simples |
| --- | --- |
| **Interfaz común de modelos** | Los modelos de distintos proveedores se usan igual: se cambia de proveedor sin reescribir el agente |
| **Definición de herramientas** | Convierte una función (nombre, descripción, tipos) en una herramienta con esquema y validación |
| **Mensajes estándar** | Un mismo formato para el usuario, el modelo y los resultados de herramientas |
| **Constructor de agentes** | El bucle ya hecho, con límites y manejo de errores |
| **Middleware** | Piezas que se añaden al bucle: límites, reintentos, aprobación humana, resumen del historial |
| **Salida estructurada, streaming, memoria** | Capacidades integradas |

**Los paquetes** (una fuente habitual de confusión):

| Paquete | Para qué |
| --- | --- |
| **langchain** | La capa de alto nivel: el constructor de agentes y el middleware |
| **langchain-core** | Las abstracciones base: mensajes, herramientas, modelos |
| **langgraph** | El **motor** debajo del agente: un grafo de pasos con persistencia y pausas |
| **Integraciones por proveedor** | Un paquete por proveedor (Anthropic, OpenAI…) |
| **langchain-classic** | La API antigua (ejecutores de agentes, cadenas). Ya no se usa en proyectos nuevos |
| **LangSmith** | Servicio **opcional** de observabilidad: ver las trazas de cada ejecución |

- **Versionado y obsolescencia:** el ecosistema cambió de forma importante con la versión 1.0; muchos tutoriales anteriores usan una API reemplazada. Conviene **comprobar la fecha** de cualquier tutorial y **fijar las versiones** del proyecto.
- **Cuándo no usarlo:** una sola llamada al modelo (basta el SDK del proveedor); necesidad de control total del bucle; máxima estabilidad. Vale la pena cuando se usan **varias de sus piezas a la vez**.
- **Servidores compatibles:** un servidor local (LM Studio, Ollama, vLLM…) que hable el formato de OpenAI se usa con la integración de OpenAI indicando la dirección del servidor.

## 6. El agente de LangChain por dentro

**Qué hace el constructor de agentes** en cada pregunta:

1. Llama al modelo con la conversación y los esquemas de las herramientas.
2. Si la respuesta trae **peticiones de herramienta**, las **ejecuta** (validando los argumentos).
3. Agrega cada resultado a la conversación como un **mensaje de herramienta**.
4. **Vuelve a llamar** al modelo con todo el historial.
5. Si el modelo responde **sin peticiones**, termina.

Conceptos:

- **Estado del agente:** la lista de mensajes de la conversación (y, opcionalmente, otros campos). Al terminar, contiene toda la **traza**.
- **Tipos de mensaje:** del usuario, del modelo (puede incluir peticiones de herramienta) y de herramienta (lleva el identificador de la petición a la que responde).
- **Traza:** el registro completo de una ejecución: qué pidió el modelo, qué devolvió cada herramienta, qué respondió. Es la herramienta principal para **depurar y auditar**.
- **Límite de pasos:** número máximo de pasos antes de cortar la ejecución. **Protege contra bucles que nunca terminan** y que gastan dinero.
- **Manejo de peticiones erróneas:** si el modelo pide una herramienta que no existe o con argumentos inválidos, el agente **no se cae**: devuelve al modelo un mensaje de error para que se corrija.
- **Middleware:** piezas que se enganchan al bucle. Ejemplos: limitar el número de llamadas a una herramienta, devolver mensajes de error seguros, reintentar, pedir aprobación humana antes de una acción, resumir el historial.
- **Modelo de juguete (*fake model*):** un "modelo" que devuelve respuestas de un guion. Permite **ver y probar el bucle sin un LLM real** (y sin gastar); es la forma habitual de probar agentes.
- **Identificador del modelo:** el modelo puede indicarse con un texto "proveedor:modelo" o con un objeto de modelo ya configurado (necesario para parámetros especiales o servidores locales).

## 7. Salida estructurada, streaming y memoria

**Salida estructurada**

- **Salida estructurada:** que el agente devuelva un **objeto validado contra un esquema** (por ejemplo, un modelo de Pydantic) en vez de texto libre. Útil cuando otro sistema consume el resultado.
- **Estrategias:** usar la **función nativa** del proveedor si existe, o, si no, una **herramienta adicional** que el modelo "llama" para entregar el resultado con el formato pedido.
- Es la misma idea de "pedir JSON y validar" de la Sesión 3, integrada en el bucle.

**Streaming**

- **Streaming:** recibir el resultado **a medida que se produce**, en lugar de esperar al final.
- **Dos niveles:** por **pasos** del agente (qué nodo corrió y qué produjo) y por **texto** (token a token).
- Mejora la **sensación de velocidad**: un agente puede tardar varios segundos.

**Memoria**

- **Por defecto, un agente no recuerda:** cada pregunta empieza de cero.
- **Checkpointer:** componente que **guarda el estado** del agente entre turnos. Con un **identificador de hilo (*thread id*)**, la misma conversación recuerda lo anterior; otro hilo es otra conversación.
- **Memoria de corto plazo:** lo que se recuerda **dentro** de una conversación.
- **Memoria de largo plazo:** lo que se recuerda **entre** conversaciones (otro tema, de la próxima semana).
- **Persistencia:** un checkpointer en memoria se pierde al reiniciar; en producción se respalda con una **base de datos** (por ejemplo, PostgreSQL).
- **El historial cuesta:** como se reenvía completo, una conversación larga **cuesta cada vez más**. Se recorta o se **resume** el historial.

## 8. Evaluar a un agente

- **Evaluar la elección de herramientas:** con preguntas fijas, comprobar qué herramientas llamó el agente y compararlas con las esperadas. Es la misma idea de las Sesiones 3 y 5: **medir con casos fijos, no "a ojo"**.
- **Herramientas "deben" y "permitidas":** las que **tienen** que llamarse y las que no estorban. **Llamar herramientas de más también es un fallo**: cuesta dinero y tiempo y, si la herramienta escribe, puede hacer daño.
- **Un agente también debe saber cuándo NO usar herramientas** (un saludo, una cuenta simple).
- **El modelo importa mucho:** con el mismo código y las mismas herramientas, un modelo pequeño (3B parámetros) y uno mediano (~12B) dieron agentes muy distintos: el pequeño llama herramientas sin necesidad, repite llamadas y malinterpreta resultados.
- **Tres habilidades de usar herramientas:** (1) **decidir** si hace falta una y cuál; (2) escribir la petición en el **formato estricto**; (3) **leer el resultado** y usarlo. Los modelos pequeños fallan sobre todo en la primera y la tercera.
- **Elegir con datos propios:** antes de elegir un modelo para un agente, evaluarlo con **tus** herramientas y **tus** preguntas.
- **Variabilidad:** con pocas preguntas y modelos no totalmente deterministas, los números son orientativos; la comparación sirve para decidir, no para prometer exactitud.
- **Costo de un agente:** suma de **todas** las llamadas al modelo de la ejecución, cada una con el historial acumulado. Se mide sumando los tokens reales.

## 9. Riesgos y seguridad

| Riesgo | Qué pasa | Defensa |
| --- | --- | --- |
| **Acción no pedida** | Un modelo débil llama a una herramienta que escribe sin que se lo pidan | **Mínimo privilegio**; descripción que limita el uso |
| **Acción repetida** | Los modelos reintentan y se repiten | Herramientas **idempotentes** |
| **Inyección indirecta** | Un resultado de herramienta o un documento contiene órdenes dirigidas al asistente | Tratar todo resultado como **dato no confiable**; mínimo privilegio; aprobación humana |
| **Bucle sin fin** | El modelo repite llamadas indefinidamente y gasta dinero | **Límite de pasos** y de llamadas por herramienta |
| **Argumentos peligrosos** | El modelo pide una acción con valores peligrosos | **Validar en el código**, con listas de valores permitidos |
| **Falla inesperada** | Una herramienta falla a mitad del bucle | Reintentos y **mensajes seguros** |
| **Fuga por errores** | Un mensaje de error revela detalles internos (servidores, usuarios) | Devolver mensajes genéricos al modelo y al usuario |

Conceptos:

- **Mínimo privilegio:** dar al agente **solo las herramientas que necesita**. Un agente de solo lectura **no puede** hacer daño con una escritura aunque el modelo lo "quiera". Es la defensa **más barata y robusta**, porque **no depende del modelo**.
- **Inyección indirecta de instrucciones:** órdenes maliciosas escondidas en datos que el agente lee (documentos o resultados de herramientas). Con herramientas es **más grave** que en un RAG: el agente puede **actuar** sobre ellas.
- **Que el modelo resista hoy no lo hace seguro:** un modelo capaz suele resistir, uno pequeño suele caer; los ataques mejoran y los modelos cambian.
- **Defensa en profundidad:** combinar varias barreras (mínimo privilegio, aprobación humana, validación, registro de la traza, límites), sabiendo que **ninguna es perfecta**.
- **Aprobación humana (*human in the loop*):** pausar el agente antes de una acción con efectos y esperar la confirmación de una persona.
- **Registro (*logging*) de la traza:** guardar qué pidió el modelo y qué se ejecutó, para auditar.
- **Superficie de ataque:** cada herramienta y cada fuente de datos que el agente lee es un punto por donde puede entrar una orden no deseada.

## 10. Glosario rápido

| Término | Definición breve |
| --- | --- |
| **Agente** | Modelo que, en un bucle, pide herramientas y lee sus resultados hasta poder responder |
| **Herramienta (*tool*)** | Función del código que el modelo puede pedir que se ejecute |
| **Tool calling / function calling** | Respuesta del modelo con una petición de herramienta en vez de texto |
| **Esquema de la herramienta** | Descripción (nombre, descripción, parámetros) que recibe el modelo |
| **Bucle del agente** | *Modelo → herramienta → modelo* hasta responder con texto |
| **Mensaje de herramienta** | Resultado de una herramienta, enlazado con la petición del modelo |
| **Traza** | Registro completo de una ejecución del agente |
| **Límite de pasos** | Máximo de pasos antes de cortar el agente |
| **Middleware** | Pieza que se engancha al bucle (límites, reintentos, aprobación) |
| **Checkpointer / hilo** | Mecanismo de memoria del agente y su identificador de conversación |
| **Salida estructurada** | Resultado validado contra un esquema, no texto libre |
| **Streaming** | Recibir pasos o texto a medida que se producen |
| **Idempotente** | Repetir la acción no cambia el resultado |
| **Mínimo privilegio** | Dar solo los permisos y herramientas necesarios |
| **Inyección indirecta** | Órdenes ocultas en datos que el agente lee |
| **Modelo de juguete** | "Modelo" con respuestas de guion, para ver y probar el bucle sin LLM |
| **LangGraph** | Motor de grafos sobre el que corre el agente de LangChain |
| **LangSmith** | Servicio opcional de observabilidad |
