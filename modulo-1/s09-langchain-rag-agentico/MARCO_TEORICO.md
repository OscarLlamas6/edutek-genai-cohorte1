# Marco teórico · Sesión 9: LangChain, características útiles y RAG agéntico

> Documento de apoyo al notebook `s09_langchain_rag_agentico.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: el código y los experimentos están en el notebook y en la práctica `langchain-agentic-rag` del repositorio `ai-agents-playground`. Basado en la documentación de LangChain 1.x (noviembre de 2026); confirma los detalles de la API en la documentación vigente.

## Índice

1. [El agente en una página](#1-el-agente-en-una-página)
2. [Mensajes y `content_blocks`](#2-mensajes-y-content_blocks)
3. [Salida estructurada](#3-salida-estructurada)
4. [Streaming](#4-streaming)
5. [Middleware e ingeniería del bucle](#5-middleware-e-ingeniería-del-bucle)
6. [Memoria de corto y largo plazo](#6-memoria-de-corto-y-largo-plazo)
7. [Contexto de ejecución y errores en herramientas](#7-contexto-de-ejecución-y-errores-en-herramientas)
8. [LangGraph por debajo y observabilidad](#8-langgraph-por-debajo-y-observabilidad)
9. [RAG agéntico](#9-rag-agéntico)
10. [Servir un agente: API, sesiones e interfaz](#10-servir-un-agente-api-sesiones-e-interfaz)
11. [Glosario rápido](#11-glosario-rápido)

---

## 1. El agente en una página

- **Agente:** sistema en el que un **modelo de lenguaje decide, dentro de un bucle, qué herramientas usar** hasta resolver una tarea. Se contrapone al **flujo fijo** (pipeline o *workflow*), donde el código decide los pasos.
- **Tool calling (llamada a herramientas):** mecanismo por el cual el modelo, en lugar de responder con texto, devuelve una **petición estructurada** ("llama a esta herramienta con estos argumentos"). **El modelo nunca ejecuta nada**: pide, y el código ejecuta y devuelve el resultado.

| Pieza | Qué es | Concepto en LangChain 1.x |
| --- | --- | --- |
| Modelo | Quien razona y decide el siguiente paso | Chat model; `init_chat_model` como interfaz común |
| Herramientas | Funciones que el modelo puede pedir | Funciones con `@tool`; el docstring es la descripción |
| Prompt de sistema | Las reglas del agente | `system_prompt` |
| Mensajes | El estado de la conversación | Lista de mensajes |
| Bucle | Repite modelo → herramienta → modelo hasta que el modelo responde sin pedir herramientas | `create_agent` |

- **Lo único que el modelo sabe de una herramienta** es su nombre, su descripción y el esquema de sus parámetros: una herramienta es, en la práctica, un *prompt disfrazado de función*.
- **Condición de salida:** el bucle termina cuando el modelo responde sin pedir herramientas, o cuando un límite lo detiene (tope de pasos, de llamadas a herramientas, de presupuesto).
- **Costo:** cada vuelta del bucle es una llamada al modelo con todo el historial; un agente hace **varias llamadas por pregunta**.
- **Analogía de infraestructura:** un bucle de reconciliación (observar → decidir → actuar → volver a observar), como el de un controlador de Kubernetes.
- **Interfaces reemplazadas en la serie 1.x:** `AgentExecutor`, `initialize_agent` y `LLMChain` pertenecen a versiones anteriores; hoy se usa `create_agent`.

## 2. Mensajes y `content_blocks`

| Tipo de mensaje | Quién lo produce | Contenido |
| --- | --- | --- |
| Sistema | El desarrollador | Reglas del agente |
| Humano | El usuario | La pregunta |
| IA | El modelo | Texto y/o peticiones de herramienta (`tool_calls`), más metadatos de uso (`usage_metadata`: tokens de entrada, salida y total) |
| Herramienta | El código | Resultado de una herramienta, ligado a la petición por su `tool_call_id` |

- **`content_blocks`:** propiedad de los mensajes que expone el contenido como una **lista estándar y tipada**, independiente del proveedor. Tipos documentados: de texto (`text`, `reasoning`, `refusal`), multimodales (`image`, `audio`, `video`, `file`), de herramientas (`tool_call`, `tool_call_chunk`, `invalid_tool_call`), de herramientas del lado del servidor y un bloque de escape para datos propios del proveedor (`non_standard`).
- **Por qué importa:** permite cambiar de proveedor sin cambiar el código que lee las respuestas.
- **`usage_metadata`:** base para calcular el costo por pregunta; en un agente hay que **sumarlo sobre todas las llamadas** del bucle.

## 3. Salida estructurada

- **Salida estructurada:** hacer que la respuesta final del agente sea un **objeto con forma conocida**, validado contra un esquema (clase Pydantic, dataclass, diccionario tipado o JSON Schema). Se obtiene en la clave `structured_response` del resultado.
- **Estrategias:**

| Estrategia | Qué hace | Cuándo |
| --- | --- | --- |
| Esquema directo | LangChain elige entre la salida nativa del proveedor y *tool calling* | Lo más simple |
| `ToolStrategy` | Trata el esquema como una **herramienta final** que el modelo debe llamar; reintenta si el modelo rompe el esquema | Cualquier modelo con *tool calling* |
| `ProviderStrategy` | Usa la salida estructurada **nativa** del proveedor | Más fiable cuando el modelo la soporta |

- **Alcance de la garantía:** el esquema asegura la **forma**, no la **verdad**. Un campo de citas bien formado puede contener identificadores inventados; la verificación del contenido es responsabilidad del código.
- **Analogía:** un contrato de API: la salida del agente se valida antes de entregarse.

## 4. Streaming

- **Streaming:** emitir el progreso del agente mientras se ejecuta, en vez de esperar el resultado final.

| Modo | Qué emite | Uso |
| --- | --- | --- |
| `updates` | El estado tras cada paso (nodo del modelo, nodo de herramientas) | Progreso y depuración |
| `messages` | Los tokens del modelo según se generan | Respuesta "en vivo" |
| `custom` | Lo que una herramienta emite a propósito | Avisos de progreso de herramientas largas |

- **Eventos (`stream_events`, protocolo v3):** forma que la documentación actual recomienda para aplicaciones nuevas; ofrece iteradores separados por proyección (mensajes, valores, llamadas a herramientas). En la versión probada (`langgraph 1.2.22`) el protocolo v3 sigue marcado como **experimental** en el código, aunque la documentación lo recomienda. `stream_events` síncrono solo admite `version="v3"`; para los esquemas v1 y v2 se usa `astream_events`.

## 5. Middleware e ingeniería del bucle

- **Middleware:** código que se **engancha en puntos del bucle** para observar, modificar o limitar al agente **sin reescribirlo**. Equivale al middleware de un servidor web o a un *sidecar*: lógica transversal separada de la lógica de negocio.
- **Ganchos:**

| Gancho | Cuándo corre |
| --- | --- |
| Antes / después del agente | Una vez por invocación |
| Antes / después del modelo | En cada llamada al modelo |
| Envoltura de la llamada al modelo | Rodea cada llamada (reintentos, cambio de petición) |
| Envoltura de la llamada a herramienta | Rodea cada ejecución (captura de errores, medición, auditoría) |
| Prompt dinámico | Genera el prompt de sistema en cada llamada |

- **Middleware incluido** (según la documentación vigente; algunos requieren versiones recientes):

| Middleware | Función |
| --- | --- |
| Límite de llamadas a herramientas | Tope por ejecución o por hilo, global o por herramienta |
| Límite de llamadas al modelo | Tope de llamadas al modelo |
| Reintento de herramientas / de modelo | Reintentos con espera creciente (*backoff*) |
| Modelo de respaldo | Cambia de modelo si el primero falla |
| Resumen de conversación | Sustituye mensajes antiguos por un resumen según un umbral |
| Aprobación humana | Pausa antes de una herramienta para que una persona apruebe, edite o rechace; requiere *checkpointer* |
| Datos personales | Enmascara, bloquea o redacta datos como correos o tarjetas |
| Errores de herramientas | Convierte excepciones en mensajes legibles para el modelo |
| Edición de contexto, selección de herramientas, lista de tareas | Gestión del contexto y de la planificación |

- **Aprobación humana (*human-in-the-loop*):** defensa estándar para herramientas que **modifican** datos. El agente se **interrumpe**, guarda su estado y espera una decisión; después **se reanuda** con la decisión (aprobar, editar o rechazar).

### 5.1. Ingeniería del bucle (*loop engineering*)

Todo bucle de agente necesita tres cosas: **qué lo dispara** (la pregunta), **cómo se sabe que terminó** (el modelo responde sin pedir más herramientas) y **qué límites lo detienen** cuando algo sale mal.

| Límite | Qué evita | Cómo se implementa |
| --- | --- | --- |
| Máximo de llamadas a herramientas | Buscar en círculos | `ToolCallLimitMiddleware` |
| Máximo de vueltas del modelo | Un bucle que nunca cierra | `ModelCallLimitMiddleware` |
| Presupuesto de tokens | Una pregunta que gasta de más | Middleware propio que suma el uso de tokens de cada respuesta |
| Falta de progreso | Repetir la misma llamada esperando otro resultado | Middleware propio que compara nombre y argumentos de las llamadas |

- **Analogía:** el `timeout`, el *circuit breaker* y el presupuesto de un servicio. Un agente sin límites es un proceso sin `ulimit`.
- Los límites propios se escriben con un gancho **después del modelo** que puede **saltar al final** del bucle; el código, no el modelo, decide terminar, y deja un mensaje con el motivo.
- Un límite sin plan de acción solo esconde el problema: hay que decidir qué hace la aplicación cuando un guardián corta (avisar, registrar, escalar a una persona).

## 6. Memoria de corto y largo plazo

- **Memoria de corto plazo:** capacidad del agente de recordar la conversación **dentro de un mismo hilo**.
- **Checkpointer:** almacén que guarda el **estado** (los mensajes) tras cada paso. En memoria sirve para aprender; en producción se usa uno **persistente** (por ejemplo, sobre Postgres).
- **`thread_id`:** identifica la conversación. Mismo identificador, continúa; otro, empieza de cero. Equivale a un id de sesión.
- **Sin checkpointer, el agente no recuerda** entre invocaciones.
- **Límite:** todo el historial viaja al modelo en cada llamada, de modo que una conversación larga **satura la ventana de contexto**, encarece y degrada las respuestas.

| Remedio | Descripción | Precaución |
| --- | --- | --- |
| Recortar | Conservar solo los mensajes recientes | Perder lo antiguo; no separar una petición de herramienta de su resultado |
| Borrar | Eliminar mensajes concretos | Mantener un historial válido para el proveedor |
| Resumir | Sustituir lo antiguo por un resumen | Cuesta una llamada; el resumen puede omitir detalles |

- **Memoria de largo plazo:** información que persiste **entre conversaciones** (preferencias, datos del usuario); se guarda en un *store*, no en el historial del hilo.

### 6.1. Memoria de largo plazo (*store*)

- **Store:** almacén de clave-valor organizado por ***namespace*** (por ejemplo, un usuario). Guarda datos que deben sobrevivir **entre conversaciones** y entre ejecuciones del programa.
- El **checkpointer** recuerda una conversación (el historial del hilo); el **store** recuerda hechos duraderos (preferencias, equipo, idioma), sin importar el hilo.

| | Corto plazo | Largo plazo |
| --- | --- | --- |
| Pieza | Checkpointer + `thread_id` | *Store* + *namespace* |
| Qué guarda | El historial de una conversación | Datos duraderos de un usuario o del sistema |
| Quién escribe | El agente, automáticamente | Herramientas: el modelo decide qué vale la pena guardar |
| Persistencia | En RAM se pierde; en producción, base de datos (p. ej. `PostgresSaver`) | Igual: en RAM se pierde; en producción, base de datos (p. ej. `PostgresStore`, en el mismo Postgres del agente) |

- La **identidad** (el *namespace*) la inyecta el código mediante el contexto de ejecución, no el modelo: así un usuario no puede leer los datos de otro.
- **Qué no guardar:** secretos, contraseñas y datos que el usuario no esperaría que se recordaran. Todo lo que se guarda es un dato personal con ciclo de vida: hay que poder consultarlo y borrarlo.

## 7. Contexto de ejecución y errores en herramientas

| Dónde vive la información | Qué guarda | Duración | Quién lo escribe |
| --- | --- | --- | --- |
| Estado | Mensajes y campos propios del agente | La conversación (con checkpointer) | El agente y las herramientas |
| Contexto | Datos de **esta** ejecución: usuario, permisos, región | Una llamada | El desarrollador, al invocar |
| *Store* | Memoria de largo plazo | Entre conversaciones | Herramientas y middleware |

- **Contexto de ejecución:** se declara con un esquema al crear el agente, se entrega al invocar y se lee desde las herramientas (mediante un parámetro especial de *runtime* que **el modelo no ve**) y desde el middleware.
- **Principio de seguridad:** la identidad y los permisos los inyecta el código, no el modelo; así una herramienta puede filtrar por usuario sin depender de lo que el modelo decida.
- **Errores en herramientas:** una herramienta fallará. Estrategias: devolver al modelo un **mensaje de error legible** (si puede corregirlo), **reintentar** (si es transitorio) o **fallar con fuerza** (si no debe "arreglarlo" el modelo).

## 8. LangGraph por debajo y observabilidad

- **LangGraph:** biblioteca para construir flujos con estado como **grafos** (nodos, aristas y ciclos). `create_agent` construye un grafo de LangGraph: un nodo del modelo, un nodo de herramientas y los ganchos del middleware.

| `create_agent` | Grafo propio (`StateGraph`) |
| --- | --- |
| El modelo decide el flujo; se acota con middleware | El desarrollador dibuja el flujo |
| Rápido y estándar | Control fino: pasos obligatorios, ramas fijas, varios agentes |

- **Criterio:** empezar con `create_agent`; bajar al grafo propio cuando ciertos pasos deban ocurrir **siempre y en un orden** o haya que coordinar varios agentes.
- **Observabilidad:** ver el camino del agente (herramientas pedidas, argumentos, tokens, costo, latencia). Opciones: la traza de mensajes, middleware propio y **LangSmith** (servicio de trazas y evaluación de LangChain; opcional).

## 9. RAG agéntico

- **RAG de flujo fijo:** el código ejecuta siempre buscar → armar prompt → generar.
- **RAG agéntico:** un RAG en el que la **búsqueda es una herramienta** del agente y el **modelo controla la recuperación**.

| Decisión | RAG fijo | RAG agéntico |
| --- | --- | --- |
| ¿Hace falta buscar? | Siempre sí | El modelo decide |
| ¿En qué fuente? | Una | Elige entre varias herramientas |
| ¿Con qué consulta? | La pregunta tal cual | La que formula el modelo |
| ¿Cuántas veces? | Una | Las necesarias, con tope |
| ¿Sirve lo recuperado? | No se evalúa | Puede juzgarlo y reformular |
| ¿Hay respuesta? | Responde con lo que llegó | Puede concluir que no la hay |

- **Patrones de referencia** (descritos en artículos de 2022 a 2024; nombres para ubicar las ideas):

| Patrón | Idea | Referencia |
| --- | --- | --- |
| ReAct | Alternar razonar, actuar y observar | Yao et al., 2022 |
| Adaptive-RAG | Enrutar según la complejidad de la pregunta | Jeong et al., 2024 |
| Corrective RAG (CRAG) | Evaluar lo recuperado y corregir si no sirve | Yan et al., 2024 |
| Self-RAG | El modelo reflexiona sobre si recuperar y sobre el respaldo de su respuesta | Asai et al., 2023 |

- **Ejemplo de la documentación de LangGraph:** un grafo en el que el modelo decide si recuperar; un paso califica la relevancia; si no sirve, se reescribe la pregunta y se reintenta; si sirve, se genera la respuesta.
- **Comparación:**

| | RAG fijo | GraphRAG | RAG agéntico |
| --- | --- | --- | --- |
| Qué cambia | Nada: es la base | La estructura de los datos | Quién decide el flujo |
| Resuelve mejor | Preguntas directas | Relaciones y preguntas globales | Fuentes múltiples, preguntas imprevisibles |
| Costo | Bajo y predecible | Alto al indexar | Alto al responder |
| Predecibilidad | Alta | Alta | Baja: el camino varía |

Las tres opciones **se combinan**: un agente puede tener como herramientas una búsqueda vectorial y una consulta a un grafo. Criterio: empezar por la más simple que cumpla el objetivo medido.

- **Costos y latencia:** una pregunta puede requerir 3 o 4 llamadas al modelo en lugar de una, cada una con todo el historial (que incluye los fragmentos recuperados). Se controla con topes de llamadas, límite de recursión, modelos económicos para decisiones simples y resultados de herramienta cortos.
- **Riesgos:**

| Riesgo | Causa | Defensa |
| --- | --- | --- |
| pgvector | Extensión de Postgres que guarda vectores y busca por similitud |
| Herramienta de escritura | Herramienta que modifica datos; debe ser acotada y de efecto limitado |
| Bitácora de consultas | Registro de cada pregunta con tokens y herramientas; lo escribe el código |
| API (FastAPI) | Interfaz HTTP del agente, con contratos Pydantic y documentación en `/docs` |
| `session_id` | Identificador de sesión en la API; es el `thread_id` del checkpointer |
| Servicio sin estado en el proceso | La conversación vive en el checkpointer (Postgres), no en la memoria del servidor |
| Streamlit | Biblioteca de interfaces web sencillas; aquí consume la API |
| Bruno | Cliente HTTP con colecciones en archivos de texto |
| Click | Biblioteca para armar una CLI con comandos |
| Ingesta | Fragmentar, vectorizar y guardar los documentos; paso aparte de las preguntas |
| Inyección indirecta | Un documento recuperado contiene instrucciones y el modelo las obedece | Tratar lo recuperado como datos; mínimo privilegio; aprobación humana para acciones |
| Bucle de búsquedas | El modelo reformula sin converger | Topes, límite de recursión, regla de abstención |
| Citas inventadas | El modelo cita lo que nunca vio | Verificar en código que cada cita apunta a algo recuperado |
| Responder sin buscar | El modelo cree saber | Prompt claro sobre cuándo buscar; medir la elección de herramienta |

- **Evaluación:** por capas, como cualquier RAG, más lo propio del agente: si **buscó cuando debía y no cuando no**, la calidad de las consultas que formuló, el número de pasos y el costo, y si la respuesta es correcta y está respaldada por citas válidas; siempre contra un conjunto fijo de preguntas y con un tamaño de muestra honesto.

### 9.x. Más herramientas que el RAG, y herramientas que escriben

- Un RAG agéntico es un agente con **la búsqueda como una herramienta entre varias**. Junto a ella pueden convivir herramientas de **consulta** (un ticket, un equipo en un inventario), de **cálculo** (un plazo en horas hábiles, que el modelo no debe hacer de cabeza) y de **escritura** (registrar algo en una base de datos). Entonces la decisión interesante del modelo es **cuál usar**, y a veces combinarlas.
- **Herramienta de lectura:** no cambia nada; se puede reintentar sin riesgo.
- **Herramienta de escritura:** cambia el mundo (una fila en una base). Su descripción debe decir **cuándo** usarla y cuándo no, y conviene que sea **aditiva** (agrega, no borra ni edita).
- **Auditoría vs. acción:** una **auditoría** (bitácora de preguntas, tokens, herramientas usadas) debe escribirla el **código**, siempre, con un gancho posterior a la ejecución; no una herramienta. Un modelo puede olvidarse de llamarla y no conoce los tokens de su propia respuesta. Una **acción** (registrar un hueco del manual) sí la decide el modelo.
- **Base vectorial en el agente:** los vectores del manual viven en **Postgres con pgvector** (la misma idea de la sesión de embeddings): la ingesta se hace una vez y cada búsqueda solo vectoriza la pregunta; la distancia coseno se calcula en la base. Las tablas de bitácora y de huecos viven en la misma base.

## 10. Servir un agente: API, sesiones e interfaz

- **Exponer un agente:** ponerlo detrás de una **API HTTP** para que lo usen otras personas y otros programas. Es la forma estándar de pasar de un notebook a un servicio.
- **API (FastAPI):** recibe `POST /chat` con `{message, session_id, user_id}` y devuelve la respuesta, las herramientas usadas, las citas y los tokens. Los contratos de entrada y de salida son modelos **Pydantic**: validan sin escribir código (422 si el cuerpo es inválido) y generan la documentación interactiva en `/docs`. Códigos HTTP con sentido: 404 sesión inexistente, 503 si falta la ingesta.
- **Sesión:** una conversación. El **`session_id`** de la API es el **`thread_id`** del checkpointer. Sin id, el servidor crea uno y lo **devuelve**; con id, continúa esa conversación. El cliente no guarda la conversación: solo el identificador.
- **Servicio sin estado en el proceso:** el proceso de la API no guarda conversaciones. El estado vive en el **checkpointer persistente** (Postgres, con `PostgresSaver`), de modo que se puede reiniciar el servidor o tener varias réplicas y la sesión continúa. Con un checkpointer en RAM, las sesiones se pierden al reiniciar y cada réplica tendría su propia memoria.
- **Analogía de infraestructura:** como un contenedor detrás de un balanceador: sin estado local, con el estado en una base de datos y una cookie de sesión en el cliente.
- **Clientes de la API:** una **interfaz** sencilla (Streamlit) que solo habla HTTP con la API; un **cliente HTTP** (Bruno) para recorrerla paso a paso con colecciones versionables; `curl`; otro programa. La interfaz **no** importa el agente.
- **CLI (Click):** comandos para tareas de operación: `ingest` (la **ingesta** del RAG), `ask`, `chat`, `log` y `serve`.
- **Ingesta como paso aparte:** fragmentar, vectorizar y guardar el manual en pgvector es costoso y se hace **una vez** (con una huella del manual y del modelo de embeddings para saber si hace falta repetirlo). Las preguntas no deberían dispararla; si falta, se responde con un error claro en vez de gastar tokens.
- **Modelo de juguete (`fake`):** un modelo determinista, sin llaves ni red, para demostrar el mecanismo (API, sesiones, bitácora, herramientas) con resultados repetibles. Demuestra el **mecanismo**, no la calidad de elección de un modelo real.

## 11. Glosario rápido

| Término | Definición |
| --- | --- |
| Agente | Modelo que decide, en un bucle, qué herramientas usar |
| Tool calling | El modelo devuelve una petición estructurada de herramienta; el código la ejecuta |
| `create_agent` | Función de LangChain 1.x que construye el agente |
| `content_blocks` | Vista estándar y tipada del contenido de un mensaje |
| Salida estructurada | Respuesta final validada contra un esquema |
| Middleware | Código que se engancha en el bucle sin reescribirlo |
| Checkpointer | Almacén del estado del agente tras cada paso |
| `thread_id` | Identificador de una conversación |
| Contexto de ejecución | Datos de una ejecución (usuario, permisos) que el modelo no controla |
| Store | Almacén de memoria de largo plazo |
| LangGraph | Biblioteca de flujos con estado como grafos |
| LangSmith | Servicio opcional de trazas y evaluación |
| RAG agéntico | RAG en el que la búsqueda es una herramienta y el modelo controla el flujo |
| Inyección indirecta | Instrucciones maliciosas escondidas en datos que el modelo lee |
