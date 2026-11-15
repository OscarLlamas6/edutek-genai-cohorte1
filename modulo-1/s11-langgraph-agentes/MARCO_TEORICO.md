# Marco teórico · Sesión 11: LangGraph, agentes como grafos

> Documento de apoyo al notebook `s11_langgraph_agentes.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: el código y los experimentos están en el notebook y en la práctica `langgraph-trello-ops-agent` del repositorio `ai-agents-playground`. Basado en la documentación de LangGraph 1.x (noviembre de 2026); confirma los detalles de la API en la documentación vigente.

## Índice

1. [Por qué un grafo](#1-por-qué-un-grafo)
2. [Estado, nodos y aristas](#2-estado-nodos-y-aristas)
3. [Aristas condicionales y ciclos](#3-aristas-condicionales-y-ciclos)
4. [El agente como grafo](#4-el-agente-como-grafo)
5. [Checkpointers e hilos](#5-checkpointers-e-hilos)
6. [Humano en el bucle: `interrupt`](#6-humano-en-el-bucle-interrupt)
7. [Store: memoria de largo plazo](#7-store-memoria-de-largo-plazo)
8. [Loop engineering: topes, errores y reintentos](#8-loop-engineering-topes-errores-y-reintentos)
9. [Streaming y observabilidad](#9-streaming-y-observabilidad)
10. [Subgrafos](#10-subgrafos)
11. [Estructura de un proyecto y pruebas](#11-estructura-de-un-proyecto-y-pruebas)
12. [Servir un agente: API, sesiones e interfaz](#12-servir-un-agente-api-sesiones-e-interfaz)
13. [El agente de operaciones sobre Trello](#13-el-agente-de-operaciones-sobre-trello)
14. [Glosario rápido](#14-glosario-rápido)
15. [Preguntas de repaso](#15-preguntas-de-repaso)

---

## 1. Por qué un grafo

- **LangGraph:** biblioteca para construir **flujos con estado** modelados como **grafos**. Es la base sobre la que LangChain construye `create_agent`: el agente "listo para usar" **es** un grafo con un nodo del modelo y un nodo de herramientas.
- **Control explícito:** en un grafo propio, **tú** dibujas el flujo (nodos, aristas y ciclos). En `create_agent`, el **modelo** decide el flujo y tú lo acotas con middleware.

| `create_agent` | `StateGraph` |
| --- | --- |
| El modelo decide el flujo | El desarrollador dibuja el flujo |
| Cambiar el comportamiento = middleware | Cambiar el comportamiento = editar el grafo |
| Rápido y estándar | Más código, más control |

- **Qué garantiza un grafo que un prompt no:** que un paso ocurra **siempre** y en cierto orden (por ejemplo, "pedir aprobación antes de escribir"). Una instrucción en el prompt es una **petición** al modelo; una arista es una **garantía** del código.
- **Cuándo usar LangGraph:** acciones con efectos que exigen aprobación o pasos fijos; ramas, ciclos o pasos obligatorios; pausas y reanudación (humano en el bucle); sobrevivir a reinicios; varios agentes o subprocesos; necesidad de probar el flujo con determinismo.
- **Cuándo no:** el agente solo lee y responde; el bucle estándar basta; la conversación es corta y no necesita persistir; es un prototipo rápido. **Regla:** empezar por lo más simple (una llamada, luego `create_agent`) y bajar al grafo cuando se necesite control que el prompt no puede garantizar.
- **Analogía de infraestructura:** un pipeline de CI/CD o un runbook automatizado: pasos fijos, condiciones, aprobaciones manuales y reanudación tras una pausa.

## 2. Estado, nodos y aristas

- **Estado (*State*):** el "documento" compartido que viaja por el grafo. Se define con un diccionario tipado (`TypedDict`) o un modelo Pydantic. Es lo que el checkpointer guarda tras cada paso.
- **Nodo:** función que recibe el estado y devuelve una **actualización parcial** (solo las claves que cambian), no el estado completo.
- **Reducer:** función que define **cómo se combina** el valor anterior de una clave con el nuevo. Sin reducer, la clave se **reemplaza**; con reducer, se **combina** (por ejemplo, concatenar listas o agregar mensajes). Los reducers también permiten que varios nodos escriban la misma clave en el mismo paso.
  - `add_messages`: agrega mensajes a la lista o **reemplaza** el que tenga el mismo identificador.
  - `MessagesState`: estado predefinido con una clave de mensajes y ese reducer.
- **Arista:** conexión que dice qué nodo sigue a cuál. `START` y `END` son los extremos del grafo.
- **Compilación:** `compile()` valida el grafo (nodos huérfanos, aristas inexistentes) y devuelve un objeto ejecutable. Es aquí donde se asocian el **checkpointer** y el **store**.
- **Ejecución:** `invoke` devuelve el estado final; `stream` entrega el progreso. El grafo se puede **inspeccionar y dibujar** (por ejemplo, como diagrama Mermaid).
- **Estado con Pydantic frente a `TypedDict`:** Pydantic valida los datos al entrar y explicita los tipos, con algo más de sobrecarga; `TypedDict` es lo habitual en estados con muchos mensajes.

## 3. Aristas condicionales y ciclos

- **Arista condicional:** una **función** que lee el estado y devuelve **el nombre del siguiente nodo** (o una clave que se traduce a un nodo). Permite ramificar el flujo **con código**, no con un prompt.
- **Ciclo:** una arista que vuelve a un nodo anterior. Un ciclo convierte un grafo en un **bucle**; el bucle de un agente es el ciclo *modelo → herramientas → modelo*.
- **Todo ciclo necesita una condición de salida** que combine el **éxito** y un **tope** (de pasos, de tiempo, de presupuesto). Un ciclo sin tope, con un modelo, es un bucle que **cuesta dinero** en cada vuelta.

## 4. El agente como grafo

- **Estructura mínima:** un nodo **modelo**, una arista condicional ("¿el último mensaje trae peticiones de herramienta?") y un nodo **herramientas** que vuelve al modelo.
- **El modelo pide; el código ejecuta.** El nodo del modelo agrega la respuesta (que puede traer peticiones); el nodo de herramientas ejecuta cada petición y agrega un mensaje de resultado por cada una, ligado a la petición por su identificador.
- **Componentes prediseñados:** LangGraph ofrece un nodo de herramientas y una condición estándar que replican este bucle. Conviene **escribir el propio nodo** cuando hace falta decidir por herramienta si requiere aprobación, llevar una bitácora o acceder al Store.
- **Clasificar las herramientas por riesgo** y dejar que **el grafo**, no el modelo, decida qué pasa con cada grupo:

| Grupo | Ejemplos | Qué hace el grafo |
| --- | --- | --- |
| Lectura | Listar tableros, buscar tarjetas, consultar el runbook | Ejecuta sin preguntar |
| Escritura | Crear tarjeta, mover, comentar | **Pausa y pide aprobación humana** |
| Memoria | Guardar una preferencia | La resuelve el grafo (acceso al Store), con lista cerrada de claves |

## 5. Checkpointers e hilos

- **Checkpoint:** foto del estado del grafo **después de cada paso**.
- **Checkpointer:** almacén de checkpoints. Implementaciones: en memoria (se pierde al cerrar; para aprender), SQLite (desarrollo y demostraciones) y Postgres (producción).
- **`thread_id`:** identificador de una conversación (hilo). Mismo identificador, la conversación continúa; otro, empieza de cero. Es un **id de sesión** y el checkpointer es el **almacén de sesiones**.
- **Qué habilita:**
  1. **Memoria de corto plazo:** el historial del hilo.
  2. **Persistencia:** un proceso nuevo, con el mismo almacén y el mismo `thread_id`, retoma la conversación.
  3. **Pausas:** se puede detener el grafo y reanudarlo después (sección 6).
  4. ***Time travel*:** listar el historial de checkpoints, **reanudar** desde uno anterior o **bifurcar** modificando su estado. La bifurcación **añade** checkpoints; no reescribe el pasado. Es como `git checkout` sobre ejecuciones y sirve para depurar.
- **Límite de la memoria de corto plazo:** todo el historial viaja al modelo en cada llamada; una conversación larga satura la ventana de contexto, encarece y empeora las respuestas. Remedios: recortar, borrar o resumir mensajes, aplicados en el nodo del modelo antes de llamarlo.

## 6. Humano en el bucle: `interrupt`

- **`interrupt(payload)`:** pausa el grafo en ese punto, **guarda el estado** y entrega el *payload* a quien invocó (se recibe en la clave `__interrupt__` del resultado).
- **`Command(resume=valor)`:** reanuda el grafo; `valor` es **lo que devuelve** `interrupt()` dentro del nodo. `Command` también permite actualizar el estado y saltar a otro nodo.
- **Requisitos:** un checkpointer y el **mismo `thread_id`** al reanudar.
- **Para qué:** aprobar, editar o rechazar acciones con efectos antes de ejecutarlas; es el equivalente a una **aprobación manual** en un pipeline. La seguridad no depende de que el modelo se porte bien: depende de una arista.
- **Reglas de uso:**
  1. Al reanudar, **el nodo se vuelve a ejecutar desde su primera línea**: lo que esté antes de `interrupt()` debe poder repetirse sin daño. Los **efectos secundarios van después**, idealmente en **otro nodo**.
  2. No envolver `interrupt()` en un manejo de excepciones genérico: funciona lanzando una excepción especial.
  3. El *payload* y la decisión deben ser serializables.
  4. Si hay varias pausas, el orden debe ser estable: los valores de reanudación se asocian por posición.
- **Decisión rica:** más que sí/no, la persona puede devolver un **motivo**; el modelo lo recibe como mensaje y puede ofrecer una alternativa.
- **Defensa en profundidad:** además de la pausa, el nodo que ejecuta las escrituras **se niega a escribir si no hay una aprobación registrada** en el estado.

## 7. Store: memoria de largo plazo

- **Store:** almacén de documentos clave-valor **compartido entre hilos**, organizado por **espacio de nombres** (*namespace*), por ejemplo "preferencias de este usuario".
- **Diferencia con el checkpointer:**

| | Checkpointer | Store |
| --- | --- | --- |
| Alcance | Un hilo | Entre hilos |
| Contenido | Estado del grafo tras cada paso | Datos que se quieren recordar |
| Analogía | La sesión | El perfil del usuario |

- **Operaciones:** guardar, leer, buscar dentro de un espacio de nombres y borrar. La búsqueda puede ser semántica si se configura un modelo de embeddings.
- **Contexto de ejecución (`context`):** datos de **una ejecución** que **pone el código** al invocar (por ejemplo, el identificador del usuario). El modelo no decide a nombre de quién actúa. El Store y el contexto llegan al nodo mediante el *runtime*.
- **Uso típico:** al comenzar cada turno, el grafo lee las preferencias del usuario y las **inserta en el prompt de sistema**.
- **Qué guardar:** preferencias y datos útiles; **nunca** contraseñas ni datos sensibles. Conviene una **lista cerrada de claves** para que el modelo no guarde cualquier cosa.
- **Implementaciones:** en memoria, SQLite (desarrollo) y Postgres (producción).

## 8. Loop engineering: topes, errores y reintentos

- ***Loop engineering*:** diseñar **cuándo se detiene** un bucle de agente y **qué hace cuando algo sale mal**. Cada vuelta con un modelo cuesta dinero y tiempo.
- **`recursion_limit`:** máximo de **superpasos** (rondas de nodos) antes de lanzar un error de recursión. Es la **última barrera**: cuando salta, el usuario ve un error. No es el diseño.
- **Topes de diseño:**

| Tope | Qué mide | Qué hace al saltar |
| --- | --- | --- |
| Número de pasos | Llamadas al modelo en esta pregunta | Un nodo de parada responde y termina |
| Presupuesto | Tokens o dinero acumulados (metadatos de uso de cada respuesta) | Igual |
| Falta de progreso | La misma petición con los mismos argumentos, repetida | Igual |

- **Nodo de parada correcto:** responde a **cada petición de herramienta pendiente** ("no ejecutada") y luego al usuario. Si no, el historial queda con peticiones sin respuesta y el siguiente turno falla con un modelo real.
- **El presupuesto es por pregunta:** los contadores se reinician al comenzar cada turno, aunque el hilo continúe.
- **Errores:** tres respuestas posibles.
  1. **Devolverlos al modelo** como mensaje legible para que corrija el argumento.
  2. **Reintentar** con espera creciente si son transitorios (red, límite de tasa); en LangGraph, una política de reintentos asociada al nodo, que solo reintenta las excepciones declaradas.
  3. **Dejar que suban** si no es algo que el modelo deba arreglar (por ejemplo, un permiso denegado).
- **Protecciones del lado del sistema externo:** lista de recursos donde se permite escribir, modo *dry-run* (valida y no escribe), sin operaciones destructivas, límites de resultados, tiempo límite por llamada y errores sin credenciales.

## 9. Streaming y observabilidad

- **Modos de `stream`:** `values` (estado completo tras cada paso), `updates` (lo que devolvió cada nodo: ideal para progreso y depuración), `messages` (tokens del modelo en vivo) y `custom` (datos que el nodo emite con un escritor de *stream*). Se pueden combinar.
- **Observabilidad mínima por pregunta:** qué nodos corrieron, qué herramientas pidió el modelo y con qué argumentos, cuántos tokens, cuánto tardó y cuánto costó.
- **Tres niveles:** el propio grafo (actualizaciones y una **bitácora** dentro del estado), instrumentación propia (por ejemplo, medir cada nodo) y **LangSmith** (servicio opcional de trazas y evaluación; se activa con variables de entorno).
- **Bitácora de auditoría:** lista acumulada (con reducer) de cada herramienta ejecutada, rechazada, fallida o detenida, con su fecha. Es lo que permite reconstruir qué hizo el agente y quién lo aprobó.

## 10. Subgrafos

- **Subgrafo:** un grafo compilado que se usa **como nodo** de otro. Encapsula un proceso con su propio estado y permite reutilizarlo.
- **Estado compartido:** las claves comunes entre padre e hijo son las que se intercambian.
- **Cuándo usarlos:** un proceso con identidad propia (varios nodos, quizá su propia pausa) o equipos distintos construyendo partes distintas. Para un agente pequeño no hacen falta.

## 11. Estructura de un proyecto y pruebas

- **Estructura recomendada:** separar el **estado**, los **nodos** y aristas, las **herramientas**, el **ensamblado del grafo** y el **sistema externo**; más un punto de entrada (línea de comandos), datos (el runbook), pruebas y configuración.
- **`build_graph(...)` recibe sus dependencias** (modelo, cliente del sistema externo, buscador, checkpointer, store): la misma función sirve en producción y en las pruebas.
- **`langgraph.json`:** archivo de configuración que registra los grafos (nombre → archivo y variable), las dependencias, el archivo de entorno y la versión de Python.
- **`langgraph dev`:** servidor local de desarrollo (paquete `langgraph-cli` con soporte en memoria; requiere Python 3.11 o superior). Expone la API del grafo en el puerto 2024 y recarga al editar el código. **El servidor aporta su propio checkpointer y store**: el grafo servido se compila sin ellos.
- **Studio:** interfaz web que se conecta al servidor local; dibuja el grafo, muestra cada paso, permite editar el estado, reanudar pausas y repetir desde un checkpoint. Puede requerir una cuenta gratuita de LangSmith.
- **Cómo se prueba un grafo:** con un **modelo guionado** (responde con mensajes preparados) y un **sistema externo falso**, el flujo es determinista y no usa red ni dinero.

| Qué probar | Cómo |
| --- | --- |
| Aprobación aceptada y rechazada | Escritura → pausa → reanudar con sí o no → comprobar si el sistema falso recibió la escritura |
| Tope de pasos y falta de progreso | Guion que nunca termina → comprobar la parada y que el historial queda válido |
| Persistencia | Mismo hilo con un checkpointer nuevo sobre el mismo archivo |
| Memoria de largo plazo | Guardar en un hilo, leer en otro; otro usuario no la ve |
| Pausa pendiente tras reinicio | Detener el proceso en la pausa, abrir otro y reanudar |
| Cliente del sistema externo | Sesión HTTP falsa: rutas, cabeceras, guardas, *dry-run* y reintentos |

- **Lo que las pruebas no cubren:** la calidad de las decisiones de un modelo real (se evalúa aparte, con un conjunto fijo de casos) ni el sistema externo real.

## 12. Servir un agente: API, sesiones e interfaz

- **Principio:** el agente es un **servicio sin estado en el proceso**; el estado vive en el checkpointer y el cliente solo lleva el **identificador de sesión**. Es el mismo principio de un servicio web detrás de un balanceador: cualquier réplica puede atender porque la sesión está en un almacén externo.
- **`session_id` = `thread_id`:** la API crea un identificador (uuid) si el cliente no envía uno, **lo devuelve**, y lo usa como `thread_id` del checkpointer. Mismo identificador, misma conversación; otro, otra conversación.
- **Persistencia real:** con un checkpointer persistente (SQLite en desarrollo, Postgres en producción), **reiniciar la API no pierde las sesiones**, ni siquiera una aprobación pendiente. Con uno en memoria sí se pierden.
- **Contrato de la API (FastAPI):** modelos de entrada y salida con validación (se ven en la documentación interactiva), y códigos HTTP con significado: 404 sesión inexistente, 403 sesión de otro usuario, 409 conflicto (sesión esperando aprobación o nada que aprobar), 422 validación, 503 si el agente no está disponible.
- **La aprobación por HTTP:** la pausa de `interrupt()` se traduce en el **estado de la respuesta** (`awaiting_approval`, con la herramienta y los argumentos propuestos); la persona decide en el cliente y una segunda petición (`approve`) reanuda el grafo con la decisión. Mientras hay una pausa, un mensaje nuevo a esa sesión se rechaza con 409.
- **Qué devuelve una respuesta útil:** el estado (`completed`, `awaiting_approval` o `halted` si saltó un tope), el texto, la acción pendiente, las herramientas usadas, las escrituras ejecutadas o rechazadas y el uso (llamadas al modelo y tokens).
- **Clientes:** un cliente HTTP con colecciones en archivos de texto (**Bruno**), que permite recorrer la API paso a paso, guardar variables (el `session_id`) entre peticiones y verificar con aserciones y pruebas, y una interfaz sencilla (**Streamlit**) que consume la API por HTTP sin importar el agente.
- **FastAPI propio o LangGraph Server:**

| | FastAPI propio | LangGraph Server (Agent Server) |
| --- | --- | --- |
| Contrato | El del desarrollador | Asistentes, hilos, ejecuciones, *store*; documentado en `/docs` |
| Persistencia | La que se elija | La aporta el servidor (en `langgraph dev`, en memoria) |
| Autenticación y middleware | A cargo del desarrollador | Del servidor; admite rutas FastAPI propias (`http.app` en `langgraph.json`) |
| Studio | No | Sí |
| Cuándo | Contrato pequeño y estable, integración con la infraestructura propia | Hilos, ejecuciones en segundo plano, *streaming*, tareas programadas y Studio sin escribirlos; despliegue gestionado |

  `langgraph dev` es para **desarrollo y depuración**, no para producción (corre en memoria). Los requisitos y la licencia del autoalojamiento del servidor deben consultarse en la documentación vigente.

## 13. El agente de operaciones sobre Trello

- **Trello:** sistema de gestión de trabajo con **tableros**, **listas** (columnas) y **tarjetas**. Su API REST es gratuita y se autentica con una **clave** (API key) y un **token** de usuario con alcance (lectura y/o escritura) y caducidad; el token no debe exponerse.
- **Operaciones del agente:**

| Tipo | Operaciones | Aprobación |
| --- | --- | --- |
| Lectura | Listar tableros y listas, ver y buscar tarjetas, consultar el runbook | No |
| Escritura | Crear tarjeta, moverla entre listas, comentarla | **Sí, siempre** |
| Memoria | Guardar el tablero por defecto | No (lista cerrada de claves) |

- **Flujo del grafo:** cargar contexto (preferencias del Store y contadores) → agente (el modelo decide) → según el resultado: terminar, parar (tope o falta de progreso), ejecutar lecturas, o **pedir aprobación y luego ejecutar** → volver al agente.
- **RAG sobre un runbook:** un manual pequeño (dónde se crean las tarjetas, cómo se nombran y se mueven) se consulta como una herramienta de lectura; así el agente **decide qué hacer según las reglas del equipo**, no según su intuición.
- **Modo simulado:** un Trello en memoria (y un modelo y unos embeddings falsos) permite probar y enseñar sin cuenta. El modo real requiere clave y token en el archivo de entorno, una **cuenta y un tablero de prueba** y una lista de tableros donde se permite escribir.
- **MCP opcional:** las herramientas de un servidor MCP pueden sumarse al agente; por **mínimo privilegio**, toda herramienta MCP se trata como escritura (pide aprobación) salvo las declaradas de solo lectura.
- **Qué lo distingue de la integración de MCP (S10):** allí el foco era **exponer** un sistema externo (un CRM) a un agente mediante un servidor; aquí el foco es **orquestar** al agente con un grafo (control, pausas, memoria, topes) sobre un sistema de gestión de trabajo.

## 14. Glosario rápido

| Término | Definición |
| --- | --- |
| LangGraph | Biblioteca de flujos con estado como grafos |
| `StateGraph` | Constructor de grafos sobre un estado compartido |
| Estado | Datos que viajan por el grafo |
| Nodo | Función estado → actualización parcial |
| Arista | Conexión entre nodos; `START` y `END` son los extremos |
| Arista condicional | Función que devuelve el nombre del siguiente nodo |
| Reducer | Función que combina el valor anterior con el nuevo de una clave |
| Checkpoint | Foto del estado tras un paso |
| Checkpointer | Almacén de checkpoints |
| `thread_id` | Identificador de una conversación |
| *Time travel* | Reanudar o bifurcar desde un checkpoint anterior |
| `interrupt` | Pausa el grafo y entrega un payload |
| `Command(resume=...)` | Reanuda un grafo pausado |
| Store | Memoria de largo plazo entre hilos |
| Contexto de ejecución | Datos de una ejecución que pone el código |
| Subgrafo | Grafo compilado usado como nodo |
| `recursion_limit` | Máximo de superpasos |
| *Loop engineering* | Diseñar cuándo y cómo se detiene un bucle |
| `langgraph.json` | Registro de grafos de un proyecto |
| `langgraph dev` | Servidor local de desarrollo |
| Studio | Interfaz web para depurar un grafo servido |
| Dry-run | Ejecución que valida y no escribe |
| `session_id` | Identificador de sesión del cliente; es el `thread_id` del checkpointer |
| `awaiting_approval` | Estado de la API con el grafo pausado en un `interrupt` |
| FastAPI | Framework web de Python con validación y documentación automática |
| Agent Server | Servidor de LangGraph (asistentes, hilos, ejecuciones, *store*) |
| Bruno | Cliente HTTP con colecciones en archivos de texto y un CLI |
| Streamlit | Biblioteca para interfaces web sencillas en Python |
| Bitácora de auditoría | Registro acumulado de lo que hizo el agente |

## 15. Preguntas de repaso

1. ¿Qué garantiza un grafo propio que `create_agent` deja en manos del modelo?
2. ¿Qué diferencia hay entre una clave del estado con reducer y una sin él?
3. ¿Qué devuelve la función de una arista condicional?
4. ¿Qué es un `thread_id` y qué ocurre si se reanuda un `interrupt` con otro?
5. ¿Por qué el efecto secundario de una escritura va **después** de `interrupt()` y, mejor, en otro nodo?
6. ¿Qué diferencia hay entre checkpointer y Store? Da un ejemplo de qué guardar en cada uno.
7. ¿Por qué `recursion_limit` no es suficiente como condición de parada?
8. ¿Qué debe hacer un nodo de parada con las peticiones de herramienta pendientes y por qué?
9. ¿Cómo se prueba, sin red, que una escritura rechazada no llegó al sistema externo?
10. ¿Qué protecciones del lado de Trello se añaden además de la aprobación humana?
11. ¿Cuándo no conviene usar LangGraph?
12. ¿Qué es el `session_id` en una API de agente y por qué la API puede reiniciarse sin perder la conversación?
13. ¿Cómo se resuelve por HTTP una pausa de `interrupt()` y qué debe responder la API si llega otro mensaje a esa sesión?
14. ¿Cuándo conviene un FastAPI propio y cuándo LangGraph Server? ¿Para qué sirve `langgraph dev`?
