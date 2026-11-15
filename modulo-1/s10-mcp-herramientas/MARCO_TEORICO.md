# Marco teórico · Sesión 10: MCP, herramientas conectadas

> Documento de apoyo al notebook `s10_mcp_herramientas.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: el código y los experimentos están en el notebook y en la práctica `langchain-mcp-advanced-tooling` del repositorio `ai-agents-playground`. Basado en la especificación de MCP (revisiones `2025-11-25`, que implementa el SDK usado, y `2026-07-28`, la más reciente), en la documentación de Claude Code y de Claude Desktop, de HubSpot y de LangChain 1.x (noviembre de 2026); confirma los detalles en la documentación vigente, porque MCP evoluciona deprisa. La seguridad de MCP se verá en el Módulo 2.

## Índice

1. [El problema N×M](#1-el-problema-nm)
2. [Arquitectura: host, cliente y servidor](#2-arquitectura-host-cliente-y-servidor)
3. [Las tres primitivas](#3-las-tres-primitivas)
4. [Transportes y ciclo de vida](#4-transportes-y-ciclo-de-vida)
5. [MCP frente al *tool calling* tradicional y el ecosistema](#5-mcp-frente-al-tool-calling-tradicional-y-el-ecosistema)
6. [Construir un servidor con `FastMCP`](#6-construir-un-servidor-con-fastmcp)
7. [El agente: `langchain-mcp-adapters`](#7-el-agente-langchain-mcp-adapters)
8. [Registrar un servidor en un cliente real](#8-registrar-un-servidor-en-un-cliente-real)
9. [Depuración](#9-depuración)
10. [Servir el agente: API, sesiones e interfaz](#10-servir-el-agente-api-sesiones-e-interfaz)
11. [El sistema externo: HubSpot CRM](#11-el-sistema-externo-hubspot-crm)
12. [Glosario rápido](#12-glosario-rápido)
13. [Preguntas de repaso](#13-preguntas-de-repaso)

---

## 1. El problema N×M

- Sin un estándar, conectar **N aplicaciones de IA** con **M sistemas** (CRM, repositorios, bases de datos, correo) exige **N × M** integraciones a medida, cada una con su formato, su autenticación y sus errores.
- **MCP (Model Context Protocol)** es un **protocolo abierto** que estandariza cómo una aplicación de IA **descubre y usa** herramientas y datos externos. Cada sistema se expone **una vez** como servidor y cada aplicación implementa **una vez** el cliente: **N + M** piezas.
- Analogías: **USB-C** (un conector común), el **Language Server Protocol** (editores y lenguajes) y, para infraestructura, un **driver/CSI** (la aplicación habla con una interfaz común; el dispositivo implementa los detalles).

## 2. Arquitectura: host, cliente y servidor

| Rol | Qué es | Ejemplo |
| --- | --- | --- |
| **Host** | La aplicación de IA que usa la persona; decide a qué servidores conectarse | Un IDE, Claude Desktop, Claude Code, un agente de LangChain |
| **Cliente** | Componente **dentro** del host que mantiene **una conexión con un servidor** | `ClientSession` del SDK, `MultiServerMCPClient` |
| **Servidor** | Programa que expone herramientas, datos y plantillas | El servidor de HubSpot de la práctica |

- Los mensajes son **JSON-RPC 2.0**: peticiones (con `id` y respuesta), respuestas y **notificaciones** (sin `id` y sin respuesta).
- Un host puede tener **varios clientes**, uno por servidor: así un mismo agente usa, a la vez, el servidor propio y otro ya hecho.

## 3. Las tres primitivas

| Primitiva | La controla | Para qué | Métodos | Ejemplo |
| --- | --- | --- | --- | --- |
| **Tools** | El **modelo** decide llamarlas | Acciones y consultas (pueden tener efectos) | `tools/list`, `tools/call` | `search_contacts`, `add_note` |
| **Resources** | La **aplicación** (o la persona, al elegirlos) decide qué cargar | Datos de solo lectura identificados por una **URI** | `resources/list`, `resources/templates/list`, `resources/read` | `docs://guia`, `clientes://{id}` |
| **Prompts** | La **persona** los invoca | Plantillas con parámetros que producen mensajes | `prompts/list`, `prompts/get` | "Revisa este código en {lenguaje}" |

- Un **recurso** puede ser **fijo** o una **plantilla de URI** (`clientes://{id}`) cuyo parámetro se rellena al pedirlo. Una *tool* hace algo; un recurso *es* un dato; un prompt *genera mensajes*.
- En **Claude Code**: las tools las llama el modelo y se nombran `mcp__<servidor>__<herramienta>`; los recursos se adjuntan con `@servidor:uri`; los prompts aparecen como comandos `/servidor:prompt`.
- Capacidades que el servidor puede pedir al **cliente**: **sampling** (que el cliente use *su* modelo), la **consulta a la persona** (que el cliente le pregunte un dato) y **roots** (conocer las carpetas con las que trabaja el cliente).
- **Anotaciones de herramientas** (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`): **metadatos** que describen la herramienta al cliente ("solo lee", "repetirla da lo mismo", "toca sistemas externos"). El cliente puede usarlos, por ejemplo, para decidir cómo mostrar la herramienta.
- **Nota de versión.** La revisión `2026-07-28` **depreca** *sampling*, *roots* y *logging* (siguen funcionando durante una ventana de al menos doce meses), reemplaza las peticiones del servidor al cliente por un patrón de **múltiples rondas** y **elimina** el apretón de manos `initialize` y las sesiones del transporte HTTP (cada petición lleva versión y capacidades). El SDK `mcp 1.30.0` implementa todavía `2025-11-25`. Los conceptos de esta sesión no cambian.

## 4. Transportes y ciclo de vida

| Transporte | Funcionamiento | Uso |
| --- | --- | --- |
| **stdio** | El cliente lanza el servidor como **subproceso** y habla por `stdin`/`stdout` | Servidores locales de un solo usuario |
| **Streamable HTTP** | Un endpoint HTTP (`/mcp`): POST para enviar y, opcionalmente, SSE para transmitir | Servidores remotos y compartidos |
| HTTP+SSE | Transporte antiguo | **Deprecado**: migrar a Streamable HTTP |

- **stdio:** `stdout` pertenece al **protocolo**. Cualquier escritura de depuración (`print`) lo corrompe. **Los logs van a `stderr`**, que el cliente captura. Con stdio, el subproceso recibe solo un **entorno básico**: las variables que necesite se pasan en `env`.
- **Ciclo de vida (revisión `2025-11-25`):** `initialize` (versión y capacidades de ambos) → notificación `initialized` → descubrimiento (`tools/list`, `resources/list`, `prompts/list`) → uso (`tools/call`, `resources/read`, `prompts/get`) → cierre.
- Un error **de la herramienta** viaja dentro de un resultado normal con `isError: true` (el modelo puede leerlo y corregirse); un error **del protocolo** viaja como error JSON-RPC (con `code`).
- Lo que el modelo ve de una herramienta es **nombre + descripción + esquema JSON de entrada**, igual que con `@tool`.

## 5. MCP frente al *tool calling* tradicional y el ecosistema

| | *Tool calling* con `@tool` | MCP |
| --- | --- | --- |
| Ubicación | Dentro de tu aplicación | En un servidor aparte (proceso, equipo o empresa distintos) |
| Descubrimiento | Registro manual en el código | Dinámico (`tools/list`) |
| Reutilización | Copiar código | Cualquier cliente MCP |

- MCP no reemplaza el *tool calling*: **lo estandariza hacia afuera**. Por debajo, el modelo sigue pidiendo herramientas con un esquema; el adaptador las presenta como `@tool`.
- **Ecosistema:** el repositorio `modelcontextprotocol/servers` (servidores de **referencia**: hora, archivos, `fetch`, git, memoria), el **registro oficial** de servidores y numerosos catálogos de la comunidad (calidad variable).
- **Servidor de referencia usado en la sesión:** `mcp-server-time` (hora y zonas horarias, solo tools de lectura), ejecutado con `uvx` y **versión fijada**.

## 6. Construir un servidor con `FastMCP`

- `FastMCP` (SDK oficial de Python) convierte **funciones decoradas** en un servidor: `@mcp.tool()`, `@mcp.resource("uri")` y `@mcp.prompt()`. El **nombre**, el **docstring** y los **tipos** forman lo que ve el modelo.
- **Los tipos con restricciones son el esquema.** Un `Annotated[str, Field(pattern=..., min_length=...)]` aparece en el JSON Schema de la herramienta y el SDK lo **valida antes** de llamar a la función; un argumento inválido vuelve al modelo como un `isError` con un mensaje legible, y puede corregirse.
- **Herramientas de lectura y de escritura:** conviene declarar la diferencia en las **anotaciones** y en la descripción ("ESCRIBE en el CRM; cada llamada crea una nota nueva").
- **Transportes del servidor:** `mcp.run()` (stdio, por defecto) o `mcp.run(transport="streamable-http")`; en un contenedor se escucha en `0.0.0.0` y el servidor debe aceptar la cabecera `Host` correspondiente.
- **Logs:** con `logging` configurado a `stderr` (nunca `print` en stdio).

## 7. El agente: `langchain-mcp-adapters`

- La librería `langchain-mcp-adapters` convierte las **tools** MCP en `@tool` de LangChain. `MultiServerMCPClient` conecta uno o varios servidores (stdio o Streamable HTTP) y `get_tools()` devuelve las herramientas; `load_mcp_tools(session)` lo hace sobre una sesión ya abierta. Los **recursos** y **prompts** se cargan con `load_mcp_resources` y `load_mcp_prompt` (blobs y mensajes de LangChain).
- Las **anotaciones** de MCP llegan en los metadatos de la herramienta (`tool.metadata`); los errores (`isError`) llegan al agente como mensajes de herramienta.
- **Multi-servidor:** un diccionario de servidores da **todas** las herramientas a un mismo agente. Con `tool_name_prefix=True` cada nombre lleva el del servidor (`crm_search_contacts`, `hora_get_current_time`) para evitar choques. Cada herramienta ocupa contexto en cada llamada: conecta solo lo necesario.
- **Aprobación humana (repaso de la S9):** `HumanInTheLoopMiddleware` pausa el agente antes de ejecutar las herramientas indicadas (aquí, `add_note`); una persona **aprueba o rechaza** y el agente continúa. Requiere *checkpointer* y `thread_id`. Funciona igual con herramientas MCP porque para LangChain son `@tool`.

## 8. Registrar un servidor en un cliente real

- Un cliente MCP necesita saber **cómo lanzar** (stdio) o **dónde encontrar** (HTTP) el servidor. Se declara en un archivo JSON con la clave `mcpServers`.
- **Claude Code (CLI):**
  - Archivo `.mcp.json` en la raíz del proyecto (**alcance de proyecto**, se comparte con el equipo; Claude Code pide aprobarlo la primera vez), o configuración **local**/**de usuario** en `~/.claude.json`.
  - Entrada **stdio**: `command`, `args`, `env` (sin `type`, se lee como stdio). Entrada **HTTP**: `type: "http"`, `url` y, opcionalmente, `headers`.
  - Admite variables de entorno con `${VAR}` y `${VAR:-valor}` en `command`, `args`, `env`, `url` y `headers`.
  - Por línea de comandos: `claude mcp add --transport stdio --scope project <nombre> -- <comando> [args...]` y `claude mcp add --transport http --scope project <nombre> <url>`; `claude mcp list` muestra el estado.
- **Claude Desktop:** archivo `claude_desktop_config.json` (macOS `~/Library/Application Support/Claude/`, Windows `%APPDATA%\Claude\`, Linux `~/.config/Claude/`), abierto desde *Settings → Developer → Edit Config*, con entradas `command`/`args`/`env`. Hay que **reiniciar** la aplicación por completo tras cambiarlo.
- **Rutas absolutas:** el directorio de trabajo con el que un cliente lanza el servidor puede no ser el tuyo; usa rutas absolutas para `command` y `args`.
- **Variables de entorno:** un servidor stdio hereda solo un subconjunto básico del entorno; lo que necesite (por ejemplo `HUBSPOT_TOKEN`) se declara en `env`.

## 9. Depuración

- Orden práctico: **(1) logs del servidor** a `stderr`; **(2) cliente mínimo** (¿arranca?, ¿lista sus herramientas?, ¿responde?); **(3) MCP Inspector** (interfaz web oficial para listar y llamar tools, resources y prompts y ver las notificaciones; la versión `2.15.0` exige Node 22.23 o superior); **(4) logs del cliente** (en Claude Desktop, `mcp.log` y `mcp-server-<nombre>.log` en `~/Library/Logs/Claude`, `%APPDATA%\Claude\logs` o `~/.config/Claude/logs`).
- **Por qué `stderr`:** en stdio, `stdout` lleva el protocolo; una línea que no sea un mensaje JSON-RPC desincroniza o rompe al cliente. Es una **regla del protocolo**. En HTTP, `stderr` no llega al cliente: se usan los logs del contenedor o del servidor y `curl`.
- **Problemas de arranque habituales:** ruta del `command` incorrecta o relativa, JSON de configuración inválido, variables de entorno que faltan, el directorio de trabajo distinto al esperado.
- **Dos tipos de error:** de la herramienta (`isError: true`, lo ve el modelo) y del protocolo (error JSON-RPC, lo ve el desarrollador).
- **Ciclo de pruebas:** los cambios en la configuración o en el código del servidor requieren reiniciar el cliente; el Inspector acelera la iteración.

## 10. Servir el agente: API, sesiones e interfaz

- **Por qué una API:** para que otra aplicación (una interfaz web, otro servicio, una persona con un cliente HTTP) use el agente sin ejecutar su código. El agente queda detrás de unos pocos endpoints; **FastAPI** valida las entradas y las salidas con modelos Pydantic y genera la documentación interactiva (`/docs`).
- **Idea central:** el agente es un **servicio sin estado en el proceso**. El historial de cada conversación vive en el **checkpointer** y el cliente solo lleva el **identificador de sesión**. Analogía de infraestructura: un servicio web con las sesiones en una base de datos; el cliente conserva la "cookie" y cualquier réplica puede continuar la conversación.
- **`session_id` = `thread_id`:** el identificador que el cliente envía es el `thread_id` con el que el agente guarda y recupera su estado. Sin id, la API crea uno (UUID) y lo **devuelve**; con un id existente, continúa esa conversación; con otro, empieza una distinta que no ve la anterior.
- **Persistencia:** un checkpointer **en memoria** pierde las sesiones al reiniciar. Con un checkpointer en **SQLite** el historial (y una aprobación pendiente) sobrevive a un reinicio de la API. Con varias réplicas hace falta un almacén compartido (por ejemplo, Postgres).
- **Endpoints de la práctica:** `GET /health`; `POST /chat` (`message`, `session_id`, `user_id` → `session_id`, `status`, `answer`, herramientas usadas, acción pendiente y uso de tokens); `GET /sessions/{id}` (historial), `GET /sessions` (de un usuario) y `DELETE /sessions/{id}`; `POST /sessions/{id}/approve`.
- **Aprobación humana por API:** si el agente quiere ejecutar `add_note`, se pausa y `POST /chat` responde `status: "awaiting_approval"` con la acción propuesta (herramienta y argumentos). Una segunda llamada, `POST /sessions/{id}/approve` con `{"approved": true|false}`, reanuda al agente. Mientras haya una aprobación pendiente, un nuevo mensaje en esa sesión responde `409`.
- **Códigos HTTP:** `200` correcto; `204` borrado; `404` sesión inexistente; `409` conflicto (aprobación pendiente o nada que aprobar); `422` entrada inválida; `503` servicio no listo (por ejemplo, no se puede abrir la base de datos).
- **Interfaz (Streamlit):** un chat sencillo que consume la API por HTTP —no importa el agente—, con el usuario y un botón "Nueva sesión" en la barra lateral, el `session_id` visible, las herramientas y los tokens bajo cada respuesta y botones Aprobar y Rechazar.
- **Bruno:** cliente HTTP cuyas colecciones son **archivos de texto** (`.bru`) que se guardan en el repositorio. Cada petición declara su URL, su cuerpo, **aserciones** (`res.status: eq 200`), **tests** en JavaScript y **scripts** (por ejemplo, guardar el `session_id` de una respuesta en una variable para las siguientes). Los **ambientes** (`environments/local.bru`) definen variables como `baseUrl`. El **CLI** (`bru run --env local`) ejecuta la colección sin interfaz, útil para automatizar. La colección de la práctica cuenta una historia en 13 pasos: salud, sesión nueva, seguimiento con memoria, historial, otra sesión aislada, aprobación, rechazo, borrado y validación.

## 11. El sistema externo: HubSpot CRM

- **HubSpot** es un CRM con plan **gratuito**. Los contactos y las **notas** (asociadas a un contacto con el tipo de asociación definido por HubSpot, `202`) se manejan por la API REST en `https://api.hubapi.com`.
- **Autenticación:** cabecera `Authorization: Bearer <token>`. Las **claves de API** heredadas ya no existen. HubSpot recomienda las **Service Keys** (credencial de cuenta para integraciones de datos sin webhooks; **beta pública desde febrero de 2026**) como reemplazo de las **apps privadas heredadas**, cuya creación está desactivada desde octubre de 2026. Los permisos se eligen al crear la clave.
- **Versiones de la API:** HubSpot usa **versiones por fecha** (por ejemplo `2026-03`, con rutas `/crm/objects/2026-03/contacts`); las rutas heredadas `/crm/v3/objects/...` siguen disponibles, pero HubSpot ha anunciado el fin de soporte de las APIs v1 a v3 con aplicación en septiembre de 2027.
- **Operaciones del servidor:** búsqueda `POST .../contacts/search` (`query`, `limit`, `properties`); lectura de un contacto por id; notas de un contacto (búsqueda de notas filtrando por `associations.contact`); creación `POST .../notes` con `hs_timestamp`, `hs_note_body` y la asociación al contacto.
- **Permisos:** `crm.objects.contacts.read` para las lecturas y `crm.objects.contacts.write` para `add_note`. Confirma en tu cuenta qué permisos exige cada endpoint de notas antes de usar el modo real.
- **Por qué este conjunto mínimo:** dos lecturas y **una** escritura aditiva (crear una nota no borra nada y se puede revisar): suficiente para ver lecturas y escrituras con la menor superficie.
- Usa una **cuenta de pruebas** y datos ficticios.

## 12. Glosario rápido

| Término | Definición |
| --- | --- |
| MCP | Protocolo abierto para que aplicaciones de IA usen herramientas y datos externos |
| Host / cliente / servidor | Aplicación de IA / conector con un servidor / programa que expone capacidades |
| JSON-RPC 2.0 | Formato de los mensajes de MCP |
| Tool / Resource / Prompt | Acción que decide el modelo / dato con URI que carga la aplicación / plantilla que elige la persona |
| Plantilla de recurso | Recurso con parámetros en la URI (`clientes://{id}`) |
| Sampling / consulta a la persona / Roots | El servidor pide al cliente: usar su modelo / preguntar un dato a la persona / conocer las carpetas con las que trabaja |
| stdio | Transporte por entrada/salida estándar de un subproceso |
| Streamable HTTP | Transporte HTTP de un endpoint para servidores remotos |
| `FastMCP` | API de alto nivel del SDK de Python para escribir servidores |
| Anotaciones de herramienta | Metadatos que describen una herramienta (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) |
| `isError` | Marca de un resultado de herramienta que falló |
| `MultiServerMCPClient` | Cliente de `langchain-mcp-adapters` que conecta varios servidores |
| `.mcp.json` | Configuración de servidores MCP de un proyecto en Claude Code |
| MCP Inspector | Interfaz web oficial para probar y depurar servidores |
| *Human-in-the-loop* | Aprobación humana antes de actuar |
| Service Key | Credencial de cuenta de HubSpot para integraciones de datos |
| FastAPI | Framework web de Python con validación Pydantic y documentación automática |
| `session_id` / `thread_id` | Identificador de una conversación: lo envía el cliente y el checkpointer lo usa para guardar y recuperar el historial |
| Checkpointer persistente | Almacén en disco (SQLite, Postgres) del estado del agente; las sesiones sobreviven a un reinicio |
| `awaiting_approval` | Estado de una sesión pausada a la espera de aprobar o rechazar una escritura |
| Streamlit | Biblioteca para interfaces web sencillas en Python |
| Bruno | Cliente HTTP con colecciones en archivos de texto, aserciones, variables y CLI |

## 13. Preguntas de repaso

1. ¿Qué problema resuelve MCP y por qué pasa de N×M a N+M?
2. Describe los tres roles (host, cliente, servidor) y da un ejemplo de cada uno.
3. ¿Quién controla cada primitiva (tools, resources, prompts)? Da un ejemplo de cada una.
4. ¿Qué diferencia hay entre un recurso fijo y una plantilla de recurso?
5. ¿Cuándo usarías stdio y cuándo Streamable HTTP? ¿Qué regla de logs aplica a stdio y por qué?
6. Enumera los mensajes del ciclo de vida de una conexión en `2025-11-25` y qué cambió en `2026-07-28`.
7. ¿Qué diferencia hay entre un error de la herramienta y un error del protocolo?
8. ¿Cómo se convierte un tipo de Python con restricciones en lo que ve el modelo? ¿Qué pasa con un argumento inválido?
9. ¿Qué describen las anotaciones de una herramienta y para qué las puede usar un cliente?
10. ¿Qué hace `langchain-mcp-adapters` y qué ventaja tiene que el agente no distinga una herramienta MCP de una `@tool`?
11. ¿Para qué sirve `tool_name_prefix` y qué costo tiene conectar muchos servidores?
12. ¿Qué datos lleva la entrada de un servidor stdio y la de un servidor HTTP en `.mcp.json`?
13. ¿Por qué se usan rutas absolutas al registrar un servidor y por qué hay que reiniciar el cliente?
14. Un servidor no aparece en el cliente: menciona tres comprobaciones, en orden.
15. ¿Qué herramienta usarías primero para depurar un servidor y por qué?
16. ¿Qué hace `HumanInTheLoopMiddleware` y por qué necesita un checkpointer?
17. ¿Qué permisos mínimos necesita el servidor de la práctica y cuál es la herramienta que escribe?
18. ¿Qué significa que el agente es un servicio sin estado en el proceso? ¿Dónde vive el estado y qué lleva el cliente?
19. ¿Qué relación hay entre `session_id` y `thread_id`? ¿Qué pasa si el cliente envía un id nuevo? ¿Y si no envía ninguno?
20. ¿Por qué un checkpointer en memoria no sirve para una API y qué cambia con SQLite?
21. Describe el flujo por HTTP de una escritura que requiere aprobación (estados, endpoints y códigos).
22. ¿Qué ventajas tiene una colección Bruno frente a `curl`? ¿Qué hace un script `post-response`?
