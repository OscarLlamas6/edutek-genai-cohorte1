# Marco teórico · Sesión 12: Automatización con n8n y cierre del Módulo 1

> Documento de apoyo al notebook `s12_n8n_automatizacion.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: los experimentos están en el notebook y los workflows en la práctica `n8n-workflows` del repositorio `ai-agents-playground`. Basado en la documentación de n8n y en la versión **2.47.7** (noviembre de 2026); n8n publica versiones casi cada semana, así que confirma los detalles en [docs.n8n.io](https://docs.n8n.io).

## Índice

1. [Qué es n8n y cuándo usarlo](#1-qué-es-n8n-y-cuándo-usarlo)
2. [Anatomía de un workflow](#2-anatomía-de-un-workflow)
3. [Credenciales, ejecuciones y depuración](#3-credenciales-ejecuciones-y-depuración)
4. [Agentes en n8n](#4-agentes-en-n8n)
5. [Salida estructurada y sub-workflows](#5-salida-estructurada-y-sub-workflows)
6. [MCP en n8n](#6-mcp-en-n8n)
7. [Patrón seguro: el agente propone, una persona aprueba, el flujo ejecuta](#7-patrón-seguro-el-agente-propone-una-persona-aprueba-el-flujo-ejecuta)
8. [Manejo de errores](#8-manejo-de-errores)
9. [Límites reales](#9-límites-reales)
10. [Operación en self-hosting](#10-operación-en-self-hosting)
11. [Seguridad](#11-seguridad)
12. [Cierre del Módulo 1 y avance del Módulo 2](#12-cierre-del-módulo-1-y-avance-del-módulo-2)
13. [Glosario rápido](#13-glosario-rápido)
14. [Preguntas de repaso](#14-preguntas-de-repaso)

---

## 1. Qué es n8n y cuándo usarlo

- **n8n:** herramienta de **automatización de flujos de trabajo** con editor visual. Se dibujan **nodos** conectados; cada uno hace una tarea (recibir un webhook, llamar a una API, decidir, llamar a un modelo). Se puede **auto-hospedar** o usar en su nube. Incluye nodos de IA (agentes, memoria, herramientas, MCP) construidos sobre LangChain.
- **Workflow de bajo código:** el flujo se define con nodos y configuración, con pequeños fragmentos de código donde hace falta. El workflow se guarda como un **JSON**.
- **Analogía de infraestructura:** un pipeline de CI/CD o un playbook de Ansible: pasos con disparadores, condiciones y manejo de errores, con un *runtime* que ya resuelve autenticación, reintentos, programación e historial.

| Criterio | Bajo código (n8n) | Código propio |
| --- | --- | --- |
| Integrar muchos servicios | Nodos listos y autenticación resuelta | Cada cliente se escribe y se mantiene |
| Primer prototipo | Horas | Días |
| Lógica compleja o algoritmos | Se vuelve un lienzo difícil de leer | Lenguaje completo |
| Pruebas y revisión de cambios | Limitadas | pytest, CI, revisión de código |
| Control fino de un agente | El nodo es una caja | LangGraph |
| Observabilidad | Historial de ejecuciones incluido | Se construye |
| Costo de operar | Un servicio más que mantener | Lo que ya se opera |

- **Regla práctica:** n8n como **pegamento y orquestación de procesos** (disparadores, integraciones, aprobaciones, notificaciones); el **núcleo lógico o de razonamiento complejo** como código, que n8n **llama** por HTTP o MCP. No son excluyentes.
- **Workflow vs agente:** el workflow tiene un camino que el autor dibuja; el agente decide el camino con un modelo. Se combinan: el workflow aporta la estructura segura y el agente actúa en un **paso acotado**.

## 2. Anatomía de un workflow

| Pieza | Definición |
| --- | --- |
| **Trigger** | Nodo que inicia el workflow: *Webhook*, *Chat Trigger*, *Schedule*, *Error Trigger*, *MCP Server Trigger*, "cuando me llaman" (sub-workflow) |
| **Nodo** | Un paso: de acción (HTTP Request, Set, Code), de lógica (IF, Switch, Merge, Wait) o de IA (AI Agent, Information Extractor, modelos, memoria, herramientas) |
| **Conexión** | Lleva los datos de un nodo al siguiente; los nodos de IA tienen conexiones especiales (modelo, memoria, herramienta, parser) |
| **Item** | Unidad de datos: cada nodo recibe una **lista de items** (`json` y, opcional, binarios) y los procesa |
| **Expresión** | Valor calculado entre `{{ }}` con JavaScript. `$json` es el item actual de la entrada; `$('Nodo')` accede a la salida de otro nodo; `$execution` da datos de la ejecución |
| **Sticky note** | Nota en el lienzo; documentación viva del workflow |

- Los **nombres de los nodos importan**: se usan en las expresiones, y el nombre de un nodo de herramienta es el que ve el modelo.
- **URL de prueba vs de producción** de un webhook: la de prueba escucha solo mientras se prueba en el editor; la de producción exige que el workflow esté **publicado**.
- Un workflow puede responder al webhook **en cuanto valida la entrada** (nodo *Respond to Webhook*) y seguir trabajando en segundo plano.

## 3. Credenciales, ejecuciones y depuración

- **Credencial:** secreto guardado **cifrado** en n8n y **referenciado** por nombre desde los nodos. El JSON exportado solo contiene la referencia (identificador y nombre), no el secreto.
- **Cifrado:** las credenciales se cifran con la clave de la instancia (`N8N_ENCRYPTION_KEY`). Sin esa clave no pueden leerse; debe respaldarse aparte.
- **Ejecución:** corrida de un workflow con los datos de entrada y salida de cada nodo, guardada para depurar. Contiene datos sensibles: se poda (`EXECUTIONS_DATA_PRUNE`) y se restringe su acceso.
- **Herramientas de depuración:** ejecutar un paso, **fijar datos** (*pin data*), lista de ejecuciones, cargar los datos de una ejecución fallida en el editor, notas y nombres claros.

## 4. Agentes en n8n

El nodo **AI Agent** es un agente con bucle de herramientas (modelo → herramienta → modelo) configurado con conexiones:

| Toma | Qué se conecta | Equivalente en LangChain |
| --- | --- | --- |
| Chat Model | OpenAI, Anthropic, Ollama, Azure…; servidores compatibles con OpenAI (LM Studio) cambiando la URL de la credencial | `model` |
| Memory | *Simple Memory* (ventana de mensajes en RAM, por `sessionId`) o memoria persistente | *Checkpointer* y `thread_id` |
| Tool | Code Tool, HTTP Request Tool, otro workflow, MCP Client Tool, integraciones | Herramientas `@tool` |
| Output Parser | Structured Output Parser | `response_format` |

- El modelo **pide** herramientas; n8n las **ejecuta**. El modelo conoce solo el **nombre, la descripción y el esquema** de cada herramienta.
- Un agente hace **varias llamadas por mensaje**: se acota con el máximo de iteraciones y modelos económicos.
- La **memoria simple** vive en el proceso: se pierde al reiniciar y no sirve con varias instancias.
- **Modelos locales:** se usan cambiando la URL de la credencial a un servidor compatible con OpenAI; se necesita un modelo con buen soporte de herramientas (≈ 12B o más).
- **Dónde poner la decisión:** camino fijo → nodos; interpretar texto libre o elegir herramientas → agente en ese paso; modificar algo → el flujo, tras una aprobación; control fino de estado o varios agentes → código propio (LangGraph) llamado desde n8n.

## 5. Salida estructurada y sub-workflows

- **Information Extractor:** extrae campos de un texto con un esquema, en una llamada y sin herramientas (clasificar un ticket, extraer entidades).
- **Structured Output Parser:** conectado al AI Agent, da forma fija a la respuesta final.
- **El esquema garantiza la forma, no la verdad:** el flujo vuelve a validar en código (valores dentro de una lista, rangos, lista blanca de acciones).
- **Sub-workflow:** workflow llamado por otro con **Execute Workflow**; entra por el disparador "cuando me llaman" y devuelve la salida del último nodo ejecutado. Un error dentro **sube** al padre. Para publicar el padre, n8n exige que los sub-workflows referenciados estén publicados.
- **Workflow como herramienta** (*Call n8n Workflow Tool*): un sub-workflow cuyas entradas rellena el modelo. Se prueba y versiona por separado y se reutiliza.

## 6. MCP en n8n

| Nodo | Rol | Descripción |
| --- | --- | --- |
| **MCP Client Tool** | Cliente | Conecta con un servidor MCP (HTTP *streamable* o SSE) y ofrece sus herramientas al AI Agent; se pueden incluir todas, algunas o todas menos algunas |
| **MCP Server Trigger** | Servidor | Expone como herramientas MCP los nodos de herramienta conectados a él, en una URL `/mcp/<ruta>`; admite autenticación por cabecera o Bearer |

- Conectar el cliente MCP con el servidor MCP de la Sesión 10 por HTTP es **opcional**.
- **Seguridad:** un servidor MCP es una superficie de ataque. Autenticación obligatoria, mínimo privilegio, herramientas con descripciones limpias y aprobación humana para escrituras. El servidor de n8n actúa con los permisos de las credenciales de las herramientas conectadas.

## 7. Patrón seguro: el agente propone, una persona aprueba, el flujo ejecuta

El triaje de tickets del curso:

1. **Webhook autenticado** → validación de formato y tamaños → respuesta `202` inmediata.
2. **Clasificación** con salida estructurada (o reglas, en modo simulado) → revalidación en código.
3. **Sub-workflow** de enriquecimiento (CMDB) con reintentos → **regla determinista** de prioridad.
4. **Switch** por severidad: crítica y alta → agente; media y baja → respuesta estándar.
5. **Agente** con herramientas de **solo lectura** que **propone** una acción; validación contra una **lista blanca**.
6. **Aprobación humana** (nodo *Wait* con formulario) con **caducidad**; sin respuesta, no se ejecuta nada.
7. Acción en **`dry_run`**, **notificación** y **registro de auditoría**; **error workflow** si algo falla.

| Defensa | Dónde |
| --- | --- |
| Autenticación del webhook | Header Auth en el nodo Webhook |
| Texto del ticket como dato no confiable | Prompt del sistema y agente sin herramientas de escritura |
| Lista blanca de acciones | Esquema de salida y validación en código |
| Aprobación humana con caducidad | Nodo Wait |
| Escritura simulada | `dry_run` |
| Límites | Tamaño del texto, máximo de iteraciones del agente, reintentos acotados |
| Auditoría | Registro final de cada ticket |

- **Modo simulado:** el flujo corre completo sin modelo ni cuentas externas (las reglas reemplazan al modelo y los servicios son webhooks dentro de n8n); sirve para clase, pruebas y casa. El modo `llm` usa los nodos de modelo.

## 8. Manejo de errores

| Nivel | Qué hace |
| --- | --- |
| **Retry On Fail** (reintento del nodo) | Reejecuta el nodo ante una falla; sirve para fallas transitorias (red, 503) |
| **On Error** | *Detener el workflow* (por defecto), *continuar* con el último dato válido, o *continuar con la salida de error* para manejarlo en una rama |
| **Error workflow** | Workflow que empieza con *Error Trigger* y corre cuando otro workflow falla; recibe el workflow, el nodo, el mensaje y el enlace a la ejecución. Se asigna en los ajustes del workflow vigilado |
| **Caducidad de esperas** | El nodo *Wait* puede limitar el tiempo de espera y seguir por la rama de "expirada" |

- **Los reintentos repiten efectos:** en lecturas son seguros; en escrituras exigen **idempotencia** o no deben usarse.
- Un error en un **sub-workflow** sube al padre; el error workflow se activa por el workflow vigilado.

## 9. Límites reales

- **Versionado:** el workflow es un JSON que se puede exportar y guardar en Git. Los diferencias son ruidosas (posiciones, identificadores). La integración de **Git y entornos** de n8n está disponible, según la documentación, en los planes **Business y Enterprise**; en la edición comunitaria el flujo es manual (exportar, versionar, importar por CLI).
- **Pruebas:** principalmente manuales (ejecutar paso, *pin data*). Se complementan con validación estática del JSON, pruebas contra los webhooks y un modo simulado. n8n ofrece funciones de evaluación para flujos con IA; su disponibilidad depende del plan y de la versión.
- **Licencia *fair-code*:** n8n usa la **Sustainable Use License** (y la *Enterprise License* para el código marcado con `.ee.`). Permite usar y modificar el software para **fines internos de negocio o personales/no comerciales**, y distribuirlo solo de forma **gratuita y no comercial**; no se pueden quitar los avisos de licencia. Ofrecerlo a terceros como producto o servicio requiere un acuerdo comercial con n8n. **No es open source** en el sentido de la OSI. (Guía general, no asesoría legal.)
- **Lógica compleja y agentes de control fino:** caen mejor en código.
- **Rendimiento y escala:** una instancia simple es un punto único de falla; el escalado requiere modo cola y más componentes.

## 10. Operación en self-hosting

| Tema | Consideración |
| --- | --- |
| Base de datos | SQLite por defecto; PostgreSQL para producción |
| Escalado | Modo cola: instancia principal y *workers* |
| Versiones | **Fijar la versión** de la imagen; leer las notas de versión; probar antes de actualizar |
| Copias de seguridad | Base de datos + `N8N_ENCRYPTION_KEY` aparte; probar la restauración |
| Datos de ejecución | Poda configurada; acceso por rol |
| Monitoreo | Endpoints de salud, logs, ejecuciones fallidas y error workflow |
| Esperas largas | Cada espera pendiente es una ejecución guardada en la base |
| Red | Detrás de un proxy inverso con TLS; editor no expuesto |

## 11. Seguridad

| Riesgo | Defensa |
| --- | --- |
| Credenciales | Almacén cifrado, mínimo privilegio, rotación, nunca en nodos ni en el JSON |
| Exposición de webhooks | URLs de producción públicas si la instancia lo es: publicar solo lo necesario, con HTTPS |
| Autenticación de webhooks | Header Auth, Basic o JWT; opción de lista de IP; sin autenticación cualquiera puede disparar el flujo |
| Cifrado | En reposo con la clave de la instancia; en tránsito con TLS; procedimiento documentado de rotación de la clave |
| Nodos que ejecutan código | Limitar quién edita workflows, bloquear nodos de riesgo y usar runners aislados |
| Variables de entorno y red interna | Opciones para bloquear el acceso a variables desde nodos y para activar protección SSRF |
| Datos en ejecuciones | Poda, control de acceso y ocultamiento de datos de ejecución |
| Inyección de instrucciones | Texto externo = dato no confiable; lista blanca; aprobación humana; agente sin escritura |
| Revisión | `n8n audit` genera un informe de riesgos de la instancia (credenciales sin uso, nodos de riesgo, etc.) |

## 12. Cierre del Módulo 1 y avance del Módulo 2

**Módulo 1 (S1 a S12):** LLM → RAG → agente con herramientas (MCP) → automatización.

| Bloque | Sesiones | Idea |
| --- | --- | --- |
| LLM y ecosistema | S1 a S3 | Tokens, contexto y costo; API; prompting y *context engineering* |
| Embeddings y RAG | S4 a S7 | Bases vectoriales; RAG simple, intermedio y avanzado (GraphRAG, evaluación) |
| Agentes | S8 y S9 | LangChain: herramientas, bucle, estructura, middleware, memoria, RAG agéntico |
| MCP | S10 | Arquitectura, seguridad y servidor propio con escrituras protegidas |
| LangGraph | S11 | Agente con RAG, memoria, herramientas y aprobación humana |
| n8n | S12 | Orquestación de procesos con workflows, agentes y aprobaciones |

**Ideas transversales:** (1) el modelo propone, el código dispone; (2) la forma no es la verdad: se verifica y se mide; (3) empezar por lo más simple y agregar autonomía donde una medición lo justifique; (4) costo, latencia y seguridad son requisitos.

**Módulo 2 · AI Engineer II: Arquitectura y Seguridad de Sistemas Agénticos** (34 h, 17 sesiones, 6 semanas): del agente a un sistema **multi-agente seguro**. Según el programa del curso: patrones multi-agente y *harness engineering*; frameworks de orquestación; comunicación entre agentes con **A2A**; orquestación avanzada, memoria y *checkpointing*; **guardrails**; seguridad de agentes (*prompt injection*, riesgos de MCP y A2A, identidad y autorización); *human-in-the-loop*, *red teaming* y evaluación; resiliencia, gobernanza y proyecto integrador.

## 13. Glosario rápido

| Término | Definición |
| --- | --- |
| n8n | Herramienta de automatización de flujos con editor visual, auto-hospedable |
| Workflow | Conjunto de nodos conectados, guardado como JSON |
| Trigger | Nodo que inicia un workflow |
| Item | Unidad de datos que viaja entre nodos |
| Expresión | Valor calculado con `{{ }}` (JavaScript) |
| Credencial | Secreto cifrado referenciado por nombre desde los nodos |
| Ejecución | Corrida de un workflow con los datos de cada nodo |
| Pin data | Fijar la salida de un nodo para iterar |
| AI Agent | Nodo de agente con modelo, memoria, herramientas y parser |
| Information Extractor | Nodo que extrae campos de un texto con un esquema |
| Sub-workflow | Workflow llamado por otro o usado como herramienta |
| MCP Client Tool / MCP Server Trigger | Cliente y servidor del Model Context Protocol en n8n |
| Wait | Pausa hasta un tiempo, un webhook o un formulario |
| Error workflow | Workflow con *Error Trigger* para fallas de otro workflow |
| Retry On Fail | Reintento del nodo |
| `dry_run` | Ejecución simulada: se registra, no se aplica |
| Lista blanca | Acciones permitidas, verificadas en código |
| Fair-code | Licencia con código visible y usos limitados |
| Modo cola | Escalado con instancia principal y *workers* |
| `N8N_ENCRYPTION_KEY` | Clave que cifra las credenciales guardadas |
| Inyección indirecta | Instrucciones maliciosas escondidas en datos que lee el modelo |

## 14. Preguntas de repaso

1. ¿Cuándo conviene n8n y cuándo código propio? Da un ejemplo de combinarlos.
2. ¿Qué es un item y cómo se leen datos del nodo actual y de otro nodo en una expresión?
3. ¿Por qué un workflow exportado no contiene secretos y qué ocurre si se pierde la clave de cifrado?
4. ¿Qué tomas tiene el nodo AI Agent y a qué concepto de LangChain equivale cada una?
5. ¿Por qué importa el nombre del nodo de una herramienta?
6. ¿Qué diferencia hay entre Information Extractor y Structured Output Parser?
7. ¿Qué papel cumple cada nodo MCP y qué riesgo se agrega al exponer un servidor?
8. En el triaje, ¿por qué el agente no tiene herramientas de escritura y qué defensas hay entre la propuesta y la acción?
9. Diferencia entre reintento del nodo, error workflow y caducidad de una espera. ¿Cuándo es peligroso reintentar?
10. ¿Qué límites de versionado y de pruebas tiene la edición comunitaria?
11. ¿Qué permite y qué no la Sustainable Use License?
12. Menciona cinco tareas de operación de un n8n auto-hospedado.
13. Un webhook de n8n está en Internet: ¿qué cuatro medidas de seguridad aplicarías?
14. ¿Qué ideas del Módulo 1 se aplican al diseño del triaje, y qué temas del Módulo 2 las profundizan?
