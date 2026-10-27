# Marco teórico · Sesión 7: RAG avanzado: técnicas, GraphRAG y evaluación

> Documento de apoyo al notebook `s07_rag_avanzado.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: el código y los experimentos están en el notebook.

## Índice

1. [De un RAG básico a uno avanzado](#1-de-un-rag-básico-a-uno-avanzado)
2. [Tipos de pregunta y dónde falla un RAG básico](#2-tipos-de-pregunta-y-dónde-falla-un-rag-básico)
3. [Técnicas avanzadas de recuperación](#3-técnicas-avanzadas-de-recuperación)
4. [GraphRAG](#4-graphrag)
5. [RAG agéntico (adelanto)](#5-rag-agéntico-adelanto)
6. [Evaluación de un RAG](#6-evaluación-de-un-rag)
7. [Alucinaciones: causas y mitigación](#7-alucinaciones-causas-y-mitigación)
8. [El juez LLM (*LLM-as-judge*)](#8-el-juez-llm-llm-as-judge)
9. [El ciclo medir, cambiar, volver a medir](#9-el-ciclo-medir-cambiar-volver-a-medir)
10. [Glosario rápido](#10-glosario-rápido)

---

## 1. De un RAG básico a uno avanzado

- **RAG básico (Sesiones 5 y 6):** fragmentar → vectorizar → buscar los *k* fragmentos más parecidos → armar el prompt → generar con citas. La Sesión 6 añadió mejor fragmentación, reescritura de la consulta, búsqueda híbrida y reranking; **no se repiten aquí**.
- **RAG avanzado:** conjunto de técnicas que se añaden **cuando se mide una falla concreta** del RAG básico. No es "más complicado = mejor": cada técnica tiene un costo (tokens, llamadas al LLM, indexado, mantenimiento) y debe justificarse con una mejora **medida**.
- **Principio rector:** *no se mejora lo que no se mide.* Por eso la sesión tiene dos mitades que se alimentan entre sí: las **técnicas** (qué cambiar) y la **evaluación** (cómo saber si funcionó).
- **Las dos capas de un RAG:** la **recuperación** (¿llegó la información necesaria al prompt?) y la **generación** (¿el modelo la usó bien?). Fallan por razones distintas y se **miden por separado**: si la recuperación falla, ningún prompt lo arregla.

## 2. Tipos de pregunta y dónde falla un RAG básico

| Tipo | Descripción | Comportamiento esperado |
| --- | --- | --- |
| **Directa** | La respuesta está en un solo lugar | Responder con cita |
| **Multi-sección** | Hacen falta datos de dos o más secciones (se combinan) | Recuperar todas y combinar |
| **Multi-salto (*multi-hop*)** | Hay que **encadenar** datos, a veces con un cálculo; un dato lleva al siguiente | Recuperar todas las piezas y razonar |
| **Ambigua** | Puede entenderse de dos o más maneras | Cubrir las interpretaciones o pedir aclaración |
| **Fuera de alcance, parecida** | Suena a algo del documento pero el dato **no está** | Abstenerse ("no sé"); es la trampa clásica |
| **Fuera de alcance, ajena** | Nada que ver con el documento | Abstenerse |
| **Relacional** | La respuesta se **construye** siguiendo relaciones entre entidades ("si cae X, ¿qué se afecta?") | Recorrer relaciones; ideal para grafos |
| **Global** | Agrega o resume sobre **todo** el corpus ("¿cuál es el servicio más crítico?") | Contar o agregar; ideal para grafos |

**Por qué falla un RAG básico en lo multi-sección y multi-salto:**

- Una pregunta con dos temas produce **un solo vector** que se parece un poco a cada tema y mucho a ninguno; los fragmentos del segundo tema quedan fuera del top-*k*.
- **Subir *k*** trae más cobertura, pero también **más ruido**: gasta tokens, sube el costo y puede confundir a un modelo pequeño.
- Los fragmentos recuperados por **similitud** no incluyen lo que **no se parece** a la pregunta, aunque sea necesario (el dato que está "al otro lado" de una relación).
- Aun con la información en el contexto, encadenar varios pasos es **razonamiento**, y un modelo pequeño puede fallar aunque la recuperación haya sido perfecta.

## 3. Técnicas avanzadas de recuperación

Cada técnica ataca una falla concreta.

### 3.1. *Small-to-big* (documento padre)

- **Dilema del tamaño del fragmento:** el fragmento **pequeño** tiene un vector preciso (habla de una sola cosa) pero poco contexto; el **grande** tiene contexto pero un vector "diluido" y mete ruido al prompt.
- ***Small-to-big* (*parent-document retrieval*):** se **indexan** fragmentos pequeños y se **busca** con ellos, pero al modelo se le **entrega** el trozo mayor al que pertenecen (su **padre**).
- **Variantes:**
  - **Por sección:** el padre es la sección completa. Más cobertura y contexto; menos precisión y más tokens.
  - **Por ventana (*sentence-window*):** el padre es el fragmento más sus vecinos inmediatos. Contexto compacto y buena precisión.
- **Se agrupa por padre:** varios hijos del mismo padre se entregan una sola vez.
- **Límite:** no ayuda si ningún hijo del segundo tema aparece entre los resultados de la búsqueda.

### 3.2. Encabezados contextuales

- **Idea:** anteponer al fragmento su contexto (título del documento, sección) **antes de vectorizarlo**, porque un fragmento suelto puede ser ambiguo ("Cuesta 60 dólares").
- Es **gratis** (no llama al LLM) y no cambia lo que se entrega al modelo.
- **Su efecto depende de los datos:** en documentos con secciones temáticas, el texto del fragmento ya nombra el tema y el efecto es pequeño; en documentos con fragmentos genéricos, es grande.

### 3.3. Descomposición de preguntas

- **Descomposición (*query decomposition*):** un LLM **separa** una pregunta compuesta en **subpreguntas** de un solo tema; se busca cada una y se **fusionan** los resultados.
- **Fusión por rango recíproco (*reciprocal rank fusion*, RRF):** cada resultado suma 1 ÷ (constante + posición) por cada lista donde aparece. Combina listas **sin comparar puntajes** (que no son comparables entre consultas distintas); lo que aparece en varias listas sube.
- **Costo:** una llamada corta al LLM por pregunta, más latencia.
- **Cuándo ayuda:** preguntas multi-sección y multi-salto. En preguntas de un solo tema no cambia casi nada.
- **Limitación:** la calidad depende del modelo que descompone; un modelo pequeño puede copiar el ejemplo del prompt o no separar bien.

### 3.4. HyDE (*Hypothetical Document Embeddings*)

- **Problema:** una **pregunta** y su **respuesta** comparten pocas palabras y quedan lejos en el espacio de vectores.
- **Técnica:** pedir al LLM que **escriba una respuesta hipotética** (aunque invente los datos) y **buscar con el vector de ese texto**, más parecido a los fragmentos del documento. El texto inventado **no se muestra** al usuario.
- **Riesgo:** si la respuesta hipotética va por un tema equivocado, arrastra la búsqueda. Con modelos pequeños puede **empeorar** la precisión; hay que medirla.
- **Familia de técnicas de transformación de la consulta:** reescritura de la consulta (Sesión 6), multi-consulta (varias paráfrasis fusionadas), descomposición y HyDE.

### 3.5. Qué no se repite

**Reescritura de la consulta, búsqueda híbrida (palabras clave + semántica) y reranking** se estudian en la Sesión 6. Se combinan con las técnicas de esta sesión.

### 3.6. Elegir técnica

Una técnica se justifica si puedes **nombrar la falla que ataca**, **medir** que la arregla y **aceptar su costo**. Si no, es complejidad sin evidencia.

## 4. GraphRAG

### 4.1. Conceptos

- **Grafo de conocimiento (*knowledge graph*):** red de **entidades** (nodos: servicios, equipos, personas, productos) unidas por **relaciones** (aristas dirigidas y con nombre: "depende de", "administrado por").
- **Tripleta:** un hecho como `(sujeto, relación, objeto)`, por ejemplo (Portal, depende de, Servicio de autenticación). Un grafo es un conjunto de tripletas.
- **Entidad:** cosa del dominio con identidad propia. **Alias:** otros nombres de la misma entidad (SSO, servicio de autenticación). **Vinculación de entidades (*entity linking*):** reconocer qué entidad del grafo menciona un texto.
- **Recorrido del grafo (*graph traversal*):** seguir aristas desde un nodo. Recorrerlas **al revés** da los dependientes ("¿quién depende de X?"); hacia adelante, las dependencias ("¿de qué depende X?").
- **Impacto o propagación:** servicios que dejan de funcionar si cae uno, es decir, todos los que dependen de él **directa o indirectamente**.
- **CMDB (*configuration management database*):** base de configuración de TI con servicios y sus relaciones; el ejemplo natural de grafo.

### 4.2. GraphRAG: qué es

**RAG que, además de fragmentos de texto, usa un grafo de conocimiento construido a partir de los documentos.** Pipeline típico:

1. **Extracción:** un LLM lee los documentos y extrae **entidades y relaciones** (tripletas).
2. **Construcción del grafo.**
3. **Consulta:** se identifican las entidades de la pregunta y se **recorre** el grafo, o se usan **resúmenes** de partes del grafo.
4. **Generación:** el contexto (hechos o resúmenes) se entrega al LLM.

Dos modos de consulta:

- **Búsqueda local:** parte de las entidades de la pregunta y explora sus **vecinos** (a pocos saltos). Para preguntas sobre cosas concretas y sus relaciones.
- **Búsqueda global:** usa **resúmenes por comunidad** para contestar preguntas sobre **todo el corpus**. **Comunidad:** grupo de nodos muy conectados entre sí, detectado con un algoritmo de agrupamiento de grafos (p. ej. Leiden).

### 4.3. Implementaciones de referencia

| Proyecto | Descripción (según su documentación) |
| --- | --- |
| **Microsoft GraphRAG** | Enfoque estructurado y jerárquico del RAG. Divide el corpus en *text units*, **extrae entidades, relaciones y afirmaciones con un LLM**, agrupa el grafo en **comunidades** (Leiden) y **genera un resumen por comunidad**, de abajo hacia arriba. Modos de consulta: **global**, **local**, **DRIFT** (local con contexto de comunidad) y **básica** (búsqueda vectorial clásica). Su documentación recomienda **ajustar los prompts** antes de fiarse de los resultados. |
| **LightRAG** (HKUDS, MIT, EMNLP 2025) | Alternativa "ligera". Guarda **grafo y vectores a la vez**; un LLM extrae entidades y relaciones. Modos: *naive*, *local*, *global*, *hybrid* y *mix* (el predeterminado). Permite **añadir documentos sin reindexar todo**. Evita los resúmenes por comunidad para reducir las llamadas al LLM y, con ello, el costo y la latencia. El almacenamiento por defecto es en memoria (solo pruebas); se recomienda PostgreSQL en producción. |

En esta sesión **no se instalan**: se explican y se reproduce a escala de juguete lo esencial (tripletas, grafo, recorrido, contexto).

### 4.4. Cuándo aporta y cuándo no

| Aporta cuando… | No compensa cuando… |
| --- | --- |
| Las preguntas son **relacionales** o de **varios saltos** | Las preguntas son directas y la respuesta está en un párrafo |
| Las preguntas son **globales** (agregar, contar, resumir el corpus) | El corpus es pequeño y cabe en el contexto |
| Hay entidades con **muchas conexiones** (CMDB, contratos, cadena de suministro, redes de personas) | Los documentos cambian a diario: **reindexar** cuesta |
| Se necesita **explicar el camino** de la respuesta | No se puede **validar** la calidad de la extracción |

### 4.5. Costos y riesgos

- **Costo de indexar:** la extracción hace **una o más llamadas al LLM por fragmento** del corpus (en corpus grandes, miles); los resúmenes por comunidad añaden más.
- **Mantenimiento:** actualizar el grafo cuando cambian los documentos; reglas de **resolución de entidades** (que "SSO" y "servicio de autenticación" sean el mismo nodo).
- **Calidad de la extracción:** es el paso **más frágil y menos medido**. Un modelo pequeño inventa relaciones, invierte su sentido o escribe la misma entidad de varias formas. **Una arista falsa contamina todas las respuestas que pasan por ella.** Se mide con **precisión y recall de tripletas** contra una muestra curada a mano.
- **Recuperar no es razonar:** aunque los hechos lleguen al prompt, un modelo pequeño puede no encadenarlos. Por eso se complementa con **recorridos calculados** (un algoritmo entrega el resultado) y con modelos más capaces.
- **Traducir la pregunta a una consulta del grafo** (texto a Cypher o SPARQL, o reglas) es otro punto frágil.

### 4.6. Vectorial, grafo y combinado

| | Búsqueda vectorial | Grafo | Combinado |
| --- | --- | --- | --- |
| Encuentra | Texto **parecido** a la pregunta | Entidades y sus **relaciones** | Ambos |
| Fuerte en | Preguntas directas, descriptivas | Impacto, dependencias, agregaciones | Cobertura amplia |
| Débil en | Relaciones implícitas, varios saltos, conteos | Preguntas sin entidades reconocibles; descripciones largas | Más contexto, más costo |

## 5. RAG agéntico (adelanto)

- **RAG agéntico:** un **agente** (un LLM con herramientas) decide **en cada paso** qué hacer: buscar en los documentos, consultar el grafo, descomponer la pregunta, o parar porque ya tiene suficiente. Itera hasta estar satisfecho.
- **Diferencia con el RAG de esta sesión:** el flujo no es fijo; el modelo elige las acciones. Las técnicas de hoy (recuperador vectorial, grafo, descomposición) se convierten en **herramientas** del agente.
- **Contras:** costo, latencia y comportamiento **menos predecibles**; hay que evaluar también las **decisiones** del agente, no solo la respuesta; y al poder actuar, exige **mínimo privilegio**.
- Se estudia con detalle en las sesiones de agentes.

## 6. Evaluación de un RAG

### 6.1. Ideas generales

- **Evaluar = medir con una vara fija.** Sin evaluación, cada cambio es una opinión.
- **Analogía con operaciones:** **SLI** (indicador que se mide), **SLO** (mínimo exigido), **prueba de regresión** (volver a medir tras cada cambio), **observabilidad** (saber en qué capa falló).
- **Golden set (conjunto de oro):** preguntas con la **respuesta correcta escrita por una persona** y la indicación de **dónde está** en el documento. Cada pregunta guarda: tipo, pregunta, respuesta esperada, secciones relevantes y datos clave.
- **Etiquetar por sección** (no por fragmento) permite comparar configuraciones con tamaños de fragmento distintos.
- **Tamaño del conjunto:** con pocas preguntas, una diferencia de una o dos preguntas es **ruido**. El conjunto sirve para **comparar ideas**; uno real necesita decenas o cientos de preguntas reales de usuarios. Debe cubrir **todos los tipos** de pregunta, incluidas las fuera de alcance.

### 6.2. Métricas de recuperación (no usan LLM)

Un fragmento es **relevante** si pertenece a una de las secciones correctas de la pregunta.

| Métrica | Pregunta que responde | Definición |
| --- | --- | --- |
| **Recall@K** | ¿Traje **todo** lo necesario? | Secciones relevantes presentes en el top-*K* ÷ secciones relevantes que existen |
| **Precision@K** | ¿Cuánto de lo que traje **sirve**? | Fragmentos relevantes del top-*K* ÷ *K* |
| **MRR** (*mean reciprocal rank*) | ¿Qué tan arriba quedó el primero útil? | Promedio de 1 ÷ posición del primer fragmento relevante (0 si no aparece) |
| **Hit@K** | ¿Apareció al menos uno relevante? | Fracción de preguntas con algún relevante en el top-*K* |

- **Tensión entre Recall y Precision:** subir *K* sube el recall y baja la precisión (más ruido). Mejorar uno sin hundir el otro exige técnicas más listas, no "traer más".
- **Con una sola sección relevante,** Recall@K coincide con hit@K; con varias, distingue "traje una" de "traje todas".
- Solo se calculan para preguntas **con respuesta** en el documento.
- **Tamaño del contexto** (palabras o tokens entregados): métrica complementaria; es la que se paga.

### 6.3. Métricas de generación (comprobaciones por código)

- **Dato correcto:** los datos clave esperados aparecen en la respuesta. Limitación: no entiende sinónimos ("no es normal" frente a "es un fraude"); da **falsos negativos**.
- **Citas válidas:** la respuesta cita y los números citados **existen** entre los fragmentos entregados. **Precisión de las citas:** fracción de lo citado que es relevante.
- **Abstención correcta ("no sé" bien dicho):** en las preguntas fuera de alcance, el modelo dice que no encuentra la información.
- **Abstención indebida:** el modelo dice "no sé" aunque la información sí llegó (o estaba disponible). Es el **costo oculto** de bajar las alucinaciones.
- **Cifras con respaldo:** toda cifra de la respuesta debe aparecer en el contexto (o en la pregunta); si no, es un indicio fuerte de invención (**un *checksum* de la alucinación**). Limitaciones: no detecta una cifra correcta puesta en el concepto equivocado ni datos inventados que no son números.
- **Cobertura de entidades (preguntas multi-salto con grafos):** recall (¿nombró todas las esperadas?) y precisión (¿solo esas?) de las entidades de la respuesta.

### 6.4. Diagnóstico por capa

Para cada pregunta se asigna la primera categoría que aplique:

| Diagnóstico | Capa | Significado |
| --- | --- | --- |
| `abstencion_por_recuperacion`, `recuperacion_incompleta` | Recuperación | La información no llegó completa al prompt |
| `abstencion_indebida` | Generación | Llegó todo, pero se abstuvo |
| `dato_incorrecto` | Generación | Llegó todo, pero el dato no está en la respuesta |
| `cifra_sin_respaldo`, `alucinacion` | Generación (alucinación) | Inventó una cifra o contestó lo que no se podía contestar |
| `sin_citas_validas` | Verificabilidad | Respondió, pero sin citas verificables |

**Qué se hace con el diagnóstico:** si los fallos son de **recuperación**, se cambia el recuperador; si son de **generación**, se cambia el prompt o el modelo. Tocar la capa equivocada no mueve nada.

### 6.5. Leer un informe

- Mirar la **capa** antes que el porcentaje.
- Mirar **juntas** las alucinaciones y la abstención indebida: un sistema que dice "no sé" a todo no alucina... y no sirve.
- Desglosar **por tipo de pregunta**: los promedios globales esconden las fallas de los tipos difíciles.
- Distinguir **ruido** de **señal**; un servidor local puede dar respuestas algo distintas entre corridas incluso con temperatura 0.

## 7. Alucinaciones: causas y mitigación

- **Alucinación:** respuesta plausible pero **no respaldada** o falsa. En un RAG casi nunca es un fallo misterioso: tiene **causas identificables**.

| Causa | Cómo se ve | Mitigación |
| --- | --- | --- |
| Recuperación pobre | El fragmento correcto no llegó; el modelo rellena | Mejorar la recuperación; abstenerse |
| El prompt no permite el "no sé" | Contesta siempre | Frase exacta de abstención; evaluar con preguntas fuera de alcance |
| Pregunta "parecida pero ausente" | El tema suena al documento, el dato no está | "No sé" explícito para lo parecido-ausente; medirlo |
| Contexto contradictorio u obsoleto | Cita fielmente un dato viejo | **Higiene de la base**: versiones, metadatos de vigencia, filtros, retirar lo obsoleto |
| El modelo ignora el contexto | Responde de "memoria" | Modelo más capaz; verificar fidelidad |
| Extrapolar o deducir | "Costaría unos 90 dólares" | Prohibir estimar; verificar cifras |

- **Fidelidad (*groundedness*) frente a corrección:**
  - **Fidelidad:** todo lo afirmado está **respaldado por el contexto**.
  - **Corrección:** lo afirmado **es verdad** (coincide con lo esperado).
  - Un RAG puede ser **fiel e incorrecto** (el contexto estaba mal u obsoleto) o **correcto e infiel** (acertó de memoria). Se miden por separado.
- **Mitigaciones en capas:** prompt con instrucciones claras de "no sé", verificación de citas y de cifras, higiene de datos, filtros por metadatos aplicados **en la recuperación** (no confiando en el prompt), umbral de similitud, modelo más capaz y revisión humana cuando el riesgo es alto. Ninguna es infalible.
- **Umbral de similitud:** descarta lo **ajeno** y ahorra llamadas al LLM, pero **no separa** las preguntas parecidas-pero-ausentes de las legítimas (puntúan en el mismo rango).
- **Principio:** una alucinación mitigada debe **comprobarse** volviendo a medir, no darse por resuelta.

## 8. El juez LLM (*LLM-as-judge*)

- **Juez LLM:** usar un modelo, con una **rúbrica**, para evaluar respuestas. Escala a miles de respuestas y entiende el lenguaje, pero es un **instrumento** que hay que **calibrar**.
- **Tipos de juez:**
  - **Fidelidad:** ¿todo lo que afirma la respuesta está en el contexto? (recibe contexto y respuesta).
  - **Corrección:** ¿coincide con la respuesta esperada? (recibe respuesta esperada y respuesta).
  - (También se usan: relevancia de la respuesta a la pregunta, relevancia del contexto recuperado.)
- **Buenas prácticas:** **un criterio por juez**; rúbrica explícita que **divide la respuesta en afirmaciones** y busca respaldo de cada una; **salida estructurada validada** (con un esquema) y reintento si es inválida; prompt **versionado**; temperatura baja.
- **Sesgos conocidos:**
  - **Indulgencia:** aprueba casi todo; el error más caro (deja pasar alucinaciones).
  - **Severidad:** rechaza de más; desperdicia tiempo de revisión.
  - **Sesgo de longitud (*verbosity bias*):** prefiere respuestas largas, aunque tengan un error escondido.
  - **Autopreferencia:** favorece respuestas de modelos parecidos a él; por eso conviene un juez **distinto** del generador.
  - **Sesgo de posición:** en comparaciones por pares, favorece la primera (o la última) respuesta.
  - **Sensibilidad al prompt:** cambia el veredicto al cambiar una frase de la rúbrica.
- **Calibración:** pasar al juez respuestas **ya etiquetadas por personas** y medir el acuerdo:
  - **Exactitud:** fracción de veredictos que coinciden con la persona. Engaña con clases desbalanceadas.
  - **Kappa de Cohen:** acuerdo **corregido por el azar** (1 = perfecto, 0 = azar). Como referencia habitual, valores de 0.6 a 0.8 se consideran acuerdo sustancial.
  - **Indulgencia (falsos aprobados) y severidad (falsos rechazados):** miden el *tipo* de error.
  - **Matriz de confusión:** aprobó bien, aprobó mal, rechazó mal, rechazó bien.
- **Un juez pequeño (3B parámetros) suele ser poco mejor que el azar** en tareas finas como detectar una cifra equivocada o una extrapolación. Si el juez es débil, la evaluación es decoración.
- **Uso recomendado:** modelo más capaz que el generador y distinto; calibrar siempre y fijar una meta antes de confiar; usarlo para **priorizar** la revisión humana, no para reemplazarla; **combinarlo con las comprobaciones por código**; versionar el golden set y las etiquetas.

## 9. El ciclo medir, cambiar, volver a medir

1. **Medir** una línea base con el golden set.
2. **Elegir una falla** (por tipo de pregunta y por capa).
3. **Cambiar una sola cosa** (prompt, tamaño de fragmento, técnica, modelo, *k*).
4. **Volver a medir** con **el mismo** conjunto.
5. **Decidir** con números: ¿señal o ruido?, ¿qué empeoró?, ¿cuánto cuesta?
6. **Registrar** el cambio (en el historial de prompts y en el informe) y repetir.

- **Prueba de regresión:** al cambiar algo, volver a correr el golden set para comprobar que **no empeoró** lo que ya funcionaba. Se automatiza en un flujo de integración continua (CI).
- **SLO como puerta de calidad:** un conjunto de reglas (por ejemplo, recall ≥ 90 %, alucinaciones ≤ 10 %) convierte el informe en una **decisión** de desplegar o no. Los valores los fija quien responde por el servicio, no la herramienta.
- **Cada cambio toca una capa:** el prompt mueve la generación; el recuperador mueve la recuperación y, a través del contexto, la generación. Por eso se mide por capas.
- **Compromisos:** puede que ninguna configuración sea la mejor en todo; elegir es decidir qué falla cuesta más (¿una respuesta inventada o un "no sé" de más?).
- **Informe:** configuración, resumen, cumplimiento del SLO, desglose por tipo de pregunta, preguntas con fallo y su capa; en JSON (para comparar) y Markdown (para revisar).

## 10. Glosario rápido

| Término | Definición breve |
| --- | --- |
| **Golden set** | Preguntas con respuesta correcta escrita por una persona; la vara de medir |
| **SLI / SLO** | Indicador que se mide / mínimo que se exige |
| **Recall@K** | Cobertura: relevantes recuperados ÷ relevantes existentes |
| **Precision@K** | Fracción del top-*K* que es relevante |
| **MRR** | Promedio de 1 ÷ posición del primer relevante |
| **Multi-sección** | Pregunta cuya respuesta combina varias secciones |
| **Multi-salto** | Pregunta que exige encadenar datos o razonar en pasos |
| ***Small-to-big*** | Buscar con fragmentos pequeños, entregar el contexto grande |
| **Documento padre** | Contexto mayor al que pertenece un fragmento |
| **Encabezado contextual** | Título o sección antepuesto al fragmento antes de vectorizar |
| **Descomposición** | Separar una pregunta en subpreguntas de un solo tema |
| **RRF** | Fusión por rango recíproco de varias listas ordenadas |
| **HyDE** | Buscar con una respuesta hipotética generada por el LLM |
| **Grafo de conocimiento** | Entidades (nodos) y relaciones (aristas) |
| **Tripleta** | Hecho `(sujeto, relación, objeto)` |
| **Entity linking** | Reconocer qué entidad del grafo menciona un texto |
| **GraphRAG** | RAG con grafo de conocimiento (búsqueda local y global) |
| **Comunidad** | Grupo de nodos muy conectados; base de los resúmenes globales |
| **Leiden** | Algoritmo de detección de comunidades en grafos |
| **CMDB** | Base de configuración de TI: servicios y sus relaciones |
| **RAG agéntico** | RAG donde un agente decide qué buscar y cuándo parar |
| **Fidelidad (*groundedness*)** | Todo lo afirmado está respaldado por el contexto |
| **Corrección** | Lo afirmado es verdad |
| **Alucinación** | Respuesta plausible pero no respaldada o falsa |
| **Abstención** | Responder "no sé" cuando el contexto no alcanza |
| **LLM-as-judge** | Evaluar respuestas con un modelo y una rúbrica |
| **Kappa de Cohen** | Acuerdo entre evaluadores corregido por el azar |
| **Indulgencia / severidad** | Falsos aprobados / falsos rechazados del juez |
| **Regresión** | Un cambio que empeora algo que ya funcionaba |
| **Higiene de la base** | Mantener la base libre de contenido obsoleto, duplicado o no confiable |
