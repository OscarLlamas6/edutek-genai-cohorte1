# Marco teórico · Sesión 7: RAG avanzado: técnicas, GraphRAG y evaluación

> Documento de apoyo al notebook `s07_rag_avanzado.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: el código y los experimentos están en el notebook.

## Índice

1. [De un RAG básico a uno avanzado](#1-de-un-rag-básico-a-uno-avanzado)
2. [Tipos de pregunta y dónde falla un RAG básico](#2-tipos-de-pregunta-y-dónde-falla-un-rag-básico)
3. [Técnicas avanzadas de recuperación](#3-técnicas-avanzadas-de-recuperación)
4. [GraphRAG](#4-graphrag)
5. [Recuperadores modernos y embeddings](#5-recuperadores-modernos-y-embeddings)
6. [RAG multimodal: audio y video](#6-rag-multimodal-audio-y-video)
7. [Evaluación de un RAG](#7-evaluación-de-un-rag)
8. [Alucinaciones: causas y mitigación](#8-alucinaciones-causas-y-mitigación)
9. [El juez LLM (*LLM-as-judge*) y los marcos de evaluación](#9-el-juez-llm-llm-as-judge-y-los-marcos-de-evaluación)
10. [Producción: caché, actualización, ciclo de vida y observabilidad](#10-producción-caché-actualización-ciclo-de-vida-y-observabilidad)
11. [El ciclo medir, cambiar, volver a medir](#11-el-ciclo-medir-cambiar-volver-a-medir)
12. [Glosario rápido](#12-glosario-rápido)
13. [Preguntas de repaso](#13-preguntas-de-repaso)

---

## 1. De un RAG básico a uno avanzado

- **RAG básico (Sesiones 5 y 6):** fragmentar → vectorizar → buscar los *k* fragmentos más parecidos → armar el prompt → generar con citas. La Sesión 6 añadió mejor fragmentación, reescritura de la consulta, búsqueda híbrida y reranking; **no se repiten aquí**. Esta sesión cubre todo lo de RAG que **no necesita un agente**; el RAG agéntico (el modelo decide los pasos) se verá en sesiones posteriores.
- **RAG avanzado:** conjunto de técnicas que se añaden **cuando se mide una falla concreta** del RAG básico. No es "más complicado = mejor": cada técnica tiene un costo (tokens, llamadas al LLM, indexado, mantenimiento) y debe justificarse con una mejora **medida**.
- **Principio rector:** *no se mejora lo que no se mide.* Por eso la sesión tiene dos mitades que se alimentan entre sí: las **técnicas** (qué cambiar: recuperación, grafos, embeddings, medios de audio y video) y la **evaluación** (cómo saber si funcionó), más lo que hace falta para **producción**.
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

### 3.2. Encabezados contextuales y *contextual retrieval*

- **Idea:** anteponer al fragmento su contexto (título del documento, sección) **antes de vectorizarlo**, porque un fragmento suelto puede ser ambiguo ("Cuesta 60 dólares").
- El **encabezado contextual** es **gratis** (no llama al LLM) y no cambia lo que se entrega al modelo.
- **Su efecto depende de los datos:** en documentos con secciones temáticas, el texto del fragmento ya nombra el tema y el efecto es pequeño; en documentos con fragmentos genéricos, es grande.

**Contextual retrieval** (el encabezado, escrito por un LLM):

- **Qué es:** un LLM recibe el **documento completo** y **un fragmento**, y escribe **una frase** que sitúa el fragmento ("Esta parte explica cuándo se suspende el servicio por falta de pago"). Esa frase se antepone al fragmento antes de vectorizarlo (y de construir un índice BM25, si lo hay).
- **Qué hace:** cambia el texto que se **vectoriza**, no el que se entrega. Cuesta **una llamada al LLM por fragmento, una vez al indexar**.
- **Para qué sirve:** corpus largos con muchos fragmentos ambiguos o documentos parecidos entre sí.
- **Analogía de infraestructura:** añadir etiquetas y una descripción a un recurso en la CMDB para que se encuentre aunque su nombre no sea claro.
- **Ejemplo a mano:** "Cuesta 60 dólares." → "Costo de la instalación sin permanencia. Cuesta 60 dólares."
- **Cuándo NO:** si los fragmentos ya nombran su tema o el encabezado basta; si el corpus es pequeño. El LLM puede **inventar** el contexto, y hay que **regenerarlo** al cambiar el documento.
- **Errores típicos:** activarlo sin medir; contextos que repiten el fragmento; no regenerarlos; olvidar que el costo es *tamaño del documento × número de fragmentos* (la **caché de prompt** del proveedor lo abarata, porque el documento se repite al inicio de cada llamada).

### 3.3. Descomposición de preguntas

- **Descomposición (*query decomposition*):** un LLM **separa** una pregunta compuesta en **subpreguntas** de un solo tema; se busca cada una y se **fusionan** los resultados.
- **Fusión por rango recíproco (*reciprocal rank fusion*, RRF):** cada resultado suma 1 ÷ (constante + posición) por cada lista donde aparece. Combina listas **sin comparar puntajes** (que no son comparables entre consultas distintas); lo que aparece en varias listas sube.
- **Costo:** una llamada corta al LLM por pregunta, más latencia.
- **Cuándo ayuda:** preguntas multi-sección y multi-salto. En preguntas de un solo tema no cambia casi nada.
- **Limitación:** la calidad depende del modelo que descompone; un modelo pequeño puede copiar el ejemplo del prompt o no separar bien.

### 3.4. HyDE (*Hypothetical Document Embeddings*)

- **Idea en una línea:** pedir al LLM una **respuesta hipotética** (aunque invente los datos) y **buscar con el vector de ese texto**, que se parece más a los fragmentos que la pregunta. El texto inventado **no se muestra** al usuario.
- **Riesgo:** si la respuesta hipotética va por un tema equivocado, arrastra la búsqueda; con modelos pequeños puede **empeorar**. Se mide antes de activarla.
- **Familia de técnicas de transformación de la consulta:** reescritura (Sesión 6), multi-consulta, descomposición y HyDE.

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

## 5. Recuperadores modernos y embeddings

### 5.1. Las familias de recuperadores

- **Qué es:** un recuperador convierte pregunta y fragmento en algo **comparable**. Lo que cambia es **qué guarda por cada texto** y **cómo compara**.

| Familia | Qué guarda por texto | Cómo compara | Ejemplos |
| --- | --- | --- | --- |
| **Léxica** | Las palabras con peso (IDF) | Palabras exactas en común | BM25 (Sesión 6) |
| **Densa, un vector** (*bi-encoder*) | **Un** vector por texto | Coseno entre el vector de la pregunta y el del fragmento | Los modelos de embeddings del curso |
| **Dispersa aprendida** | Un vector del tamaño del vocabulario, casi todo ceros: un **peso por palabra**, con **expansión** a palabras relacionadas que no aparecen | Producto punto de pesos; se indexa como BM25 | **SPLADE** |
| **Interacción tardía** (*late interaction*) | **Un vector por token** | Cada token de la pregunta busca su **mejor coincidencia** en el documento (**MaxSim**) y se suman | **ColBERT**; **ColPali / ColQwen** (imágenes de páginas) |

- **Qué hace cada una:** BM25 *cuenta palabras*; el bi-encoder *resume un texto en un punto*; SPLADE *reparte pesos entre palabras y sus parientes*; la interacción tardía *deja que cada palabra de la pregunta busque su pareja*. **ColPali** aplica la última idea a la **imagen de una página** (cada región de la imagen es un "token"), sin pasar por OCR.
- **Para qué sirve conocerlas:** saber qué hay cuando bi-encoder + BM25 + reranking se queda corto, y entender documentaciones que hablan de "vectores dispersos" o "multivector".
- **Analogía de infraestructura:** BM25 es un `grep` ponderado; el bi-encoder busca por la *etiqueta* general del recurso; SPLADE es un `grep` al que alguien añadió sinónimos a cada documento; la interacción tardía compara *línea por línea* en vez de comparar el resumen.
- **Ejemplo a mano (SPLADE, ilustrativo):** "Cuesta 60 dólares la instalación" se guardaría con pesos como `instalación 2.1 · cuesta 1.4 · dólares 1.2 · precio 0.9 · costo 0.8`; las dos últimas palabras **no aparecen** en el texto. (Pesos inventados para ilustrar.)
- **Ejemplo a mano (MaxSim):** dos tokens de pregunta; cada uno toma su mejor similitud con los tokens del documento (por ejemplo 0.99 y 0.99) y se suman (1.98). Un documento más largo, con las mismas dos palabras relevantes y relleno, **conserva** el puntaje; un único vector promedio se **diluye**.
- **Cuándo NO:** si BM25 + densa + reranking ya cumplen el SLO. La interacción tardía **multiplica el almacenamiento** (un vector por token); SPLADE añade un modelo al indexado y a la consulta; ColPali exige un modelo de visión y suele necesitar GPU.
- **Errores típicos:** cambiar de familia sin medir con las preguntas propias; olvidar el costo de almacenamiento del multivector; mezclar vectores de modelos distintos.
- **En el curso:** se explican y se calcula MaxSim a mano; **no se implementan** (necesitan sus propios modelos).

### 5.2. Elegir el modelo de embeddings

- **Qué es:** el modelo fija la **calidad**, el **idioma** y las **dimensiones** de los vectores. Los vectores de modelos distintos **no se pueden mezclar**; cambiar de modelo obliga a reindexar todo.
- **Cómo se elige:** con **tus** preguntas, en **tu** idioma, midiendo Recall@K y MRR. Un modelo centrado en inglés recupera peor en español.
- **Prefijos:** algunos modelos esperan prefijos distintos para documentos y consultas; omitirlos baja la calidad.
- **Cuándo NO:** no elegir por una tabla de resultados de otro idioma o dominio.

### 5.3. Dimensiones y Matryoshka

- **Qué es:** un embedding de 768 números cuesta 768 × 4 = 3 072 bytes en `float32`. Un modelo entrenado con **Matryoshka (*Matryoshka Representation Learning*, MRL)** concentra la información en los **primeros** números: un prefijo del vector ya es un buen vector corto, como las muñecas rusas.
- **Qué hace:** se **trunca** a menos dimensiones (por ejemplo 256 o 128) y se **renormaliza** (se divide cada vector entre su nueva longitud) para que el coseno siga siendo un coseno. La **pregunta** y los **fragmentos** deben tener las mismas dimensiones.
- **Para qué sirve:** menos memoria, disco y tiempo de búsqueda, y bases vectoriales más baratas.
- **Analogía de infraestructura:** bajar la resolución de un video para ahorrar ancho de banda: se elige un nivel y se vigila que siga siendo útil.
- **Ejemplo a mano:** 768 → 256 dimensiones son 3 veces menos bytes por vector; con un millón de fragmentos, 3.07 GB pasan a 1.02 GB.
- **Cuándo NO:** si el corpus es pequeño (no hay ahorro real); si el modelo **no** se entrenó con Matryoshka (la calidad cae mucho más). Que un modelo **declare** soporte no sustituye a **verificarlo** con el corpus y las preguntas propias.
- **Errores típicos:** truncar sin renormalizar; truncar solo los fragmentos; decidir con pocas preguntas o de otro dominio; mezclar dimensiones en un índice.

### 5.4. Cuantización

- **Qué es:** guardar cada número del vector con **menos bits**. **int8:** un byte en vez de cuatro (4 veces menos). **Binaria:** un bit por dimensión, solo el signo (32 veces menos), con comparación por distancia de Hamming.
- **Qué hace:** reduce memoria y acelera la comparación a cambio de algo de precisión numérica. Se combina con Matryoshka (por ejemplo 256 dimensiones en int8 = 256 bytes por vector).
- **Para qué sirve:** corpus grandes donde el almacenamiento y la memoria son el cuello de botella; se suele **reordenar** después un conjunto pequeño de candidatos con el vector completo.
- **Analogía de infraestructura:** reducir la profundidad de color de una imagen, o comprimir logs con pérdida controlada.
- **Ejemplo a mano:** un valor 0.37 con escala (máximo ÷ 127) se guarda como un entero entre −127 y 127; al leerlo se multiplica por la escala.
- **Cuándo NO:** corpus pequeño; calidad crítica sin margen para reordenar; el binario pierde más en corpus donde las diferencias son sutiles.
- **Errores típicos:** cuantizar sin medir; olvidar guardar la escala; confundir cuantización (números más pequeños) con índice aproximado (HNSW, otra optimización que se puede sumar).

## 6. RAG multimodal: audio y video

### 6.1. El pipeline estándar

1. **Extraer el audio** (`ffmpeg`: sin imagen, 16 kHz, mono).
2. **Transcribir** con un modelo de **ASR** (*automatic speech recognition*), por ejemplo **Whisper** (en CPU, con `faster-whisper`). El resultado son **segmentos con `inicio`, `fin` y `texto`**.
3. **Fragmentar con tiempo:** juntar segmentos en **ventanas** de unos 45 a 60 segundos (con un poco de solape). Cada fragmento conserva su inicio y su fin.
4. **Vectorizar** con los mismos embeddings del resto del curso y **guardar** en el almacén, con el tiempo como metadato.
5. **Recuperar** y **responder con cita de tiempo** (`[mm:ss–mm:ss]`), para que una persona pueda ir al reproductor y verificar.

- **Qué es:** un RAG de audio o video es el mismo RAG con un **parser nuevo** (el ASR).
- **Para qué sirve:** preguntar a reuniones, charlas, capacitaciones y llamadas, y obtener la respuesta **con el minuto**.
- **Analogía de infraestructura:** los logs con `timestamp`: lo valioso de una línea no es solo el texto sino cuándo ocurrió. Perder el tiempo es perder la correlación. El ASR es el *parser*: si falla, todo lo demás hereda el error.
- **Ejemplo a mano:** ocho segmentos de 6 a 8 segundos se juntan en una ventana de `03:00` a `03:52`; la cita es `[03:00–03:52]`.
- **Preguntar en otro idioma:** con un modelo de embeddings **multilingüe**, una pregunta en español puede recuperar fragmentos en inglés; el LLM responde en español. Hay que **medirlo**, no suponerlo.

### 6.2. Video: audio más fotogramas

- **El audio** se trata igual que arriba y suele contener casi toda la información.
- **Los fotogramas clave** (uno cada N segundos, o al cambiar la escena) se pueden **describir** con un modelo con visión o con **OCR**, y se **alinean por tiempo** con la transcripción: para el fragmento de `08:29` a `08:52` se asocia el fotograma más cercano. Es un complemento **opcional**.
- **Qué indexar, según el caso:**

| Enfoque | Cuándo conviene | Limitación |
| --- | --- | --- |
| Solo transcripción | Charlas, reuniones, podcasts, llamadas | Pierde lo que solo se ve |
| Transcripción + fotogramas | Diapositivas, demostraciones, pizarras | Visión local lenta y débil con letra manuscrita o baja resolución |
| OCR de fotogramas | Capturas y diapositivas con texto claro | Mal con manuscrita; ruido de interfaces |
| Embeddings de la imagen (ColPali / ColQwen) | Documentos visuales: PDF con tablas, gráficos y diseño | Modelo de visión, GPU y mucho almacenamiento |

### 6.3. Documentos visuales: ColPali y ColQwen

- **Qué es:** modelos que producen **embeddings de la imagen de una página** (un vector por región) y usan **interacción tardía** (MaxSim) para compararla con la pregunta. Evitan el paso de OCR y de extracción de tablas.
- **Cuándo conviene:** cuando el OCR o la extracción de tablas te cuestan respuestas medidas, o el significado depende del diseño.
- **Cuándo NO:** texto limpio (un Markdown, un PDF con buena capa de texto); equipos sin GPU; almacenamiento limitado.
- En el curso **se explica, no se implementa**.

### 6.4. Limitaciones honestas

- **La transcripción es lo que sostiene el RAG.** Si se transcribe mal, no se encuentra bien.
- **El ASR se equivoca con el vocabulario técnico y los nombres propios** ("ChatGPT" puede salir como otra palabra). Se mitiga con **`initial_prompt`** (se le da a Whisper el vocabulario que debe esperar) y con la revisión de una muestra, pero **no se elimina**.
- **Visión local débil:** un modelo con visión que corre en CPU es lento y lee mal la escritura manuscrita a baja resolución.
- **Costo en cómputo:** transcribir es CPU/GPU; se hace una vez y se **guarda** el resultado.
- **Privacidad y derechos:** una grabación puede contener datos personales; los medios de terceros no se publican en el repositorio.
- **Errores típicos:** perder los tiempos al fragmentar; ventanas demasiado cortas o demasiado largas; no revisar la transcripción; transcribir de nuevo en cada ejecución.

## 7. Evaluación de un RAG

### 7.1. Ideas generales

- **Evaluar = medir con una vara fija.** Sin evaluación, cada cambio es una opinión.
- **Analogía con operaciones:** **SLI** (indicador que se mide), **SLO** (mínimo exigido), **prueba de regresión** (volver a medir tras cada cambio), **observabilidad** (saber en qué capa falló).
- **Golden set (conjunto de oro):** preguntas con la **respuesta correcta escrita por una persona** y la indicación de **dónde está** en el documento. Cada pregunta guarda: tipo, pregunta, respuesta esperada, secciones relevantes y datos clave.
- **Etiquetar por sección** (no por fragmento) permite comparar configuraciones con tamaños de fragmento distintos.
- **Tamaño del conjunto:** con pocas preguntas, una diferencia de una o dos preguntas es **ruido**. El conjunto sirve para **comparar ideas**; uno real necesita decenas o cientos de preguntas reales de usuarios. Debe cubrir **todos los tipos** de pregunta, incluidas las fuera de alcance.

### 7.2. Métricas de recuperación (no usan LLM)

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

### 7.3. Métricas de generación (comprobaciones por código)

- **Dato correcto:** los datos clave esperados aparecen en la respuesta. Limitación: no entiende sinónimos ("no es normal" frente a "es un fraude"); da **falsos negativos**.
- **Citas válidas:** la respuesta cita y los números citados **existen** entre los fragmentos entregados. **Precisión de las citas:** fracción de lo citado que es relevante.
- **Abstención correcta ("no sé" bien dicho):** en las preguntas fuera de alcance, el modelo dice que no encuentra la información.
- **Abstención indebida:** el modelo dice "no sé" aunque la información sí llegó (o estaba disponible). Es el **costo oculto** de bajar las alucinaciones.
- **Cifras con respaldo:** toda cifra de la respuesta debe aparecer en el contexto (o en la pregunta); si no, es un indicio fuerte de invención (**un *checksum* de la alucinación**). Limitaciones: no detecta una cifra correcta puesta en el concepto equivocado ni datos inventados que no son números.
- **Cobertura de entidades (preguntas multi-salto con grafos):** recall (¿nombró todas las esperadas?) y precisión (¿solo esas?) de las entidades de la respuesta.

### 7.4. Diagnóstico por capa

Para cada pregunta se asigna la primera categoría que aplique:

| Diagnóstico | Capa | Significado |
| --- | --- | --- |
| `abstencion_por_recuperacion`, `recuperacion_incompleta` | Recuperación | La información no llegó completa al prompt |
| `abstencion_indebida` | Generación | Llegó todo, pero se abstuvo |
| `dato_incorrecto` | Generación | Llegó todo, pero el dato no está en la respuesta |
| `cifra_sin_respaldo`, `alucinacion` | Generación (alucinación) | Inventó una cifra o contestó lo que no se podía contestar |
| `sin_citas_validas` | Verificabilidad | Respondió, pero sin citas verificables |

**Qué se hace con el diagnóstico:** si los fallos son de **recuperación**, se cambia el recuperador; si son de **generación**, se cambia el prompt o el modelo. Tocar la capa equivocada no mueve nada.

### 7.5. Leer un informe

- Mirar la **capa** antes que el porcentaje.
- Mirar **juntas** las alucinaciones y la abstención indebida: un sistema que dice "no sé" a todo no alucina... y no sirve.
- Desglosar **por tipo de pregunta**: los promedios globales esconden las fallas de los tipos difíciles.
- Distinguir **ruido** de **señal**; un servidor local puede dar respuestas algo distintas entre corridas incluso con temperatura 0.

## 8. Alucinaciones: causas y mitigación

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

## 9. El juez LLM (*LLM-as-judge*) y los marcos de evaluación

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

### 9.1. Marcos de evaluación del ecosistema

- **Qué son:** bibliotecas y plataformas que traen **métricas, jueces y paneles** ya hechos. Según su documentación: **RAGAS** (métricas de RAG como fidelidad, relevancia de la respuesta, precisión y recall del contexto, varias con un LLM como juez), **TruLens** (instrumenta la app y evalúa con funciones de retroalimentación; la "tríada del RAG": relevancia del contexto, fundamentación y relevancia de la respuesta), **DeepEval** (pruebas de LLM al estilo de `pytest`) y **Phoenix** de Arize (observabilidad con trazas y evaluaciones, sobre OpenTelemetry).
- **Cuándo usarlos:** cuando ya entiendes tus métricas y necesitas **estandarizar**, **graficar**, **correr en CI** o **comparar** entre equipos.
- **Por qué en el curso se miden a mano:** cada métrica de esos marcos es por dentro **un conteo o un prompt de juez** como los que se escriben aquí; si no sabes qué mide, no sabes cuándo se equivoca. Muchas usan un **LLM como juez**, así que heredan la necesidad de **calibrarlo**. Y cambian rápido: se consulta su documentación vigente antes de adoptar uno. No se añaden dependencias.
- **Errores típicos:** adoptar un marco y confiar en sus números sin validar unos casos a mano; no calibrar el juez que el marco usa por dentro.

## 10. Producción: caché, actualización, ciclo de vida y observabilidad

### 10.1. Caché

| Caché | Qué guarda | Cuándo acierta | Riesgo principal |
| --- | --- | --- | --- |
| **Exacta** | Respuesta por texto de pregunta idéntico | Preguntas repetidas al carácter | Casi nunca acierta con lenguaje natural |
| **Semántica** | Respuesta por pregunta **parecida** (embedding + **umbral** de similitud) | Paráfrasis | **Falso acierto**: pregunta casi igual con **otra** respuesta |
| **De prompt / prefijo** | El inicio ya procesado de un prompt, en el proveedor | Prompts que comparten el comienzo | Exige poner **lo estable primero y lo variable al final** |
| **De embeddings** | Vectores de textos ya vectorizados | Reindexar o repetir consultas | Memoria |
| **De recuperación** | Fragmentos devueltos por consulta | Consultas repetidas | Datos obsoletos |

- **Qué es la caché semántica:** antes de llamar al LLM, se vectoriza la pregunta y se compara con las preguntas ya respondidas; si la similitud supera un **umbral**, se devuelve la respuesta guardada.
- **Qué hace:** ahorra la generación (la etapa que más tiempo y dinero consume).
- **Cómo se elige el umbral:** **midiendo**, con paráfrasis (que deben acertar) y con preguntas casi iguales de otra respuesta (que deben fallar); depende del modelo de embeddings. Se mide la **tasa de aciertos** y la **tasa de falsos aciertos**.
- **Analogía de infraestructura:** un CDN o un Redis delante de la API, con sus reglas: **TTL**, **invalidación**, **clave de caché** y el temor a servir contenido viejo.
- **Ejemplo a mano:** se guardó "¿Cuánto cuesta el plan de 300 Mbps?"; llega "¿Cuál es el precio mensual del plan de 300 Mbps?" (similitud alta): acierto. Llega "¿Cuánto cuesta el plan de 600 Mbps?" (casi igual, otra respuesta): si el umbral es bajo, falso acierto.
- **Cuándo NO:** respuestas que dependen de **quién** pregunta (permisos, cuenta) o de datos que cambian en minutos.
- **Errores típicos:** umbral "a ojo"; claves sin la **versión del índice y del prompt**; sin TTL ni invalidación; la pregunta del usuario al inicio del prompt (arruina la caché de prefijo).

### 10.2. Latencia y costo

- **Etapas de una petición:** embedding de la consulta, búsqueda, (reranking), armado del prompt y **generación**. La generación suele dominar el tiempo y el costo.
- **Palancas:** caché; menos contexto (menor *k*, *small-to-big*); un modelo más económico para las preguntas fáciles; *streaming* para la latencia percibida; caché de prefijo.
- **Se mide por percentiles** (p50, p95), no por promedios, y por etapa.
- **El costo del contexto:** cada fragmento entregado se paga en tokens de entrada en **cada** pregunta.

### 10.3. Actualización incremental e invalidación

- **Qué es:** cuando cambia un documento, se **re-vectoriza solo lo que cambió**. Cada fragmento se identifica por el **hash de su contenido**; el que ya existe se reutiliza, el nuevo se vectoriza y el que desapareció se elimina.
- **Para qué sirve:** que el índice no quede viejo sin pagar un reindexado completo.
- **Analogía de infraestructura:** `rsync` frente a copiar todo con `scp`; un *build* incremental frente a uno completo.
- **Ejemplo a mano:** un precio cambia en un párrafo: cambia el hash de 1 fragmento de 26; se vectoriza 1 y se reutilizan 25.
- **Invalidación:** las cachés (semántica, de recuperación) pueden seguir sirviendo la respuesta vieja: se invalidan por TTL o al cambiar el documento que las originó.
- **Cuándo NO bastan:** si cambias de **modelo de embeddings** o de **estrategia de fragmentado**, todos los hashes valen y se reindexa todo.
- **Errores típicos:** reindexar todo "por si acaso"; olvidar borrar los fragmentos de documentos retirados; no invalidar las cachés.

### 10.4. Ciclo de vida del índice

- **Versionar** el índice: modelo de embeddings, tamaño y estrategia de fragmentado, esquema de metadatos. La versión viaja en las **trazas** y en la **clave de caché**.
- **Un cambio estructural = un índice nuevo**, construido al lado (como un despliegue azul/verde), evaluado con el **golden set** y el **SLO**, y solo entonces activado; el anterior se conserva para volver atrás.
- **Vigencia y retiros:** metadatos de vigencia, borrado real de lo retirado (derecho al olvido).
- **Deriva:** una muestra de preguntas reales pasa periódicamente por el golden set y el juez; si la calidad cae, algo cambió.

### 10.5. Observabilidad

- **Qué trazar en cada petición:** identificador (anonimizado); **versiones** (prompt, modelo, embeddings, índice, parámetros); **recuperación** (ids y puntajes, acierto de caché); **generación** (tokens, costo, motivo de parada); **tiempos por etapa**; **calidad** (citas válidas, "no sé", veredicto del juez en una muestra, voto del usuario); y **privacidad** (qué se guarda y por cuánto tiempo).
- **Para qué sirve:** reproducir un caso, saber qué capa falló, controlar el costo y detectar regresiones.
- **Herramientas:** trazado con OpenTelemetry y plataformas como Langfuse o Phoenix; son lo mismo que los logs y métricas de siempre aplicados a un RAG.
- **Errores típicos:** registrar la pregunta completa con datos personales sin política de retención; medir solo el promedio; no registrar las versiones.

## 11. El ciclo medir, cambiar, volver a medir

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

## 12. Glosario rápido

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
| **Fidelidad (*groundedness*)** | Todo lo afirmado está respaldado por el contexto |
| **Corrección** | Lo afirmado es verdad |
| **Alucinación** | Respuesta plausible pero no respaldada o falsa |
| **Abstención** | Responder "no sé" cuando el contexto no alcanza |
| **LLM-as-judge** | Evaluar respuestas con un modelo y una rúbrica |
| **Kappa de Cohen** | Acuerdo entre evaluadores corregido por el azar |
| **Indulgencia / severidad** | Falsos aprobados / falsos rechazados del juez |
| **Regresión** | Un cambio que empeora algo que ya funcionaba |
| **Higiene de la base** | Mantener la base libre de contenido obsoleto, duplicado o no confiable |
| **Contextual retrieval** | Un LLM escribe una frase de contexto por fragmento antes de vectorizarlo |
| **Bi-encoder** | Modelo que convierte cada texto en un vector por separado y compara con coseno |
| **SPLADE** | Recuperación dispersa aprendida: pesos por palabra, con expansión a palabras relacionadas |
| **Interacción tardía (*late interaction*)** | Un vector por token y comparación palabra a palabra |
| **MaxSim** | Para cada token de la pregunta, su mejor similitud en el documento; se suman |
| **ColBERT** | Modelo de recuperación de texto con interacción tardía |
| **ColPali / ColQwen** | Interacción tardía sobre la imagen de una página, sin OCR |
| **Matryoshka (MRL)** | Embeddings entrenados para que un prefijo del vector sea un buen vector corto |
| **Renormalizar** | Dividir un vector entre su longitud para que vuelva a medir 1 |
| **Cuantización** | Guardar cada número del vector con menos bits (int8, 1 bit) |
| **ASR / Whisper** | Reconocimiento automático de voz / modelo abierto de ASR |
| **Segmento** | Tramo de habla con inicio, fin y texto, que devuelve el ASR |
| **Ventana de tiempo** | Conjunto de segmentos consecutivos que forma un fragmento de un audio o video |
| **`initial_prompt`** | Vocabulario que se le da a Whisper para que espere ciertos términos |
| **Fotograma clave** | Imagen extraída de un video cada cierto tiempo, para describirla o leerla con OCR |
| **Caché semántica** | Reutilizar la respuesta de una pregunta parecida (umbral de coseno) |
| **Falso acierto (de caché)** | La caché responde a una pregunta casi igual que pedía otra cosa |
| **Caché de prompt / prefijo** | El proveedor reutiliza el procesamiento del inicio de un prompt repetido |
| **TTL** | Tiempo de vida de una entrada de caché |
| **Actualización incremental** | Re-vectorizar solo los fragmentos cuyo hash de contenido cambió |
| **Traza** | Registro de las etapas, tiempos, versiones y costo de una petición |
| **RAGAS / TruLens / DeepEval / Phoenix** | Marcos de evaluación y observabilidad del ecosistema |

## 13. Preguntas de repaso

1. **¿Por qué Recall@K sube y Precision@K baja al aumentar K?** Más fragmentos cubren más secciones relevantes, pero también meten más ruido.
2. **¿Qué ataca *small-to-big* y qué no puede arreglar?** Ataca la falta de contexto del fragmento correcto; no arregla un segundo tema que no aparece en los resultados.
3. **¿En qué se parecen y en qué se diferencian el encabezado contextual y el *contextual retrieval*?** Ambos anteponen contexto antes de vectorizar; el primero es el título de la sección (gratis) y el segundo lo escribe un LLM (una llamada por fragmento al indexar).
4. **¿Por qué la pregunta "si falla X, ¿qué se afecta?" es difícil para un RAG vectorial?** La respuesta no está escrita en ningún fragmento: hay que construirla recorriendo relaciones al revés.
5. **¿Qué costos de GraphRAG no se ven en una demo?** Extraer el grafo (llamadas por fragmento), mantenerlo y validar la extracción; una arista falsa contamina todos los recorridos que pasan por ella.
6. **¿Qué guarda por texto un bi-encoder, SPLADE y ColBERT?** Un vector; un vector disperso con un peso por palabra (incluidas las relacionadas); y un vector por token.
7. **¿Qué precio paga la interacción tardía?** Almacenamiento: un vector por token en vez de uno por fragmento.
8. **¿Qué es Matryoshka y qué hay que hacer después de truncar?** Embeddings cuyo prefijo ya es un buen vector corto; hay que renormalizar y truncar también la pregunta.
9. **¿Cuándo no tiene sentido cuantizar o truncar?** Con corpus pequeños y cuando no se ha verificado el efecto con las preguntas propias.
10. **En un RAG de audio, ¿por qué se guardan `inicio` y `fin`?** Para citar `mm:ss` y que una persona verifique la respuesta en el reproductor.
11. **¿Cómo se mitigan los errores de ASR con vocabulario técnico?** Con `initial_prompt`, un modelo de ASR más grande y la revisión de una muestra; no se eliminan del todo.
12. **¿Qué se indexa en un video y qué aporta cada parte?** La transcripción aporta casi todo el contenido; los fotogramas (con visión u OCR) aportan lo que solo se ve, si se justifica midiendo.
13. **¿Cuándo conviene ColPali?** Con documentos visuales (tablas, gráficos, diseño) donde el OCR cuesta respuestas medidas y se dispone de GPU y almacenamiento.
14. **¿Un RAG con 0 % de alucinaciones y 40 % de "no sé" indebido es mejor que otro con 5 % y 5 %?** No necesariamente: un sistema que se abstiene de todo no alucina y no sirve; se miran juntas.
15. **¿Qué diferencia hay entre fidelidad y corrección?** Fidelidad: lo dicho está en el contexto. Corrección: lo dicho es verdad. Un contexto obsoleto produce respuestas fieles e incorrectas.
16. **¿Cómo se sabe si un juez LLM sirve?** Calibrándolo contra etiquetas humanas (exactitud, kappa, tipo de error) con una meta fijada de antemano.
17. **¿Por qué en el curso se miden las métricas a mano si existen RAGAS o TruLens?** Para entender qué mide cada una; los marcos se adoptan después, validando unos casos a mano.
18. **¿Qué es un falso acierto de la caché semántica y cómo se controla?** Responder con una respuesta guardada a una pregunta casi igual que pedía otra cosa; se controla con un umbral calibrado, midiendo la tasa de falsos aciertos.
19. **¿Por qué la pregunta va al final del prompt si se quiere aprovechar la caché de prefijo?** Porque solo se reutiliza el comienzo idéntico; lo que cambia en cada petición debe ir al final.
20. **¿Qué se evita con la actualización incremental y qué sigue siendo necesario?** Reindexar todo; sigue siendo necesario invalidar cachés, borrar lo retirado y reindexar entero si cambia el modelo de embeddings.
21. **¿Qué se traza en cada petición de un RAG?** Versiones, fragmentos y puntajes, tiempos por etapa, tokens y costo, señales de calidad y la política de privacidad.
