# Marco teórico · Sesión 7: RAG avanzado y GraphRAG

> Documento de apoyo al notebook `s07_rag_avanzado.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: el código está en el notebook y la implementación completa, en el proyecto `rag-avanzado` del repositorio `rag-playground`.

## Índice

1. [Qué hace "avanzado" a un RAG](#1-qué-hace-avanzado-a-un-rag)
2. [Cómo se mide](#2-cómo-se-mide)
3. [Mejorar la pregunta: descomposición, RRF y HyDE](#3-mejorar-la-pregunta-descomposición-rrf-y-hyde)
4. [Darle contexto al fragmento](#4-darle-contexto-al-fragmento)
5. [Otras familias de recuperadores y vectores más pequeños](#5-otras-familias-de-recuperadores-y-vectores-más-pequeños)
6. [GraphRAG](#6-graphrag)
7. [Alucinaciones, fidelidad y corrección](#7-alucinaciones-fidelidad-y-corrección)
8. [El juez LLM y su calibración](#8-el-juez-llm-y-su-calibración)
9. [Producción](#9-producción)
10. [La implementación de referencia: `rag-avanzado`](#10-la-implementación-de-referencia-rag-avanzado)
11. [Glosario rápido](#11-glosario-rápido)
12. [Preguntas de repaso](#12-preguntas-de-repaso)

---

## 1. Qué hace "avanzado" a un RAG

- **RAG (*Retrieval-Augmented Generation*, generación aumentada por recuperación):** técnica que busca fragmentos relevantes en tus documentos y se los entrega a un modelo como contexto para que responda apoyándose en ellos y **citándolos**.
- **RAG básico:** fragmentar → vectorizar → buscar los *k* fragmentos más parecidos → generar. Es un **flujo fijo**: el código decide cada paso.
- **RAG avanzado:** el mismo flujo con **mejoras** que se añaden cuando una medición muestra una falla concreta. No es una tecnología distinta ni "más complicado = mejor".
- **Criterio de una mejora justificada:** (1) se puede nombrar la falla que ataca, (2) se midió antes, (3) se vuelve a medir después. Sin eso es complejidad sin evidencia.
- **Costo:** cada técnica cuesta llamadas al modelo, tokens, tiempo de indexado, mantenimiento o complejidad.

**Las tres palancas:**

| Palanca | Qué se cambia | Técnicas |
| --- | --- | --- |
| La pregunta | Lo que se busca | Descomposición, HyDE |
| Lo que se recupera | Qué entra al prompt y cómo se ordena | Fusión RRF, contexto en fragmentos, *small-to-big*, otras familias de recuperadores |
| La estructura del conocimiento | Cómo se guardan los datos | GraphRAG |

## 2. Cómo se mide

### 2.1. El conjunto de oro (*golden set*)

Lista fija de preguntas, cada una con la **respuesta correcta escrita por una persona**, **dónde está** la respuesta (secciones o entidades), los **datos clave** que debe contener y un **tipo** (directa, de varios temas, de varios saltos, ambigua, fuera de alcance). Funciona como una **prueba de regresión**: se vuelve a correr al cambiar el prompt, el fragmentado o el modelo. Con pocas preguntas, una o dos de diferencia son **ruido**: sirve para comparar ideas, no para prometer exactitud.

### 2.2. Dos capas medidas por separado

| Capa | Pregunta | Quién falla |
| --- | --- | --- |
| **Recuperación** | ¿Llegó al prompt lo necesario? | Buscador, fragmentado, grafo |
| **Generación** | ¿El modelo respondió bien con lo recibido? | Prompt, modelo |

Separarlas indica **dónde actuar**: si la información no llegó, ningún prompt lo arregla.

### 2.3. Métricas de recuperación

| Métrica | Pregunta | Cálculo |
| --- | --- | --- |
| **Recall@K** | ¿Traje todo lo necesario? | necesarios en el top-*K* ÷ necesarios que existen |
| **Precision@K** | ¿Cuánto de lo traído sirve? | relevantes del top-*K* ÷ *K* |
| **MRR** | ¿Qué tan arriba quedó el primero útil? | promedio de 1 ÷ posición del primer relevante (0 si no aparece) |
| **hit@K** | ¿Apareció al menos un relevante en el top-*K*? | 1 o 0, promediado |

Subir *K* sube el recall y baja (casi siempre) la precisión. El ruido cuesta tokens y puede confundir al modelo.

### 2.4. Otras medidas

| Medida | Qué mide | Capa |
| --- | --- | --- |
| **Cobertura del contexto** | Fracción de entidades o hechos esperados que llegó al prompt | Recuperación |
| **Recall y precisión de la respuesta** | Fracción de lo esperado que el modelo nombró; fracción de lo nombrado que era esperado | Generación |
| **Respuesta completa** | Nombró todo lo esperado y solo eso | Generación |
| **Citas válidas** | Cada `[n]` apunta a un elemento que existe | Generación |
| **Abstención correcta / indebida** | Dijo "no sé" cuando debía / cuando no debía | Generación |

## 3. Mejorar la pregunta: descomposición, RRF y HyDE

### 3.1. Descomposición de preguntas

Un modelo separa una pregunta compuesta en **subpreguntas**, cada una sobre un solo tema; cada una se busca por separado y los resultados se juntan.

- **Ataca:** preguntas que mezclan temas (un solo vector queda "a medio camino").
- **Costo:** una llamada corta al modelo por pregunta.
- **Ayuda cuando:** la pregunta realmente tiene varias partes. **No ayuda** en preguntas directas y puede **empeorar** si las subpreguntas no aportan información nueva (solo añaden ruido).
- **Riesgo conocido:** los modelos pequeños **copian los ejemplos** del prompt de descomposición.

### 3.2. Fusión por rango recíproco (RRF)

- **RRF (*Reciprocal Rank Fusion*, Cormack, Clarke y Büttcher, 2009):** une varias listas ordenadas en una. Cada elemento suma **1 ÷ (c + posición)** por cada lista donde aparece; *c* vale 60 por convención.
- **Usa solo posiciones**, porque los puntajes de búsquedas distintas no son comparables.
- Lo que aparece en **varias listas** suma varias veces y sube: es una votación por posición (analogía: un quórum).

### 3.3. HyDE

**HyDE** (*Hypothetical Document Embeddings*, Gao et al., 2022): en lugar de buscar con la pregunta, el modelo escribe una **respuesta hipotética** y se busca con el vector de ese texto, que se parece más a los fragmentos que la pregunta. A veces ayuda y a veces empeora: se mide, no se activa por defecto.

## 4. Darle contexto al fragmento

Un fragmento suelto puede ser ambiguo ("Cuesta 60 dólares": ¿qué cuesta?).

| Técnica | Qué hace | Costo |
| --- | --- | --- |
| **Encabezado contextual** | Antepone el título de la sección antes de vectorizar | Cero |
| **Recuperación contextual** (*contextual retrieval*, Anthropic, 2024) | Un modelo lee el documento y cada fragmento y escribe una frase que lo sitúa; se antepone antes de vectorizar | Una llamada por fragmento, una vez al indexar; la caché de prompt la abarata |
| ***Small-to-big*** (documento padre) | Se **busca** con fragmentos pequeños y se **entrega** el trozo grande al que pertenecen | Más tokens por consulta |

- **Dilema del tamaño:** el fragmento pequeño tiene un vector preciso pero poco contexto; el grande tiene contexto pero un vector diluido y más ruido. *Small-to-big* resuelve el dilema: pequeño para buscar, grande para responder.
- **Variantes del padre:** la **sección completa** (más cobertura y tokens) o una **ventana** de vecinos (más compacto).
- **Contexto escrito por un LLM:** ayuda sobre todo con fragmentos genéricos o documentos que se parecen entre sí; si el texto ya nombra su tema, casi no se nota. Se mide antes de pagarlo.

## 5. Otras familias de recuperadores y vectores más pequeños

### 5.1. Familias de recuperadores

| Familia | Idea | Fortaleza | Costo |
| --- | --- | --- | --- |
| **Léxica (BM25)** | Cuenta palabras; pesa más las raras | Códigos y nombres exactos | Casi ninguno |
| **Densa (*bi-encoder*)** | Un vector por texto; se compara por coseno | Significado y paráfrasis | Un vector por fragmento |
| **Dispersa aprendida (SPLADE**, Formal et al., 2021) | Vector enorme casi todo ceros: un peso por palabra, con palabras relacionadas añadidas por el modelo | Coincidencia exacta con algo de significado; se indexa como BM25 | Modelo propio |
| **Interacción tardía (ColBERT**, Khattab y Zaharia, 2020) | Un vector por palabra; cada palabra de la pregunta busca su mejor pareja (*MaxSim*) y se suman | Precisión en documentos largos | Mucho más almacenamiento |
| **Interacción tardía visual (ColPali**, Faysse et al., 2024) | La misma idea con regiones de la imagen de una página | PDF con tablas y gráficos sin OCR | Modelo de visión; suele pedir GPU |

Cambiar de familia se justifica si la búsqueda por palabras, la vectorial y el reordenamiento ya no cumplen el objetivo de calidad medido con tus preguntas.

### 5.2. Vectores más pequeños

| Técnica | Idea | Riesgo |
| --- | --- | --- |
| **Matryoshka** (Kusupati et al., 2022) | El modelo se entrena para que los primeros *d* números sean un buen vector pequeño; se **trunca** y se **renormaliza** | No todos los modelos se truncan igual: se verifica con tus datos |
| **Cuantización** | Guardar cada número con menos bits (int8: 4 veces menos que float32; 1 bit: 32 veces menos) | Pérdida de precisión en el orden de los resultados |

Un vector de 768 dimensiones en float32 pesa 3 072 bytes; con un millón de fragmentos son unos 3 GB. Cambiar de modelo o de dimensiones obliga a **reindexar todo**.

## 6. GraphRAG

### 6.1. Definiciones

| Término | Definición |
| --- | --- |
| **Grafo de conocimiento** | Red de **nodos** (entidades del dominio) y **aristas** (relaciones entre ellas) |
| **Entidad** | Una cosa del dominio: servicio, equipo, persona, contrato |
| **Relación** | Vínculo entre dos entidades: *depende de*, *se ejecuta en*, *administrado por* |
| **Tripleta** | Un hecho: (sujeto, relación, objeto) |
| **Procedencia** | Sección y oración del documento de donde salió el hecho |
| **GraphRAG** | RAG que, además de fragmentos de texto, usa un grafo construido a partir de los documentos |

- **Diferencia de fondo:** un buscador vectorial responde "¿qué texto se parece a esto?"; un grafo responde "¿qué está conectado con esto?" y permite **recorrer, contar y agregar**.
- **Analogía:** `grep` sobre runbooks frente a consultar una **CMDB**. "¿Qué se afecta si cae X?" no está escrito en ningún runbook: se **calcula** recorriendo dependencias (análisis de impacto).

### 6.2. Construir el grafo (el paso frágil)

| Paso | Qué hace |
| --- | --- |
| **Extracción** | Un modelo lee cada sección y propone tripletas |
| **Normalización** | Lleva los nombres a su forma oficial: artículos, alias y relaciones invertidas |
| **Validación** | Rechaza lo que no cumple reglas: relación permitida, entidades conocidas, tipos coherentes, evidencia en el texto de la sección |
| **Procedencia** | Cada hecho se guarda con su sección y oración |
| **Medición** | Se compara con un grafo de referencia hecho a mano: **precisión** (¿qué fracción de lo extraído es correcto?) y **recall** (¿qué fracción de lo correcto se extrajo?) |

- La validación **sube la precisión**, pero **no recupera** lo que el modelo no vio: si faltan aristas, los recorridos se cortan.
- Una **arista falsa pero bien formada** no la atrapa ninguna regla y **contamina** todos los recorridos que pasan por ella.
- **Resolución de entidades:** que "SSO" y "Servicio de autenticación" sean el mismo nodo es parte del problema de mantener el grafo.

### 6.3. Búsqueda local

Responde preguntas sobre **cosas concretas**.

1. **Enlazar entidades:** identificar en la pregunta las entidades que existen en el grafo.
2. **Recorrer:** seguir aristas desde ellas un número de **saltos**.
3. **Entregar:** al modelo, una lista de **hechos con su procedencia** y los fragmentos de origen.
4. **Citar y verificar:** el modelo cita hechos y fragmentos; el **código** comprueba que cada cita apunte a algo que existe.

- **Sentido del recorrido:** "¿qué depende de X?" recorre las aristas **al revés**; "¿de qué depende X?" va **hacia adelante**.
- **Saltos:** pocos dejan cadenas cortas; muchos traen ruido.
- Si la pregunta **no nombra ninguna entidad** del grafo, no hay punto de partida.
- **En una base relacional**, "quién depende de X, directa o indirectamente" es una **consulta recursiva** (CTE recursiva). Una **base de grafos** compensa con millones de aristas, consultas de patrones complejos o algoritmos de grafos habituales.

### 6.4. Búsqueda global

Responde preguntas sobre **todo el corpus** (grupos, agregaciones), que un buscador de los *k* mejores nunca ve completo.

1. **Comunidades:** grupos de nodos muy conectados entre sí, detectados con **Louvain** (Blondel et al., 2008) o **Leiden** (Traag et al., 2019).
2. **Resumen:** un modelo escribe un resumen por comunidad (una llamada por comunidad, al indexar).
3. **Mapa:** cada resumen aporta una respuesta parcial.
4. **Reducción:** las respuestas parciales se combinan en la final.

Es la parte más cara y más frágil: su calidad depende de los resúmenes y un modelo pequeño los puede escribir con errores que se propagan.

### 6.5. Modo híbrido y límite del grafo

- **Híbrido:** hechos del grafo y fragmentos vectoriales en **un solo contexto numerado**. Mejor cobertura; más tokens y más ruido.
- **Recuperar no es razonar:** que los hechos correctos estén en el prompt no garantiza que el modelo los encadene. Contar, unir o calcular el cierre de una dependencia conviene hacerlo con **código sobre el grafo**.

### 6.6. Cuándo sí y cuándo no

| Aporta cuando | No compensa cuando |
| --- | --- |
| Preguntas relacionales o de varios saltos | Preguntas directas (el vector ya acierta) |
| Entidades muy conectadas (CMDB, contratos, cadena de suministro) | Corpus pequeño que cabe en el contexto |
| Hay que explicar el camino de la respuesta | Documentos que cambian a diario |
| Inventario de entidades estable | No se puede medir la extracción |

**Costos que una demostración no muestra:**

| Costo | Qué es |
| --- | --- |
| **Indexar** | Una o más llamadas por fragmento para extraer y una por comunidad para resumir |
| **Mantener** | Volver a extraer y resolver entidades cuando cambia un documento |
| **Validar** | La extracción es el paso más frágil y el menos medido |

### 6.7. Implementaciones de referencia (según su documentación)

| Proyecto | Descripción |
| --- | --- |
| **Microsoft GraphRAG** (Edge et al., 2024, *From Local to Global*) | Divide el corpus en unidades de texto; extrae entidades, relaciones y afirmaciones; agrupa el grafo jerárquicamente con **Leiden**; genera informes por comunidad. Modos de consulta: **global**, **local**, **DRIFT** (local con información de comunidades) y **básico** (búsqueda vectorial top-*k*) |
| **LightRAG** (HKUDS; EMNLP 2025; licencia MIT) | Extrae entidades y relaciones con **descripciones**; guarda un grafo y vectores de fragmentos, entidades y relaciones (diseño de dos capas). Modos: *naive*, *local*, *global*, *hybrid* y *mix*. Permite actualizar sin reprocesar todo el corpus |

Ambos evolucionan rápido: conviene consultar su documentación vigente.

## 7. Alucinaciones, fidelidad y corrección

- **Alucinación:** respuesta plausible pero **no respaldada** por las fuentes. En un RAG tiene causas identificables.

| Causa | Mitigación |
| --- | --- |
| Recuperación pobre | Mejorar la recuperación |
| Prompt sin salida de "no sé" | Regla explícita de abstención |
| Pregunta parecida pero ausente | Abstención; no extrapolar |
| Contexto obsoleto o contradictorio | Vigencia en metadatos y filtro al buscar |
| El modelo ignora el contexto | Mejor modelo; exigir citas |
| Extrapolar ("podemos inferir…") | Prohibirlo en el prompt |

- **Umbral de similitud:** descarta lo ajeno, pero las preguntas parecidas pero ausentes puntúan en el mismo rango que las legítimas; ningún umbral las separa bien.
- **Trampa de la medición:** un modelo que dice "no sé" a todo no alucina nunca y no sirve. Se miran **juntos** el "no sé" indebido y las alucinaciones.
- **Fidelidad (*groundedness*):** todo lo dicho está respaldado por el contexto. **Corrección:** lo dicho es verdad. Una respuesta puede ser **fiel e incorrecta** si el contexto estaba desactualizado; la mitigación está en los **datos** (vigencia y filtros), no en el prompt.
- **Comprobaciones por código:** dato clave presente, citas válidas, **cifras con respaldo** (un número que no aparece en el contexto delata una invención) y abstención correcta. Son el piso, no el techo.

## 8. El juez LLM y su calibración

- **Juez LLM (*LLM-as-judge*):** otro modelo que evalúa respuestas con una **rúbrica** y devuelve un veredicto estructurado.
- **Buenas prácticas:** un criterio por juez, rúbrica explícita, salida estructurada validada y prompt versionado.

| Sesgo | Efecto |
| --- | --- |
| Indulgencia | Aprueba todo; deja pasar alucinaciones |
| Severidad | Rechaza de más |
| Longitud | Favorece respuestas largas |
| Autopreferencia | Favorece a modelos parecidos a él |
| Sensibilidad a la rúbrica | Cambia de veredicto al cambiar una frase |

- **Calibración:** compararlo con **etiquetas humanas** (con casos trampa) usando **exactitud** y **kappa de Cohen** (acuerdo descontando el azar: 1 es perfecto, 0 es azar).
- **El modelo del juez pesa más que su prompt**; se recalibra al cambiar de modelo o de rúbrica.
- **Marcos de evaluación** de código abierto (RAGAS, TruLens, DeepEval, Phoenix): traen métricas y jueces hechos; por dentro son conteos o prompts de juez, así que heredan sus sesgos.

## 9. Producción

| Tema | Idea | Cuidado |
| --- | --- | --- |
| **Caché exacta** | Misma pregunta, misma respuesta | Casi nunca acierta con lenguaje natural |
| **Caché de prompt** | El proveedor reutiliza el procesamiento del comienzo del prompt | Lo estable va primero y la pregunta al final |
| **Caché semántica** | Devuelve la respuesta de una pregunta parecida ya respondida | **Falso acierto**: una pregunta casi igual que pide otra cosa recibe la respuesta equivocada; se calibra con umbral alto midiendo aciertos y falsos aciertos |
| **Actualización incremental** | Hash del contenido de cada fragmento; solo se vectoriza lo nuevo o modificado | Invalidar las cachés al cambiar el documento |
| **Ciclo de vida del índice** | Un cambio de modelo o de fragmentado se prueba en un índice nuevo al lado (despliegue azul/verde) | Medir con el conjunto de oro antes de cambiar el tráfico |
| **Observabilidad** | Por petición: versiones de prompt, modelo e índice; fragmentos con puntajes; tiempos por etapa; tokens y costo; señales de calidad | Privacidad de lo registrado |

Analogías: caché semántica = CDN para preguntas; actualización incremental = `rsync` frente a `scp` completo.

## 10. La implementación de referencia: `rag-avanzado`

Proyecto del repositorio `rag-playground` (FastAPI + interfaz web; PostgreSQL + pgvector). Es de un solo turno y de pipeline fijo.

| Concepto | Dónde se ve |
| --- | --- |
| Encabezado contextual, recuperación contextual | `ingest.py`, `rag.py`; pestaña Ingesta |
| Descomposición + RRF | `rag.py`, `fusion.py`; casilla Descomponer |
| Grafo curado y extraído; normalización, validación y medición | `graph.py`; pestaña Grafo |
| Búsqueda local con CTE recursiva | `graphstore.py`, modo `graph_local` |
| Búsqueda global con comunidades | `graph.py`, modo `graph_global` |
| Modo híbrido | modo `hybrid_graph` |
| Citas verificadas | `rag.py` |
| Cobertura del contexto y evaluación por configuración | `evaluation.py`; pestaña Evaluación |
| Trazas por etapa | Panel "Cómo se buscó" |

**No muestra** (solo teoría): *small-to-big*, HyDE, SPLADE, ColBERT/ColPali, Matryoshka y cuantización, Recall@K/Precision@K/MRR sobre un conjunto de oro de recuperación, alucinaciones provocadas, juez LLM, caché semántica ni actualización incremental.

## 11. Glosario rápido

| Término | Definición |
| --- | --- |
| **RAG** | Buscar fragmentos en tus documentos y dárselos al modelo para que responda citando |
| **Golden set** | Preguntas con respuesta correcta escrita por una persona, para medir |
| **Recall@K / Precision@K / MRR** | Cobertura, limpieza y posición del primer resultado útil |
| **Cobertura del contexto** | Fracción de lo esperado que llegó al prompt |
| **Descomposición** | Separar una pregunta compuesta en subpreguntas |
| **RRF** | Fusión de listas por posición |
| **HyDE** | Buscar con una respuesta hipotética |
| **Contextual retrieval** | Anteponer a cada fragmento una frase que lo sitúa en el documento |
| **Small-to-big** | Buscar pequeño, entregar grande |
| **SPLADE / ColBERT / ColPali** | Disperso aprendido / interacción tardía / interacción tardía visual |
| **Matryoshka / cuantización** | Vectores truncables / menos bits por número |
| **Grafo de conocimiento** | Entidades (nodos) y relaciones (aristas) |
| **Tripleta** | (sujeto, relación, objeto) |
| **Procedencia** | Sección y oración de donde salió un hecho |
| **Búsqueda local / global** | Recorrer vecinos de entidades / usar resúmenes de comunidades |
| **Comunidad** | Grupo de nodos muy conectados |
| **Mapa-reducción** | Respuestas parciales y combinación final |
| **CTE recursiva** | Consulta SQL que se llama a sí misma; sirve para recorrer relaciones |
| **Fidelidad / corrección** | Respaldado por el contexto / verdadero |
| **Juez LLM** | Modelo que evalúa con una rúbrica |
| **Kappa de Cohen** | Acuerdo descontando el azar |
| **Caché semántica / falso acierto** | Reutilizar respuesta de pregunta parecida / servir la respuesta de otra pregunta casi igual |

## 12. Preguntas de repaso

1. ¿Qué tres condiciones justifican una mejora en un RAG avanzado?
2. ¿Por qué se miden por separado la recuperación y la generación?
3. Un buscador devuelve `[A, X, Y, B]` y se necesitaba `{A, B}`: ¿cuánto vale el Recall@2 y la Precision@4?
4. ¿Por qué RRF no usa puntajes?
5. ¿Qué ataca cada una de las tres palancas?
6. ¿Qué diferencia hay entre una tripleta que la validación rechaza y una arista falsa bien formada?
7. ¿Qué sentido de recorrido responde "¿qué depende de X?" y cuál "¿de qué depende X?"?
8. ¿Cuándo no compensa GraphRAG? Da dos casos.
9. ¿Qué significa "recuperar no es razonar" y qué se hace al respecto?
10. ¿Puede una respuesta ser fiel e incorrecta? ¿Cómo se mitiga?
11. ¿Qué mide el kappa de Cohen que no mida la exactitud?
12. ¿Qué es un falso acierto en una caché semántica?
