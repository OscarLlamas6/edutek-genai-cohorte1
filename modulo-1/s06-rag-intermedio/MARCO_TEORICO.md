# Marco teórico · Sesión 6: RAG intermedio, chunking y recuperación

> Documento de apoyo al notebook `s06_rag_intermedio.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: el código y los experimentos están en el notebook y en la práctica.

## Índice

1. [De un RAG simple a uno intermedio](#1-de-un-rag-simple-a-uno-intermedio)
2. [Ingesta y parsing de documentos](#2-ingesta-y-parsing-de-documentos)
3. [OCR](#3-ocr)
4. [Fragmentación (chunking)](#4-fragmentación-chunking)
5. [Búsqueda léxica: BM25](#5-búsqueda-léxica-bm25)
6. [Búsqueda híbrida y fusión RRF](#6-búsqueda-híbrida-y-fusión-rrf)
7. [Metadatos y filtros](#7-metadatos-y-filtros)
8. [Reranking](#8-reranking)
9. [Reescritura y expansión de la consulta](#9-reescritura-y-expansión-de-la-consulta)
10. [Evaluar la recuperación](#10-evaluar-la-recuperación)
11. [Glosario rápido](#11-glosario-rápido)

---

## 1. De un RAG simple a uno intermedio

El RAG de la Sesión 5 hacía un solo recorrido: fragmentar un documento ordenado, vectorizar, buscar los *k* más parecidos y generar con citas. Un RAG intermedio añade piezas **antes** y **durante** la búsqueda, sin cambiar la idea central: el LLM solo ve lo que la recuperación le entrega.

| Problema real | Pieza que lo ataca |
| --- | --- |
| Los documentos no son Markdown (Word, web, PDF) | Ingesta multiformato; OCR |
| Los fragmentos cortan o diluyen las ideas | Estrategias de chunking |
| La búsqueda por significado falla con identificadores exactos | Búsqueda léxica (BM25) e híbrida |
| La respuesta depende de quién pregunta | Metadatos y filtros |
| El fragmento correcto aparece en el puesto 5 y solo se envían 3 | Reranking |
| La pregunta no se entiende sola ("¿y los domingos?") | Reescritura de la consulta |

Ideas centrales:

- **La calidad del RAG se decide casi siempre antes del LLM**: en la ingesta y en la recuperación.
- **Cada mejora se mide.** Una técnica que añade un LLM suma latencia, costo y un punto de falla; se queda solo si los números lo justifican.
- **Sigue siendo un flujo fijo, no un agente**: ninguna de estas piezas decide por sí misma qué herramienta usar.

## 2. Ingesta y parsing de documentos

- **Ingesta:** proceso que convierte documentos en fragmentos vectorizados guardados en un almacén: extraer, fragmentar, vectorizar, guardar.
- **Parsing (extracción):** obtener el texto, y su **estructura**, a partir de un formato de archivo.
- ***Garbage in, garbage out*:** si la extracción mete basura, todo lo posterior trabaja con basura.

| Formato | Qué hay que cuidar al extraer |
| --- | --- |
| **Markdown** | Casi nada: ya tiene estructura (encabezados). |
| **HTML** | Descartar lo que no es contenido (scripts, estilos, menús de navegación, pie de página, anuncios) y conservar títulos, párrafos y listas. |
| **DOCX** | Respetar los **estilos de título** (la estructura del documento) y convertir las **tablas** en texto. |
| **PDF con texto** | El PDF guarda letras en posiciones, no párrafos ni tablas: las líneas se cortan por el ancho de la hoja y **las tablas se aplanan**. Se suele tomar una sección por página. |
| **PDF escaneado** | No tiene texto: es una imagen. Requiere OCR. |

Conceptos relacionados:

- **Sección:** unidad lógica de un documento (un título con su texto, o una página de un PDF). Es la base sobre la que se fragmenta.
- **Estructura:** la jerarquía de títulos. Conservarla permite citar ("Parte B › Compensaciones") y etiquetar con metadatos.
- **Tabla aplanada:** una tabla convertida en texto corrido; la información sigue, pero pierde filas y columnas, y un corte de fragmento puede separar un dato de su encabezado.
- **Extracción con *document AI*:** servicios o librerías que reconocen tablas y diseño, o modelos con visión que leen la página como imagen. Son más precisos con documentos complejos y más caros.
- **Aviso de lectura:** un buen lector avisa cuando no pudo extraer algo (por ejemplo, una página sin texto) en lugar de ingerir un documento vacío en silencio.

## 3. OCR

- **OCR (*Optical Character Recognition*):** convertir una imagen de texto en texto.
- Se necesita con **escaneos, fotos de documentos y PDFs generados como imagen**.

| Opción | Cuándo conviene | Precauciones |
| --- | --- | --- |
| **Motor local de código abierto** (p. ej. Tesseract) | Texto impreso limpio; privacidad total | Rinde mal con fotos torcidas, manuscritos y tablas |
| **OCR de la nube** | Volumen, tablas y formularios | Costo por página; los documentos salen de tu red |
| **Modelo con visión** | Documentos complejos | Más caro por página; puede equivocarse con cifras, por lo que hay que verificar |

Ideas clave:

- Un escaneo ingerido **sin OCR no existe** para el RAG, y nadie lo notará si no se detecta: contar las páginas sin texto en cada ingesta es la forma más simple de descubrirlo.
- El OCR comete errores (una "l" por un "1", un decimal perdido): para datos críticos conviene validar o conservar el documento original como fuente de la cita.

## 4. Fragmentación (chunking)

El **fragmento** (*chunk*) es la **unidad de búsqueda** (lo que se vectoriza y se compara con la pregunta) y la **unidad de contexto** (lo que se copia al prompt y cuesta tokens). Tiene dos presiones opuestas:

| Si el fragmento es… | Al buscar | Al responder |
| --- | --- | --- |
| **Pequeño** | Vector preciso (un solo tema) | Poco contexto; la respuesta puede quedar partida entre dos fragmentos |
| **Grande** | Vector diluido (mezcla temas) | Contexto completo, pero más tokens, más costo y más ruido |

No existe un tamaño correcto universal: **se mide**.

### Estrategias

| Estrategia | Cómo parte | Fortaleza | Debilidad |
| --- | --- | --- | --- |
| **Tamaño fijo** | Ventanas de N palabras (o tokens) sin mirar la gramática | La más simple y rápida | Corta ideas y datos por la mitad |
| **Por oraciones** | Junta oraciones hasta un tamaño y repite las últimas (solape) | Fragmentos legibles; buen punto de partida | El tamaño depende de la longitud de las oraciones |
| **Por sección** | Un fragmento por sección o por página | Respeta la estructura; sin cortes | Las secciones largas diluyen el vector y cuestan más contexto |
| **Recursivo** | Corta por el separador más grande que quepa (párrafo, línea, oración, palabra) y junta vecinos hasta el tamaño | Respeta la estructura cuando puede y se adapta cuando no; muy usado | Requiere elegir separadores y tamaño |
| **Semántico** | Mide la similitud entre oraciones consecutivas y corta donde el tema cambia | Fragmentos coherentes | Más cómputo en la ingesta (vectoriza cada oración); no siempre mejora los resultados |
| **Padre-hijo** | Se busca con fragmentos pequeños (hijos) y se entrega el fragmento grande (padre) al que pertenecen | Precisión al buscar y contexto al responder | Entrega contexto grande; depende de que los padres sean coherentes |

### Conceptos asociados

- **Solape (*overlap*):** repetir el final de un fragmento al inicio del siguiente para no cortar una idea justo en el límite. Demasiado solape duplica información y aumenta el costo.
- **Tamaño de fragmento:** se expresa en palabras o en tokens. Valores habituales: de decenas a pocos cientos de tokens; el límite superior lo marca también el contexto máximo del modelo de embeddings.
- **Metadatos del fragmento:** fuente, sección, posición, tipo de archivo, audiencia, fecha. Permiten citar, filtrar y depurar.
- **Texto indexado con contexto:** anteponer el título de la sección al texto antes de vectorizar ayuda a desambiguar fragmentos sueltos.
- **Costo del contexto:** número de palabras (tokens) que se envían al LLM por consulta. Una estrategia con fragmentos grandes puede "ganar" en aciertos solo porque entrega más texto: por eso se compara también el contexto.

## 5. Búsqueda léxica: BM25

- **Búsqueda léxica:** busca por **palabras exactas**, no por significado.
- **BM25:** algoritmo clásico de búsqueda léxica (usado por Lucene, Elasticsearch y OpenSearch). Puntúa más a un fragmento cuando contiene las palabras de la consulta, sobre todo si son raras y si aparecen varias veces.

Componentes:

| Componente | Qué aporta |
| --- | --- |
| **Frecuencia del término** | Más apariciones, más puntaje, con **rendimientos decrecientes** (parámetro `k1`). |
| **IDF** (*inverse document frequency*) | Las palabras raras en el corpus pesan más que las comunes. Un código como `OPT-1107` pesa mucho; "el" casi nada. |
| **Normalización por longitud** | Evita favorecer a los fragmentos largos (parámetro `b`). |

Procesamiento previo del texto (**tokenización**):

- Pasar a minúsculas y quitar acentos.
- Quitar **palabras vacías** (*stopwords*) como "el", "de", "que".
- Conservar los **códigos completos** (`E-217`, `0800-26337`) además de sus partes.
- **Stemming** (reducir una palabra a su raíz) para juntar "cancelar" y "cancelación". Una versión ingenua (recortar a las primeras letras) funciona sin librerías, pero es tosca y no debe aplicarse a códigos.

| | Búsqueda semántica | BM25 |
| --- | --- | --- |
| Entiende paráfrasis y sinónimos | Sí | No |
| Encuentra un código o número exacto | A veces confunde códigos parecidos | Sí |
| Necesita un modelo de embeddings | Sí | No |
| Costo de indexar | Vectorizar | Casi nulo |

Cada una **cubre el punto ciego de la otra**.

## 6. Búsqueda híbrida y fusión RRF

- **Búsqueda híbrida:** ejecutar la búsqueda léxica y la semántica y **combinar** sus resultados.
- **Problema de las escalas:** el coseno (0 a 1) y el puntaje BM25 (0 a 20 o más) no son comparables; sumarlos no tiene sentido.
- **RRF (*Reciprocal Rank Fusion*):** ignora los puntajes y usa solo la **posición** en cada lista. Cada lista aporta `peso / (c + posición)` a cada fragmento; se suman los aportes y se reordena. Un fragmento bien ubicado en ambas listas sube al primer lugar. La constante `c` (60 por tradición) suaviza la diferencia entre las primeras posiciones.

Ideas clave:

- RRF funciona como una **votación** entre buscadores.
- Los **pesos** de cada buscador importan: con pesos iguales, la híbrida puede quedar **entre** las dos y no superar a la mejor.
- **Sobreajuste:** ajustar los pesos con las mismas preguntas con que se evalúa da resultados optimistas; en serio se separan preguntas de **ajuste** y de **prueba**.
- La híbrida **gana más cuando el corpus crece**, porque la búsqueda semántica empieza a confundir identificadores parecidos.
- Es una función estándar en bases como Elasticsearch, OpenSearch, Weaviate, Pinecone (vectores dispersos y densos) y PostgreSQL (búsqueda de texto completo junto con pgvector).

## 7. Metadatos y filtros

- **Metadatos:** datos sobre cada fragmento (fuente, sección, tipo de archivo, audiencia, fecha, permisos). Se asignan en la **ingesta**.
- **Filtro por metadatos:** restricción que se aplica **dentro** de la búsqueda (solo candidatos con `publico = negocio`), antes de que nada llegue al prompt.

Usos:

| Uso | Ejemplo |
| --- | --- |
| **Calidad** | Dos fragmentos casi idénticos con respuestas distintas según el tipo de cliente; el filtro evita que el LLM mezcle ambas. |
| **Seguridad** | Que cada usuario o inquilino vea solo sus documentos (control de acceso en la recuperación). |
| **Frescura** | Quedarse con la versión vigente de un documento. |
| **Ámbito** | Buscar solo en un manual, un producto o un idioma. |

Reglas importantes:

1. **Filtrar en la búsqueda, no en el prompt.** Pedirle al modelo "no muestres datos de otros clientes" no es un control de acceso: el dato ya llegó al modelo.
2. El filtro lo decide el **código** (a partir de la sesión del usuario), nunca el modelo ni el texto de la pregunta.
3. Los metadatos usados para filtrar se asignan en la ingesta y se **auditan**.
4. Las bases reales (cláusulas `WHERE` en PostgreSQL, parámetro `filter` en Pinecone) aplican el filtro dentro del índice.

## 8. Reranking

El **reranking** añade una **segunda etapa**: después de recuperar muchos candidatos de forma rápida, un modelo más preciso los **reordena** y se envían al prompt solo los mejores.

```
Etapa 1 · Recuperar (rápida, amplia): búsqueda híbrida → N candidatos
Etapa 2 · Reordenar (lenta, fina): modelo que lee pregunta + candidato → los k mejores
```

| | Bi-encoder (embeddings) | Cross-encoder (reranker) |
| --- | --- | --- |
| Cómo lee | Cada texto **por separado**; luego se comparan vectores | Pregunta y fragmento **juntos** |
| Velocidad | Muy rápido (se precalcula) | Lento (una pasada por candidato) |
| Precisión | Buena | Mejor |
| Uso | Buscar entre miles o millones | Reordenar decenas |

Opciones de reranker:

- **Cross-encoder especializado** (modelos tipo `bge-reranker`, o servicios como Cohere, Voyage o Jina): más rápido, barato y fiable.
- **LLM como reranker:** no requiere nada nuevo, pero es más lento y caro, y un modelo pequeño puede ordenar peor que el orden original. Elegir **solo el mejor** candidato suele ser más fiable que ordenar la lista completa.

Limitaciones:

- Solo reordena lo que la etapa 1 trajo: **si el fragmento correcto no está entre los candidatos, no se rescata**. Se sube el número de candidatos y se baja el *k* final.
- Añade **latencia y costo** en el camino crítico de la respuesta.
- Como cualquier paso con LLM, debe **justificarse con mediciones**.

## 9. Reescritura y expansión de la consulta

El usuario no escribe consultas de búsqueda: escribe como habla.

| Técnica | Qué hace | Cuándo ayuda | Riesgo |
| --- | --- | --- | --- |
| **Reescritura de la consulta** (*query rewriting*) | Convierte la última pregunta de una conversación en una consulta **autocontenida** usando el historial ("¿y los domingos?" → "¿a qué hora atienden por teléfono los domingos?") | Preguntas de seguimiento, pronombres, referencias | Perder o inventar un dato del historial |
| **Multi-consulta** (*multi-query*) | Genera varias formas de preguntar lo mismo, busca con todas y fusiona con RRF | El usuario usa palabras distintas a las del documento | Las reformulaciones pueden inventar temas y arrastrar fragmentos irrelevantes |
| **HyDE** (*Hypothetical Document Embeddings*) | Pide al LLM una **respuesta hipotética** y vectoriza esa respuesta en lugar de la pregunta | Preguntas cortas o vagas | Si el modelo inventa, la búsqueda se desvía |
| **Descomposición** | Divide una pregunta compleja en sub-preguntas | Preguntas con varias partes | Más llamadas y más latencia |

Buenas prácticas:

- Reescribir **solo cuando hay historial**.
- **Registrar la consulta reescrita**: cuando el RAG falla, es lo primero que se revisa.
- Pedir al modelo que **conserve códigos y nombres** y que **no responda** la pregunta, solo la reformule.
- Los modelos pequeños hacen estas tareas peor; se evalúa con el modelo que se usará en producción.

## 10. Evaluar la recuperación

Evaluar la recuperación **por separado de la generación** permite saber si el problema es "no encontró el dato" o "lo encontró y respondió mal".

| Métrica | Definición |
| --- | --- |
| **hit@k** | La respuesta correcta aparece entre los *k* primeros resultados. |
| **MRR** (*Mean Reciprocal Rank*) | Promedio de `1 / posición` del primer acierto. 1.0 = siempre primero; 0.5 = suele ser el segundo; 0 = nunca aparece. Premia estar arriba, no solo estar. |
| **Costo del contexto** | Palabras o tokens promedio que se envían al LLM por consulta. |

Conceptos asociados:

- **Conjunto de preguntas de prueba:** preguntas reales o realistas, cada una con la información que debe aparecer en el fragmento correcto (la **clave**). Conviene mezclar tipos: paráfrasis, códigos exactos, tablas, preguntas dependientes del tipo de usuario y preguntas de conversación.
- **Evaluar por tipo de pregunta:** un promedio general oculta dónde falla cada técnica; separar por tipo revela, por ejemplo, que BM25 gana en códigos y pierde en paráfrasis.
- **Ruido estadístico:** con pocas preguntas, una o dos de diferencia no son una señal. Las diferencias grandes y consistentes sí.
- **Sesgo del tamaño:** una métrica que solo pregunta "¿el fragmento contiene la respuesta?" favorece a los fragmentos grandes; por eso se mira junto con el costo del contexto.
- **Sobreajuste a las preguntas de prueba:** ajustar parámetros con las mismas preguntas de evaluación infla los resultados.
- **No determinismo:** los pasos con LLM pueden variar uno o dos aciertos entre corridas.

## 11. Glosario rápido

| Término | Definición breve |
| --- | --- |
| **Ingesta** | Extraer, fragmentar, vectorizar y guardar documentos |
| **Parsing** | Extraer el texto y la estructura de un archivo |
| **OCR** | Convertir una imagen de texto en texto |
| **Chunk / chunking** | Fragmento de un documento / proceso de partir documentos |
| **Overlap (solape)** | Texto repetido entre fragmentos consecutivos |
| **Chunking recursivo** | Cortar por el separador más grande que quepa |
| **Chunking semántico** | Cortar donde cambia el tema según la similitud entre oraciones |
| **Padre-hijo** | Buscar con fragmentos pequeños y entregar el grande al que pertenecen |
| **BM25** | Algoritmo de búsqueda por palabras (frecuencia, IDF, longitud) |
| **IDF** | Qué tan rara es una palabra en el corpus |
| **Stemming** | Reducir una palabra a su raíz |
| **Búsqueda híbrida** | Combinar búsqueda léxica y semántica |
| **RRF** | Fusión de listas usando solo la posición de cada resultado |
| **Metadatos** | Datos sobre cada fragmento (fuente, sección, audiencia…) |
| **Filtro** | Restricción por metadatos aplicada dentro de la búsqueda |
| **Bi-encoder** | Modelo que vectoriza cada texto por separado (embeddings) |
| **Cross-encoder** | Modelo que lee pregunta y fragmento juntos; más preciso y lento |
| **Reranking** | Reordenar los candidatos con un modelo más preciso |
| **Query rewriting** | Reescribir la consulta como una pregunta autocontenida |
| **Multi-query** | Buscar con varias reformulaciones y fusionar |
| **HyDE** | Vectorizar una respuesta hipotética en lugar de la pregunta |
| **hit@k** | Acierto entre los *k* primeros resultados |
| **MRR** | Promedio de 1/posición del primer acierto |
| **Costo del contexto** | Tokens enviados al LLM por consulta |
