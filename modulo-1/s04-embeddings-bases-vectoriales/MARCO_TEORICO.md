# Marco teórico · Sesión 4: Embeddings y bases de datos vectoriales

> Documento de apoyo al notebook `s04_embeddings_bases_vectoriales.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: el código y los experimentos están en el notebook.

## Índice

1. [Embedding](#1-embedding)
2. [Espacio vectorial y dimensión](#2-espacio-vectorial-y-dimensión)
3. [Medidas de similitud](#3-medidas-de-similitud)
4. [Búsqueda semántica vs. por palabras clave](#4-búsqueda-semántica-vs-por-palabras-clave)
5. [Modelos de embeddings: reglas prácticas](#5-modelos-de-embeddings-reglas-prácticas)
6. [Evaluación de un buscador](#6-evaluación-de-un-buscador)
7. [Base de datos vectorial](#7-base-de-datos-vectorial)
8. [Búsqueda exacta vs. aproximada (ANN)](#8-búsqueda-exacta-vs-aproximada-ann)
9. [Neon y pgvector](#9-neon-y-pgvector)
10. [Pinecone](#10-pinecone)
11. [Cómo elegir entre opciones](#11-cómo-elegir-entre-opciones)
12. [Glosario rápido](#12-glosario-rápido)

---

## 1. Embedding

- **Embedding (incrustación):** **vector numérico** (lista de números) que representa el **significado** de un texto (o imagen, audio…). Textos con significado parecido producen vectores cercanos.
- **Modelo de embeddings:** modelo especializado en convertir texto en vectores. Es distinto de un modelo generativo: no escribe texto, solo produce representaciones.
- **Semántica:** relativa al significado. Un embedding captura similitud *semántica*, no coincidencia literal de palabras.
- **Representación densa:** vector donde casi todos los valores son distintos de cero y cada dimensión mezcla rasgos aprendidos (opuesto a las representaciones dispersas, como el conteo de palabras).
- **Intuición:** imagina un mapa donde cada frase es un punto; frases con ideas parecidas quedan juntas, aunque usen palabras diferentes (por ejemplo "no puedo entrar a mi cuenta" y "olvidé mi contraseña").
- **Reducción de dimensionalidad / proyección a 2D:** técnicas (PCA, t-SNE, UMAP) para dibujar en una pantalla vectores de cientos de dimensiones; sirve para visualizar, pero **pierde información**.

## 2. Espacio vectorial y dimensión

- **Vector:** lista ordenada de números; geométricamente, un punto o una flecha en un espacio.
- **Dimensión:** cantidad de números del vector (p. ej. 768, 1024, 1536, 3072). **Es fija para cada modelo**: un texto de 3 palabras y uno de 3 páginas producen vectores del mismo tamaño.
- **Espacio de embeddings:** el "mapa" donde viven los vectores de un modelo concreto.
- **Más dimensiones:** potencialmente más matices, pero **más espacio de almacenamiento, más memoria y búsquedas más lentas**.
- **Tamaño en disco (estimación):** `n.º de vectores × dimensiones × bytes por número` (un número en coma flotante de 32 bits ocupa 4 bytes).
- **Float32:** número de coma flotante de 32 bits, el formato habitual de los vectores.
- **Normalización:** escalar un vector para que tenga longitud 1; simplifica y acelera el cálculo de la similitud coseno.

## 3. Medidas de similitud

- **Similitud coseno:** mide el **ángulo** entre dos vectores, ignorando su longitud. Va de −1 a 1 (en la práctica con textos suele estar entre 0 y 1); más cerca de 1 = más parecidos. Es la medida más usada en búsqueda semántica.
- **Distancia coseno:** `1 − similitud coseno`; a menor distancia, más parecidos.
- **Distancia euclidiana (L2):** distancia "en línea recta" entre dos puntos.
- **Producto punto (*inner product*):** suma de los productos de componentes; si los vectores están normalizados equivale al coseno.
- **Operadores de pgvector:** `<->` (L2), `<#>` (producto interno negativo) y `<=>` (distancia coseno).
- **Top-k:** los *k* resultados más parecidos a la consulta.
- **Umbral de similitud:** valor mínimo exigido para aceptar un resultado.

## 4. Búsqueda semántica vs. por palabras clave

- **Búsqueda por palabras clave (léxica):** encuentra documentos que **contienen** los términos buscados (p. ej. BM25, `LIKE`, índices de texto completo). Falla con sinónimos, paráfrasis y errores ortográficos.
- **Búsqueda semántica:** compara **significado** mediante embeddings; encuentra resultados relevantes aunque no compartan palabras.
- **Búsqueda híbrida:** combina léxica y semántica para obtener lo mejor de ambas.

Proceso de un buscador semántico:

1. **Vectorizar todos los documentos** (una vez, por adelantado) → *indexación*.
2. **Vectorizar la pregunta** cuando llega.
3. **Calcular la similitud** con cada documento y devolver los *k* más parecidos.

## 5. Modelos de embeddings: reglas prácticas

1. **No se pueden mezclar modelos.** Los vectores de dos modelos viven en "mapas" distintos; compararlos carece de sentido (y si las dimensiones difieren, ni siquiera es posible).
2. **La base vectorial se crea con una dimensión fija.** Si cambias de modelo, debes **volver a vectorizar todo** y recrear el índice.
3. **Algunos modelos exigen prefijos** distintos para documentos y consultas (p. ej. `search_document:` / `search_query:`). Olvidarlos baja la calidad.
4. **Elige un modelo adecuado a tu idioma:** un modelo multilingüe o entrenado en español rinde mejor con textos en español.

Otros términos:

- **Embedding de consulta vs. de documento:** algunos modelos los generan con instrucciones o prefijos diferentes (asimetría).
- **Modelo local vs. API:** los embeddings pueden generarse con un servicio de pago o con un modelo que corre en tu equipo.
- **Costo de embeddings:** se cobra por tokens procesados; es muy inferior al de los modelos generativos, pero no es cero.
- **Fragmento a vectorizar:** qué texto se convierte en vector (título, contenido, ambos) afecta el resultado; debe medirse.

## 6. Evaluación de un buscador

La calidad de una búsqueda se **mide con números**, no a ojo:

- **Conjunto de evaluación:** preguntas con el documento correcto esperado.
- **Hit@1:** proporción de preguntas en que el documento correcto aparece **en el primer resultado**.
- **Hit@3 (o Hit@k):** proporción en que el correcto aparece **entre los 3 (k) primeros**.
- **Recall@k:** fracción de los documentos relevantes que logra recuperar entre los *k* primeros.
- **Precisión@k:** fracción de los *k* resultados que son realmente relevantes.
- **Latencia de búsqueda:** tiempo en devolver resultados; se mide al comparar almacenes.

## 7. Base de datos vectorial

- **Base de datos vectorial:** sistema que **almacena vectores junto con sus metadatos y permite buscar los más cercanos a una consulta** de forma eficiente y a escala.
- **Por qué existe:** comparar la consulta contra **todos** los vectores (*fuerza bruta*) es correcto pero su costo crece linealmente con el tamaño de la colección; con millones de vectores resulta lento.
- **Metadatos (*metadata*):** datos adicionales asociados a cada vector (fuente, fecha, categoría, autor…).
- **Filtro por metadatos:** restringir la búsqueda a los vectores que cumplen condiciones (por ejemplo, solo documentos de cierta categoría).
- **Colección / índice / tabla:** contenedor de vectores del mismo tipo y dimensión (el nombre depende del producto).
- **Upsert:** insertar un registro o actualizarlo si ya existe.
- **Persistencia:** que los datos sobrevivan al reinicio del programa (a diferencia de un almacén en memoria).
- **Almacén en memoria:** estructura temporal (por ejemplo, una matriz en RAM); útil para aprender o prototipar, no para producción.
- **Escalabilidad:** capacidad de crecer en volumen y en consultas manteniendo el rendimiento.

## 8. Búsqueda exacta vs. aproximada (ANN)

- **Búsqueda exacta (*exact / brute force / kNN*):** compara con todos los vectores; resultado perfecto pero lento a gran escala.
- **ANN (*Approximate Nearest Neighbors*, vecinos más cercanos aproximados):** usa un **índice** que sacrifica un poco de exactitud a cambio de gran velocidad.
- **Recall (de un índice ANN):** porcentaje de los vecinos verdaderos que el índice logra devolver; el equilibrio es **velocidad vs. precisión**.
- **HNSW (*Hierarchical Navigable Small World*):** índice basado en un **grafo en capas**; se navega de lo general a lo específico para llegar rápido a los vecinos. Muy rápido y con buen recall, pero consume más memoria y tarda más en construirse.
- **IVFFlat (*Inverted File*):** agrupa los vectores en **clústeres** (listas) y, al buscar, solo examina los más cercanos a la consulta. Más ligero, pero requiere entrenar con datos y su recall depende de cuántas listas se revisan.
- **Parámetros de índice:** `m` y `ef_construction` (HNSW, calidad del grafo), `ef_search` (HNSW, esfuerzo al consultar), `lists` y `probes` (IVFFlat).
- **`EXPLAIN`:** comando SQL que muestra el plan de ejecución, útil para comprobar si una consulta **usa el índice**.

## 9. Neon y pgvector

- **PostgreSQL:** base de datos relacional de código abierto, muy extendida.
- **pgvector:** **extensión** de PostgreSQL que añade el tipo `vector`, operadores de distancia e índices (HNSW, IVFFlat). Permite guardar embeddings junto a los datos relacionales y usar SQL.
- **Neon:** servicio de PostgreSQL **en la nube, serverless**, con pgvector disponible.
- **Serverless / escala a cero:** la base se **suspende cuando no hay actividad** y se reactiva al recibir consultas. Ventaja: ahorro de costos. Inconveniente: la primera consulta tras la pausa puede ser más lenta (*arranque en frío*).
- **Rama (*branch*):** copia ligera de la base para pruebas, característica de Neon.
- **Cadena de conexión (*connection string*):** URL con usuario, contraseña, host y base de datos para conectarse; es un secreto, igual que una API key.
- **SQL:** lenguaje estándar para consultar bases relacionales.
- **Ventaja del enfoque:** una sola base para datos transaccionales y vectores, con filtros SQL, `JOIN`s y transacciones.
- **Plan gratuito:** con límites de almacenamiento y cómputo (verificar siempre la página de precios vigente).

## 10. Pinecone

- **Pinecone:** servicio **gestionado y especializado** en bases de datos vectoriales; no hay servidor que administrar.
- **Index (índice):** contenedor principal de vectores con una dimensión y métrica fijadas.
- **Namespace:** partición **lógica dentro de un índice**; permite separar datos (por cliente, entorno o fuente) y consultar solo uno.
- **Record (registro):** un vector con su ID y sus metadatos.
- **Serverless:** modalidad en la que se paga por uso y almacenamiento, sin dimensionar servidores.
- **Consistencia eventual:** tras escribir (*upsert*), los datos pueden **tardar un instante en ser visibles** en las búsquedas. No es un error: hay que esperar o reintentar al verificar.
- **Diferencia index vs. namespace:** el índice define la dimensión y la métrica; los namespaces solo dividen los datos dentro de ese índice.
- **Plan gratuito:** con límites de almacenamiento, lecturas y escrituras (verificar la página de precios vigente).

## 11. Cómo elegir entre opciones

| Opción | Cuándo conviene | Consideraciones |
| --- | --- | --- |
| **En memoria** | Aprender, prototipos, pocos documentos | No persiste; no escala |
| **Neon + pgvector** | Ya usas PostgreSQL; necesitas filtros SQL, `JOIN`s y transacciones; volumen moderado | Tú gestionas índices y consultas; arranque en frío si escala a cero |
| **Pinecone** | Vectores a gran escala, poca operación, búsqueda como servicio | Dependencia de un proveedor; modelo de costos propio |

Otras opciones: Chroma, Qdrant, Weaviate, Milvus, FAISS (librería), OpenSearch/Elasticsearch con vectores, entre otras.

Conceptos que se verán pronto:

- **Búsqueda híbrida** (léxica + semántica) y **reranking** (reordenar los mejores resultados con un modelo más preciso).
- **Fragmentación (*chunking*):** partir documentos largos en trozos antes de vectorizar (Sesión 5).

## 12. Glosario rápido

| Término | Definición breve |
| --- | --- |
| **Embedding** | Vector que codifica significado |
| **Dimensión** | Longitud del vector; fija por modelo |
| **Similitud coseno** | Ángulo entre vectores; cercano a 1 = parecidos |
| **Búsqueda semántica** | Búsqueda por significado, no por palabras |
| **Top-k** | Los *k* resultados más cercanos |
| **Hit@k** | % de aciertos con el correcto entre los *k* primeros |
| **Base vectorial** | Almacén optimizado para buscar vectores cercanos |
| **ANN** | Búsqueda aproximada de vecinos cercanos |
| **HNSW** | Índice de grafo en capas, rápido y preciso |
| **IVFFlat** | Índice por clústeres |
| **pgvector** | Extensión de PostgreSQL para vectores |
| **Neon** | PostgreSQL serverless en la nube |
| **Pinecone** | Base vectorial gestionada |
| **Namespace** | Partición lógica dentro de un índice de Pinecone |
| **Metadatos** | Datos asociados a cada vector, usables como filtro |
| **Consistencia eventual** | Los datos escritos tardan en ser visibles |
| **Upsert** | Insertar o actualizar |
