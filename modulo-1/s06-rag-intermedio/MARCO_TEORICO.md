# Marco teórico · Sesión 6: RAG intermedio, chunking y recuperación

> Documento de apoyo al notebook `s06_rag_intermedio.ipynb`. Reúne las definiciones y conceptos de la sesión. No contiene código: el código y los experimentos están en el notebook y en la práctica. Las secciones sobre Docling (parsing con *layout*, OCR en la práctica y chunking híbrido) son **opcionales**: Docling no forma parte del entorno base de la sesión. Cada concepto sigue la misma estructura: **qué es, qué hace, para qué sirve, analogía de infraestructura, ejemplo a mano, cuándo no usarlo y errores típicos**.

## Índice

1. [De un RAG simple a uno intermedio](#1-de-un-rag-simple-a-uno-intermedio)
2. [Ingesta y parsing de documentos](#2-ingesta-y-parsing-de-documentos)
3. [OCR](#3-ocr)
4. [Parsing moderno con análisis de layout](#4-parsing-moderno-con-análisis-de-layout)
5. [OCR en la práctica](#5-ocr-en-la-práctica)
6. [Cómo elegir un parser](#6-cómo-elegir-un-parser)
7. [Fragmentación (chunking)](#7-fragmentación-chunking)
8. [Chunking híbrido de Docling](#8-chunking-híbrido-de-docling)
9. [Búsqueda léxica: BM25](#9-búsqueda-léxica-bm25)
10. [Búsqueda híbrida y fusión RRF](#10-búsqueda-híbrida-y-fusión-rrf)
11. [Metadatos y filtros](#11-metadatos-y-filtros)
12. [Reranking](#12-reranking)
13. [Reescritura y expansión de la consulta](#13-reescritura-y-expansión-de-la-consulta)
14. [Evaluar la recuperación](#14-evaluar-la-recuperación)
15. [Glosario ampliado](#15-glosario-ampliado)
16. [Preguntas de repaso con respuesta](#16-preguntas-de-repaso-con-respuesta)

---

## 1. De un RAG simple a uno intermedio

El RAG de la Sesión 5 hacía un solo recorrido: fragmentar un documento ordenado, vectorizar, buscar los *k* más parecidos y generar con citas. Un RAG intermedio añade piezas **antes** y **durante** la búsqueda, sin cambiar la idea central: el LLM solo ve lo que la recuperación le entrega.

| Problema real | Pieza que lo ataca |
| --- | --- |
| Los documentos no son Markdown (Word, web, PDF) | Ingesta multiformato |
| Una tabla de un PDF se aplana, o las columnas se mezclan | Parser con análisis de *layout* (Docling) |
| La página es una imagen (escaneo) | OCR (con sus errores) |
| Los fragmentos cortan o diluyen las ideas | Estrategias de chunking |
| Quieres fragmentos que respeten los encabezados, midan lo que admite el modelo de embeddings y se puedan citar con página | Chunking híbrido de Docling (jerarquía + tokens) |
| La búsqueda por significado falla con identificadores exactos | Búsqueda léxica (BM25) e híbrida |
| La respuesta depende de quién pregunta | Metadatos y filtros |
| El fragmento correcto aparece en el puesto 5 y solo se envían 3 | Reranking |
| La pregunta no se entiende sola ("¿y los domingos?") | Reescritura de la consulta |

### El pipeline completo

```
INGESTA (una vez, o cuando el documento cambia)

 archivo ─► EXTRAER texto ─► OCR (si es ─► FRAGMENTAR ─► vectorizar ─► guardar fragmentos
            + estructura      escaneo)      (estrategia      │        + metadatos + índice BM25
               │                │            y tamaño)       │
               ▼                ▼                ▼
        formatos distintos,  la página es    cortes que cortan
        basura (menús,       una imagen      o diluyen ideas
        pies de página)

CONSULTA (en cada pregunta)

 pregunta ─► REESCRIBIR ─► BUSCAR ───────────────────► FILTRAR ─► REORDENAR ─► prompt ─► LLM ─► respuesta
             (si hay        semántica + BM25 + RRF      por         (rerank)    con         con
             historial)     │                           metadatos                citas       citas
                │           ▼                              │            │
                ▼     códigos Y paráfrasis                 ▼            ▼
          "¿y los     (cada método tiene        quién pregunta   el correcto va
          domingos?"   un punto ciego)          cambia la        5.º y solo
          no se                                 respuesta        enviamos 3
          entiende sola

 MEDIR (hit@k, MRR, contexto) se aplica a TODAS las cajas.
```

Ideas centrales:

- **La calidad del RAG se decide casi siempre antes del LLM**: en la ingesta y en la recuperación.
- **Cada mejora se mide.** Una técnica que añade un LLM suma latencia, costo y un punto de falla; se queda solo si los números lo justifican.
- **Sigue siendo un flujo fijo, no un agente**: ninguna de estas piezas decide por sí misma qué herramienta usar.

**Analogía de infraestructura.** Un sistema de observabilidad: una búsqueda de texto libre (`grep`) encuentra el código exacto pero no sabe que "caída de voltaje" y "voltaje bajo" son lo mismo; una búsqueda semántica sí, pero se pierde con identificadores. Un buen sistema usa las dos, filtra por entorno o equipo (`env=prod`) y deja que un revisor reordene los candidatos antes de actuar.

**Cómo decidir qué piezas activar** (de lo más barato y seguro a lo más caro): ingesta (y, si lo mides necesario, parser con layout / OCR) → chunking → filtros → híbrida → reranking → reescritura / multi-consulta / HyDE. Cada pieza se añade **solo si el síntoma lo pide** y la medición lo confirma.

| Síntoma que ves | Primera pieza a probar | Cómo confirmar |
| --- | --- | --- |
| Faltan documentos o salen con basura | Ingesta / OCR | Muestra del texto extraído; páginas sin texto |
| Las tablas del PDF salen aplanadas o las columnas mezcladas | Parser con *layout* (Docling) | Comparar la tabla extraída con la del PDF; preguntas sobre tablas |
| Un escaneo sale con errores o sin texto | Mejor fuente; OCR revisado | Comparar por palabras con un texto bueno; contar páginas casi vacías |
| Respuestas cortadas o ruido en el contexto | Chunking (o padre-hijo) | hit@k **y contexto** |
| Necesitas citar con página y encabezados, o controlar el tamaño en tokens | Chunking híbrido de Docling | Metadatos del fragmento; tokens medidos con el tokenizador del modelo |
| No encuentra códigos, números o nombres | BM25 / híbrida | Resultados por tipo de pregunta |
| Mezcla respuestas de clientes o versiones | Metadatos y filtros | Preguntas dependientes del tipo de cliente, con y sin filtro |
| El correcto aparece, pero lejos del primer puesto | Reranking | MRR y hit@3 antes y después |
| Falla en preguntas de seguimiento o vagas | Reescritura / multi-consulta | Preguntas conversacionales con y sin reescribir |

## 2. Ingesta y parsing de documentos

- **Ingesta:** proceso que convierte documentos en fragmentos vectorizados guardados en un almacén: extraer, fragmentar, vectorizar, guardar.
- **Parsing (extracción):** obtener el texto, y su **estructura**, a partir de un formato de archivo.
- ***Garbage in, garbage out*:** si la extracción mete basura, todo lo posterior trabaja con basura.
- **Audio y video también son fuentes.** Una charla grabada o un video contienen conocimiento que ningún lector de texto ve. Se **transcriben con Whisper** (voz a texto) y la transcripción entra al mismo pipeline. Se verán en la siguiente sesión.

| Formato | Qué hay que cuidar al extraer |
| --- | --- |
| **Markdown** | Casi nada: ya tiene estructura (encabezados). |
| **HTML** | Descartar lo que no es contenido (scripts, estilos, menús de navegación, pie de página, anuncios) y conservar títulos, párrafos y listas. |
| **DOCX** | Respetar los **estilos de título** (la estructura del documento) y convertir las **tablas** en texto. |
| **PDF con texto** | El PDF guarda letras en posiciones, no párrafos ni tablas: las líneas se cortan por el ancho de la hoja y **las tablas se aplanan**. Se suele tomar una sección por página. |
| **PDF escaneado** | No tiene texto: es una imagen. Requiere OCR (secciones 3 y 5). |

Un **parser con análisis de layout** (sección 4) reconstruye tablas y orden de lectura y trae OCR; es opcional y más pesado.

Conceptos relacionados:

- **Sección:** unidad lógica de un documento (un título con su texto, o una página de un PDF). Es la base sobre la que se fragmenta.
- **Estructura:** la jerarquía de títulos. Conservarla permite citar ("Parte B › Compensaciones") y etiquetar con metadatos.
- **Tabla aplanada:** una tabla convertida en texto corrido; la información sigue, pero pierde filas y columnas, y un corte de fragmento puede separar un dato de su encabezado.
- **Extracción con *document AI*:** servicios o librerías que reconocen tablas y diseño, o modelos con visión que leen la página como imagen. Son más precisos con documentos complejos y más caros. **Docling** es una opción local y de código abierto (sección 4).
- **Aviso de lectura:** un buen lector avisa cuando no pudo extraer algo (por ejemplo, una página sin texto) en lugar de ingerir un documento vacío en silencio.

### Aterrizando la ingesta

- **Qué es.** El camino completo de archivos a fragmentos buscables; el *parsing* es su primer tramo. Hay un lector por formato, y todos entregan la misma estructura: un documento con secciones `(título, texto)`, un tipo y avisos.
- **Qué hace.** Separa el contenido del ruido, conserva la jerarquía de títulos y convierte tablas y páginas en texto utilizable.
- **Para qué sirve.** Es la **puerta de entrada** del RAG: lo que no entra limpio no se puede arreglar después, ni con mejores embeddings ni con mejores prompts.
- **Analogía de infraestructura.** Es el **parser del pipeline de logs** (el *grok* o el *shipper* que da formato a los eventos antes de indexarlos). Si parte un *stack trace* en eventos sueltos o descarta un campo, ningún dashboard ni alerta puede compensarlo.
- **Ejemplo a mano.** Una página web contiene el contenido ("El plan Familiar cuesta 39 dólares"), un menú ("Inicio · Planes · Contacto") y un pie ("Síguenos en redes sociales"). Si el lector no los separa, los tres se vectorizan, y una pregunta sobre "redes" puede traer el pie como respuesta. En el ejemplo de la sesión, la guía técnica pasa de 956 a 839 palabras al descartar el ruido.
- **Cuándo no complicarse.** Con Markdown limpio o texto plano el lector es trivial. Los extractores de tablas, *document AI* o modelos con visión se justifican solo cuando se **mide** que las tablas o el diseño cuestan respuestas.
- **Errores típicos.** (1) Asumir que "se leyó bien" sin mirar una muestra del texto. (2) No tener avisos (páginas vacías, secciones de 0 palabras). (3) Perder la estructura de títulos y luego no poder citar ni filtrar. (4) Ingerir una vez y olvidar que, si el documento cambia, hay que **reingestar**.

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
- El OCR comete errores (una "l" por un "1", un decimal perdido): para datos críticos conviene validar o conservar el documento original como fuente de la cita. La sección 5 muestra errores **medidos** y qué se puede (y qué no) mitigar.

### Aterrizando el OCR

- **Qué es.** Un motor que mira una imagen y devuelve el texto que cree leer, normalmente con un nivel de confianza por palabra.
- **Qué hace.** Preprocesa la imagen (enderezar, limpiar), reconoce las letras y arma líneas y párrafos. Su salida es texto **con posibles errores**.
- **Para qué sirve.** Para que un papel escaneado pase de ser una foto a ser un documento más del RAG.
- **Analogía de infraestructura.** Digitalizar un **runbook que solo existe impreso**: antes de poder buscar en él hay que transcribirlo, y como toda transcripción, hay que revisar las cifras y los códigos.
- **Ejemplo a mano.** Si el OCR lee `E-217` como `E-2l7` (una "l" en lugar de un 1), la búsqueda por código exacto (BM25) **no lo encuentra** y la semántica tampoco: el texto "existe" pero ya no coincide. Con una cifra pasa lo mismo: `39,00` leído como `3900` cambia la respuesta.
- **Cuándo no.** Si el PDF ya tiene capa de texto, no hace falta OCR; usarlo solo añade errores y costo. Se detecta porque el extractor devuelve texto; si devuelve vacío en una página, ahí sí.
- **Errores típicos.** (1) No detectar los escaneos y descubrir meses después que "faltaban" documentos. (2) Confiar ciegamente en las cifras del OCR. (3) Descartar el original: conserva el archivo como fuente de la cita.

## 4. Parsing moderno con análisis de *layout*

### Docling de un vistazo

| | |
| --- | --- |
| **Qué es** | Una librería de **código abierto** para convertir documentos (PDF, Word, PowerPoint, Excel, HTML, imágenes, audio) en un documento **estructurado** que se puede exportar a Markdown o JSON |
| **Quién lo hace** | Nació en **IBM Research Zurich** (equipo *AI for knowledge*), se abrió en 2024 y hoy lo aloja la **LF AI & Data Foundation** |
| **Licencia** | El código es **MIT**. Cada modelo que descarga tiene **su propia licencia**: se revisan antes de usarlo en un producto |
| **Dónde corre** | En tu máquina (local, sin enviar el documento a un servicio), en CPU o GPU |
| **Qué hace** | Análisis de layout, orden de lectura, tablas, OCR para escaneos y chunking por jerarquía y tokens (sección 8) |
| **Qué NO hace** | No vectoriza ni busca (eso sigue siendo tu RAG); no garantiza un OCR perfecto (en la sesión, `Rojas` salió como `Fojas`); en la versión probada su transcripción de audio **no conserva los tiempos** en el documento, por eso para citar `mm:ss` se usa un transcriptor aparte; no es liviano (varios cientos de MB de modelos la primera vez) |
| **Cuándo NO usarlo** | Markdown, HTML o PDF de texto corrido sin tablas ni columnas: un lector simple es más rápido y no instala nada (ver sección 6) |

*Fuente: documentación oficial del proyecto (github.com/docling-project/docling). Las cifras de tiempo y tamaño son las medidas en la sesión.*

- **Análisis de layout** (*document layout analysis*): reconocer, en la **imagen** de una página, las regiones (título, párrafo, lista, tabla, figura, pie de página), su tipo y su **orden de lectura**.
- **Parser con layout** (por ejemplo **Docling**): un convertidor que aplica ese análisis y, si hace falta, OCR, y entrega un documento **estructurado** (jerarquía de títulos, textos, tablas y la página de cada elemento) en lugar de texto plano.
- **Por qué hace falta:** un PDF **no tiene párrafos ni tablas, tiene letras en posiciones**. Un extractor simple las lee en el orden en que se guardaron, y de ahí salen las líneas cortadas, las **tablas aplanadas** y las columnas mezcladas.

**Cómo funciona (en cinco pasos).**

| Paso | Qué hace |
| --- | --- |
| 1. Layout | Un modelo de visión detecta regiones y su tipo |
| 2. Orden de lectura | Ordena las regiones (columnas, notas, pies) como las leería una persona |
| 3. Tablas | Otro modelo reconoce filas, columnas y celdas combinadas |
| 4. OCR | Solo en las regiones que son imagen sin texto |
| 5. Resultado | Un documento con jerarquía, tipos de elemento y página, exportable a Markdown o JSON |

En Markdown, HTML y DOCX no hace falta mirar la página: el formato **ya trae** su estructura y se lee directamente. El análisis de layout y el OCR solo se pagan en **PDF e imágenes**.

### Tres tipos de PDF

| Tipo | Cómo se reconoce | Qué hacer |
| --- | --- | --- |
| **Con texto** (generado por un programa) | El extractor devuelve texto | Extractor directo; parser con layout **sin OCR** si hay tablas o columnas |
| **Escaneado** (una imagen por página) | Cero texto en la página | OCR |
| **Mixto** (páginas con texto y páginas escaneadas, o texto sobre una imagen) | Hay páginas con texto y páginas vacías | Decidir **por página**; OCR solo donde falta el texto |

Un caso que sorprende: hay PDF **con capa de texto** (por ejemplo, informes con diseño tipo presentación) que un parser con layout, sin OCR, trata como imagen y devuelve casi vacíos. La estrategia "primero sin OCR y reintentar con OCR si casi no hay texto" cubre también ese caso.

### Aterrizando el parsing con layout

- **Qué es.** Un lector de documentos que "ve" la página: reconoce regiones y tipos, ordena la lectura y reconstruye las tablas.
- **Qué hace.** Convierte el archivo en un documento estructurado y lo exporta (Markdown, JSON) o lo entrega ya listo para fragmentar, con la **página** y los **encabezados** de cada elemento.
- **Para qué sirve.** Para recuperar lo que un extractor de texto pierde: **tablas con estructura**, **orden de lectura** correcto y texto de **escaneos**; y para citar con página y sección.
- **Analogía de infraestructura.** Pasar de hacer `grep` sobre logs en texto plano a usar un **parser con esquema**: el primero ve líneas; el segundo entiende qué es cada campo. Y el OCR es **reconstruir datos desde una captura de pantalla** de una interfaz que no exporta logs: se puede, con margen de error.
- **Ejemplo a mano.** Una "página" de juguete es una lista de palabras con coordenadas, guardadas columna por columna: `Zona, Norte, Sur, Dia, 5, 12, Horario, 01:00, 00:00`. Leídas en ese orden salen mezcladas. Agrupando por altura (filas) y ordenando por posición horizontal (columnas) reaparece una tabla de 3 filas × 3 columnas. Un parser con layout hace eso mismo, con un modelo de visión que además ve bordes, celdas combinadas y varias columnas. En el boletín de la sesión, `pypdf` entrega la tabla como una sola línea y Docling como una tabla de 4 filas × 3 columnas (encabezado y tres zonas).
- **Cuándo no.** Con Markdown limpio, HTML sencillo o PDF de texto corrido sin tablas ni columnas, los lectores simples son más rápidos y no instalan nada. Docling cuesta CPU, cientos de MB de modelos y una dependencia grande.
- **Errores típicos.** (1) Activar OCR en PDF que ya tienen texto. (2) Dar por buena una tabla sin mirar cómo quedó. (3) No instalar los modelos antes de una demostración. (4) No guardar cómo se leyó cada documento (con o sin OCR) para poder revisarlo.

## 5. OCR en la práctica

La sección 3 explicó qué es el OCR. Aquí, cómo se comporta **de verdad**.

**Motores.**

| Motor | Dónde corre | Comentario |
| --- | --- | --- |
| **RapidOCR** | Local | El que Docling usa por defecto; viene incluido |
| **EasyOCR**, **Tesseract** | Local | Alternativas (Tesseract necesita el programa instalado en el sistema) |
| **OCR de la nube** (Azure, Google, AWS) | Servicio | Mejor con volumen, formularios y tablas; costo por página y el documento sale de tu red |
| **Modelo con visión** | Servicio o local | Para documentos complejos; más caro y puede "corregir" o inventar cifras |

**Errores típicos del OCR.** Confusiones entre letras parecidas: **"OpenAl" en lugar de "OpenAI"** (una "l" minúscula por una "I" mayúscula, que se observó probando el OCR con un informe de diseño tipo presentación), `E-217` leído como `E-2l7`, `Rojas` como `Fojas`; cifras (`39,00` leído como `3900`); signos de puntuación cambiados (`:` por `.`); palabras pegadas (`AX-2200Metros`); mayúsculas alteradas (`Equipo Instalado`); y, con imágenes malas, **texto ilegible o ningún texto**.

**Qué se midió en la sesión** (acta escaneada de ejemplo, que es un escaneo limpio y sintético; el OCR de Docling con RapidOCR; las versiones degradadas se generan con Pillow a partir de la misma página):

| Versión del escaneo | Parecido con el texto bueno | Qué cambió |
| --- | --- | --- |
| Original (OCR de referencia) | — | `AX-2200Metros`: el código quedó pegado a la palabra siguiente, y BM25 ya no tiene el término `ax-2200` |
| Limpia, re-renderizada (control) | 100 % | Nada |
| Leve (2° de inclinación, mitad de píxeles, algo de ruido) | 99 % | Una mayúscula (`Instalado:`) |
| Media (3°, 35 % de los píxeles, ruido y desenfoque) | 98 % | **`Rojas` → `Fojas`**: el apellido ya no se puede buscar |
| Fuerte (4°, 25 % de los píxeles, mucho ruido) | 0 % | **No devuelve texto** |

Lecciones: (1) un "98 % de parecido" suena a casi perfecto y **el 2 % que falla puede ser justo el dato que se pregunta**; (2) mide **qué palabras** cambiaron, no un solo porcentaje; (3) con una resolución muy baja **no hay nada que reconocer**; (4) con BM25, un solo carácter cambiado vuelve invisible un código.

**Mitigaciones.** Enderezar la imagen, ampliarla, mejorar el contraste, o **escanear de nuevo a mayor resolución**. En las pruebas de la sesión, **ninguna fue una receta**: enderezar la versión "media" recuperó `Rojas` pero introdujo `Famáliar`; ampliar ×3 no recuperó `Rojas` y añadió cambios de puntuación; con la versión "fuerte", ampliar o enderezar devolvió **texto ilegible** (parecido de 10 % y 27 %), peor que un vacío porque parece texto y entra al índice. Lo que se perdió al escanear no lo recupera el OCR: **mejora la fuente antes que el pre-proceso**, y mide con tu documento.

**Cuándo se justifica.** Solo cuando la página **no tiene capa de texto** (o el parser la trata como imagen). Estrategia práctica: primero sin OCR (rápido) y reintentar con OCR si el resultado casi no tiene texto; guardar en un metadato que la página salió de OCR.

**Costo.** Local: **cero dinero**, pero tiempo y espacio. Medido en la máquina de pruebas (CPU, sin GPU): la primera conversión de un PDF carga los modelos (7 s el boletín; 3 s el OCR del acta después de eso; 23 s el boletín cuando además se descargaron los modelos), después ≈ 1 a 2 s por PDF pequeño. Los modelos pesan ≈ 500 MB (layout y tablas) más ≈ 60 MB de OCR, y el entorno pasa de ≈ 390 MB a ≈ 1.8 GB con PyTorch (solo CPU). En la nube: costo por página y los documentos salen de tu red.

### Aterrizando el OCR en la práctica

- **Qué es.** El OCR como un paso **con errores medibles**, no como una función que "devuelve el texto".
- **Qué hace.** Produce texto aproximado que hay que validar, con distintos modos de fallo (cambio de letras, palabras pegadas, texto ilegible, nada).
- **Para qué sirve.** Para que un papel sea un documento más del RAG, sabiendo **qué datos pueden estar mal**.
- **Analogía de infraestructura.** Un **OCR es un sensor ruidoso**: los datos llegan, pero con una tasa de error. Igual que no se alerta sobre una métrica sin validar el sensor, no se responde con un código leído por OCR sin validarlo.
- **Ejemplo a mano.** Un escaneo dice "Cliente: Familia Rojas". El OCR devuelve "Familia Fojas". La búsqueda por el apellido "Rojas" deja de coincidir con BM25. El documento "existe", pero esa consulta no lo encuentra.
- **Cuándo no.** PDF con capa de texto; documentos donde basta un resumen (no se necesitan códigos ni cifras exactas); datos críticos que no se pueden validar.
- **Errores típicos.** (1) Confiar en un porcentaje global. (2) No distinguir "ningún texto" de "texto ilegible". (3) Pre-procesar sin medir. (4) Tirar el original. (5) No registrar qué texto salió de OCR.

## 6. Cómo elegir un parser

| Situación | Lector simple (a mano) | Parser con layout (Docling) | Servicio de la nube o modelo con visión |
| --- | --- | --- | --- |
| Markdown, HTML o DOCX sencillos | **Suficiente** | Funciona, aporta poco | Excesivo |
| PDF con texto corrido | **Suficiente** | Opcional | Excesivo |
| PDF con **tablas o columnas** | Tablas aplanadas | **Recomendado** | Si el volumen o la complejidad lo piden |
| PDF **escaneado** de buena calidad | Solo avisa | **OCR incluido** | Para formularios, manuscritos o tablas difíciles |
| Documento **sensible** que no puede salir de tu red | **Sí** | **Sí** (corre local) | No, salvo acuerdo y revisión |
| Entorno **ligero** (sin PyTorch ni modelos) | **Sí** | No | Depende del servicio |
| **Citar con página** y encabezados | Solo "una sección por página" | **Sí** | Depende |

**Reglas para decidir.**
1. Empieza con lo simple y **mide** (muestra del texto extraído, preguntas con tablas).
2. Sube al parser con layout cuando midas que **las tablas, el orden de lectura o los escaneos** cuestan respuestas.
3. Sube a un servicio o modelo con visión cuando el parser local no alcance (manuscritos, formularios complejos) y el documento pueda salir de tu red.
4. Un parser nuevo cambia el **formato** del texto (por ejemplo, cómo se escribe una tabla): **revisa las claves de tu conjunto de pruebas**, que pueden depender del formato del lector anterior.

### Aterrizando la elección del parser

- **Qué es.** Una decisión de ingeniería: costo de complejidad frente a calidad de lo extraído.
- **Para qué sirve.** Para no pagar (en tiempo, dependencias y errores de OCR) por algo que no necesitas, ni quedarte corto cuando lo necesitas.
- **Analogía de infraestructura.** Elegir entre un script de `awk`, un parser con esquema y un servicio gestionado de observabilidad: empiezas por el más simple que cubre el caso y cambias cuando el caso lo exige y lo puedes demostrar.
- **Ejemplo a mano.** Tu corpus tiene 90 % de Markdown y un PDF con una tabla de tarifas. Opciones: lectores simples para todo y Docling solo para ese PDF (lectura por tipo de archivo). No instalas PyTorch en todo el entorno si solo un PDF lo necesita (por eso en la sesión el grupo `docling` es **opcional**).
- **Cuándo no cambiar.** Si los números no mejoran con el parser nuevo y no necesitas sus extras (página, tablas, OCR).
- **Errores típicos.** (1) Elegir por moda. (2) Cambiar de parser sin volver a medir. (3) No medir el costo de tiempo por documento. (4) Olvidar que un parser distinto da un texto distinto (y fragmentos distintos).

## 7. Fragmentación (chunking)

El **fragmento** (*chunk*) es la **unidad de búsqueda** (lo que se vectoriza y se compara con la pregunta) y la **unidad de contexto** (lo que se copia al prompt y cuesta tokens). Tiene dos presiones opuestas:

| Si el fragmento es… | Al buscar | Al responder |
| --- | --- | --- |
| **Pequeño** | Vector preciso (un solo tema) | Poco contexto; la respuesta puede quedar partida entre dos fragmentos |
| **Grande** | Vector diluido (mezcla temas) | Contexto completo, pero más tokens, más costo y más ruido |

No existe un tamaño correcto universal: **se mide**.

### Aterrizando el chunking

- **Qué es.** Partir cada sección en fragmentos, decidiendo dónde cortar y qué repetir (el solape).
- **Qué hace.** Define qué compite en la búsqueda y cuánto texto llega al modelo.
- **Para qué sirve.** Es, después de la ingesta, la decisión que más mueve la calidad de la recuperación y el costo del contexto.
- **Analogía de infraestructura.** Partir **logs multilínea**: si cortas cada N bytes sin mirar el formato, un *stack trace* queda a la mitad y ninguna mitad sirve; si agrupas por evento, cada registro es legible. Y está el tamaño: muy pequeño pierde el contexto de la excepción; muy grande mezcla eventos que no tienen relación.
- **Ejemplo a mano.** Una sección de 200 palabras trata tres temas (precio, instalación, garantía). Como **un solo fragmento**, su vector es una mezcla de los tres y la pregunta "¿cuánto cuesta?" se compara contra esa mezcla; pero si la respuesta cruza precio e instalación, todo está en el mismo trozo. Como **tres fragmentos de ≈65 palabras**, cada vector es de un solo tema y la búsqueda es más precisa, pero una respuesta que cruza dos temas puede quedar partida. En el laboratorio de la sesión, enviar secciones enteras cuesta ≈ 345 palabras de contexto por pregunta frente a ≈ 160 con fragmentos de 70 palabras (más del doble).
- **Cuándo no complicarse.** Con una FAQ donde cada entrada ya es corta y autónoma, cada entrada **es** el fragmento: partirla más solo daña.
- **Errores típicos.** (1) Elegir el tamaño por costumbre. (2) Cortar por caracteres y separar un número de su unidad. (3) Mucho solape (duplica texto y costo) o ninguno. (4) Olvidar los metadatos: sin fuente y sección no se puede citar ni filtrar.

### Estrategias

| Estrategia | Cómo parte | Fortaleza | Debilidad |
| --- | --- | --- | --- |
| **Tamaño fijo** | Ventanas de N palabras (o tokens) sin mirar la gramática | La más simple y rápida | Corta ideas y datos por la mitad |
| **Por oraciones** | Junta oraciones hasta un tamaño y repite las últimas (solape) | Fragmentos legibles; buen punto de partida | El tamaño depende de la longitud de las oraciones |
| **Por sección** | Un fragmento por sección o por página | Respeta la estructura; sin cortes | Las secciones largas diluyen el vector y cuestan más contexto |
| **Recursivo** | Corta por el separador más grande que quepa (párrafo, línea, oración, palabra) y junta vecinos hasta el tamaño | Respeta la estructura cuando puede y se adapta cuando no; muy usado | Requiere elegir separadores y tamaño |
| **Semántico** | Mide la similitud entre oraciones consecutivas y corta donde el tema cambia | Fragmentos coherentes | Más cómputo en la ingesta (vectoriza cada oración); no siempre mejora los resultados |
| **Padre-hijo** | Se busca con fragmentos pequeños (hijos) y se entrega el fragmento grande (padre) al que pertenecen | Precisión al buscar y contexto al responder | Entrega contexto grande; depende de que los padres sean coherentes |
| **Híbrido de Docling** (opcional, sección 8) | Parte por la jerarquía del documento y por un límite de tokens (parte lo que excede, une lo pequeño) | Fragmentos coherentes, citables (encabezados y página) y acotados en tokens | Requiere Docling; no es la búsqueda híbrida |

### Cada estrategia, aterrizada

| Estrategia | Qué hace en detalle | Analogía de infraestructura | Cuándo no usarla |
| --- | --- | --- | --- |
| **Fijo** | Avanza N palabras menos un solape de 10 palabras (con el solape por defecto), sin mirar puntuación | `split -b`: corta por bytes sin respetar líneas | Casi siempre; solo para texto sin ninguna estructura |
| **Oraciones** | Suma oraciones hasta N palabras y repite las últimas *solape* oraciones en el fragmento siguiente | Agrupar líneas de log por evento | Si una sola oración excede N o el texto no tiene puntuación |
| **Sección** | No corta: usa la estructura del documento | Un runbook completo por archivo | Secciones largas: diluyen el vector y encarecen el contexto |
| **Recursivo** | Prueba los separadores en orden (párrafo, línea, oración, palabra) y baja de nivel solo si el trozo no cabe | Rotar logs respetando límites naturales | Con texto limpio y corto da casi lo mismo que oraciones |
| **Semántico** | Vectoriza cada oración, calcula el coseno entre vecinas y corta en el 25 % de uniones más flojas, con tope de N palabras | Detectar el cambio de servicio en un log mezclado | Si importa el costo de la ingesta (una vectorización por oración) o no mejora la medición |
| **Padre-hijo** | Hijos de ≈35 palabras por oración; cada uno guarda su sección como padre; al responder se reemplaza cada hijo por su padre sin repetirlo | Buscar por la alerta y abrir el runbook completo | Si los padres son enormes (el LLM recibe ruido y costo) |

**Ejemplo a mano.** El texto *"El pago vence el día 15. Si no se paga hasta el día 15, se cobra un recargo de 3 dólares. Si sigue sin pagar el día 25, el servicio se suspende."* con ventanas de 18 palabras: el fragmentado **fijo** puede terminar en "… el día 25, el servicio se" (mitad de frase); **oraciones** y **recursivo** terminan en un punto; **padre-hijo** busca por "recargo de 3 dólares" y entrega la sección de pagos completa.

### Aterrizando padre-hijo

- **Qué es.** Un patrón de dos niveles: los hijos (pequeños) se vectorizan y se buscan; el padre (la sección) es lo que se entrega.
- **Qué hace.** Cada hijo guarda el identificador y el texto de su padre. En la consulta se buscan los hijos más parecidos (conviene traer unos 10), se reemplaza cada uno por su padre y se eliminan duplicados.
- **Para qué sirve.** Resuelve la tensión de las dos presiones: precisión al buscar y contexto completo al responder.
- **Analogía.** Una alerta apunta a su runbook; o el índice de un libro: buscas la entrada, lees la página.
- **Ejemplo a mano.** Una sección de pagos de 49 palabras se parte en 4 hijos (de 9 a 14 palabras, uno por oración). La pregunta "¿cuánto es el recargo?" coincide con el hijo del "recargo de 3 dólares"; se entrega la sección completa, que además dice cuándo se suspende y cómo reactivar. Si dos hijos de la misma sección están entre los primeros, el padre se envía **una sola vez**.
- **Cuándo no.** Con secciones gigantes (un libro sin títulos), el "padre" sería todo el libro. Antes hay que definir mejores padres.
- **Errores típicos.** (1) Olvidar que el contexto sube (≈ 364 palabras en el laboratorio). (2) Buscar solo *k* hijos y quedarse con menos de *k* padres tras quitar duplicados. (3) Padres mal formados que arrastran ruido.

### Conceptos asociados

- **Solape (*overlap*):** repetir el final de un fragmento al inicio del siguiente para no cortar una idea justo en el límite. Demasiado solape duplica información y aumenta el costo.
- **Tamaño de fragmento:** se expresa en palabras o en tokens. Valores habituales: de decenas a pocos cientos de tokens; el límite superior lo marca también el contexto máximo del modelo de embeddings.
- **Metadatos del fragmento:** fuente, sección, posición, tipo de archivo, audiencia, fecha. Permiten citar, filtrar y depurar.
- **Texto indexado con contexto:** anteponer el título de la sección al texto antes de vectorizar ayuda a desambiguar fragmentos sueltos.
- **Costo del contexto:** número de palabras (tokens) que se envían al LLM por consulta. Una estrategia con fragmentos grandes puede "ganar" en aciertos solo porque entrega más texto: por eso se compara también el contexto.

## 8. Chunking híbrido de Docling

> ⚠️ **Terminología.** El **chunking híbrido** de Docling (`HybridChunker`) **no es** la **búsqueda híbrida** (BM25 + vectores, sección 10). Comparten la palabra "híbrido" y nada más: el primero decide **dónde cortar** los documentos (en la ingesta); la segunda decide **cómo buscar** (en la consulta). Se pueden usar las dos a la vez.

- **Chunking híbrido:** fragmentar usando **dos criterios combinados**: la **jerarquía del documento** (encabezados y elementos) y un **límite de tokens**.
- **Cómo funciona:** (1) parte de los elementos del documento (párrafo, ítem de lista, tabla) y cada fragmento **lleva los encabezados** que lo contienen y las **páginas** donde está; (2) **parte** los que exceden el máximo de tokens; (3) **une** los vecinos pequeños que comparten encabezados mientras quepan.
- **Token:** unidad en que un modelo cuenta el texto. No es una palabra; cada modelo tiene su **tokenizador**.
- **`contextualize`:** devuelve **encabezados + texto** del fragmento. Es lo que conviene **vectorizar**.
- **Tokenizador coincidente:** el chunker debe contar con el tokenizador del **mismo modelo de embeddings** que usarás; si no, "128 tokens" puede ser más o menos para el modelo real.

### Aterrizando el chunking híbrido

- **Qué es.** Un fragmentador que respeta la estructura del documento y, a la vez, controla el tamaño en tokens.
- **Qué hace.** Genera fragmentos coherentes (no cruzan los límites de sus elementos ni parten tablas que caben), acotados en tokens y con metadatos (`encabezados`, páginas).
- **Para qué sirve.** Para obtener fragmentos **citables** y **del tamaño que el modelo de embeddings admite**, sin escribir un divisor propio.
- **Analogía de infraestructura.** Empaquetar logs por evento con un **tamaño máximo de lote**: respetas los límites naturales (un evento completo, un stack trace entero); el evento enorme se trocea y los eventos diminutos del mismo servicio se agrupan hasta llenar el lote.
- **Ejemplo a mano.** Con "tokens" de juguete (una palabra, un token) y un máximo de 14: dos párrafos cortos de la sección *Pagos* (6 palabras cada uno, más el encabezado) se **unen** en un fragmento de 13 tokens; un párrafo de 26 palabras se **parte** en dos fragmentos de 14; y *Garantía* queda **aparte** (5 tokens) porque tiene otro encabezado, aunque quepa.
- **Cuándo no.** Texto plano sin estructura (la jerarquía no aporta); documentos ya cortos y autónomos; entornos donde no puedes instalar Docling.
- **Errores típicos.** (1) Confundirlo con la búsqueda híbrida. (2) Usar un tokenizador distinto al del modelo de embeddings. (3) Vectorizar solo el texto, sin encabezados. (4) Olvidar que el máximo **cuenta también los encabezados**. (5) Elegir un máximo demasiado bajo.

**Qué se midió en la sesión** (modelo `nomic-embed-text-v2-moe`, tokenizador del mismo modelo, 22 preguntas base, búsqueda semántica; los cinco documentos leídos con Docling y tablas en Markdown):

| Configuración | Fragmentos | hit@1 | hit@3 | MRR | Contexto (palabras) |
| --- | --- | --- | --- | --- | --- |
| `docling-hibrido`, 64 tokens | 106 | 17/22 | 20/22 | 0.84 | ≈ 71 |
| **`docling-hibrido`, 128 tokens** (referencia) | 55 | **18/22** | **22/22** | **0.91** | ≈ 158 |
| `docling-hibrido`, 256 tokens | 37 | 19/22 | 22/22 | 0.92 | ≈ 243 |
| Oraciones de 70 palabras (línea base) | 59 | 18/22 | 21/22 | 0.87 | ≈ 160 |

Con 128 tokens los fragmentos miden de 38 a 126 tokens (media 81); los de "oraciones de 70 palabras" miden en media 77 tokens con el mismo tokenizador. Resultado: **empate técnico con "oraciones"** (misma hit@1; una pregunta más en hit@3; cuatro centésimas de MRR, que con 22 preguntas son ruido). **Su valor no es ganar esta prueba**, sino: un solo parser para todos los formatos (con OCR: el acta escaneada queda buscable), la **tabla entera con sus filas** en un fragmento, **citas con página y encabezados**, y el tamaño controlado en tokens.

**Una trampa de evaluación.** Con el formato por defecto de Docling para tablas (cada celda como una frase "columna = valor"), la pregunta de la tabla del Word deja de coincidir con su `clave` (escrita con el formato de tabla de nuestro lector) y el resultado baja a 16/22 (MRR 0.84). Moraleja: una **clave atada al formato de un lector** puede penalizar a otro; al cambiar de parser, revisa las claves.

## 9. Búsqueda léxica: BM25

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

### Aterrizando BM25

- **Qué es.** Un puntaje por palabras: dada una consulta, asigna un número a cada fragmento según qué palabras de la consulta contiene.
- **Qué hace.** Para cada palabra de la consulta que aparece en el fragmento suma un aporte con tres ingredientes: TF con rendimientos decrecientes (`k1 = 1.5`), IDF (rareza en el corpus) y normalización por longitud (`b = 0.75`).
- **Para qué sirve.** Para lo que los embeddings hacen mal: identificadores exactos (`OPT-1107`, `E-217`, un SKU, un teléfono), nombres propios y cifras. Además es barato: no necesita modelo ni vectores.
- **Analogía de infraestructura.** Un **`grep` que sabe ponderar**. El IDF es como una alerta: si salta en **todos** los hosts no dice nada; si salta en **uno**, es una señal fuerte. Los rendimientos decrecientes son "10 alertas del mismo host no son 10 veces más graves que una". La normalización evita que un log gigante gane solo por tener más líneas.
- **Ejemplo a mano.**
  - *TF* (fragmento de largo promedio, `k1 = 1.5`): una aparición aporta **1.00**; dos, **1.43**; tres, **1.67**; diez, **2.17**. Diez apariciones no valen diez veces.
  - *IDF* (corpus de 10 fragmentos): "el" (en los 10) → **0.05**; "plan" (en 6) → **0.53**; "opt-1107" (en 1) → **1.99**. La palabra rara pesa unas 40 veces más.
  - *Longitud*: con la misma palabra una vez, un fragmento del doble del largo medio aporta ≈ **0.69**; uno de la mitad, ≈ **1.29**.
- **Cuándo no.** No entiende significado: "contraseña" y "clave" son palabras distintas. Falla con paráfrasis sin palabras en común, y una palabra común del dominio ("casa") puede llevarla a una sección equivocada solo porque aparece.
- **Errores típicos.** (1) Un tokenizador que rompe los códigos. (2) Aplicar *stemming* a los códigos. (3) No quitar palabras vacías. (4) Esperar que "entienda" sinónimos.

| Gana la… | Cuando la pregunta… |
| --- | --- |
| **semántica** | usa otras palabras que el documento (paráfrasis, intención) |
| **BM25** | contiene un código, número o nombre exacto |
| **híbrida** | puede ser de cualquiera de los dos tipos (lo normal en producción) |

Referencia medida en la sesión (oraciones de 70 palabras): en las preguntas léxicas, semántica 8/11 y BM25 **11/11**; en las semánticas, semántica 8/9 y BM25 6/9.

## 10. Búsqueda híbrida y fusión RRF

- **Búsqueda híbrida:** ejecutar la búsqueda léxica y la semántica y **combinar** sus resultados.
- **Problema de las escalas:** el coseno (0 a 1) y el puntaje BM25 (0 a 20 o más) no son comparables; sumarlos no tiene sentido.
- **RRF (*Reciprocal Rank Fusion*):** ignora los puntajes y usa solo la **posición** en cada lista. Cada lista aporta `peso / (c + posición)` a cada fragmento; se suman los aportes y se reordena. Un fragmento bien ubicado en ambas listas sube al primer lugar. La constante `c` (60 por tradición) suaviza la diferencia entre las primeras posiciones.

Ideas clave:

- RRF funciona como una **votación** entre buscadores.
- Los **pesos** de cada buscador importan: con pesos iguales, la híbrida puede quedar **entre** las dos y no superar a la mejor.
- **Sobreajuste:** ajustar los pesos con las mismas preguntas con que se evalúa da resultados optimistas; en serio se separan preguntas de **ajuste** y de **prueba**.
- La híbrida **gana más cuando el corpus crece**, porque la búsqueda semántica empieza a confundir identificadores parecidos.
- Es una función estándar en bases como Elasticsearch, OpenSearch, Weaviate, Pinecone (vectores dispersos y densos) y PostgreSQL (búsqueda de texto completo junto con pgvector).

### Aterrizando la híbrida y RRF

- **Qué es.** La búsqueda híbrida ejecuta dos buscadores y combina sus listas; RRF es la forma de combinarlas usando solo posiciones.
- **Qué hace.** Cada lista da a cada fragmento `peso / (60 + posición)`; se suman los aportes y se reordena.
- **Para qué sirve.** Para obtener lo mejor de ambos: encontrar la paráfrasis y el código exacto sin decidir de antemano qué tipo de pregunta es.
- **Analogía de infraestructura.** Dos sistemas de monitoreo (métricas y logs) dan cada uno su ranking de "servicios sospechosos", con escalas distintas. Sumar esos números no tiene sentido; lo robusto es mirar en qué puesto puso cada uno a cada servicio. El que aparece arriba en **los dos** es el primero que investigas.
- **Ejemplo a mano (el de la lámina de clase).** BM25 ordena **A, C, B**; la búsqueda por significado ordena **B, A, D**.

  | Fragmento | Aporte BM25 | Aporte significado | Suma |
  | --- | --- | --- | --- |
  | **A** | 1/(60+1) = 0.01639 | 1/(60+2) = 0.01613 | **0.0325** |
  | **B** | 1/(60+3) = 0.01587 | 1/(60+1) = 0.01639 | **0.0323** |
  | **C** | 1/(60+2) = 0.01613 | (no aparece) | **0.0161** |
  | **D** | (no aparece) | 1/(60+3) = 0.01587 | **0.0159** |

  La híbrida queda **A, B, C, D**: ni C ni D ganan porque aparecen en una sola lista.
- **La constante 60.** Con `c = 0`, el primer puesto vale el doble que el segundo (1 contra 0.5): un solo voto manda. Con `c = 60`, el primero y el segundo valen casi lo mismo (0.0164 contra 0.0161): para ganar hay que estar bien en varias listas.
- **Cuándo no.** En un corpus pequeño la híbrida con pesos 1:1 puede quedar entre las dos (18/22 en la sesión, igual a la semántica sola). Con pesos que favorecen a BM25 (semántica 1, BM25 2) llegó a 20/22 y MRR 0.91, pero esos pesos se eligieron con las mismas preguntas: es una ilustración, no una prueba.
- **Errores típicos.** (1) Normalizar y sumar puntajes. (2) Fusionar listas muy cortas (cada buscador debería aportar al menos 10 candidatos). (3) Aplicar el filtro después de fusionar en lugar de dentro de cada búsqueda.

## 11. Metadatos y filtros

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

### Aterrizando metadatos y filtros

- **Qué es.** Los metadatos son etiquetas sobre cada fragmento; el filtro es una restricción sobre esas etiquetas dentro de la búsqueda.
- **Qué hace.** Los metadatos se asignan al ingerir (por ejemplo, `publico` sale del título de la sección del Word) y se usan al consultar para descartar candidatos **antes** de ordenar.
- **Para qué sirve.** Calidad, seguridad, frescura y ámbito (tabla anterior).
- **Analogía de infraestructura.** Las **etiquetas de Kubernetes o Prometheus** (`env=prod`, `team=pagos`) y el **control de acceso / seguridad por fila** de una base de datos. Filtrar en la búsqueda es poner la ACL en la base; "filtrar en el prompt" es pedirle al usuario que prometa no mirar las filas ajenas.
- **Ejemplo a mano.** Hay dos fragmentos casi iguales: "reparamos en un máximo de **48 horas**" (`publico = residencial`) y "reparamos en un máximo de **4 horas**" (`publico = negocio`). Para un cliente Negocio, sin filtro la búsqueda devuelve ambos y el LLM puede mezclarlos; con `publico = negocio`, el de 48 horas ni se considera. En la sesión, con cuatro preguntas que dependen del tipo de cliente, sin filtro la semántica acierta 3 de 4 y con filtro 4 de 4.
- **Filtrar antes, no después.** Si se buscan los 3 mejores y luego se filtra, puede quedar **cero** resultados (los 3 mejores eran de otra audiencia). Filtrar dentro de la búsqueda devuelve los 3 mejores que cumplen la regla.
- **Cuándo no basta.** El filtro decide qué se puede ver, no qué es más relevante. Y si el metadato está mal asignado en la ingesta, esconde documentos correctos o muestra los equivocados.
- **Errores típicos.** (1) Filtrar en el prompt. (2) Dejar que el LLM o el texto de la pregunta decidan el filtro. (3) No auditar los metadatos. (4) No definir un valor por defecto (`general`) para lo que no tiene audiencia.

## 12. Reranking

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

### Aterrizando el reranking

- **Qué es.** Una segunda etapa que reordena los candidatos de la primera. La primera es rápida y amplia; la segunda, lenta y fina, y solo ve unas pocas decenas.
- **Qué hace.** Un modelo lee la pregunta y el fragmento **juntos** y le pone una nota de relevancia. Los embeddings leen cada texto por separado y comparan vectores; un cross-encoder compara con ambos textos en la mano, así que ve matices que el vector perdió.
- **Para qué sirve.** Para rescatar el fragmento correcto cuando quedó en el puesto 5 y solo se envían 3 al LLM. Pasa de "recall" (que esté entre los candidatos) a "precisión" (que esté arriba).
- **Analogía de infraestructura.** Una **ACL de red** que decide en microsegundos por IP y puerto (barata, amplia), seguida de un **IDS con inspección profunda** que lee el contenido del paquete (cara, precisa) **solo** sobre lo que pasó el primer filtro. Nadie corre inspección profunda sobre todo el tráfico.
- **Ejemplo a mano.** La etapa 1 trae cinco candidatos: **C3, C1, C5, C2, C4**. El correcto es **C2**, en el puesto 4. Si se envían solo 3 al LLM (C3, C1, C5), no está y el LLM no puede responder. Con reranking, el modelo lee la pregunta junto a cada candidato, pone a C2 primero y se envían **C2, C3, C1**. Mejora hit@3 y MRR. Pero si el correcto no entró entre los cinco, no existe para el reranker.
- **Por qué un modelo pequeño puede empeorar.** Ordenar una lista completa es una tarea difícil (comparar todo contra todo y devolver un formato exacto). Un modelo de ≈3B parámetros se equivoca y desordena algo que ya estaba bien: en las mediciones de la sesión, ordenar todo bajó de 18/22 a 15/22 en hit@1. Elegir solo el mejor es una tarea más simple y rindió mejor (16 a 17/22). Un cross-encoder especializado o un modelo grande suele ser claramente mejor, pero **se mide con los datos propios**.
- **Cuándo no.** Si hit@3 ya es alto y la etapa 1 deja lo correcto arriba, el reranking solo añade costo.
- **Errores típicos.** (1) Pocos candidatos en la etapa 1. (2) Reordenar con un LLM pequeño sin medir. (3) Ignorar la latencia, que es una llamada extra en el camino de la respuesta. (4) Mandar fragmentos enormes al reranker (se recortan).

## 13. Reescritura y expansión de la consulta

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

### Aterrizando las técnicas sobre la consulta

| | Reescritura | Multi-consulta | HyDE |
| --- | --- | --- | --- |
| **Qué es** | Pregunta autocontenida a partir del historial | Varias formas de preguntar lo mismo | Respuesta hipotética como consulta |
| **Qué hace** | "¿Y los domingos?" → "¿A qué hora atienden por teléfono los domingos?" | Busca con la original y N variantes y fusiona con RRF | Vectoriza el texto inventado, que se parece más a un documento que una pregunta |
| **Para qué sirve** | Seguimiento, pronombres, referencias | El usuario usa otras palabras ("multa" frente a "penalización") | Preguntas cortas o vagas |
| **Analogía de infraestructura** | Un ticket que dice "sigue igual": el ingeniero de turno lo reescribe con el historial antes de buscar en la base de conocimiento | Buscar en la wiki con sinónimos: "caída", "outage", "incidente" | Buscar un runbook escribiendo primero cómo imaginas que sería su solución |
| **Cuándo no** | Si no hay historial | Con un modelo pequeño sin medir | Como mejora "segura" por defecto |
| **Error típico** | Perder el nombre del plan al reescribir | Variantes que inventan temas | Dejar que el modelo invente datos |

- **Ejemplo a mano (reescritura).** Historial: el cliente preguntó "¿A qué hora atienden por teléfono de lunes a sábado?" y el asistente respondió "De 7:00 a 22:00". Nueva pregunta: "¿Y los domingos?". Sin reescribir, el buscador recibe tres palabras sin tema. Reescrita, recibe la pregunta completa y encuentra el horario.
- **Qué cuesta cada una.** Reescritura: una llamada pequeña por pregunta, solo con historial. Multi-consulta: una llamada para generar las variantes y N+1 búsquedas. HyDE: una llamada para generar el texto hipotético. Todas añaden latencia y un punto de falla.
- **Medido en la sesión (modelo local pequeño).** La reescritura con búsqueda híbrida pasó de 1/4 a 2/4 en hit@1, y en una de las cuatro el modelo perdió el nombre del plan ("¿Y trae IP fija?" quedó como "¿Trata de IP fija?"). La multi-consulta **empeoró** de 8/9 a 5/9 porque las variantes inventaron temas. Con solo 4 y 9 preguntas es una ilustración, no una prueba.

## 14. Evaluar la recuperación

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

### Aterrizando las métricas

- **Qué es.** Tres números y un costo que resumen qué tan bien busca el RAG, sin llamar al LLM, sobre preguntas donde se sabe qué texto debe aparecer.
- **Qué hace cada una.** hit@1: ¿quedó primero? hit@k: ¿está entre los k que se enviarían? MRR: ¿qué tan arriba quedó? Contexto: ¿cuánto texto se envía?
- **Para qué sirve.** Para decidir con números y no con la impresión de haber probado dos ejemplos.
- **Analogía de infraestructura.** Un **conjunto de pruebas de regresión**: las preguntas son los casos de prueba y hit@k es el porcentaje que pasa. MRR es como medir en qué posición aparece el servicio correcto en una lista de resultados de monitoreo: no basta con que esté, importa que esté arriba. El contexto es el costo de ejecución de cada prueba.
- **Ejemplo a mano (4 preguntas).** El primer acierto quedó en el puesto 1, 1, 2 y "no apareció". Rangos recíprocos: 1.00, 1.00, 0.50 y 0.00. **MRR = (1 + 1 + 0.5 + 0) ÷ 4 = 0.625**; hit@1 = 2/4; hit@3 = 3/4. Si en una pregunta los 3 fragmentos tenían 50, 60 y 40 palabras, su contexto es 150.
- **Cuándo no basta.** Miden recuperación, no si el LLM responde bien (Sesión 7). Con 22 preguntas, 1 o 2 aciertos de diferencia son ruido. hit@1 y MRR premian a los fragmentos grandes, por eso se mira el contexto al lado.
- **Errores típicos.** (1) Declarar ganador por una pregunta. (2) Comparar MRR sin mirar el contexto. (3) Cambiar varias cosas a la vez. (4) Ajustar con las mismas preguntas con que se evalúa.

Cifras de referencia de la sesión (modelo `nomic-embed-text-v2-moe`, 22 preguntas base, búsqueda semántica): línea base (oraciones de 70 palabras) 18/22 en hit@1, 21/22 en hit@3, MRR 0.87, ≈ 160 palabras de contexto; por sección 20/22, MRR 0.95, ≈ 345 palabras; tamaño fijo de 40 palabras 13/22; padre-hijo 22/22, MRR 1.00, ≈ 364 palabras; `docling-hibrido` de 128 tokens (opcional) 18/22, 22/22 en hit@3, MRR 0.91, ≈ 158 palabras (empate técnico con "oraciones"). Pueden variar en uno o dos aciertos con otro equipo.

## 15. Glosario ampliado

| Término | Definición breve |
| --- | --- |
| **Ingesta** | Extraer, fragmentar, vectorizar y guardar documentos |
| **Parsing** | Extraer el texto y la estructura de un archivo |
| **Sección** | Unidad lógica de un documento: un título con su texto o una página de un PDF |
| **Aviso de lectura** | Señal de que el lector no pudo extraer algo (p. ej. una página sin texto) |
| **Reingesta** | Volver a procesar un documento cuando cambia, para no servir contenido viejo |
| **Layout (análisis de)** | Reconocer las regiones de una página (título, párrafo, tabla, figura) y su orden de lectura |
| **Orden de lectura** | La secuencia en que una persona leería las regiones de una página |
| **Docling** | Convertidor de documentos con layout, tablas y OCR; produce un `DoclingDocument` |
| **`DoclingDocument`** | Documento estructurado: jerarquía de títulos, textos, tablas y la página de cada elemento |
| **PDF con texto / escaneado / mixto** | Con capa de texto; solo imágenes; o con páginas de ambos tipos |
| **Capa de texto** | El texto que un PDF guarda además de su aspecto visual |
| **OCR** | Convertir una imagen de texto en texto |
| **Motor de OCR** | Programa que reconoce texto en imágenes (RapidOCR, EasyOCR, Tesseract, servicios de la nube) |
| **RapidOCR** | Motor de OCR local que Docling usa por defecto |
| **Error de OCR** | Letra o símbolo mal reconocido ("OpenAl" por "OpenAI", `Fojas` por `Rojas`), palabras pegadas o texto ilegible |
| **Document AI** | Servicios que reconocen tablas y diseño de documentos complejos |
| **Chunk / chunking** | Fragmento de un documento / proceso de partir documentos |
| **Unidad de búsqueda** | Lo que se vectoriza y se compara con la pregunta (el fragmento) |
| **Unidad de contexto** | Lo que se copia al prompt y cuesta tokens (también el fragmento) |
| **Overlap (solape)** | Texto repetido entre fragmentos consecutivos |
| **Chunking recursivo** | Cortar por el separador más grande que quepa |
| **Chunking semántico** | Cortar donde cambia el tema según la similitud entre oraciones |
| **Padre-hijo** | Buscar con fragmentos pequeños (hijos) y entregar el grande (padre) |
| **Chunking híbrido (Docling)** | Fragmentar por jerarquía del documento y límite de tokens; **no** es la búsqueda híbrida |
| **`HybridChunker`** | La clase de Docling que implementa ese chunking |
| **Token** | Unidad en que un modelo cuenta el texto; no es una palabra y depende del tokenizador |
| **Tokenizador** | Programa que convierte texto en tokens; cada modelo tiene el suyo |
| **`contextualize`** | Antepone los encabezados al texto de un fragmento: lo que conviene vectorizar |
| **BM25** | Algoritmo de búsqueda por palabras (frecuencia, IDF, longitud) |
| **TF** | Frecuencia del término: cuántas veces aparece una palabra en el fragmento |
| **IDF** | Qué tan rara es una palabra en el corpus |
| **`k1` y `b`** | Parámetros de BM25: saturación de la frecuencia y normalización por longitud |
| **Tokenización** | Convertir un texto en los términos que usa el buscador |
| **Stopwords** | Palabras vacías ("el", "de", "que") que no ayudan a buscar |
| **Stemming** | Reducir una palabra a su raíz |
| **Búsqueda léxica** | Búsqueda por palabras exactas |
| **Búsqueda semántica** | Búsqueda por significado con embeddings |
| **Búsqueda híbrida** | Combinar búsqueda léxica y semántica |
| **RRF** | Fusión de listas usando solo la posición de cada resultado |
| **Sobreajuste** | Ajustar parámetros con las mismas preguntas con que se evalúa |
| **Metadatos** | Datos sobre cada fragmento (fuente, sección, audiencia…) |
| **Filtro** | Restricción por metadatos aplicada dentro de la búsqueda |
| **Seguridad por fila** | Control de acceso aplicado dentro de la consulta, no en la capa de presentación |
| **Bi-encoder** | Modelo que vectoriza cada texto por separado (embeddings) |
| **Cross-encoder** | Modelo que lee pregunta y fragmento juntos; más preciso y lento |
| **Reranking** | Reordenar los candidatos con un modelo más preciso |
| **Candidatos** | Resultados de la etapa 1 que el reranker puede reordenar |
| **Query rewriting** | Reescribir la consulta como una pregunta autocontenida |
| **Multi-query** | Buscar con varias reformulaciones y fusionar |
| **HyDE** | Vectorizar una respuesta hipotética en lugar de la pregunta |
| **Descomposición** | Dividir una pregunta compleja en sub-preguntas |
| **hit@k** | Acierto entre los *k* primeros resultados |
| **MRR** | Promedio de 1/posición del primer acierto |
| **Costo del contexto** | Tokens enviados al LLM por consulta |
| **Clave** | Texto que debe aparecer en un fragmento para que ese fragmento contenga la respuesta |
| **Línea base** | Resultado de la versión actual contra el que se compara cada mejora |

## 16. Preguntas de repaso con respuesta

1. **¿Qué dos presiones opuestas hay al elegir el tamaño de un fragmento?** Un fragmento pequeño da un vector preciso pero poco contexto; uno grande da contexto completo pero un vector diluido, más tokens y más ruido.
2. **¿Por qué la estrategia por sección "gana" en MRR y por qué eso puede engañar?** Porque una sección entera casi siempre contiene la clave; pero entrega más del doble de contexto (≈ 345 frente a ≈ 160 palabras), así que gana en parte por ser grande.
3. **Da un ejemplo donde BM25 gane a la semántica y otro donde pierda.** Gana con un código exacto como `OPT-1107`; pierde con una paráfrasis sin palabras en común, como "llevarme el servicio a otra casa" (que significa traslado).
4. **¿Por qué RRF usa la posición y no el puntaje?** Porque el coseno (0 a 1) y BM25 (0 a 20 o más) están en escalas incomparables; la posición es comparable entre buscadores.
5. **Calcula el MRR si el primer acierto quedó en los puestos 1, 2, 4 y "no apareció".** (1 + 0.5 + 0.25 + 0) ÷ 4 = **0.4375**.
6. **¿Por qué filtrar por permisos en la búsqueda y no en el prompt?** Porque en el prompt el dato ya llegó al modelo y no es un control de acceso; el filtro en la búsqueda impide que ese dato entre al contexto.
7. **¿Qué diferencia hay entre un bi-encoder y un cross-encoder, y cuál usarías para reordenar 8 candidatos?** El bi-encoder lee cada texto por separado (rápido, sirve para buscar entre miles); el cross-encoder lee pregunta y fragmento juntos (preciso, lento). Para reordenar 8 candidatos, el cross-encoder.
8. **¿Cuándo no reescribirías la consulta?** Cuando no hay historial: añade un LLM, latencia y riesgo de perder datos sin necesidad.
9. **Un reranker no mejora tus resultados. ¿Qué dos cosas revisas primero?** Que la etapa 1 traiga suficientes candidatos (si el correcto no entra, no se rescata) y que el modelo reranker sea capaz para esa tarea (un modelo pequeño puede desordenar lo que ya estaba bien).
10. **¿Cómo te das cuenta de que a tu RAG le faltan documentos escaneados?** Contando las páginas sin texto en cada ingesta y avisando; un escaneo sin OCR no existe para el RAG.
11. **¿Qué tiene realmente un PDF, y qué añade el análisis de layout?** Tiene letras en posiciones (no párrafos, filas ni columnas). El análisis de layout reconoce las regiones y su tipo, reconstruye el orden de lectura y las tablas, y decide dónde hace falta OCR.
12. **Un PDF casi no devuelve texto. ¿Qué haces?** Compruebas si tiene capa de texto. Si no la tiene (escaneo) o el parser lo trata como imagen, reintentas con OCR; si la tiene, no usas OCR (añade errores). Guardas que esa página salió de OCR.
13. **Un OCR da 98 % de parecido con el texto bueno. ¿Está bien?** No necesariamente: el 2 % puede ser justo el apellido o el código que se pregunta (`Rojas` → `Fojas`). Se miden las **palabras que cambiaron**, no un solo porcentaje, y se validan los datos críticos.
14. **¿Siempre ayuda enderezar o ampliar la imagen antes del OCR?** No. En las pruebas de la sesión, enderezar recuperó un apellido pero introdujo otro error; ampliar no lo recuperó; con una imagen muy degradada, ambas devolvieron texto ilegible. Lo que se perdió al escanear no lo recupera el OCR: se mejora la fuente y se mide.
15. **¿Qué diferencia hay entre el chunking híbrido de Docling y la búsqueda híbrida?** El chunking híbrido decide **dónde cortar** (jerarquía del documento + límite de tokens, en la ingesta). La búsqueda híbrida decide **cómo buscar** (BM25 + vectores con RRF, en la consulta). Se pueden usar juntos.
16. **¿Por qué el tokenizador del chunker debe ser el del modelo de embeddings?** Porque los tokens se cuentan distinto en cada modelo; con otro tokenizador, "128 tokens" puede ser más o menos para el modelo real y el límite engaña.
17. **`docling-hibrido` empató con "oraciones" en el laboratorio. ¿Para qué sirve entonces?** Para lo que la nota no mide: un solo parser con OCR para todos los formatos, tablas enteras con sus filas, citas con página y encabezados, y tamaño controlado en tokens. Y se compara con el mismo criterio de siempre: hit@k, MRR y contexto.
18. **¿Cuándo eliges un lector simple y cuándo un parser con layout?** Lector simple para Markdown, HTML, DOCX sencillos y PDF de texto corrido (rápido, sin dependencias). Parser con layout cuando mides que las tablas, el orden de lectura o los escaneos te cuestan respuestas, y puedes asumir CPU, modelos y una dependencia grande.

---

**Implementación formal para estudiar:** `rag-playground/rag-intermedio` (repo `/home/oscar/Desktop/Personal/EDUTEK/GEN-AI/rag-playground`).
