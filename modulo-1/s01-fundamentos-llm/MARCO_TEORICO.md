# Marco teórico · Sesión 1: Fundamentos del LLM y ventanas de contexto

> Documento de apoyo al notebook `s01_fundamentos_llm.ipynb`. Reúne las definiciones y conceptos que se usan a lo largo de la sesión. No contiene código: el código y los experimentos están en el notebook.

## Índice

1. [Inteligencia artificial y sus ramas](#1-inteligencia-artificial-y-sus-ramas)
2. [Breve historia de la IA generativa](#2-breve-historia-de-la-ia-generativa)
3. [Modelo de lenguaje y LLM](#3-modelo-de-lenguaje-y-llm)
4. [Arquitectura transformer y atención](#4-arquitectura-transformer-y-atención)
5. [Token y tokenizador](#5-token-y-tokenizador)
6. [Ventana de contexto](#6-ventana-de-contexto)
7. [Costo de una llamada](#7-costo-de-una-llamada)
8. [Parámetros de generación](#8-parámetros-de-generación)
9. [Sistemas deterministas vs. probabilísticos](#9-sistemas-deterministas-vs-probabilísticos)
10. [API, API key y statelessness](#10-api-api-key-y-statelessness)
11. [Cuándo no usar un LLM](#11-cuándo-no-usar-un-llm)
12. [Glosario rápido](#12-glosario-rápido)

---

## 1. Inteligencia artificial y sus ramas

- **Inteligencia artificial (IA):** campo de la informática que busca construir sistemas capaces de realizar tareas que normalmente requieren inteligencia humana (percibir, razonar, decidir, comunicarse).
- **Aprendizaje automático (*machine learning*, ML):** subconjunto de la IA en el que el sistema **aprende patrones a partir de datos** en lugar de seguir reglas escritas a mano.
- **Aprendizaje profundo (*deep learning*):** subconjunto del ML que usa **redes neuronales con muchas capas**. Es la base de los modelos actuales de lenguaje, imagen y audio.
- **IA generativa:** modelos que **producen contenido nuevo** (texto, imagen, audio, código) en lugar de solo clasificar o predecir una etiqueta.
- **Red neuronal:** función matemática formada por capas de unidades simples (neuronas) con **parámetros** (pesos) que se ajustan durante el entrenamiento.
- **Parámetro (peso):** número interno del modelo que se ajusta al entrenar. Se suele hablar del "tamaño" de un modelo en miles de millones de parámetros (7B, 70B…).

Relación de conjuntos: **IA ⊃ ML ⊃ Deep learning ⊃ IA generativa (LLM incluidos)**.

## 2. Breve historia de la IA generativa

La historia suele resumirse en olas, donde cada una resolvió el límite de la anterior:

| Era | Idea central | Límite que dejó |
| --- | --- | --- |
| **IA simbólica / sistemas expertos** | Reglas escritas a mano por humanos | No escala; frágil ante casos nuevos |
| **Aprendizaje estadístico** | Aprender patrones de datos (p. ej. *n-gramas*, modelos probabilísticos) | Contexto muy corto; poca comprensión |
| **Redes neuronales profundas** | Representaciones aprendidas, más datos y cómputo (GPU) | Procesar secuencias largas era difícil (RNN/LSTM) |
| **Transformers (2017)** | Mecanismo de **atención**; paralelizable y escalable | Costo de entrenamiento muy alto |
| **LLM y asistentes (2020 →)** | Escalar datos, cómputo y parámetros; alinear con instrucciones humanas | Alucinaciones, costo, seguridad |

Conceptos relacionados:

- **Invierno de la IA:** periodo de recortes de financiación e interés tras expectativas infladas que la tecnología no cumplió. Recuerda que el entusiasmo va en ciclos.
- **N-grama:** modelo que predice la siguiente palabra mirando solo las *n−1* palabras anteriores, a partir de conteos en un corpus. Es el antecesor sencillo de los LLM: la idea de "predecir la siguiente palabra" es antigua; lo que cambió es la **escala** y la **arquitectura**.
- **Corpus:** colección grande de textos usada para entrenar o analizar un modelo.
- **Ley de escalado (*scaling law*):** observación empírica de que el rendimiento mejora de forma predecible al aumentar datos, cómputo y parámetros.

## 3. Modelo de lenguaje y LLM

- **Modelo de lenguaje:** modelo que asigna **probabilidades a secuencias de texto**; en la práctica, estima qué es probable que venga a continuación dado lo anterior.
- **LLM (*Large Language Model*, modelo de lenguaje de gran tamaño):** modelo de lenguaje basado en redes neuronales (hoy, casi siempre transformers) con **miles de millones de parámetros**, entrenado con enormes cantidades de texto.
- **Idea central:** un LLM **predice el siguiente token**. Genera texto de forma **autorregresiva**: predice un token, lo añade a la entrada, y vuelve a predecir, hasta terminar.
- **Autorregresivo:** cada salida depende de las salidas anteriores del propio modelo.
- **Distribución de probabilidad:** en cada paso el modelo no entrega "la" palabra, sino una probabilidad para **cada token posible** del vocabulario; luego se *muestrea* (elige) uno según los parámetros de generación.
- **Entrenamiento (*pre-training*):** fase en la que el modelo aprende a predecir el siguiente token sobre un corpus masivo. Es muy costosa y ocurre una vez (o pocas).
- **Ajuste fino (*fine-tuning*) / alineación:** entrenamientos posteriores para que el modelo siga instrucciones, sea útil y evite respuestas dañinas (p. ej. con retroalimentación humana, RLHF).
- **Inferencia:** uso del modelo ya entrenado para generar respuestas. Es lo que ocurre cada vez que haces una llamada.
- **Fecha de corte de conocimiento (*knowledge cutoff*):** fecha hasta la cual el modelo vio datos de entrenamiento; no sabe nada posterior por sí mismo.
- **Alucinación:** respuesta que **suena convincente pero es falsa o inventada**. Ocurre porque el modelo optimiza lo *plausible*, no lo *verdadero*.

> Consecuencia importante: un LLM **no es una base de datos ni un buscador**; no "consulta" hechos, genera texto plausible a partir de patrones aprendidos.

## 4. Arquitectura transformer y atención

A muy alto nivel, el flujo interno de un LLM es:

1. El texto se parte en **tokens**.
2. Cada token se convierte en un **vector** (lista de números) llamado *embedding*.
3. Varias capas de **atención** deciden qué partes del texto son relevantes para predecir lo que sigue.
4. La salida es una **distribución de probabilidad** sobre todos los tokens posibles.

Definiciones:

- **Transformer:** arquitectura de red neuronal (Vaswani et al., 2017, *Attention Is All You Need*) basada en el mecanismo de atención; permite procesar toda la secuencia en paralelo y es la base de los LLM actuales.
- **Atención (*attention*):** mecanismo que permite que cada token "mire" a los demás tokens del contexto y pondere cuáles importan más para su representación.
- **Embedding:** representación numérica (vector) de un token o texto, en la que elementos con significado parecido quedan cerca. Se profundiza en la Sesión 4.
- **Vocabulario:** conjunto fijo de tokens que el modelo conoce (decenas a cientos de miles).
- **Logits:** puntuaciones brutas que el modelo calcula para cada token antes de convertirlas en probabilidades.
- **Softmax:** función que convierte los logits en probabilidades que suman 1.

## 5. Token y tokenizador

- **Token:** unidad mínima de texto que procesa un LLM. Puede ser una palabra completa, un trozo de palabra (subpalabra), un signo de puntuación, un espacio o incluso bytes sueltos. **No equivale a una palabra.**
- **Tokenización:** proceso de dividir el texto en tokens y asignar a cada uno un **identificador numérico** (ID) del vocabulario. El modelo trabaja con IDs, no con letras.
- **Tokenizador (*tokenizer*):** componente (con su propio vocabulario y reglas) que realiza la tokenización. **Cada familia de modelos tiene el suyo**, por lo que el mismo texto puede dar distinta cantidad de tokens.
- **BPE (*Byte Pair Encoding*):** algoritmo común de tokenización por subpalabras: parte de caracteres/bytes y fusiona iterativamente los pares más frecuentes del corpus hasta formar el vocabulario.
- **Subpalabra (*subword*):** fragmento intermedio entre carácter y palabra; permite representar palabras raras combinando piezas conocidas.
- **Regla aproximada:** en inglés, 1 token ≈ 4 caracteres ≈ ¾ de palabra. Es solo una estimación.

Por qué importa:

1. **Se cobra por token**, tanto de entrada como de salida.
2. **El límite del modelo se mide en tokens** (ventana de contexto).
3. **El idioma pesa:** el español suele necesitar **más tokens** que el inglés para decir lo mismo, porque los tokenizadores se entrenaron con mucho más texto en inglés. El **código** y los textos con símbolos también tienden a consumir muchos tokens.

## 6. Ventana de contexto

- **Contexto:** todo el texto que el modelo "ve" en una llamada: instrucciones, historial de la conversación, documentos adjuntos y la propia respuesta que va generando.
- **Ventana de contexto (*context window*):** **cantidad máxima de tokens** que el modelo puede manejar en una sola llamada, sumando **entrada + salida**.
- **Tokens de entrada (*input tokens* / *prompt tokens*):** los que envías al modelo.
- **Tokens de salida (*output tokens* / *completion tokens*):** los que el modelo genera.

Tres consecuencias:

1. **Todo cuenta:** entrada y salida comparten la misma ventana.
2. **Más contexto = más costo y más latencia**, y no siempre mejor calidad: un contexto enorme o lleno de ruido puede hacer que el modelo "se pierda".
3. **No es memoria:** el modelo **no recuerda nada entre llamadas**. Si quieres que sepa algo, se lo reenvías.

Efecto práctico: en un chat, **cada turno reenvía todo el historial**, por lo que el costo acumulado de una conversación crece más rápido que linealmente (un chat de 20 turnos cuesta bastante más que 20 veces un turno aislado).

## 7. Costo de una llamada

Los proveedores cobran **por millón de tokens** (MTok), con precio distinto para entrada y salida:

```
costo = (tokens_entrada × precio_entrada + tokens_salida × precio_salida) / 1 000 000
```

Ideas clave:

- **Entrada y salida se cobran distinto:** los tokens de salida suelen costar varias veces más que los de entrada.
- **Precio por token ≠ costo por tarea:** un modelo caro que acierta a la primera puede salir más barato que uno barato que necesita varios intentos. Conviene medir el **costo por tarea completada**.
- **Latencia:** tiempo que tarda en llegar la respuesta; crece con el tamaño del contexto y de la salida.
- **Presupuesto / límite de gasto:** tope mensual configurable en la consola del proveedor; buena práctica obligatoria al trabajar con APIs de pago.

## 8. Parámetros de generación

Controlan **cómo se elige el siguiente token** a partir de la distribución de probabilidad:

- **Muestreo (*sampling*):** elegir el siguiente token de forma aleatoria ponderada por su probabilidad.
- **Decodificación voraz (*greedy*):** elegir siempre el token más probable.
- **`temperature` (temperatura):** reescala las probabilidades. Valores **bajos** (≈0) hacen la salida más **predecible y conservadora**; valores **altos** la hacen más **variada y creativa** (y más propensa a errores).
- **`top_p` (*nucleus sampling*):** el modelo solo considera el conjunto mínimo de tokens cuya probabilidad acumulada alcanza *p*; descarta la "cola" improbable.
- **`top_k`:** solo se consideran los *k* tokens más probables.
- **`max_tokens`:** tope de tokens que puede generar la respuesta; si se alcanza, la respuesta **se corta** (aunque esté a mitad de frase).
- **Secuencias de parada (*stop sequences*):** textos que, al generarse, detienen la respuesta.
- **Semilla (*seed*):** valor para intentar reproducir el muestreo aleatorio (no siempre garantizado por todos los proveedores).

Matiz importante: `temperature=0` **reduce** la variación pero **no garantiza** salidas idénticas (puede haber diferencias por el cómputo numérico, cambios de versión del modelo, etc.). Además, algunos modelos (por ejemplo los de razonamiento) **no aceptan** ciertos parámetros como `temperature`.

## 9. Sistemas deterministas vs. probabilísticos

- **Sistema determinista:** misma entrada → misma salida, siempre. Es el software tradicional; se prueba con aserciones exactas (`assertEqual`).
- **Sistema probabilístico (o estocástico):** misma entrada → salidas que pueden variar. Es el caso de un LLM.
- **Evaluación (*evals*):** en vez de una única prueba exacta, se mide el comportamiento sobre **muchos casos** y se reportan porcentajes (tasa de acierto, formato válido, etc.). Es el sustituto del test unitario clásico.
- **Reproducibilidad:** capacidad de obtener el mismo resultado al repetir; en LLM se busca de forma aproximada, no absoluta.

Cambio mental clave: **un LLM se evalúa, no se prueba con igualdad exacta.**

## 10. API, API key y *statelessness*

- **API (*Application Programming Interface*):** interfaz para que un programa use un servicio de otro. Aquí, para enviar texto a un modelo y recibir su respuesta.
- **API key (clave de API):** credencial secreta que identifica y autoriza (y factura) al usuario. **Nunca** se escribe en el código ni se sube a Git: se guarda en un archivo `.env` o en una variable de entorno.
- **Variable de entorno:** valor de configuración que vive fuera del código, en el sistema o en un archivo `.env`.
- **Proveedor:** empresa que ofrece los modelos vía API (por ejemplo Anthropic u OpenAI).
- **Mensaje y roles:** una llamada se compone de mensajes con un rol: *system* (instrucciones generales), *user* (lo que pide el usuario) y *assistant* (lo que responde el modelo).
- **Sin estado (*stateless*):** la API no guarda memoria de llamadas previas; el cliente debe **reenviar el historial** en cada llamada.
- **Uso (*usage*):** metadatos que devuelve la API con los tokens de entrada y salida consumidos; sirven para calcular el costo.
- **Motivo de parada (*stop reason*):** indica por qué terminó la respuesta (fin natural, límite de `max_tokens`, etc.).

## 11. Cuándo no usar un LLM

Un LLM no es la mejor herramienta cuando se necesita:

- **Exactitud garantizada** (cálculos, fechas, validaciones, reglas legales o contables).
- **Resultados reproducibles al 100%.**
- **Consultas sobre datos estructurados** que una base de datos o una expresión regular resuelven de forma exacta, barata y rápida.
- **Muy baja latencia o muy bajo costo por operación a gran volumen.**

Regla práctica: si existe una solución **determinista, simple y barata**, úsala; reserva el LLM para tareas con **lenguaje natural ambiguo** donde la flexibilidad compense su incertidumbre.

## 12. Glosario rápido

| Término | Definición breve |
| --- | --- |
| **IA generativa** | Modelos que crean contenido nuevo |
| **LLM** | Modelo de lenguaje grande que predice el siguiente token |
| **Token** | Unidad de texto que procesa el modelo (no es una palabra) |
| **Tokenizador** | Componente que convierte texto en tokens/IDs |
| **Ventana de contexto** | Máximo de tokens por llamada (entrada + salida) |
| **Transformer** | Arquitectura basada en atención, base de los LLM |
| **Embedding** | Vector numérico que representa significado |
| **Inferencia** | Uso del modelo entrenado para generar |
| **Temperature** | Controla aleatoriedad/variedad de la salida |
| **top_p / top_k** | Limitan los tokens candidatos al muestrear |
| **max_tokens** | Tope de longitud de la respuesta |
| **Alucinación** | Respuesta plausible pero falsa |
| **Stateless** | La API no recuerda llamadas previas |
| **Probabilístico** | La misma entrada puede dar salidas distintas |
| **API key** | Credencial secreta de acceso al servicio |
