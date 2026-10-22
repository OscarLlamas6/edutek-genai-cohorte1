# Crear tu cuenta de Pinecone (base de datos vectorial)

## 1. ¿Qué es Pinecone?

**Pinecone** es una **base de datos vectorial administrada**: un servicio en la nube diseñado solo para guardar vectores y encontrar los más parecidos a uno dado. No instalas ni administras servidores; usas su **API** (en el curso, su SDK de Python).

Su modo **serverless** separa almacenamiento y cómputo y cobra **por lo que usas** (lecturas, escrituras y almacenamiento). Pinecone gestiona el índice interno por ti. En el curso es nuestra **base de datos vectorial #2**, para comparar con Neon.

### Vocabulario

| Término | Qué es |
| --- | --- |
| **Index** (índice) | El contenedor principal; se crea con una **dimensión** fija (la del modelo de embeddings) y una **métrica** (coseno) |
| **Namespace** | Partición lógica dentro de un índice; las búsquedas se hacen dentro de un namespace |
| **Record** (registro) | `id` + vector + `metadata` (datos extra, p. ej. el texto original) |
| **Read / write units** | La unidad con que Pinecone mide lo que consumes |
| **API key** | La credencial que identifica tu cuenta |

## 2. ¿Cómo se paga y qué límites tiene?

| Plan | Precio | Límites / tarifas |
| --- | --- | --- |
| **Starter (gratis)** | US$ 0 | Hasta 2 GB de almacenamiento, 2 M de unidades de escritura al mes, 1 M de unidades de lectura al mes, 5 índices, 100 namespaces por índice; **solo AWS `us-east-1`** |
| **Standard** | Mínimo de uso de US$ 50 al mes, pago por uso por encima | Almacenamiento ≈ US$ 0.33 por GB-mes; lecturas ≈ US$ 16–18 por millón; escrituras ≈ US$ 4–4.50 por millón; todas las regiones |
| **Enterprise** | Mínimo de uso de US$ 500 al mes | Para grandes organizaciones |

**Para el curso usamos solo el plan Starter.** Verás que la página de precios menciona una **prueba de 3 semanas del plan Standard con US$ 300 en créditos**: **no la necesitas ni la actives**; puede requerir datos de pago. Pinecone no indica en su página de precios si el plan Starter pide tarjeta; si en tu registro te la pide, **detente y avisa en clase** antes de ingresarla.

> **Importante:** dentro del curso los índices se crean en **AWS, región `us-east-1`** (única disponible en Starter). El repositorio ya lo usa por defecto.

## 3. Crear la cuenta

1. Entra a **[pinecone.io](https://www.pinecone.io)** y pulsa **Sign up** (o *Start for free*).
2. Elige registrarte con **Google**, **GitHub** o **correo y contraseña**.
3. Verifica tu correo si te lo pide.
4. Responde las preguntas iniciales (rol, uso) con lo que quieras; son opcionales o de marketing.
5. Si aparece la selección de plan, elige **Starter** (gratuito). **No** elijas Standard ni la prueba.

Al terminar entrarás a la **consola** (console), el panel de control de Pinecone.

## 4. Crear la API key

1. En la consola abre el menú **API keys** (lateral).
2. Si ya existe una llave por defecto, puedes usarla; si no, pulsa **Create API key**, ponle el nombre `edutek` y créala.
3. **Copia el valor** de la llave (lo verás completo solo al crearla; luego se oculta parcialmente).

> 🔐 **La API key es una credencial.** Quien la tenga puede usar tu cuenta. No la compartas, no la pegues en un chat o captura de pantalla ni la subas a Git. Si se filtra, bórrala en la consola y crea otra.

## 5. Guardarla en tu `.env`

En la raíz del repositorio, abre (o crea con `cp .env.example .env`) el archivo `.env` y añade:

```
PINECONE_API_KEY=pega-aqui-tu-llave
```

Opcionalmente puedes fijar otros valores (por defecto ya son estos):

```
PINECONE_INDEX=edutek-cohorte1
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

Sin comillas ni espacios alrededor del `=`. El `.env` está en el `.gitignore`. Si el notebook ya estaba abierto, **reinicia el kernel**.

## 6. Verificar que funciona

Desde la carpeta `modulo-1/s04-embeddings-bases-vectoriales/`:

```bash
uv run python -c "
from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv(usecwd=True))
import os
from pinecone import Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
print('Índices existentes:', [i.name for i in pc.list_indexes()])
"
```

Si imprime una lista (vacía o con nombres), la llave funciona. El **índice se crea automáticamente** la primera vez que el notebook o la práctica lo necesitan (tarda alrededor de un minuto); no tienes que crearlo a mano.

## 7. Ver tus datos y tu consumo

- **Indexes:** en la consola verás `edutek-cohorte1`, su dimensión y su métrica.
- **Namespaces / registros:** al abrir el índice puedes ver los namespaces (`faqs`, `rag`, `edutek`) y cuántos registros tiene cada uno.
- **Usage / Billing:** muestra las unidades de lectura y escritura y el almacenamiento consumidos.

## 8. Cómo evitar cobros

- Quédate en el plan **Starter**; no actives Standard ni la prueba de créditos.
- No agregues método de pago si no es necesario.
- Revisa *Usage* de vez en cuando.
- Borra el índice cuando termines el curso (ver abajo).

## 9. Limpieza y cierre

- **Vaciar un namespace:** desde el código, `almacen.vaciar()`, o en la consola.
- **Borrar el índice:** en la consola, *Indexes* → tu índice → *Delete*; o desde el código, `almacen.eliminar()`. Con el plan gratuito hay un máximo de 5 índices: borra los que no uses.
- **Si se filtró tu API key:** en *API keys*, bórrala y crea otra; actualiza tu `.env`.

## Problemas frecuentes

| Síntoma | Causa y solución |
| --- | --- |
| `401` / `Unauthorized` | API key mal copiada o borrada; crea otra y actualiza el `.env` |
| La primera ejecución tarda ~1 min | Se está creando el índice; es normal |
| `dimension` distinta a la del índice | Cambiaste de modelo de embeddings: borra el índice (`almacen.eliminar()`) y reindexa |
| No aparecen resultados justo después de insertar | Consistencia eventual; el código espera unos segundos a que los registros sean visibles |
| Error de región o nube no soportada | El plan Starter solo admite AWS `us-east-1`; deja `PINECONE_CLOUD=aws` y `PINECONE_REGION=us-east-1` |
| Máximo de índices alcanzado | Borra un índice que no uses (límite de 5 en Starter) |
