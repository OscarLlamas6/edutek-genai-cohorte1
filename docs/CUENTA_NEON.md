# Crear tu cuenta de Neon (PostgreSQL + pgvector)

**Tiempo:** ≈ 10 minutos · **Costo:** US$ 0 (plan Free, sin tarjeta) · **Se usa en:** Sesiones 4 y 5

> Las pantallas de Neon cambian con frecuencia. Si algún botón no está donde dice esta guía, busca uno equivalente; el flujo es el mismo. Los datos de planes y precios de este documento se verificaron en octubre de 2026: confirma siempre en [neon.com/pricing](https://neon.com/pricing).

## 1. ¿Qué es Neon?

**Neon** es **PostgreSQL como servicio, "serverless"**: una base de datos PostgreSQL en la nube que no tienes que instalar ni administrar. Tiene dos rasgos distintivos:

- **Escala a cero:** si nadie la usa durante unos minutos, el cómputo se suspende (y deja de consumir horas). La primera consulta después de la pausa tarda un poco más.
- **Ramas (*branches*):** puedes clonar una base de datos como si fuera una rama de Git.

Neon incluye **pgvector**, la extensión de PostgreSQL que permite guardar vectores y buscar por similitud. En el curso la usamos como nuestra **base de datos vectorial #1**.

## 2. ¿Cómo se paga y qué límites tiene?

| Plan | Precio | Cómputo | Almacenamiento | Notas |
| --- | --- | --- | --- | --- |
| **Free** | US$ 0 / mes, **sin tarjeta** | 100 CU-horas por proyecto | 1 GB por proyecto | Hasta 100 proyectos; autosuspensión a los 5 min (no se puede desactivar) |
| **Launch** | Pago por uso, sin mínimo mensual | ≈ US$ 0.106 por CU-hora | ≈ US$ 0.35 por GB-mes | Autosuspensión desactivable |
| **Scale** | Pago por uso, sin mínimo mensual | ≈ US$ 0.222 por CU-hora | ≈ US$ 0.35 por GB-mes | Más proyectos y autosuspensión configurable |

**CU-hora:** una *compute unit* equivale a ≈ 1 vCPU y 4 GB de RAM; una CU-hora es esa capacidad encendida una hora. Una base en la configuración mínima (0.25 CU) activa durante 4 horas consume 1 CU-hora. Como la base se suspende sola cuando no se usa, **100 CU-horas alcanzan de sobra** para el curso.

**Para el curso usamos solo el plan Free.** No necesitas tarjeta, así que **no puedes recibir un cobro por accidente**. Si agotas la cuota, la base se limita hasta el mes siguiente.

## 3. Crear la cuenta

1. Entra a **[neon.com](https://neon.com)** y pulsa **Sign up** (Registrarse).
2. Elige cómo registrarte: con **GitHub**, **Google** o con **correo y contraseña**. Cualquiera sirve.
3. Si te pide verificar el correo, abre el mensaje y confirma.
4. Es posible que te pida un nombre para tu organización o cuenta; escribe el que quieras.

## 4. Crear un proyecto

Un **proyecto** de Neon contiene tu base de datos.

1. Pulsa **Create project** (Crear proyecto) si no se abrió solo el asistente.
2. **Nombre:** `edutek-cohorte1`.
3. **Versión de Postgres:** deja la que viene por defecto (o 17, 16).
4. **Región:** elige la más cercana a ti (por ejemplo, *AWS US East* si estás en Centroamérica). Cuanto más cerca, menor la latencia.
5. Pulsa **Create**. En unos segundos verás el panel del proyecto.

## 5. Obtener la cadena de conexión

1. En el panel del proyecto busca el botón **Connect** (Conectar).
2. Elige la rama `main`, la base de datos `neondb` y el rol que aparece por defecto.
3. Verás una URL como:

   ```
   postgresql://usuario:clave@ep-xxxx-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```

   - Si el host contiene **`-pooler`**, es la conexión **con pool** (varias conexiones cortas). Para el curso sirve cualquiera de las dos.
   - Pulsa **Show password** si la clave aparece oculta, y copia la **URL completa** con el botón de copiar.

> 🔐 **La cadena de conexión es una credencial.** Quien la tenga puede leer, modificar y borrar tu base. **No la compartas, no la pegues en un chat o captura de pantalla, ni la subas a Git.**

## 6. Guardarla en tu `.env`

En la raíz del repositorio, abre (o crea con `cp .env.example .env`) el archivo `.env` y añade:

```
NEON_DATABASE_URL=postgresql://usuario:clave@ep-xxxx-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require
```

- Sin comillas y sin espacios alrededor del `=`.
- El archivo `.env` está en el `.gitignore`: no se sube a Git.
- Si el notebook ya estaba abierto, **reinicia el kernel** para que lea el cambio.

## 7. Verificar que funciona

Desde la carpeta `modulo-1/s04-embeddings-bases-vectoriales/`:

```bash
uv run python -c "
from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv(usecwd=True))
import os, psycopg
with psycopg.connect(os.environ['NEON_DATABASE_URL']) as c:
    print(c.execute('SELECT version()').fetchone()[0])
"
```

Debe imprimir algo como `PostgreSQL 17.x ...`. También puedes abrir el notebook y ejecutar la Parte 6: la celda de conexión muestra ✅.

La **extensión pgvector** se activa sola desde el código (`CREATE EXTENSION IF NOT EXISTS vector`); no necesitas hacer nada en el panel.

## 8. Cómo ver tus datos y tu consumo

- **SQL Editor:** en el panel de Neon, abre *SQL Editor* y ejecuta, por ejemplo, `SELECT count(*) FROM faqs;` para ver lo que guardó el notebook.
- **Tables:** el panel permite explorar las tablas creadas (`faqs`, `rag`, `fragmentos`).
- **Uso:** en *Billing* / *Usage* verás las CU-horas y el almacenamiento consumidos del mes.

## 9. Cómo evitar cobros

- Quédate en el plan **Free** y **no agregues una tarjeta**: sin tarjeta no hay cobro posible.
- No cambies de plan salvo que quieras hacerlo.
- Revisa el consumo de vez en cuando en *Usage*.

## 10. Limpieza y cierre

- **Borrar datos de una tabla:** en el SQL Editor, `DROP TABLE faqs;` (o usa `vaciar()` desde el código).
- **Borrar el proyecto completo:** *Project settings* → *Delete project*. Se pierde todo, sin vuelta atrás.
- **Si se filtró tu cadena de conexión:** en el panel, restablece la contraseña del rol (*Reset password*) y actualiza tu `.env`.

## Problemas frecuentes

| Síntoma | Causa y solución |
| --- | --- |
| `connection refused` / `timeout` | Revisa la URL completa y tu conexión a internet; algunas redes bloquean el puerto 5432 |
| `password authentication failed` | La clave cambió o está mal copiada; copia de nuevo la URL desde *Connect* |
| La primera consulta tarda | La base estaba suspendida; despierta en ~1 segundo o más |
| `extension "vector" is not available` | Raro en Neon; revisa que el proyecto sea de Neon y no una base local |
| `SSL required` | Mantén `?sslmode=require` al final de la URL |
| Cuota agotada | Espera al mes siguiente o crea otro proyecto (el plan Free admite varios) |
