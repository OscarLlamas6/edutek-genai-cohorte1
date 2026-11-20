# Crear tu cuenta de HubSpot (CRM gratis) y conectarla al servidor MCP

**Tiempo:** ≈ 15 minutos · **Costo:** US$ 0 (CRM gratuito, sin tarjeta) · **Se usa en:** Sesión 10 (MCP)

> Las pantallas de HubSpot cambian con frecuencia. Si un botón no está donde dice esta guía, busca uno equivalente; el flujo es el mismo. Los datos de esta guía se contrastaron con la documentación pública de HubSpot en noviembre de 2026; **nada se probó contra una cuenta real**, así que los pasos 5 y 6 son los más propensos a necesitar un ajuste (hay una tabla de problemas al final).
>
> **Es opcional.** Toda la sesión funciona sin cuenta, con `HUBSPOT_MODE=fake` (un CRM simulado en memoria). Crea la cuenta solo si quieres ver el agente leer y escribir en un CRM de verdad.

## 1. ¿Qué es HubSpot y por qué lo usamos?

**HubSpot** es un **CRM** (*Customer Relationship Management*): el sistema donde una empresa guarda sus **contactos** (personas), **empresas**, **negocios** y las **notas** de cada conversación. Es el ejemplo clásico de "sistema externo" al que un agente se conecta.

En la sesión el servidor MCP hace dos **lecturas** (buscar contactos y leer sus notas) y una **escritura** (crear una nota asociada a un contacto). Por eso necesitamos un sistema real donde una escritura tenga un efecto que se pueda ver.

## 2. Reglas de oro antes de empezar

- **Solo datos ficticios.** Los contactos de prueba son inventados (dominios `example.com`, teléfonos de ficción). No metas clientes, correos ni teléfonos reales.
- **Cuenta de pruebas, no la de tu empresa.** Crea una cuenta nueva para el curso. Si usas una cuenta que ya tiene datos, el agente podría leerlos y escribir en ellos.
- **El token es una contraseña.** No lo pegues en un chat, un correo ni el código. Va solo en tu `.env`, que no se sube a Git.

## 3. Crear la cuenta

1. Entra a **[hubspot.com/products/crm](https://www.hubspot.com/products/crm)** y pulsa **Get free CRM** (Obtener CRM gratis).
2. Regístrate con tu correo o con Google, y confirma el correo si te lo pide. **No pide tarjeta.**
3. Completa el asistente inicial con lo mínimo (nombre, empresa ficticia como `Curso Edutek`, tamaño). Puedes saltar los pasos opcionales con **Skip** u **Omitir**.
4. Cuando llegues al panel principal del CRM, la cuenta está lista.

El plan gratuito incluye hasta 2 usuarios y 1 000 contactos de marketing, de sobra para el curso.

## 4. Cargar los contactos de prueba

Son los mismos cuatro del modo simulado (Ana Morales, Luis Pérez, Marta Solís y Diego Ruiz). Hay dos formas:

**A. Importar el archivo (más rápido).**

1. En el menú, entra a **CRM → Contactos**.
2. Pulsa **Importar** (arriba a la derecha) → **Iniciar una importación** → **Archivo de computadora** → **Un archivo** → **Contactos**.
3. Elige `ai-agents-playground/langchain-mcp-advanced-tooling/data/hubspot_contactos.csv`.
4. En el mapeo de columnas, comprueba que cada columna del CSV quede asignada a su propiedad (nombre, apellido, correo, teléfono, empresa, cargo, etapa del ciclo de vida) y termina la importación.

**B. A mano.** En **Contactos → Crear contacto**, escribe los cuatro con los datos del CSV (abre el archivo para copiarlos).

Después, abre un contacto y **anota su ID**: está en la dirección del navegador, `.../record/0-1/`**`123456789`**. En HubSpot los ids son números largos; **no** son los `101`, `102`… del modo simulado. No importa: el agente los obtiene con la búsqueda.

## 5. Crear una Service Key

Una **Service Key** es una credencial que da a un programa (nuestro servidor MCP) permiso para usar la API de tu cuenta, con **solo los permisos que le des**. Es el reemplazo recomendado de las "apps privadas" heredadas, cuya creación HubSpot desactivó.

1. En el menú principal, entra a **Desarrollo → Claves → Service keys**. (También se llega por **Configuración → Integraciones → Service Keys**.)
2. Pulsa **Crear service key**.
3. **Nombre:** `mcp-curso`.
4. En **Scopes** (permisos), agrega solo estos:
   - `crm.objects.contacts.read` (leer contactos)
   - `crm.objects.contacts.write` (necesario para asociar una nota a un contacto)
   - Si HubSpot ofrece permisos específicos de notas (`crm.objects.notes.read` / `.write` u otros equivalentes), agrégalos también. La lista oficial está en [developers.hubspot.com/scopes](https://developers.hubspot.com/scopes).
5. Crea la clave y **copia el token** (empieza por `pat-`). **Cópialo ahora:** puede que luego solo puedas rotarlo.

> Necesitas ser **Super Admin** (o tener acceso a *Herramientas de desarrollador*) para crear Service Keys. Si abriste la cuenta tú, lo eres.

> Las Service Keys están en **beta pública**: el nombre del menú o la lista de scopes pueden haber cambiado. Si no encuentras la opción, busca "Service keys" en la barra de búsqueda de HubSpot.

## 6. Guardar el token en tu `.env`

En la carpeta `ai-agents-playground/langchain-mcp-advanced-tooling`, copia la plantilla y edítala:

```bash
cp .env.example .env
```

```
HUBSPOT_MODE=live
HUBSPOT_TOKEN=pat-na1-...           # tu token, sin comillas ni espacios
HUBSPOT_API_VERSION=2026-03         # si algo da 404, prueba v3
```

El archivo `.env` no se sube a Git. El código nunca imprime el token.

## 7. Verificar que funciona (antes de la clase)

```bash
uv run python check_hubspot.py                      # solo lecturas
uv run python check_hubspot.py --write 123456789    # además, crea UNA nota de prueba (usa el id de tu contacto)
```

Cada paso muestra ✅ o ❌ y, si falla, **qué corregir**. Resultado esperado:

```
HubSpot · versión de API: 2026-03 · token: 70 caracteres (no se muestra)

Lecturas
  ✅ Buscar contactos (POST …/contacts/search)
     encontré 1: Ana Morales (id 123456789)
  ✅ Leer un contacto por id (GET …/contacts/{id})
  ✅ Leer sus notas (POST …/notes/search)
     0 nota(s)

Escritura
  ✅ Crear una nota de prueba en el contacto 123456789
```

Abre el contacto en HubSpot y verás la nota en su línea de tiempo. Si todo salió ✅, ya puedes usar el agente con tu CRM:

```bash
uv run python main.py --trace "Resume las notas de Ana Morales."
```

## 8. Problemas frecuentes

| Síntoma | Causa probable | Qué hacer |
| --- | --- | --- |
| `401` | Token mal copiado, con espacios o comillas, o de otra cuenta | Copia el token otra vez; si no, crea una Service Key nueva |
| `403` (`MISSING_SCOPES`) | A la clave le falta un permiso | Edita la Service Key y agrega el scope que mencione HubSpot |
| `404` en la búsqueda de contactos | La ruta con fecha no aplica a tu cuenta | `HUBSPOT_API_VERSION=v3` (ruta heredada) |
| La búsqueda devuelve 0 contactos | No importaste el CSV, o la búsqueda es por «Morales» y cambiaste el apellido | Revisa **CRM → Contactos** |
| ✅ en lecturas y ❌ al leer o escribir notas | Los permisos de notas son distintos a los de contactos | Agrega los scopes de notas (paso 5) |
| `429` | Demasiadas peticiones seguidas | Espera unos segundos |
| La búsqueda tarda en mostrar un contacto recién creado | El índice de búsqueda de HubSpot se actualiza con unos segundos de retraso | Espera un momento y repite |
| No encuentro "Service keys" | Menú distinto en tu cuenta | Usa la barra de búsqueda de HubSpot o la guía oficial de [Service keys](https://developers.hubspot.com/docs/apps/developer-platform/build-apps/authentication/account-service-keys) |

Si no logras que funcione antes de la clase, no pasa nada: cambia `HUBSPOT_MODE=fake` y la sesión se hace completa con el CRM simulado.

## 9. Limpieza al terminar el curso

1. **Elimina o rota la Service Key** (Desarrollo → Claves → Service keys), para que el token deje de funcionar.
2. Borra los contactos de prueba si quieres (Contactos → seleccionar → Eliminar).
3. Borra `HUBSPOT_TOKEN` de tu `.env`.
