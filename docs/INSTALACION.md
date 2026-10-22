# Guía de instalación

Esta guía te deja con todo listo para las sesiones del Curso I. Funciona en **Linux**, **macOS** y **Windows (con WSL2)**.

> **Tiempo estimado:** 20 a 30 minutos. Hazlo **antes** de la primera clase.

## 1. Qué necesitas instalar

| Programa | Para qué | Sesiones | ¿Obligatorio? |
| --- | --- | --- | --- |
| **Git** | Clonar el repositorio | Todas | Sí |
| **uv** | Gestionar Python, entornos virtuales y dependencias | Todas | **Recomendado** (hay alternativas, ver sección 5) |
| **Python 3.12** | Lenguaje de las sesiones 1 a 3 | 1 a 3 | Sí. Con uv se instala solo; con otro gestor lo instalas tú |
| **Una API key de Anthropic u OpenAI** | Llamar a un modelo (de pago, centavos) | Todas | **Sí, de uno de los dos**: el repo funciona con ambos y se elige con `LLM_PROVIDER` ([guía](PROVEEDORES.md)) |
| **LM Studio** (u Ollama, llama-server…) | Ejecutar un modelo open-weight en tu equipo | 2 | Recomendado: el repo usa **LM Studio**, pero puedes usar otro programa local ([ver sección 8](#8-modelos-open-weight-locales-sesión-2)) |
| **Un editor** (VS Code recomendado) | Abrir notebooks y código | Todas | Recomendado |

Más adelante en el curso se sumarán otras herramientas (por ejemplo **Docker** en las sesiones de bases vectoriales y n8n). Se avisará en el README de cada sesión.

## 2. Sistema operativo

### Linux
Usa tu terminal normal. Necesitas `git` y `curl` (`sudo apt install git curl` en Ubuntu/Debian).

### macOS
Usa la Terminal. Instala las herramientas de línea de comandos con `xcode-select --install` (incluye Git).

### Windows
Recomendamos **WSL2** (Windows Subsystem for Linux) con Ubuntu: todo el curso se ejecuta como en Linux.

```powershell
# En PowerShell como administrador
wsl --install -d Ubuntu
```

Reinicia, abre "Ubuntu" desde el menú Inicio y sigue las instrucciones de **Linux**. Trabaja **dentro** del sistema de archivos de WSL (por ejemplo `~/edutek-genai-cohorte1`), no en `/mnt/c/...`, porque es mucho más rápido.

> Si no puedes usar WSL2, uv funciona en Windows nativo (ver abajo), pero los comandos de esta guía están escritos para shells tipo Unix (bash/zsh).

## 3. Instalar `uv` (recomendado)

> **¿Ya usas pyenv, venv, virtualenv, conda, Anaconda u otro gestor?** Puedes seguir usándolo: uv **no es obligatorio**. Lo recomendamos porque todo el material del repositorio está preparado para él y es la forma más rápida de que todos tengamos exactamente lo mismo. Si prefieres tu herramienta, salta a la [sección 5](#5-alternativas-a-uv).

`uv` es un gestor de paquetes y de versiones de Python, muy rápido, que reemplaza a `pip`, `venv` y `pyenv` en un solo programa.

**Linux / macOS / WSL2:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell, sin WSL):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Cierra y abre la terminal, y verifica:

```bash
uv --version
```

Si aparece un número de versión, está listo. (Otras formas de instalar uv, como `brew` o `pipx`, están en su documentación oficial.)

## 4. Cómo funciona uv con Python (lo esencial)

Cada sesión es un **proyecto uv** y trae tres archivos que definen su entorno:

| Archivo | Qué contiene |
| --- | --- |
| `.python-version` | La versión de Python de **esa sesión** (por ejemplo `3.12`) |
| `pyproject.toml` | Las dependencias que necesita la sesión |
| `uv.lock` | Las versiones **exactas** instaladas, para que todos tengamos lo mismo |

### El comando que lo hace todo

Dentro de la carpeta de una sesión:

```bash
uv sync
```

Este comando, en una sola vez:

1. **Descarga la versión de Python indicada** en `.python-version` si no la tienes (no necesitas instalar Python tú).
2. **Crea el entorno virtual** `.venv/` dentro de la carpeta.
3. **Instala las dependencias** exactas de `uv.lock`.

No hace falta "activar" el entorno: usa `uv run <comando>` y uv lo ejecuta dentro del entorno correcto.

```bash
uv run python --version        # Python de ESTA sesión
uv run jupyter lab             # abre JupyterLab con las dependencias de la sesión
uv run python archivo.py       # ejecuta un script
```

### Comandos útiles de uv

| Qué quieres hacer | Comando |
| --- | --- |
| Ver las versiones de Python disponibles/instaladas | `uv python list` |
| Instalar una versión de Python sin crear un proyecto | `uv python install 3.12` |
| **Fijar** la versión de Python de una carpeta | `uv python pin 3.12` (escribe `.python-version`) |
| Crear/actualizar el entorno desde `uv.lock` | `uv sync` |
| Añadir una dependencia al proyecto | `uv add <paquete>` |
| Quitar una dependencia | `uv remove <paquete>` |
| Recalcular `uv.lock` tras editar `pyproject.toml` | `uv lock` |
| Ejecutar algo en el entorno del proyecto | `uv run <comando>` |
| Borrar y recrear el entorno desde cero | `rm -rf .venv && uv sync` |

### ¿Por qué una versión de Python por sesión?

Cada sesión fija **su** versión en `.python-version` y en `requires-python` de `pyproject.toml`. Así el material de una sesión sigue funcionando aunque más adelante el curso avance a otra versión, y todos los alumnos corren exactamente lo mismo. Hoy las sesiones 1 a 3 usan **Python 3.12**.

Si en algún momento quieres probar otra versión en una sesión **en tu copia**:

```bash
cd modulo-1/s01-fundamentos-llm
uv python pin 3.13        # cambia .python-version
# ajusta `requires-python` en pyproject.toml si hace falta
uv sync                   # recrea el entorno con esa versión
```

## 5. Alternativas a uv

uv es lo que usamos en clase y lo que verás en todos los comandos del repositorio (`uv sync`, `uv run ...`). **Pero no estás obligado a usarlo.** Cualquier gestor de entornos sirve mientras cumplas dos cosas:

1. **Usar la versión de Python de la sesión** (hoy, **3.12**; está en `.python-version`).
2. **Instalar las dependencias de la sesión.** Cada carpeta de sesión trae un `requirements.txt` con las versiones exactas (se genera desde `uv.lock`), así que funciona con `pip` en cualquier gestor.

> **Qué pierdes sin uv:** que uv descargue Python por ti, el `uv sync` de un solo paso y la garantía de que tu entorno sea idéntico al del instructor. Si algo falla y no usas uv, es posible que te pidamos que reproduzcas el problema con uv para descartar diferencias de entorno.

En todos los casos, `uv run <comando>` de los README se reemplaza por ejecutar `<comando>` con **tu entorno activado**. Por ejemplo, `uv run python practica/calculadora_tokens.py ...` pasa a ser `python practica/calculadora_tokens.py ...`, y `uv run jupyter lab` pasa a ser `jupyter lab`.

### venv (incluido en Python)

```bash
cd modulo-1/s01-fundamentos-llm
python3.12 -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

### virtualenv

```bash
pip install virtualenv
virtualenv -p python3.12 .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### pyenv (gestiona versiones de Python) + venv

```bash
pyenv install 3.12                   # si aún no la tienes
pyenv local 3.12                     # fija la versión en esta carpeta
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### conda / Anaconda / Miniconda

```bash
conda create -n edutek-s01 python=3.12 -y
conda activate edutek-s01
pip install -r requirements.txt      # usa pip dentro del entorno conda
```

Crea **un entorno por sesión** (por ejemplo `edutek-s01`, `edutek-s02`): así cada una tiene sus propias dependencias, como en el repositorio.

### Otros (poetry, pipenv, pdm…)

Funcionan con los datos de `pyproject.toml` (sección `dependencies`) o con `requirements.txt`. Recuerda usar Python 3.12.

### Notebooks con un entorno que no es el de uv

- **JupyterLab:** lanza `jupyter lab` desde el entorno activado.
- **VS Code:** elige como kernel el intérprete de tu entorno (`Select Kernel → Python Environments`).
- Los notebooks verifican que la versión de Python sea 3.12. Si usas otra, la primera celda fallará a propósito: instala 3.12 o, bajo tu responsabilidad, borra esa verificación.
- Las dependencias de Jupyter (`jupyterlab`, `ipykernel`, `ipywidgets`) ya están en `requirements.txt`.

> Los `requirements.txt` se regeneran con `uv export --no-hashes --no-emit-project -o requirements.txt` cuando cambia `pyproject.toml`. La fuente de verdad es `uv.lock`.

## 6. Clonar el repositorio

```bash
git clone <URL-del-repositorio> edutek-genai-cohorte1
cd edutek-genai-cohorte1
```

## 7. Tu API key (Anthropic u OpenAI)

> 💰 **FYI:** es un servicio de pago por uso. Más detalles en [COSTOS.md](COSTOS.md).
>
> **El repositorio funciona con dos proveedores, Anthropic y OpenAI**, y se elige con una variable de entorno (`LLM_PROVIDER`). Necesitas la llave de **uno** de los dos. Detalles y diferencias en [PROVEEDORES.md](PROVEEDORES.md). ¿Gemini, Groq, OpenRouter u otro? También puedes adaptarlo (ver esa guía).

1. Entra a la consola de tu proveedor (**Anthropic Platform** u **OpenAI Platform**) y crea una cuenta.
2. Agrega un **saldo pequeño** (con unos pocos dólares alcanza para todo el Curso I) o configura facturación.
3. **Configura un límite de gasto mensual** en la consola. Es tu red de seguridad.
4. Crea una **API key** y cópiala. Solo se muestra una vez.
5. En la raíz del repositorio:

   ```bash
   cp .env.example .env
   ```

   Abre `.env`, elige el proveedor y pega **su** llave:

   ```
   # Con Anthropic (por defecto)
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-...

   # Con OpenAI
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-...
   ```

6. **Nunca** pegues la llave en un notebook, en el código, en un chat o en un commit. `.env` ya está en el `.gitignore`.
7. Si cambias el `.env`, **reinicia el kernel** del notebook (las variables se leen al iniciar).

Un solo archivo `.env` en la raíz sirve para todas las sesiones.

## 8. Modelos open-weight locales (Sesión 2)

En la Sesión 2 comparamos un modelo propietario (por API: Anthropic u OpenAI) con un modelo **open-weight que corre en tu propio equipo**. Hay varios programas para eso; **el repositorio y el notebook están preparados para LM Studio**, pero puedes usar el que prefieras.

| Programa | Qué es | Interfaz | Servidor local (por defecto)* | Para quién |
| --- | --- | --- | --- | --- |
| **LM Studio** (el del repositorio; [guía](LMSTUDIO.md)) | Aplicación de escritorio para buscar, descargar y probar modelos, con servidor local integrado | **Gráfica (muy didáctica)** | `http://localhost:1234/v1` | Quien quiere ver el modelo, los tokens y los parámetros en una interfaz |
| **Ollama** | Servicio con línea de comandos que descarga y ejecuta modelos | Terminal | `http://localhost:11434/v1` | Quien prefiere la terminal |
| **llama-server** (de llama.cpp) | Servidor ligero del motor de inferencia en el que se basan otras herramientas | Terminal | `http://localhost:8080/v1` | Quien quiere control fino y entender el motor por dentro |
| Otros (vLLM, Jan, GPT4All…) | Alternativas, algunas pensadas para servidores con GPU | Varía | Varía | Entornos más avanzados (vLLM se retoma en el Curso III) |

\* Los puertos y rutas por defecto pueden cambiar entre versiones: confírmalos en la documentación de cada programa.

Todos ejecutan los mismos tipos de modelos (muchos en formato **GGUF**) y comparten la idea: **descargas un modelo y lo consumes en `localhost`**. Y los cuatro exponen una API **compatible con el formato de OpenAI**, por eso el repositorio usa **una sola función**, `llamar_local()` en `cliente_llm.py`, que funciona con cualquiera de ellos: solo cambias `LOCAL_LLM_URL` y `LOCAL_LLM_MODEL` en tu `.env`.

### LM Studio (el que usa el repositorio)

Resumen (la guía completa, con la terminal `lms` y la solución de problemas, está en [LMSTUDIO.md](LMSTUDIO.md)):

1. **Instálalo** desde su página oficial (lmstudio.ai). Es una aplicación de escritorio para Windows, macOS y Linux.
2. **Descarga un modelo de chat pequeño** (≈ 2 GB): *Llama 3.2 3B Instruct* (en la terminal: `lms get llama-3.2-3b-instruct`).
3. **Activa el servidor local:** pantalla *Local Model API* → *Local API server* en **Running** y *Just-in-time model loading* activado. Copia la **Base URL**: suele ser `http://localhost:1234/v1`, pero **puede ser otro puerto**; usa el que muestre tu pantalla.
4. **Configura tu `.env`:**

   ```
   LOCAL_LLM_URL=http://localhost:1234/v1
   LOCAL_LLM_MODEL=llama-3.2-3b-instruct
   ```

5. **Verifica** (con tu puerto): `curl http://localhost:1234/v1/models`. Debe incluir `llama-3.2-3b-instruct`.

> **Usas WSL2 con LM Studio en Windows:** el servidor corre en Windows y tu código en WSL2, así que `localhost` puede no alcanzarlo. Prueba con la IP del host de Windows, activa la red en modo espejo de WSL2, o ejecuta el notebook directamente en Windows. Ajusta `LOCAL_LLM_URL` en tu `.env`.

### Usar Ollama o llama-server en lugar de LM Studio

Funcionan igual; solo apunta el repositorio a su servidor en tu `.env`:

```
# Ollama (tras: ollama pull llama3.2:3b)
LOCAL_LLM_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=llama3.2:3b

# llama-server (ejemplo)
LOCAL_LLM_URL=http://localhost:8080/v1
LOCAL_LLM_MODEL=el-nombre-del-modelo
```

- **Ollama:** instálalo desde su página oficial (en Linux: `curl -fsSL https://ollama.com/install.sh | sh`), descarga el modelo con `ollama pull llama3.2:3b` y, si no corre como servicio, ejecuta `ollama serve`.
- **llama-server:** viene con llama.cpp; consulta su documentación para arrancarlo con un archivo de modelo GGUF.

> **¿No puedes instalar nada?** (equipo corporativo bloqueado). No pasa nada: el notebook de la Sesión 2 detecta que no hay servidor local y omite esa parte; puedes trabajar en pareja o seguir la pantalla del instructor.

## 9. Abrir los notebooks

Tienes dos formas; usa la que prefieras.

### Opción A · JupyterLab (navegador)

```bash
cd modulo-1/s01-fundamentos-llm
uv sync
uv run jupyter lab
```

Se abre el navegador. Abre el archivo `.ipynb` de la carpeta.

### Opción B · VS Code

1. Instala la extensión **Python** y la extensión **Jupyter** de Microsoft.
2. Abre la **carpeta de la sesión** (`File → Open Folder…` en `s01-fundamentos-llm`).
3. Abre el `.ipynb`.
4. Arriba a la derecha elige **Select Kernel → Python Environments →** `.venv` (Python 3.12).

> Si abres la carpeta raíz del repositorio en vez de la carpeta de la sesión, VS Code no encontrará el `.venv` correcto. Abre siempre **la carpeta de la sesión**.

### Cómo se ejecuta un notebook

- `Shift + Enter` ejecuta la celda seleccionada y pasa a la siguiente.
- Ejecuta las celdas **en orden**, de arriba hacia abajo.
- Si algo se rompe: `Kernel → Restart Kernel and Run All Cells`.

## 10. Verificar que todo funciona

```bash
cd modulo-1/s01-fundamentos-llm
uv sync
uv run python -c "import sys, anthropic, tiktoken; print('Python', sys.version.split()[0], '· anthropic', anthropic.__version__)"
```

Debe imprimir algo como `Python 3.12.x · anthropic 1.x`. Para verificar tu API key **sin gastar nada** abre el notebook de la Sesión 1 y ejecuta la celda opcional "Contar con el tokenizador real de Claude" (el conteo de tokens es gratuito).

## 11. Problemas frecuentes

| Síntoma | Causa probable | Solución |
| --- | --- | --- |
| `uv: command not found` | La terminal no recargó el PATH (o no instalaste uv; es opcional, ver sección 5) | Cierra y vuelve a abrir la terminal |
| El notebook dice "Esta sesión está fijada a Python 3.12" | El kernel no es el del `.venv` | Selecciona el kernel `.venv` (ver sección 9) y reinicia el kernel |
| `ModuleNotFoundError: No module named 'anthropic'` | Se ejecutó fuera del entorno | Activa tu entorno (o usa `uv run ...`) y selecciona su kernel |
| `Falta ANTHROPIC_API_KEY` / `Falta OPENAI_API_KEY` | No existe `.env`, está vacío, o la llave no corresponde a `LLM_PROVIDER` | `cp .env.example .env`, pega la llave del proveedor elegido y reinicia el kernel |
| `LLM_PROVIDER='...' no es válido` | Error de escritura en el `.env` | Usa `anthropic` u `openai` |
| `401 authentication_error` | Llave mal copiada o revocada | Crea una nueva llave y actualiza `.env` |
| `429` / `rate_limit` | Demasiadas peticiones o saldo agotado | Espera unos segundos; revisa tu saldo y límites |
| Aviso "unauthenticated requests to the HF Hub" | Descarga de un tokenizador sin token de Hugging Face | Es **inofensivo**; ignóralo |
| `No hay servidor local en http://localhost:1234/v1` | El servidor de LM Studio no está activo, o la URL es otra | Abre LM Studio → *Developer* → **Start Server**; revisa `LOCAL_LLM_URL` (con WSL2, ver sección 8) |
| `El modelo '...' no aparece en el servidor local` | El identificador no coincide con el de LM Studio | Copia el identificador exacto que muestra LM Studio en `LOCAL_LLM_MODEL` |
| Descarga lenta/fallida de Python o paquetes | Proxy o firewall corporativo | Configura `HTTPS_PROXY`, o trabaja en un equipo personal |
| En WSL2, todo va muy lento | El proyecto está en `/mnt/c/...` | Muévelo al sistema de archivos de WSL (`~/`) |

## 12. Cuentas de la Semana 2 (Neon y Pinecone)

Las sesiones de bases de datos vectoriales usan dos servicios con **plan gratuito**. Crea las cuentas siguiendo las guías (≈ 10 minutos cada una):

- [`CUENTA_NEON.md`](CUENTA_NEON.md): PostgreSQL en la nube con pgvector.
- [`CUENTA_PINECONE.md`](CUENTA_PINECONE.md): base de datos vectorial administrada.

También necesitas el **modelo de embeddings** en LM Studio (ver [`LMSTUDIO.md`](LMSTUDIO.md#2b-descargar-un-modelo-de-embeddings-sesiones-4-a-6)). Si no logras crear una cuenta, puedes seguir las sesiones con el almacén en memoria.
