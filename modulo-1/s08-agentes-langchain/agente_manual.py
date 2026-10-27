"""El bucle de un agente HECHO A MANO, con el SDK del proveedor (sin LangChain).

Sirve para una sola cosa: que veas QUÉ hace `create_agent` por ti. Léelo de arriba abajo; es corto.

LAS CUATRO PIEZAS DEL TOOL CALLING
    1. DESCRIBES las herramientas al modelo con un esquema JSON (nombre, descripción, parámetros).
    2. El modelo responde, en vez de texto, con una PETICIÓN: "llama a estado_servicio con servicio='app-movil'".
       El modelo NO ejecuta nada. Solo pide.
    3. TU CÓDIGO ejecuta la función y le devuelve el resultado como un mensaje más.
    4. Repites desde el paso 2 hasta que el modelo responde con texto (o te quedas sin pasos).

Cada API dibuja esas piezas con otro formato:

                  Chat Completions (LM Studio y         Responses (OpenAI)                   Messages (Anthropic)
                  otros servidores compatibles)
    esquema       tools=[{"type":"function",            tools=[{"type":"function",           tools=[{"name", "description",
                  "function":{…}}]                      "name", "parameters"…}]              "input_schema"}]
    petición      message.tool_calls[i].function        ítem "function_call" en              bloques "tool_use" (input = dict),
                  (args = texto JSON)                   response.output (args = texto JSON)  stop_reason="tool_use"
    resultado     mensaje role="tool" + tool_call_id    ítem "function_call_output"          mensaje user con bloque "tool_result"
                                                        + call_id                            + tool_use_id

LangChain unifica estas diferencias: tu código de agente es el mismo con cualquier proveedor.

💰 COSTO: cada vuelta del bucle es una llamada al modelo (de pago con Anthropic/OpenAI; gratis con local).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import servicios
from cliente_llm import costo_usd, modelo_economico, modelo_local, url_local

# ───────────────────────── Pieza 1: el esquema de las herramientas (escrito a mano) ─────────────────────────
# Es EXACTAMENTE lo que `@tool` genera por ti a partir de la firma y el docstring de una función.
# (En el notebook lo comparas con `convert_to_openai_tool(estado_servicio)`.)

ESQUEMAS = [
    {
        "name": "listar_servicios",
        "description": "Lista TODOS los servicios de Andes Fibra con su estado actual. Úsala para preguntas generales ('¿hay fallas?').",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "estado_servicio",
        "description": "Consulta el estado actual y el detalle de UN servicio de Andes Fibra. Úsala cuando pregunten por un servicio concreto.",
        "parameters": {
            "type": "object",
            "properties": {
                "servicio": {
                    "type": "string",
                    "description": "Identificador del servicio: portal-mi-andes, app-movil, pagos-en-linea, chat-soporte, telefonia-soporte, red-zona-norte o red-zona-sur.",
                }
            },
            "required": ["servicio"],
        },
    },
]

INSTRUCCIONES = (
    "Eres el asistente de soporte de Andes Fibra. Para saber si un servicio funciona AHORA, usa las herramientas; "
    "nunca lo supongas. Si la pregunta no necesita herramientas, responde directo. Cuando una herramienta ya te dio "
    "lo que necesitas, responde sin repetir la llamada. Responde en español, breve."
)


# ───────────────────────── Una sola vuelta, para mirar la petición cruda ─────────────────────────


def primera_peticion(pregunta: str, proveedor: str) -> dict:
    """UNA llamada al modelo con las herramientas descritas. Devuelve lo que el modelo "dijo" sin ejecutar nada.

    Es lo más importante de entender del tool calling: el modelo no responde con texto sino con una PETICIÓN
    (`nombre` + `argumentos`). No ejecutó nada ni consultó nada: solo pidió.
    """
    if proveedor == "anthropic":
        import anthropic

        modelo = modelo_economico("anthropic")
        r = anthropic.Anthropic(max_retries=2).messages.create(
            model=modelo, max_tokens=1024, system=INSTRUCCIONES, messages=[{"role": "user", "content": pregunta}],
            tools=[{"name": e["name"], "description": e["description"], "input_schema": e["parameters"]} for e in ESQUEMAS])
        peticiones = [b for b in r.content if b.type == "tool_use"]
        return {"modelo": modelo, "parada": r.stop_reason, "crudo": r.model_dump_json(indent=2),
                "nombre": peticiones[0].name if peticiones else None, "argumentos": peticiones[0].input if peticiones else None,
                "texto": "".join(b.text for b in r.content if b.type == "text")}
    import openai

    if proveedor == "openai":
        modelo = modelo_economico("openai")
        r = openai.OpenAI(max_retries=2).responses.create(
            model=modelo, instructions=INSTRUCCIONES, input=[{"role": "user", "content": pregunta}],
            tools=[{"type": "function", "strict": False, **e} for e in ESQUEMAS])
        llamada = next((i for i in r.output if i.type == "function_call"), None)
        return {"modelo": modelo, "parada": r.status, "crudo": json.dumps([i.model_dump(exclude_none=True) for i in r.output], ensure_ascii=False, indent=2),
                "nombre": llamada.name if llamada else None, "argumentos": json.loads(llamada.arguments or "{}") if llamada else None,
                "texto": r.output_text or ""}
    cliente = openai.OpenAI(base_url=url_local(), api_key=os.getenv("LOCAL_LLM_API_KEY", "no-se-usa"), max_retries=0, timeout=300)
    modelo = modelo_local()
    r = cliente.chat.completions.create(
        model=modelo, messages=[{"role": "system", "content": INSTRUCCIONES}, {"role": "user", "content": pregunta}],
        tools=[{"type": "function", "function": e} for e in ESQUEMAS], temperature=0)
    m = r.choices[0].message
    llamada = m.tool_calls[0] if m.tool_calls else None
    return {"modelo": modelo, "parada": r.choices[0].finish_reason, "crudo": m.model_dump_json(indent=2, exclude_none=True),
            "nombre": llamada.function.name if llamada else None,
            "argumentos": json.loads(llamada.function.arguments or "{}") if llamada else None, "texto": m.content or ""}


# ───────────────────────── Pieza 3: quién ejecuta de verdad ─────────────────────────


def ejecutar_herramienta(nombre: str, argumentos: dict) -> str:
    """Tu código ejecuta la función que el modelo pidió. Un nombre desconocido o argumentos mal formados
    se devuelven como TEXTO de error, para que el modelo pueda corregirse en la siguiente vuelta."""
    try:
        if nombre == "listar_servicios":
            return servicios.listar()
        if nombre == "estado_servicio":
            return servicios.estado(argumentos["servicio"])
    except (KeyError, TypeError) as error:
        return f"Error: argumentos inválidos para {nombre} ({type(error).__name__}: {error})."
    return f"Error: la herramienta '{nombre}' no existe. Herramientas disponibles: {', '.join(e['name'] for e in ESQUEMAS)}."


# ───────────────────────── La traza ─────────────────────────


@dataclass
class CorridaManual:
    respuesta: str = ""
    pasos: list[str] = field(default_factory=list)       # lo que fue pasando, en orden
    llamadas: int = 0                                     # vueltas del bucle (llamadas al modelo)
    tokens_entrada: int = 0
    tokens_salida: int = 0
    segundos: float = 0.0
    terminó: bool = False                                 # False si se acabaron los pasos sin respuesta final
    error: str = ""                                       # si el servidor rechazó la petición, el motivo
    modelo: str = ""
    proveedor: str = ""

    @property
    def costo(self) -> float:
        return 0.0 if self.proveedor == "local" else costo_usd(self.modelo, self.tokens_entrada, self.tokens_salida)


def _anotar(corrida: CorridaManual, linea: str, mostrar: bool) -> None:
    corrida.pasos.append(linea)
    if mostrar:
        print(linea)


# ───────────────────────── Pieza 2 y 4 con LM Studio (Chat Completions) ─────────────────────────


def bucle_chat_completions(pregunta: str, max_pasos: int = 8, mostrar: bool = True) -> CorridaManual:
    """El bucle con el SDK de OpenAI apuntando a LM Studio (API Chat Completions, la que hablan casi todos los servidores locales)."""
    import openai

    cliente = openai.OpenAI(base_url=url_local(), api_key=os.getenv("LOCAL_LLM_API_KEY", "no-se-usa"), max_retries=0, timeout=300)
    modelo, extra = modelo_local(), {"temperature": 0}
    herramientas = [{"type": "function", "function": e} for e in ESQUEMAS]
    mensajes = [{"role": "system", "content": INSTRUCCIONES}, {"role": "user", "content": pregunta}]

    corrida = CorridaManual(modelo=modelo, proveedor="local")
    t0 = time.perf_counter()
    _anotar(corrida, f"👤 Usuario        │ {pregunta}", mostrar)
    for _ in range(max_pasos):                                   # ← el bucle: máximo `max_pasos` vueltas
        try:
            r = cliente.chat.completions.create(model=modelo, messages=mensajes, tools=herramientas, **extra)
        except openai.BadRequestError as error:
            # Pasa con modelos locales pequeños: la petición de herramienta que escribieron no respeta el formato
            # estricto y LM Studio la rechaza (HTTP 400). Un modelo de proveedor casi nunca falla así.
            corrida.error = str(getattr(error, "message", error))[:200]
            _anotar(corrida, f"⚠️ El servidor rechazó la salida del modelo (HTTP 400): {corrida.error}", mostrar)
            break
        corrida.llamadas += 1
        corrida.tokens_entrada += r.usage.prompt_tokens
        corrida.tokens_salida += r.usage.completion_tokens
        mensaje = r.choices[0].message
        mensajes.append(mensaje)                                 # el historial crece: el modelo no tiene memoria propia

        if not mensaje.tool_calls:                               # sin peticiones → es la respuesta final
            corrida.respuesta, corrida.terminó = (mensaje.content or "").strip(), True
            _anotar(corrida, f"🤖 Modelo responde│ {corrida.respuesta}", mostrar)
            break
        for llamada in mensaje.tool_calls:                       # puede pedir varias a la vez
            argumentos = json.loads(llamada.function.arguments or "{}")   # ¡llegan como TEXTO JSON!
            _anotar(corrida, f"🤖 Modelo pide    │ {llamada.function.name}({argumentos})", mostrar)
            resultado = ejecutar_herramienta(llamada.function.name, argumentos)   # ← TU código ejecuta
            _anotar(corrida, f"🔧 Resultado      │ {resultado[:160].replace(chr(10), ' ')}", mostrar)
            mensajes.append({"role": "tool", "tool_call_id": llamada.id, "content": resultado})
    corrida.segundos = time.perf_counter() - t0
    return corrida


# ───────────────────────── Pieza 2 y 4 con OpenAI (Responses) ─────────────────────────


def bucle_responses(pregunta: str, max_pasos: int = 8, mostrar: bool = True) -> CorridaManual:
    """El bucle con la API Responses de OpenAI (la misma que usa `llamar_openai` en la Sesión 2).

    Diferencias con Chat Completions: la respuesta es una lista de ÍTEMS (`r.output`): puede haber ítems de
    razonamiento, de llamada a función (`function_call`) y de mensaje. Los reenviamos tal cual y añadimos
    un `function_call_output` por cada llamada.
    """
    import openai

    cliente = openai.OpenAI(max_retries=2)
    modelo = modelo_economico("openai")
    # `strict: False` porque nuestros esquemas no declaran additionalProperties=false (el modo estricto lo exige).
    herramientas = [{"type": "function", "strict": False, **e} for e in ESQUEMAS]
    entrada: list = [{"role": "user", "content": pregunta}]

    corrida = CorridaManual(modelo=modelo, proveedor="openai")
    t0 = time.perf_counter()
    _anotar(corrida, f"👤 Usuario        │ {pregunta}", mostrar)
    for _ in range(max_pasos):
        r = cliente.responses.create(model=modelo, instructions=INSTRUCCIONES, input=entrada, tools=herramientas)
        corrida.llamadas += 1
        corrida.tokens_entrada += r.usage.input_tokens
        corrida.tokens_salida += r.usage.output_tokens
        entrada += r.output                                       # el historial crece: reenviamos lo que dijo el modelo

        llamadas = [item for item in r.output if item.type == "function_call"]
        if not llamadas:
            corrida.respuesta, corrida.terminó = (r.output_text or "").strip(), True
            _anotar(corrida, f"🤖 Modelo responde│ {corrida.respuesta}", mostrar)
            break
        for llamada in llamadas:
            argumentos = json.loads(llamada.arguments or "{}")     # ¡llegan como TEXTO JSON!
            _anotar(corrida, f"🤖 Modelo pide    │ {llamada.name}({argumentos})", mostrar)
            resultado = ejecutar_herramienta(llamada.name, argumentos)    # ← TU código ejecuta
            _anotar(corrida, f"🔧 Resultado      │ {resultado[:160].replace(chr(10), ' ')}", mostrar)
            entrada.append({"type": "function_call_output", "call_id": llamada.call_id, "output": resultado})
    corrida.segundos = time.perf_counter() - t0
    return corrida


# ───────────────────────── Pieza 2 y 4 con Anthropic (Messages) ─────────────────────────


def bucle_anthropic(pregunta: str, max_pasos: int = 8, mostrar: bool = True) -> CorridaManual:
    """El bucle con el SDK de Anthropic. Fíjate en las diferencias de formato con la versión de OpenAI."""
    import anthropic

    cliente = anthropic.Anthropic(max_retries=2)
    modelo = modelo_economico("anthropic")
    herramientas = [{"name": e["name"], "description": e["description"], "input_schema": e["parameters"]} for e in ESQUEMAS]
    mensajes: list[dict] = [{"role": "user", "content": pregunta}]

    corrida = CorridaManual(modelo=modelo, proveedor="anthropic")
    t0 = time.perf_counter()
    _anotar(corrida, f"👤 Usuario        │ {pregunta}", mostrar)
    for _ in range(max_pasos):
        r = cliente.messages.create(model=modelo, max_tokens=1024, system=INSTRUCCIONES, tools=herramientas, messages=mensajes)
        corrida.llamadas += 1
        corrida.tokens_entrada += r.usage.input_tokens
        corrida.tokens_salida += r.usage.output_tokens
        mensajes.append({"role": "assistant", "content": r.content})

        if r.stop_reason != "tool_use":                          # aquí la señal es el motivo de parada, no una lista vacía
            corrida.respuesta = "".join(b.text for b in r.content if b.type == "text").strip()
            corrida.terminó = True
            _anotar(corrida, f"🤖 Modelo responde│ {corrida.respuesta}", mostrar)
            break
        resultados = []
        for bloque in r.content:
            if bloque.type != "tool_use":
                continue
            _anotar(corrida, f"🤖 Modelo pide    │ {bloque.name}({bloque.input})", mostrar)   # input ya llega como dict
            resultado = ejecutar_herramienta(bloque.name, bloque.input)
            _anotar(corrida, f"🔧 Resultado      │ {resultado[:160].replace(chr(10), ' ')}", mostrar)
            resultados.append({"type": "tool_result", "tool_use_id": bloque.id, "content": resultado})
        mensajes.append({"role": "user", "content": resultados})  # los resultados vuelven como un mensaje "user"
    corrida.segundos = time.perf_counter() - t0
    return corrida


def bucle(pregunta: str, proveedor: str, max_pasos: int = 8, mostrar: bool = True) -> CorridaManual:
    """Elige la versión según el proveedor: "anthropic", "openai" o "local"."""
    if proveedor == "anthropic":
        return bucle_anthropic(pregunta, max_pasos, mostrar)
    if proveedor == "openai":
        return bucle_responses(pregunta, max_pasos, mostrar)
    return bucle_chat_completions(pregunta, max_pasos, mostrar)
