"""Un agente con LangChain: crear el modelo, crear el agente, preguntar y ver qué pasó.

QUÉ ES UN AGENTE (en una línea)
    Un modelo que, en un BUCLE, puede pedir que se ejecuten herramientas y leer sus resultados,
    hasta que decide que ya puede responder.

        pregunta ─► MODELO ──¿pide una herramienta?──► sí ─► tu código la EJECUTA ─► resultado ─┐
                      ▲                                                                           │
                      └───────────────────────────── (vuelve al modelo con el resultado) ◄───────┘
                                  │ no
                                  ▼
                              respuesta final

    LangChain hace el bucle por ti: `create_agent(modelo, herramientas)`. El archivo
    agente_manual.py hace lo mismo "a mano" con el SDK del proveedor, para que veas qué hay debajo.

CÓMO ELEGIR EL MODELO (en tu .env)
    AGENT_PROVIDER=anthropic | openai | local   (si no existe, se usa LLM_PROVIDER; por defecto anthropic)
    LLM_MODEL=...                                (opcional: otro modelo de proveedor)
    LOCAL_LLM_URL / LOCAL_LLM_MODEL              (modelo local: LM Studio, Ollama…)

💰 COSTO
    Un agente hace VARIAS llamadas al modelo por cada pregunta (una para decidir, otra por cada
    resultado de herramienta, y la final), y en cada una reenvía TODO el historial. Por eso cuesta
    más que un chatbot. Con el modelo económico de un proveedor, una pregunta típica ≈ US$ 0.0005 a
    0.002. Con el modelo local, 0. `registrar_corrida()` suma los tokens reales de todas las llamadas.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

import openai
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from cliente_llm import Gasto, ResultadoLLM, costo_usd, hay_api_key, local_disponible, modelo_economico, modelo_local, url_local, variable_api_key

PROVEEDORES_AGENTE = ("anthropic", "openai", "local")

# Un agente sigue instrucciones como cualquier prompt: aquí van las reglas de CUÁNDO usar cada herramienta
# y CUÁNDO parar. Sin la última regla, los modelos pequeños repiten la misma llamada en bucle.
INSTRUCCIONES = """Eres el asistente de soporte de Andes Fibra, un proveedor ficticio de internet por fibra óptica.

Reglas:
- Para saber si un servicio funciona AHORA, usa las herramientas de estado. Nunca lo supongas.
- Si el usuario dice que algo no le funciona, revisa primero el estado del servicio afectado y, si hace falta, consulta el manual.
- Para políticas, precios, horarios y procedimientos, usa consultar_manual y responde solo con lo que devuelva.
- Si la pregunta no necesita herramientas (saludos, cuentas simples), responde directamente sin llamarlas.
- Cuando una herramienta ya te dio lo que necesitas, responde al usuario con eso. No repitas la misma llamada.
- Responde en español, breve y claro. No inventes datos que las herramientas no hayan devuelto."""

# Cuántos pasos (modelo → herramienta → modelo…) se permiten antes de cortar. Protege contra bucles
# que nunca terminan (y que cuestan dinero). LangChain lo llama "recursion_limit".
LIMITE_PASOS = 12


# ───────────────────────── Elegir y crear el modelo ─────────────────────────


def proveedor_agente(preferido: str | None = None) -> str:
    """AGENT_PROVIDER, o si no existe LLM_PROVIDER, o anthropic."""
    p = (preferido or os.getenv("AGENT_PROVIDER") or os.getenv("LLM_PROVIDER", "anthropic")).strip().lower()
    if p not in PROVEEDORES_AGENTE:
        raise ValueError(f"Proveedor '{p}' no válido. Usa: {', '.join(PROVEEDORES_AGENTE)}.")
    return p


def agente_disponible(proveedor: str | None = None) -> tuple[bool, str]:
    """¿Puedo usar este proveedor ahora? (llave configurada, o servidor local activo)."""
    p = proveedor_agente(proveedor)
    if p == "local":
        return local_disponible(modelo_local())
    return (True, "ok") if hay_api_key(p) else (False, f"falta {variable_api_key(p)} en tu .env")


def nombre_modelo(proveedor: str | None = None) -> str:
    """Nombre del modelo que se usará (para mostrarlo y para calcular el costo)."""
    p = proveedor_agente(proveedor)
    return modelo_local() if p == "local" else modelo_economico(p)


def crear_modelo(proveedor: str | None = None) -> BaseChatModel:
    """Un modelo de chat de LangChain para el proveedor elegido.

    Los tres usan la misma interfaz (`invoke`, `stream`, `bind_tools`), por eso el resto del código no cambia:
        anthropic → ChatAnthropic   (paquete langchain-anthropic)
        openai    → ChatOpenAI      (paquete langchain-openai; con los modelos gpt-6 y herramientas usa la API Responses)
        local     → ChatOpenAI apuntando a LM Studio (base_url): LM Studio habla el formato Chat Completions de OpenAI

    No fijamos `temperature` en los modelos de proveedor: algunos modelos recientes solo aceptan
    el valor por defecto (la Sesión 2 lo explica). En el modelo local sí: 0 para que sea repetible.
    """
    p = proveedor_agente(proveedor)
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=modelo_economico(p), max_tokens=1024)
    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=modelo_economico(p))
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=modelo_local(),
        base_url=url_local(),
        api_key=os.getenv("LOCAL_LLM_API_KEY", "no-se-usa"),   # el servidor local no la valida, pero el SDK exige un texto
        temperature=0,
        max_retries=0,
        timeout=300,                                           # sin GPU, un modelo local puede tardar
    )


def crear_agente(herramientas: Sequence, proveedor: str | None = None, modelo: BaseChatModel | None = None,
                 instrucciones: str = INSTRUCCIONES, **opciones):
    """El agente: `create_agent(modelo, herramientas, system_prompt=…)`. Devuelve un grafo compilado.

    `opciones` pasa tal cual lo demás que acepta create_agent: `response_format` (salida estructurada),
    `checkpointer` (memoria entre turnos), `middleware` (límites, reintentos, aprobación humana…).
    """
    return create_agent(modelo or crear_modelo(proveedor), tools=list(herramientas), system_prompt=instrucciones, **opciones)


# ───────────────────────── Preguntar y mirar qué pasó ─────────────────────────


def preguntar(agente, pregunta: str, hilo: str | None = None, limite_pasos: int = LIMITE_PASOS) -> dict:
    """Una pregunta al agente. Devuelve el estado final: `resultado["messages"]` es TODA la conversación.

    `hilo` (thread_id) solo importa si el agente tiene checkpointer: con el mismo hilo, el agente recuerda
    los turnos anteriores. `limite_pasos` corta los bucles (lanza GraphRecursionError si se excede).
    """
    config: dict = {"recursion_limit": limite_pasos}
    if hilo:
        config["configurable"] = {"thread_id": hilo}
    return agente.invoke({"messages": [{"role": "user", "content": pregunta}]}, config)


def texto_de(mensaje: BaseMessage) -> str:
    """El texto de un mensaje, sin importar si el proveedor lo entrega como texto o como bloques."""
    if isinstance(mensaje.content, str):
        return mensaje.content
    return "".join(b.get("text", "") for b in mensaje.content if isinstance(b, dict) and b.get("type") == "text")


def respuesta_final(mensajes: Sequence[BaseMessage]) -> str:
    return texto_de(mensajes[-1]).strip() if mensajes else ""


def herramientas_usadas(mensajes: Sequence[BaseMessage]) -> list[str]:
    """Nombres de las herramientas que el modelo pidió, en orden (con repeticiones)."""
    return [llamada["name"] for m in mensajes if isinstance(m, AIMessage) for llamada in m.tool_calls]


def mostrar_mensajes(mensajes: Sequence[BaseMessage]) -> None:
    """Imprime la conversación completa: es la "traza" del agente. Aquí se ve cada decisión del modelo."""
    for m in mensajes:
        if isinstance(m, HumanMessage):
            print(f"👤 Usuario        │ {texto_de(m)}")
        elif isinstance(m, AIMessage):
            for llamada in m.tool_calls:
                print(f"🤖 Modelo pide    │ {llamada['name']}({llamada['args']})")
            if texto_de(m).strip():
                print(f"🤖 Modelo responde│ {texto_de(m).strip()}")
        elif isinstance(m, ToolMessage):
            resultado = texto_de(m).replace("\n", " ")
            print(f"🔧 Resultado      │ {resultado[:160]}{'…' if len(resultado) > 160 else ''}")


def uso_de_tokens(mensajes: Sequence[BaseMessage]) -> tuple[int, int]:
    """(tokens de entrada, tokens de salida) sumados sobre TODAS las llamadas al modelo del agente."""
    entrada = salida = 0
    for m in mensajes:
        if isinstance(m, AIMessage) and m.usage_metadata:
            entrada += m.usage_metadata.get("input_tokens", 0)
            salida += m.usage_metadata.get("output_tokens", 0)
    return entrada, salida


def llamadas_al_modelo(mensajes: Sequence[BaseMessage]) -> int:
    """Cuántas veces se llamó al modelo (= cuántos AIMessage hay)."""
    return sum(isinstance(m, AIMessage) for m in mensajes)


def registrar_corrida(gasto: Gasto, mensajes: Sequence[BaseMessage], proveedor: str, segundos: float) -> ResultadoLLM:
    """Convierte una corrida del agente en un ResultadoLLM y la suma al contador de gasto del curso."""
    modelo = nombre_modelo(proveedor)
    entrada, salida = uso_de_tokens(mensajes)
    return gasto.registrar(ResultadoLLM(
        texto=respuesta_final(mensajes),
        modelo=f"local:{modelo}" if proveedor == "local" else modelo,
        tokens_entrada=entrada,
        tokens_salida=salida,
        segundos_total=segundos,
        costo=0.0 if proveedor == "local" else costo_usd(modelo, entrada, salida),
        proveedor=proveedor,
    ))


def correr(agente, pregunta: str, proveedor: str, gasto: Gasto | None = None, hilo: str | None = None,
           mostrar: bool = True) -> dict:
    """preguntar() + medir el tiempo + (opcional) mostrar la traza y registrar el gasto. Devuelve el estado final."""
    t0 = time.perf_counter()
    try:
        estado = preguntar(agente, pregunta, hilo)
    except GraphRecursionError:
        print(f"⚠️ El agente no terminó en {LIMITE_PASOS} pasos y se cortó (límite de pasos: protege tu bolsillo de los bucles).")
        return {"messages": [], "error": "limite_de_pasos"}
    except openai.BadRequestError as error:
        # Modelos locales pequeños: la petición de herramienta no respeta el formato y LM Studio la rechaza (HTTP 400).
        print(f"⚠️ El servidor rechazó la salida del modelo (HTTP 400): {str(getattr(error, 'message', error))[:160]}")
        return {"messages": [], "error": "salida_rechazada"}
    segundos = time.perf_counter() - t0
    # Con memoria (hilo), `estado["messages"]` trae TODA la conversación; lo de esta pregunta empieza en el último mensaje del usuario.
    todos = estado["messages"]
    inicio = max((i for i, m in enumerate(todos) if isinstance(m, HumanMessage)), default=0)
    mensajes = todos[inicio:]
    if gasto is not None:
        registrar_corrida(gasto, mensajes, proveedor, segundos)
    if mostrar:
        mostrar_mensajes(mensajes)
        entrada, salida = uso_de_tokens(mensajes)
        print(f"   ({llamadas_al_modelo(mensajes)} llamadas al modelo · {entrada} tokens de entrada · {salida} de salida · {segundos:.1f} s)")
    return estado


# ───────────────────────── Un modelo de juguete (sin LLM, sin costo) ─────────────────────────


class ModeloDeJuguete(GenericFakeChatModel):
    """Un "modelo" que no piensa: devuelve, en orden, las respuestas de un guion que tú escribes.

    Sirve para ver el bucle del agente sin depender de ningún LLM: tú decides en qué momento el
    "modelo" pide una herramienta y cuándo responde, y LangChain hace el resto (ejecutar la herramienta,
    agregar su resultado a la conversación, volver a llamar al modelo). También es la forma de PROBAR un
    agente sin gastar: los tests de agentes suelen usar modelos falsos como este.
    """

    def bind_tools(self, tools, **kwargs):   # un modelo real recibe aquí los esquemas; el de juguete los ignora
        return self


def pedir(herramienta: str, **argumentos) -> AIMessage:
    """Un paso del guion: el "modelo" pide ejecutar `herramienta` con esos argumentos."""
    return AIMessage(content="", tool_calls=[{"name": herramienta, "args": argumentos, "id": f"llamada_{herramienta}"}])


def responder(texto: str) -> AIMessage:
    """Un paso del guion: el "modelo" responde con texto (y el bucle termina)."""
    return AIMessage(content=texto)


def modelo_de_juguete(*guion: AIMessage) -> ModeloDeJuguete:
    return ModeloDeJuguete(messages=iter(guion))


# ───────────────────────── Evaluar la ELECCIÓN de herramientas ─────────────────────────


def evaluar_herramientas(agente, preguntas: list[dict], proveedor: str, gasto: Gasto | None = None,
                         disponibles: Sequence[str] | None = None, mostrar: bool = True) -> dict:
    """¿El agente eligió las herramientas correctas? Una prueba por pregunta, sin juzgar el texto de la respuesta.

    Cada pregunta trae `deben` (herramientas que SÍ tiene que llamar) y `permitidas` (otras que no estorban). Pasa la
    prueba si llamó todas las `deben` y NINGUNA fuera de `deben` + `permitidas`: llamar herramientas de más también
    es un fallo (cuesta dinero, tarda y, si la herramienta escribe, puede hacer daño). Es la misma idea de las
    Sesiones 3 y 5: medir con casos fijos en vez de "a ojo". `disponibles` limita la prueba a las preguntas
    cuyas herramientas existen en el agente.
    """
    resultados = []
    for p in preguntas:
        if disponibles is not None and not set(p["deben"]) <= set(disponibles):
            continue                                                   # la herramienta no está en este agente
        estado = correr(agente, p["pregunta"], proveedor, gasto, mostrar=False)
        usadas = herramientas_usadas(estado["messages"])
        ok = bool(estado["messages"]) and set(p["deben"]) <= set(usadas) <= set(p["deben"]) | set(p["permitidas"])
        resultados.append({"id": p["id"], "pregunta": p["pregunta"], "usadas": usadas, "ok": ok,
                           "llamadas": llamadas_al_modelo(estado["messages"]), "error": estado.get("error", "")})
        if mostrar:
            print(f"{'✅' if ok else '❌'} {p['id']} {p['pregunta'][:58]:<58} → {usadas or 'sin herramientas'}"
                  + (f"  ⚠️ {resultados[-1]['error']}" if resultados[-1]["error"] else ""))
    aciertos = sum(r["ok"] for r in resultados)
    if mostrar:
        print(f"\nElección correcta de herramientas: {aciertos}/{len(resultados)}")
    return {"aciertos": aciertos, "total": len(resultados), "resultados": resultados}


# ───────────────────────── Streaming: ver la respuesta mientras se escribe ─────────────────────────


def transmitir(agente, pregunta: str, hilo: str | None = None, limite_pasos: int = LIMITE_PASOS) -> str:
    """Imprime la respuesta del agente token a token y devuelve el texto completo.

    `stream_mode="messages"` entrega pares (trozo, metadatos). Mostramos solo el texto que escribe el MODELO
    (nodo "model"); los resultados de las herramientas llegan como ToolMessage y no se imprimen aquí.
    En la práctica, el streaming mejora la sensación de velocidad: el usuario lee mientras el modelo escribe.
    """
    config: dict = {"recursion_limit": limite_pasos}
    if hilo:
        config["configurable"] = {"thread_id": hilo}
    texto = ""
    for trozo, meta in agente.stream({"messages": [{"role": "user", "content": pregunta}]}, config, stream_mode="messages"):
        if meta.get("langgraph_node") == "model" and isinstance(trozo, BaseMessage):
            nuevo = texto_de(trozo)
            if nuevo:
                print(nuevo, end="", flush=True)
                texto += nuevo
    print()
    return texto
