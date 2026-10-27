"""Práctica S8 · Bot de estado de servicios (un agente con herramientas, con LangChain).

PREGUNTA QUE RESPONDE
    ¿Puedo dar a un modelo herramientas (consultar el estado de un servicio, buscar en un manual) y dejar
    que ÉL decida cuándo usarlas, en vez de programar yo cada camino?

    (A diferencia del RAG de la Sesión 5, aquí NO hay un flujo fijo: el modelo elige qué herramienta usar,
    con qué argumentos, cuántas veces, o ninguna. Ese poder de decidir es lo que hace que sea un agente.)

QUÉ HACE
    Un asistente de soporte de "Andes Fibra" (empresa ficticia) con estas herramientas:
        listar_servicios   estado de todos los servicios                (lee)
        estado_servicio    detalle de UN servicio                       (lee)
        consultar_manual   busca en el manual de soporte (el RAG de la Sesión 5, como herramienta)   (lee)
        reportar_falla     crea un ticket (simulado)                    (ESCRIBE: solo con --herramientas todas)

    Modos:
        pregunta           una pregunta; muestra la traza (qué pidió el modelo, qué devolvieron las herramientas)
        --chat             conversación con memoria (el mismo hilo recuerda los turnos anteriores)
        --a-mano           el mismo flujo con el SDK del proveedor, SIN LangChain (para ver qué hace por ti)
        --evaluar          mide si el agente ELIGE las herramientas correctas (7 preguntas con respuesta esperada; llamar herramientas de más también es un fallo)
        --ver-esquemas     muestra lo que el modelo recibe de cada herramienta (nombre, descripción, parámetros)

CÓMO EJECUTARLO (desde la carpeta s08-agentes-langchain/)
    uv run python practica/bot_estado_servicios.py "¿Cómo está la app móvil?"
    uv run python practica/bot_estado_servicios.py "¿Hay algún servicio caído?"
    uv run python practica/bot_estado_servicios.py "No puedo pagar con tarjeta, ¿qué hago?"
    uv run python practica/bot_estado_servicios.py --a-mano "¿Cómo está la app móvil?"
    uv run python practica/bot_estado_servicios.py --herramientas todas "Abre un ticket: los pagos con tarjeta fallan desde las 5 pm"
    uv run python practica/bot_estado_servicios.py --chat
    uv run python practica/bot_estado_servicios.py --evaluar
    uv run python practica/bot_estado_servicios.py --proveedor local "¿Cuánto es el 15 % de 39?"
    uv run python practica/bot_estado_servicios.py --ver-esquemas

💰 COSTO
    - Un agente hace varias llamadas al modelo por pregunta (decidir, leer cada resultado, responder).
      Con el modelo económico de Anthropic u OpenAI: ≈ US$ 0.0005 a 0.002 por pregunta.
      --evaluar (7 preguntas): ≈ US$ 0.005 a 0.01.
    - Con --proveedor local (LM Studio): sin costo por token. Un modelo de 3B parámetros elige peor las
      herramientas; para esta práctica rinde mucho mejor uno de ~12B (LOCAL_LLM_MODEL en tu .env).
    - Los embeddings del manual son locales: gratis.
    - El programa muestra los tokens y el costo de cada pregunta.

IDEAS PARA EXPERIMENTAR (ver practica/README.md)
    - Cambia la descripción de una herramienta (herramientas.py) y mira si el modelo la elige distinto.
    - Edita datos/servicios.json (pon el portal en "caido") y vuelve a preguntar: el agente lee el mundo, no lo recuerda.
    - Compara --herramientas lectura con --herramientas todas: ¿qué pasa cuando el modelo tiene la herramienta que escribe?
    - Compara proveedores con la misma pregunta y la misma traza.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Este script vive en practica/, pero los módulos están una carpeta más arriba.
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from dotenv import find_dotenv, load_dotenv  # noqa: E402 (el import va tras ajustar sys.path)

load_dotenv(find_dotenv(usecwd=True))

from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import agente_manual  # noqa: E402
import herramientas  # noqa: E402
import servicios  # noqa: E402
from agentes import (  # noqa: E402
    agente_disponible, correr, crear_agente, evaluar_herramientas, nombre_modelo, proveedor_agente,
)
from cliente_llm import Gasto  # noqa: E402

# Qué herramientas recibe el agente. Mínimo privilegio: por defecto, ninguna que escriba.
CONJUNTOS = {
    "estado": [herramientas.listar_servicios, herramientas.estado_servicio],
    "lectura": herramientas.SOLO_LECTURA,
    "todas": herramientas.TODAS,
}


def ver_esquemas(conjunto: list) -> None:
    """Lo que el modelo REALMENTE ve de cada herramienta. Si el modelo elige mal, empieza por aquí."""
    for h in conjunto:
        print(json.dumps(convert_to_openai_tool(h)["function"], ensure_ascii=False, indent=2), "\n")


def modo_a_mano(pregunta: str, proveedor: str) -> None:
    c = agente_manual.bucle(pregunta, proveedor)
    print(f"   ({c.llamadas} llamadas al modelo · {c.tokens_entrada} tokens de entrada · {c.tokens_salida} de salida"
          f" · {c.segundos:.1f} s · US$ {c.costo:.6f})")


def modo_chat(agente, proveedor: str, gasto: Gasto) -> None:
    """Un bucle de conversación. El agente tiene checkpointer, así que con el mismo hilo recuerda lo anterior."""
    print("Chat con el asistente de Andes Fibra. Escribe 'salir' para terminar.\n")
    hilo = f"chat-{int(time.time())}"
    while True:
        try:
            pregunta = input("Tú › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pregunta.lower() in ("salir", "exit", "quit", ""):
            break
        correr(agente, pregunta, proveedor, gasto, hilo=hilo)
        print()


def modo_evaluar(agente, proveedor: str, gasto: Gasto, conjunto: list) -> None:
    preguntas = json.loads((RAIZ / "datos" / "preguntas_agente.json").read_text(encoding="utf-8"))
    nombres = [h.name for h in conjunto]
    servicios.reiniciar_tickets()
    evaluar_herramientas(agente, preguntas, proveedor, gasto, disponibles=nombres)
    omitidas = [p["id"] for p in preguntas if not set(p["deben"]) <= set(nombres)]
    if omitidas:
        print(f"(Preguntas omitidas por falta de la herramienta: {', '.join(omitidas)}. Usa --herramientas todas.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bot de estado de servicios: un agente con herramientas (LangChain).")
    ap.add_argument("pregunta", nargs="?", help="la pregunta para el agente")
    ap.add_argument("--proveedor", choices=["anthropic", "openai", "local"], help="por defecto AGENT_PROVIDER o LLM_PROVIDER")
    ap.add_argument("--herramientas", choices=list(CONJUNTOS), default="lectura",
                    help="estado = solo las de estado · lectura = + manual (por defecto) · todas = + reportar_falla (escribe)")
    ap.add_argument("--chat", action="store_true", help="conversación con memoria")
    ap.add_argument("--a-mano", action="store_true", help="el bucle con el SDK del proveedor, sin LangChain")
    ap.add_argument("--evaluar", action="store_true", help="mide la elección de herramientas con 7 preguntas")
    ap.add_argument("--ver-esquemas", action="store_true", help="muestra lo que el modelo recibe de cada herramienta")
    args = ap.parse_args()

    conjunto = CONJUNTOS[args.herramientas]
    if args.ver_esquemas:
        ver_esquemas(conjunto)
        return
    if not (args.pregunta or args.chat or args.evaluar):
        ap.error("escribe una pregunta, o usa --chat, --evaluar o --ver-esquemas")

    proveedor = proveedor_agente(args.proveedor)
    ok, motivo = agente_disponible(proveedor)
    if not ok:
        sys.exit(f"No puedo usar el proveedor '{proveedor}': {motivo}.")
    print(f"Proveedor: {proveedor} · modelo: {nombre_modelo(proveedor)} · herramientas: {[h.name for h in conjunto]}\n")

    if args.a_mano:
        if not args.pregunta:
            ap.error("--a-mano necesita una pregunta")
        modo_a_mano(args.pregunta, proveedor)
        return

    gasto = Gasto()
    # El checkpointer es lo que da memoria al agente; sin él, cada pregunta empieza de cero.
    agente = crear_agente(conjunto, proveedor, checkpointer=InMemorySaver() if args.chat else None)
    if args.chat:
        modo_chat(agente, proveedor, gasto)
    elif args.evaluar:
        modo_evaluar(agente, proveedor, gasto, conjunto)
    else:
        correr(agente, args.pregunta, proveedor, gasto)
    if servicios.TICKETS:
        print(f"\nTickets creados (simulados): {servicios.TICKETS}")
    gasto.resumen()


if __name__ == "__main__":
    main()
