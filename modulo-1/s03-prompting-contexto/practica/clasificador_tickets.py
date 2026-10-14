"""Práctica S3 · Clasificador de tickets.

PREGUNTA QUE RESPONDE
    ¿Puedo convertir el texto libre de un usuario en datos confiables usando un
    prompt VERSIONADO y una VALIDACIÓN de la salida?

QUÉ HACE
    1. Carga una plantilla de prompt desde un archivo versionado (prompts/…/vN.j2).
    2. La rellena con el ticket, las categorías permitidas y los ejemplos.
    3. Llama al modelo y recibe texto.
    4. Valida ese texto contra un esquema: si cumple, devuelve datos; si no, falla
       con un mensaje claro (nunca deja pasar un dato inválido).
    5. Con --evaluar, mide una versión del prompt sobre 10 casos de prueba.

LOS TRES MÓDULOS QUE USA
    cliente_llm.py   llama al LLM de tu proveedor (reintentos, medición de costo)
    plantillas.py    carga y rellena las plantillas versionadas
    validacion.py    valida la salida del modelo

CÓMO EJECUTARLO (desde la carpeta s03-prompting-contexto/)
    uv run python practica/clasificador_tickets.py "El switch del piso 3 reinicia sin parar"
    uv run python practica/clasificador_tickets.py "No puedo entrar a la VPN" --version v2
    uv run python practica/clasificador_tickets.py --evaluar --version v3
    uv run python practica/clasificador_tickets.py --evaluar --version v1     # compara v1 vs v3

💰 COSTO
    Cada ticket clasificado = UNA llamada al modelo económico de tu proveedor (Anthropic u OpenAI,
    según LLM_PROVIDER en tu .env) ≈ US$ 0.0001.
    --evaluar corre 10 casos = 10 llamadas ≈ US$ 0.001. Evaluar las 3 versiones ≈ US$ 0.003.

IDEAS PARA EXPERIMENTAR (ver practica/README.md)
    - Crea prompts/clasificador_tickets/v4.j2, anota el cambio en prompts/CHANGELOG.md
      y compara con --evaluar si mejoró o empeoró.
    - Añade casos difíciles a datos/casos_tickets.json.
"""

import argparse
import json
import sys
from pathlib import Path

# RAIZ = la carpeta de la sesión (s03-prompting-contexto/). Este script vive en practica/,
# así que subimos un nivel. La usamos para encontrar los módulos, los prompts y los datos
# sin importar desde qué carpeta ejecutes el comando.
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from dotenv import find_dotenv, load_dotenv  # noqa: E402 (el import va tras ajustar sys.path)

from cliente_llm import Gasto, llamar_llm, modelo_economico  # noqa: E402
from plantillas import renderizar, versiones  # noqa: E402
from validacion import CATEGORIAS, ClasificacionTicket, SalidaInvalida, validar_clasificacion  # noqa: E402

NOMBRE_PROMPT = "clasificador_tickets"   # carpeta dentro de prompts/
# El modelo económico del proveedor activo (claude-haiku-5-5 con Anthropic, gpt-6-luna con OpenAI)
# es suficiente para clasificar. Se puede cambiar con LLM_MODEL en el .env.


def clasificar(ticket: str, version: str, gasto: Gasto) -> ClasificacionTicket:
    """Clasifica UN ticket y devuelve un objeto validado (💰 hace una llamada de pago).

    Lanza SalidaInvalida si el modelo responde algo que no cumple el esquema.
    """
    # 1) Ejemplos para las versiones few-shot (v3). Las versiones que no los usan los ignoran.
    ejemplos = json.loads(
        (RAIZ / "prompts" / NOMBRE_PROMPT / "ejemplos.json").read_text(encoding="utf-8")
    )

    # 2) Prompt = plantilla versionada + variables. Si falta una variable, renderizar() falla
    #    en voz alta ANTES de gastar dinero (plantillas.py usa StrictUndefined).
    prompt = renderizar(NOMBRE_PROMPT, version, ticket=ticket, categorias=CATEGORIAS, ejemplos=ejemplos)

    # 3) Llamada al modelo.
    #    - streaming=False: no necesitamos ver el texto fluir, solo el resultado final.
    #    - max_tokens=400: tope de salida; la respuesta JSON es corta, esto protege el costo.
    #    - esfuerzo="minimo": clasificar es simple, no hace falta que el modelo razone mucho
    #      (el razonamiento se cobra como tokens de salida).
    resultado = llamar_llm(
        prompt, modelo=modelo_economico(), max_tokens=400, streaming=False,
        esfuerzo="minimo",
    )
    gasto.registrar(resultado)   # suma al contador de gasto de esta ejecución

    # 4) Validación: JSON válido NO es lo mismo que JSON correcto. Esta función comprueba
    #    la estructura y que categoría y prioridad sean valores permitidos.
    return validar_clasificacion(resultado.texto)


def evaluar(version: str, gasto: Gasto) -> None:
    """Mide una versión del prompt sobre los casos de datos/casos_tickets.json.

    Dos métricas por versión:
      - JSON válido:        ¿la salida pasó la validación del esquema?
      - Categoría correcta: ¿coincide con la esperada (o con alguna aceptable)?
    Es una mini-evaluación: la idea central de la Semana 3 del curso.
    """
    casos = json.loads((RAIZ / "datos" / "casos_tickets.json").read_text(encoding="utf-8"))
    validos = correctos = 0

    for caso in casos:
        try:
            obj = clasificar(caso["texto"], version, gasto)
        except SalidaInvalida as error:
            # La respuesta no cumplió el esquema: lo contamos como inválida y seguimos.
            print(f"{caso['id']}  ❌ salida inválida: {error}")
            continue

        validos += 1
        # Algunos casos son ambiguos a propósito y traen varias respuestas aceptables.
        aceptables = caso.get("aceptables", [caso["esperada"]])
        acierto = obj.categoria in aceptables
        correctos += acierto
        marca = "✅" if acierto else "⚠️"
        print(f"{caso['id']}  {marca} {obj.categoria:<9} (esperada: {caso['esperada']})")

    total = len(casos)
    print(f"\n{version}: JSON válido {validos}/{total} · categoría correcta {correctos}/{total}")
    print("Con tan pocos casos, 1 punto de diferencia puede ser ruido: es una demostración de método.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Clasificador de tickets")
    ap.add_argument("ticket", nargs="?", help="Texto del ticket a clasificar")
    ap.add_argument("--version", default="v3", choices=versiones(NOMBRE_PROMPT),
                    help="Versión de la plantilla de prompt")
    ap.add_argument("--evaluar", action="store_true",
                    help="Corre los 10 casos de prueba en lugar de clasificar un ticket")
    args = ap.parse_args()

    # Carga el .env de la raíz del repositorio (LLM_PROVIDER y la API key).
    load_dotenv(find_dotenv(usecwd=True))

    gasto = Gasto()   # acumula el costo real de las llamadas de esta ejecución

    if args.evaluar:
        evaluar(args.version, gasto)
    elif args.ticket:
        try:
            print(clasificar(args.ticket, args.version, gasto).model_dump_json(indent=2))
        except SalidaInvalida as error:
            # Qué hacer con una salida inválida es una decisión de diseño: aquí fallamos con
            # un mensaje claro. Otras opciones: reintentar explicando el error (pedir_json de
            # la Sesión 2) o enviar el ticket a una persona.
            print("❌ El modelo devolvió una salida inválida:", error)
    else:
        ap.error("Indica un ticket o usa --evaluar")

    gasto.resumen()   # imprime cuántas llamadas se hicieron y cuánto costaron
    return 0


if __name__ == "__main__":
    sys.exit(main())
