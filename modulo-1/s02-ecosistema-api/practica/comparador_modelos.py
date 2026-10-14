"""Práctica S2 · Comparador de modelos.

PREGUNTA QUE RESPONDE
    ¿Qué diferencia hay, en respuesta, tokens, tiempo y costo, entre un modelo
    propietario (por API: Anthropic u OpenAI) y uno open-weight (local, con LM Studio)?

QUÉ HACE
    1. Envía el MISMO prompt a dos modelos, uno detrás del otro.
    2. Muestra cada respuesta EN VIVO (streaming).
    3. Al final imprime una tabla: tokens, TTFT, tiempo total, tokens/segundo y costo.
    Los reintentos ante errores transitorios (429, 5xx, red) los resuelve
    cliente_llm.py, que es el módulo reutilizable de la sesión.

CÓMO EJECUTARLO (desde la carpeta s02-ecosistema-api/)
    uv run python practica/comparador_modelos.py "Explica qué es un balanceador de carga en 3 frases"
    uv run python practica/comparador_modelos.py "..." --a api --b local
    uv run python practica/comparador_modelos.py "..." --a anthropic:claude-haiku-5-5 --b openai:gpt-6-luna
    uv run python practica/comparador_modelos.py "..." --a api --b local:llama-3.2-3b-instruct

CÓMO ELEGIR CADA MODELO (--a y --b)
    "api"                 → el modelo económico del proveedor activo (LLM_PROVIDER en tu .env).
    "<proveedor>:<modelo>" → un modelo concreto: anthropic:... u openai:...
    "<modelo>"            → un modelo del proveedor activo (por ejemplo, gpt-6-luna).
    "local"               → el modelo local por defecto (LOCAL_LLM_MODEL en tu .env).
    "local:<nombre>"      → un modelo local concreto (el nombre que muestra LM Studio).
    Los modelos locales corren en TU computadora, sin cobro por token, y necesitan el servidor activo.

💰 COSTO
    Cada modelo por API (Anthropic u OpenAI) que uses CONSUME DINERO (centavos). Con el modelo
    económico y --max-tokens 400 una ejecución cuesta una fracción de centavo (≈ US$ 0.0003).
    El modelo local no cobra por token, pero usa tu CPU, tu RAM y tu electricidad.

IDEAS PARA EXPERIMENTAR (ver practica/README.md)
    - Prueba prompts de distinto tipo: cálculo, redacción, código, español vs inglés.
    - Baja --max-tokens a 50 y observa cómo se corta la respuesta.
"""

import argparse
import sys
from pathlib import Path

# Este script vive en practica/, pero cliente_llm.py está una carpeta más arriba.
# Añadimos esa carpeta al path para poder hacer `from cliente_llm import ...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import find_dotenv, load_dotenv  # noqa: E402 (el import va tras ajustar sys.path)

from cliente_llm import (  # noqa: E402
    PRECIOS,
    PROVEEDORES,
    ResultadoLLM,
    llamar_llm,
    llamar_local,
    local_disponible,
    modelo_local,
    proveedor_activo,
)


def imprimir_en_vivo(trozo: str) -> None:
    """Callback de streaming: muestra cada trozo de texto apenas llega, sin salto de línea."""
    print(trozo, end="", flush=True)


def llamar(modelo: str, prompt: str, max_tokens: int) -> ResultadoLLM:
    """Envía `prompt` al modelo indicado y devuelve un ResultadoLLM con todas las métricas."""
    print(f"\n=== {modelo} ===")

    # ── Caso 1: modelo open-weight local ("local" o "local:<nombre>") ─────────
    if modelo == "local" or modelo.startswith("local:"):
        # Sin nombre explícito se usa el modelo por defecto (LOCAL_LLM_MODEL o el de cliente_llm.py).
        nombre = modelo.removeprefix("local:") if modelo != "local" else modelo_local()

        # Verificamos ANTES de llamar que el servidor local corre y ofrece el modelo,
        # para dar un mensaje útil ("abre LM Studio y pulsa Start Server") en lugar de un error críptico.
        disponible, mensaje = local_disponible(nombre)
        if not disponible:
            sys.exit(mensaje)

        return llamar_local(prompt, modelo=nombre, max_tokens=max_tokens,
                            al_recibir_texto=imprimir_en_vivo)

    # ── Caso 2: modelo por API (esto cuesta dinero) ───────────────────────────
    # "api" = modelo económico del proveedor activo; "<proveedor>:<modelo>" = uno concreto.
    # El esfuerzo "minimo" apaga o reduce el razonamiento interno de los modelos que "piensan":
    # ese razonamiento se cobra como salida y retrasa el primer token.
    proveedor, nombre = None, None
    if modelo != "api":
        proveedor, _, resto = modelo.partition(":")
        if proveedor in PROVEEDORES:
            nombre = resto
        else:                       # sin prefijo: es un modelo del proveedor activo
            proveedor, nombre = None, modelo
    return llamar_llm(prompt, proveedor=proveedor, modelo=nombre or None, max_tokens=max_tokens,
                      al_recibir_texto=imprimir_en_vivo, esfuerzo="minimo")


def imprimir_tabla(resultados: list[ResultadoLLM]) -> None:
    """Compara los resultados en una tabla: una fila por modelo."""
    print("\n\n" + "=" * 79)
    print(f"{'modelo':<28}{'tok in':>7}{'tok out':>8}{'TTFT s':>8}{'total s':>9}{'tok/s':>8}{'USD':>11}")
    print("-" * 79)
    for r in resultados:
        # TTFT = tiempo hasta el primer token (lo que el usuario espera antes de ver algo).
        # tok/s = velocidad de generación una vez que empezó a responder.
        # Pueden faltar si la respuesta vino vacía, por eso se protegen con un guion.
        tok_s = f"{r.tokens_por_segundo:.1f}" if r.tokens_por_segundo else "-"
        ttft = f"{r.ttft:.2f}" if r.ttft else "-"
        print(f"{r.modelo:<28}{r.tokens_entrada:>7}{r.tokens_salida:>8}{ttft:>8}"
              f"{r.segundos_total:>9.2f}{tok_s:>8}{r.costo:>11.6f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Comparador de modelos")
    ap.add_argument("prompt", help="Texto que se envía a los dos modelos")
    ap.add_argument("--a", default="api",
                    help=f"Modelo A: 'api' (el económico de tu proveedor), '<proveedor>:<modelo>' o un modelo. Con precio conocido: {', '.join(PRECIOS)}")
    ap.add_argument("--b", default="local",
                    help="Modelo B: 'local' o 'local:<nombre>' (en tu computadora), o igual que --a")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="Tope de tokens de salida: es el control más directo del costo")
    args = ap.parse_args()

    # Carga el .env de la raíz del repositorio (LLM_PROVIDER, la API key y la configuración local).
    load_dotenv(find_dotenv(usecwd=True))

    # Llamamos a los dos modelos con exactamente el mismo prompt y el mismo tope.
    resultados = [llamar(modelo, args.prompt, args.max_tokens) for modelo in (args.a, args.b)]

    imprimir_tabla(resultados)
    return 0


if __name__ == "__main__":
    sys.exit(main())
