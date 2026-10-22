"""Práctica S5 · Un RAG simple con citas.

PREGUNTA QUE RESPONDE
    ¿Puedo hacer que un LLM responda preguntas sobre MIS documentos, citando de dónde sale
    cada dato y diciendo "no sé" cuando el documento no lo contiene?

    (Esto NO es un agente: es un flujo fijo de pasos. No decide nada, no usa herramientas.)

QUÉ HACE
    1. --ingestar: lee datos/manual_soporte.md, lo parte en fragmentos, los vectoriza y los guarda
       en el almacén elegido (memoria, Neon o Pinecone).
    2. Con una pregunta: busca los k fragmentos más parecidos, arma el prompt (plantilla versionada
       prompts/rag_respuesta/vN.j2), llama al LLM y muestra la respuesta, las FUENTES citadas y el costo.
    3. --evaluar: mide la recuperación (¿llegó la sección correcta al prompt?) sin llamar al LLM.

CÓMO EJECUTARLO (desde la carpeta s05-fundamentos-rag/)
    uv run python practica/rag_simple.py --almacen neon --ingestar
    uv run python practica/rag_simple.py --almacen neon "¿Cuánto cuesta el plan de 300 Mbps?"
    uv run python practica/rag_simple.py --almacen neon "¿Ofrecen televisión por cable?"      # fuera de alcance
    uv run python practica/rag_simple.py --almacen memoria --ingestar "¿Hasta cuándo puedo pagar sin recargo?"
    uv run python practica/rag_simple.py --almacen neon --evaluar --estrategia oraciones --max-palabras 40
    uv run python practica/rag_simple.py --almacen neon --proveedor local "¿A qué hora atienden los domingos?"
    uv run python practica/rag_simple.py --almacen neon --proveedor todos "¿Qué pasa si no pago a tiempo?"   # compara

💰 COSTO
    - Embeddings: gratis con el modelo local.
    - Cada pregunta con Anthropic u OpenAI = 1 llamada al modelo económico ≈ US$ 0.0005.
    - Con --proveedor local (LM Studio): sin costo por token.
    - --evaluar y --ingestar no llaman al LLM.

IDEAS PARA EXPERIMENTAR (ver practica/README.md)
    - Cambia --estrategia y --max-palabras y compara con --evaluar.
    - Cambia --version v1 por v2 (otro prompt) y compara las respuestas.
    - Usa --umbral para no llamar al LLM cuando nada se parece a la pregunta.
    - Escribe tu propio documento .md (con secciones ##) y pásalo con --documento.
"""

import argparse
import json
import sys
from pathlib import Path

# Este script vive en practica/, pero los módulos están una carpeta más arriba.
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from dotenv import find_dotenv, load_dotenv  # noqa: E402 (el import va tras ajustar sys.path)

from almacenes import crear_almacen  # noqa: E402
from embeddings import modelo_embeddings, proveedor_embeddings  # noqa: E402
from rag import evaluar_recuperacion, ingestar, proveedor_generacion, proveedores_disponibles, responder  # noqa: E402


def mostrar(r) -> None:
    print(f"\nPregunta: {r.pregunta}")
    print("Fragmentos recuperados (lo que vio el modelo):")
    for i, f in enumerate(r.fragmentos, start=1):
        marca = "★" if i in r.citas else " "
        print(f"  {marca}[{i}] {f.puntaje:.3f}  {f.metadatos['seccion']}")
    print(f"\nRespuesta:\n  {r.texto}")
    if r.citas_invalidas:
        print(f"\n⚠️  Citó fragmentos que NO existen: {sorted(r.citas_invalidas)} (posible invención)")
    if r.sin_citas:
        print("\n⚠️  Respondió sin citar ningún fragmento: no se puede verificar.")
    if r.fuentes:
        print("\nFuentes citadas:")
        for f in r.fuentes:
            print(f"  - {f.metadatos['fuente']} · {f.metadatos['seccion']}")
    if r.llamada:
        print(f"\nLLM: {r.llamada.modelo} · {r.llamada.tokens_entrada} tokens de entrada, "
              f"{r.llamada.tokens_salida} de salida · US$ {r.llamada.costo:.6f} · {r.llamada.segundos_total:.1f} s")
    else:
        print("\n(No se llamó al LLM: nada superó el umbral.)")


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG simple con citas")
    ap.add_argument("pregunta", nargs="?", help="Pregunta sobre el documento")
    ap.add_argument("--almacen", choices=("memoria", "neon", "pinecone"), default="memoria")
    ap.add_argument("--proveedor", choices=("anthropic", "openai", "local", "todos"),
                    help="Quién genera la respuesta; 'todos' repite la pregunta con cada proveedor disponible y los compara")
    ap.add_argument("--documento", default=str(RAIZ / "datos" / "manual_soporte.md"))
    ap.add_argument("--ingestar", action="store_true", help="Vectoriza y guarda el documento antes de preguntar")
    ap.add_argument("--evaluar", action="store_true", help="Mide la recuperación con datos/preguntas_evaluacion.json")
    ap.add_argument("--estrategia", choices=("seccion", "oraciones", "fijo"), default="oraciones")
    ap.add_argument("--max-palabras", type=int, default=70)
    ap.add_argument("--solape", type=int, default=1)
    ap.add_argument("-k", type=int, default=4, help="Fragmentos que se envían al modelo")
    ap.add_argument("--version", default="v1", help="Versión del prompt (prompts/rag_respuesta/)")
    ap.add_argument("--umbral", type=float, help="Similitud mínima del mejor fragmento para llamar al LLM")
    args = ap.parse_args()
    load_dotenv(find_dotenv(usecwd=True))

    print(f"Embeddings: {proveedor_embeddings()} · {modelo_embeddings()} · almacén: {args.almacen}")
    almacen = crear_almacen(args.almacen)

    if args.ingestar or (args.almacen == "memoria" and (args.pregunta or args.evaluar)):
        fragmentos = ingestar(almacen, args.documento, estrategia=args.estrategia,
                              max_palabras=args.max_palabras, solape=args.solape)
        print(f"Ingestados {len(fragmentos)} fragmentos ({args.estrategia}, ~{args.max_palabras} palabras, solape {args.solape}).")
    elif args.pregunta or args.evaluar:
        # Asegura que la estructura exista con la dimensión del modelo actual.
        from embeddings import vector_consulta
        almacen.preparar(len(vector_consulta("prueba")))
        if almacen.contar() == 0:
            print("El almacén está vacío: ejecuta primero con --ingestar.")
            return 1

    if args.evaluar:
        preguntas = json.loads((RAIZ / "datos" / "preguntas_evaluacion.json").read_text(encoding="utf-8"))
        r = evaluar_recuperacion(almacen, preguntas, k=args.k)
        print(f"\nRecuperación sobre {r['total']} preguntas: hit@1 = {r['hit1']}/{r['total']} · "
              f"hit@{args.k} = {r[f'hit{args.k}']}/{r['total']}")
        for id_, pregunta, esperada, obtenida in r["fallos"]:
            print(f"  ⚠️ {id_} {pregunta!r}\n      esperada: {esperada}\n      primeras: {obtenida}")
        print("Con tan pocas preguntas, 1 punto de diferencia es ruido: es una demostración del método.")

    if args.pregunta:
        if args.proveedor == "todos":
            # Mismos fragmentos y mismo prompt; solo cambia el modelo que redacta. Útil para comparar calidad y costo.
            proveedores = [p for p, ok in proveedores_disponibles().items() if ok]
            print(f"Comparando proveedores disponibles: {', '.join(proveedores) or 'ninguno'}")
        else:
            proveedores = [args.proveedor]
        for p in proveedores:
            print(f"\n══════ Generación con: {proveedor_generacion(p)} ══════")
            mostrar(responder(almacen, args.pregunta, k=args.k, version=args.version,
                              proveedor=p, umbral=args.umbral))
    elif not (args.ingestar or args.evaluar):
        ap.error("Indica una pregunta, --ingestar o --evaluar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
