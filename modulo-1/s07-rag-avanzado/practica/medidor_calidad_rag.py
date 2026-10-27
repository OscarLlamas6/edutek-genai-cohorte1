"""Práctica S7 · Medidor de calidad RAG.

PREGUNTA QUE RESPONDE
    ¿Mi RAG es bueno... y cómo lo sé? Si cambio el prompt, el tamaño de los fragmentos o la técnica de
    recuperación, ¿mejora, empeora o da igual? ¿Puedo desplegar el cambio?

QUÉ HACE
    1. Arma el RAG (ingesta + la técnica de recuperación que elijas) sobre datos/manual_soporte.md.
    2. Lo evalúa con el golden set (datos/golden_set.json: 24 preguntas de 6 tipos, o las 10 "rápidas"):
         RECUPERACIÓN → Recall@K, Precision@K, MRR     (sin LLM, gratis)
         GENERACIÓN   → dato correcto, citas válidas, "no sé" bien dicho, cifras con respaldo
         JUEZ (opcional) → fidelidad y/o corrección con un LLM
    3. Imprime el informe: resumen, desglose por tipo de pregunta, DIAGNÓSTICO POR CAPA (¿falló la
       recuperación o la generación?) y si cumple el SLO.
    4. Guarda el informe en informes/ (JSON para comparar, Markdown para leer).
    5. Con --comparar-con informes/anterior.json muestra qué mejoró y qué empeoró (prueba de regresión).
    6. Con --exigir-slo termina con código 1 si no cumple el SLO: sirve como puerta de calidad en CI.

CÓMO EJECUTARLO (desde la carpeta s07-rag-avanzado/)
    uv run python practica/medidor_calidad_rag.py --sin-generacion                 # solo recuperación: gratis y en segundos
    uv run python practica/medidor_calidad_rag.py --rapido --version v3            # 10 preguntas
    uv run python practica/medidor_calidad_rag.py --version v3 --nombre base       # las 24, y guarda informes/base.json
    uv run python practica/medidor_calidad_rag.py --tecnica small-to-big -k 2 --version v3 --comparar-con informes/base.json
    uv run python practica/medidor_calidad_rag.py --tecnica descomposicion --version v3
    uv run python practica/medidor_calidad_rag.py --version v3 --juez fidelidad    # añade el juez LLM
    uv run python practica/medidor_calidad_rag.py --version v3 --exigir-slo        # código de salida 1 si no cumple

💰 COSTO
    - --sin-generacion: US$ 0 (solo embeddings locales).
    - Con generación: 1 llamada al LLM por pregunta (24 preguntas ≈ US$ 0.012 con el modelo económico de un
      proveedor; US$ 0 con RAG_PROVIDER=local). La descomposición y HyDE suman 1 llamada corta por pregunta.
    - --juez suma hasta 1 llamada del juez por respuesta (ver JUEZ_PROVIDER y JUEZ_MODEL).

IDEAS PARA EXPERIMENTAR (ver practica/README.md)
    - Compara --version v1, v2 y v3 del prompt con el mismo golden set.
    - Compara --tecnica basica, small-to-big y descomposicion.
    - Escribe tu propio golden set (mismo formato JSON) y pásalo con --golden (y tu documento con --documento).
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
from avanzado import RecuperadorDescompuesto, RecuperadorHyDE, RecuperadorSmallToBig, ingestar_pequenos  # noqa: E402
from embeddings import modelo_embeddings, proveedor_embeddings  # noqa: E402
from evaluacion import (  # noqa: E402
    CAPA_DEL_DIAGNOSTICO, ETIQUETAS_RESUMEN, cargar_golden, evaluar_rag, guardar_informe, imprimir_resumen, tabla_por_tipo,
    verificar_slo,
)
from rag import ingestar  # noqa: E402

# Métricas donde MENOS es mejor (para marcar bien las flechas al comparar).
MENOS_ES_MEJOR = {"alucinaciones", "abstencion_indebida"}


def comparar_con(ruta: Path, resumen: dict) -> None:
    """Compara el resumen actual con el de un informe anterior: la prueba de regresión."""
    anterior = json.loads(ruta.read_text(encoding="utf-8"))["resumen"]
    print(f"\nComparación con {ruta.name} (diferencia en puntos porcentuales; con tan pocas preguntas, menos de 3 puntos es ruido):")
    for clave, nombre in ETIQUETAS_RESUMEN.items():
        if clave in resumen and clave in anterior:
            delta = (resumen[clave] - anterior[clave]) * 100
            mejor = delta < 0 if clave in MENOS_ES_MEJOR else delta > 0
            marca = " ≈ (ruido)" if abs(delta) < 3 else (" ▲ mejora" if mejor else " ▼ EMPEORA")
            print(f"  {nombre:<22} {anterior[clave] * 100:5.0f} % → {resumen[clave] * 100:5.0f} %   {delta:+5.0f}{marca}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Medidor de calidad RAG")
    ap.add_argument("--golden", default=str(RAIZ / "datos" / "golden_set.json"), help="Preguntas de evaluación (JSON)")
    ap.add_argument("--documento", default=str(RAIZ / "datos" / "manual_soporte.md"))
    ap.add_argument("--rapido", action="store_true", help="Solo las preguntas marcadas como rápidas (10)")
    ap.add_argument("--tecnica", choices=("basica", "small-to-big", "ventana", "descomposicion", "hyde"), default="basica",
                    help="Cómo se recuperan los fragmentos")
    ap.add_argument("--estrategia", choices=("seccion", "oraciones", "fijo"), default="oraciones")
    ap.add_argument("--max-palabras", type=int, default=70, help="Tamaño de fragmento de la técnica básica")
    ap.add_argument("--solape", type=int, default=1)
    ap.add_argument("-k", type=int, default=4, help="Fragmentos (o padres, con small-to-big) que se entregan al modelo")
    ap.add_argument("--version", default="v3", help="Versión del prompt (prompts/rag_respuesta/)")
    ap.add_argument("--umbral", type=float, help="Similitud mínima del mejor fragmento para llamar al LLM")
    ap.add_argument("--proveedor", choices=("anthropic", "openai", "local"), help="Quién genera (por defecto RAG_PROVIDER)")
    ap.add_argument("--sin-generacion", action="store_true", help="Solo mide la recuperación (gratis, sin LLM)")
    ap.add_argument("--juez", choices=("fidelidad", "correccion", "ambos"), help="Añade un juez LLM")
    ap.add_argument("--salida", default=str(RAIZ / "informes"), help="Carpeta donde se guardan los informes")
    ap.add_argument("--nombre", help="Nombre de los archivos del informe (por defecto, la fecha y hora)")
    ap.add_argument("--comparar-con", help="Informe JSON anterior con el que comparar")
    ap.add_argument("--exigir-slo", action="store_true", help="Código de salida 1 si no se cumple el SLO")
    args = ap.parse_args()
    load_dotenv(find_dotenv(usecwd=True))

    golden = cargar_golden(args.golden, solo_rapido=args.rapido)
    print(f"Embeddings: {proveedor_embeddings()} · {modelo_embeddings()} · técnica: {args.tecnica} · {len(golden)} preguntas")

    # 1. Ingesta y recuperador, según la técnica elegida.
    almacen, recuperador = crear_almacen("memoria"), None
    if args.tecnica in ("small-to-big", "ventana"):
        modo = "seccion" if args.tecnica == "small-to-big" else "ventana"
        fragmentos = ingestar_pequenos(almacen, args.documento, max_palabras=35)
        recuperador = RecuperadorSmallToBig(almacen, args.documento, fragmentos, modo=modo)
        print(f"Ingestados {len(fragmentos)} fragmentos pequeños; se entregan {args.k} 'padres' ({modo}).")
    else:
        fragmentos = ingestar(almacen, args.documento, estrategia=args.estrategia,
                              max_palabras=args.max_palabras, solape=args.solape)
        print(f"Ingestados {len(fragmentos)} fragmentos ({args.estrategia}, ~{args.max_palabras} palabras, solape {args.solape}).")
        if args.tecnica == "descomposicion":
            recuperador = RecuperadorDescompuesto(almacen, args.proveedor)
        elif args.tecnica == "hyde":
            recuperador = RecuperadorHyDE(almacen, args.proveedor)

    # 2. Evaluación (recuperación, generación y, si se pide, juez).
    print("Evaluando…")
    informe = evaluar_rag(almacen, golden, k=args.k, version=args.version, proveedor=args.proveedor, umbral=args.umbral,
                          con_generacion=not args.sin_generacion, juez=args.juez, recuperador=recuperador,
                          tecnica=args.tecnica)

    # 3. Informe.
    print()
    imprimir_resumen(informe, "Medidor de calidad RAG")
    print("\nPor tipo de pregunta:\n" + tabla_por_tipo(informe))
    fallos = [f for f in informe.filas if f.diagnostico not in ("ok", "sin_generacion")]
    if fallos:
        print("\nPreguntas con fallo y en qué capa:")
        for f in fallos:
            print(f"  {f.id:<5}{f.diagnostico:<28}{CAPA_DEL_DIAGNOSTICO[f.diagnostico]:<26}{f.pregunta[:60]}")

    resumen = informe.resumen()
    cumple = True
    if not args.sin_generacion:
        print("\n¿Cumple el SLO?")
        for f in verificar_slo(resumen):
            cumple &= f["cumple"]
            print(f"  {'✅' if f['cumple'] else '❌'} {ETIQUETAS_RESUMEN.get(f['metrica'], f['metrica']):<22} {f['valor'] * 100:5.0f} %   (regla {f['regla']})")

    # 4. Guardar y comparar.
    ruta_json, ruta_md = guardar_informe(informe, args.salida, args.nombre)
    print(f"\nInforme guardado en {ruta_json} y {ruta_md}")
    if args.comparar_con:
        comparar_con(Path(args.comparar_con), resumen)

    if args.exigir_slo and not cumple:
        print("\nSLO NO cumplido: el cambio no debería desplegarse.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
