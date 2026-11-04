"""Práctica S6 · Laboratorio de chunking.

PREGUNTA QUE RESPONDE
    Con MIS documentos y MIS preguntas, ¿qué estrategia de fragmentado (y qué tamaño) recupera mejor?
    ¿Y cuánto "contexto" (palabras que se enviarían al LLM, o sea, dinero) cuesta cada una?

QUÉ HACE
    1. Lee todos los documentos de una carpeta (Markdown, HTML, DOCX, PDF) y avisa de los que no se pudieron leer.
    2. Para cada combinación estrategia × tamaño:
         fragmenta → vectoriza → construye el índice (vectores + BM25) → hace las preguntas → mide.
    3. Imprime una tabla ordenada de mejor a peor (por MRR) con hit@1, hit@k, MRR y el contexto promedio.
    4. Con --detalle muestra qué preguntas falló la mejor configuración, que es lo que de verdad enseña.

    Además de las estrategias de `fragmentado.py`, puedes probar `docling-hibrido` (OPCIONAL, necesita Docling:
    `uv sync --group docling`): lee TODOS los documentos con Docling (un solo parser, con OCR para el acta escaneada)
    y los parte con su `HybridChunker` (jerarquía de encabezados + límite de TOKENS). Su tamaño se da en tokens
    (`--tokens`), no en palabras. ⚠️ No tiene relación con la búsqueda híbrida (`--modo hibrido`): aquella combina
    BM25 y vectores al BUSCAR; esta decide dónde CORTAR los documentos.

    No llama al LLM: solo usa el modelo de embeddings (local por defecto). Es gratis y se repite cuantas veces quieras.

CÓMO EJECUTARLO (desde la carpeta s06-rag-intermedio/)
    uv run python practica/laboratorio_chunking.py
    uv run python practica/laboratorio_chunking.py --modo hibrido
    uv run python practica/laboratorio_chunking.py --estrategias fijo,oraciones,recursivo --tamanos 40,70
    uv run python practica/laboratorio_chunking.py --estrategias padre-hijo,seccion --tipos semantica,lexica
    uv run python practica/laboratorio_chunking.py --documentos mi_carpeta/ --preguntas mis_preguntas.json
    uv run python practica/laboratorio_chunking.py --detalle
    uv run python practica/laboratorio_chunking.py --estrategias oraciones,docling-hibrido --tokens 64,128,256

💰 COSTO
    Cero con el modelo de embeddings local. Con EMBEDDINGS_PROVIDER=openai, cada configuración vectoriza el corpus
    completo (unos cientos de fragmentos cortos): fracciones de centavo en total.

CÓMO LEER LA TABLA
    hit@1   en cuántas preguntas el PRIMER fragmento contiene la respuesta.
    hit@k   en cuántas la contiene alguno de los k primeros.
    MRR     1.0 = la respuesta siempre es el primer resultado; 0.5 = suele ser el segundo; 0 = nunca aparece.
    ctx     palabras promedio de los k fragmentos: es lo que pagarías en tokens al armar el prompt.
    tam.    palabras por fragmento (o tokens, en `docling-hibrido`: se indica con "tok").
    Con pocas preguntas, 1 o 2 aciertos de diferencia son RUIDO. Busca diferencias grandes y mira el contexto.

IDEAS PARA EXPERIMENTAR (ver practica/README.md)
    - Cambia --modo (vectorial, bm25, hibrido) y fíjate en qué tipo de pregunta cambia el resultado (--tipos).
    - Prueba tamaños muy pequeños (20) y muy grandes (200).
    - Añade TU documento a datos/ y escribe 5 preguntas con su `clave` en datos/preguntas_recuperacion.json.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Este script vive en practica/, pero los módulos están una carpeta más arriba.
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from dotenv import find_dotenv, load_dotenv  # noqa: E402 (el import va tras ajustar sys.path)

load_dotenv(find_dotenv(usecwd=True))

from embeddings import embeddings_disponibles, modelo_embeddings, proveedor_embeddings  # noqa: E402
from fragmentado import ESTRATEGIAS, fragmentar_corpus, fragmentar_padre_hijo  # noqa: E402
from ingesta import leer_carpeta  # noqa: E402
from parsing_docling import docling_disponible, fragmentar_carpeta_hibrido  # noqa: E402 (no importa Docling hasta usarlo)
from recuperacion import MODOS, construir_indice, evaluar_busqueda, expandir_a_padre  # noqa: E402

DOCLING = "docling-hibrido"


def configuraciones(estrategias: list[str], tamanos: list[int], tokens: list[int] | None = None) -> list[tuple[str, int]]:
    """Todas las combinaciones estrategia × tamaño. 'seccion' no depende del tamaño: se prueba una sola vez.
    `docling-hibrido` usa su propia lista de tamaños, en TOKENS (`tokens`)."""
    salida = []
    for e in estrategias:
        if e == DOCLING:
            salida.extend((e, t) for t in (tokens or [128]))
            continue
        for t in ([tamanos[0]] if e == "seccion" else tamanos):
            salida.append((e, t))
    return salida


def probar(documentos, estrategia: str, tamano: int, modo: str, preguntas: list[dict], k: int,
           carpeta: str | None = None) -> dict:
    """Construye el índice con UNA configuración y mide el buscador con todas las preguntas."""
    if estrategia == DOCLING:
        # Docling lee la CARPETA completa (todos los formatos, con OCR) y corta por jerarquía + `tamano` TOKENS.
        fragmentos = fragmentar_carpeta_hibrido(carpeta, max_tokens=tamano)
        indice = construir_indice(fragmentos)

        def buscador(p, k):
            return MODOS[modo](indice, p["pregunta"], k, p.get("filtro"))
    elif estrategia == "padre-hijo":
        # Se BUSCA con hijos pequeños (de `tamano` palabras) y se ENTREGA el padre (la sección completa).
        fragmentos = [h for d in documentos for h in fragmentar_padre_hijo(d, tamano)]
        indice = construir_indice(fragmentos)

        def buscador(p, k):
            return expandir_a_padre(MODOS[modo](indice, p["pregunta"], 10, p.get("filtro")))[:k]
    else:
        fragmentos = fragmentar_corpus(documentos, estrategia=estrategia, max_palabras=tamano)
        indice = construir_indice(fragmentos)

        def buscador(p, k):
            return MODOS[modo](indice, p["pregunta"], k, p.get("filtro"))

    r = evaluar_busqueda(preguntas, buscador, k)
    r.update(estrategia=estrategia, tamano=tamano, fragmentos=len(fragmentos))
    return r


def etiqueta(r: dict) -> str:
    """'oraciones con 70 palabras'; 'seccion' no usa tamaño; 'docling-hibrido' se mide en tokens."""
    if r["estrategia"] == "seccion":
        return "seccion"
    return f"{r['estrategia']} con {r['tamano']} " + ("tokens" if r["estrategia"] == DOCLING else "palabras")


def main() -> None:
    ap = argparse.ArgumentParser(description="Laboratorio de chunking: compara estrategias de fragmentado midiendo la recuperación.")
    ap.add_argument("--documentos", default=str(RAIZ / "datos"), help="carpeta con los documentos (.md, .html, .docx, .pdf)")
    ap.add_argument("--preguntas", default=str(RAIZ / "datos" / "preguntas_recuperacion.json"))
    ap.add_argument("--estrategias", default="fijo,oraciones,recursivo,seccion,semantico",
                    help="separadas por coma: " + ", ".join(ESTRATEGIAS) + ", padre-hijo, " + DOCLING
                         + f" (si Docling está instalado, `{DOCLING}` se añade sola)")
    ap.add_argument("--tamanos", default="40,70,120", help="palabras por fragmento, separadas por coma")
    ap.add_argument("--tokens", default="128", help=f"tokens por fragmento de `{DOCLING}`, separados por coma")
    ap.add_argument("--modo", choices=list(MODOS), default="vectorial", help="cómo se busca")
    ap.add_argument("--tipos", default="semantica,lexica,tabla",
                    help="tipos de pregunta a usar (semantica, lexica, tabla, filtro, conversacional)")
    ap.add_argument("-k", type=int, default=3, help="cuántos fragmentos se recuperan")
    ap.add_argument("--detalle", action="store_true", help="muestra las preguntas que falló la mejor configuración")
    ap.add_argument("--csv", help="guarda la tabla en este archivo CSV")
    args = ap.parse_args()

    ok, detalle = embeddings_disponibles()
    print(("✅ " if ok else "❌ ") + detalle)
    if not ok:
        sys.exit(1)

    # 1) Leer los documentos y avisar de los que no se pudieron leer (p. ej. un PDF escaneado).
    documentos = leer_carpeta(args.documentos)
    print(f"\nDocumentos ({len(documentos)}):")
    for d in documentos:
        palabras = sum(len(t.split()) for _, t in d.secciones)
        print(f"  • {d.fuente:<30} {d.tipo:<5} {len(d.secciones):>2} secciones · {palabras:>4} palabras")
        for aviso in d.avisos:
            print(f"      ⚠️ {aviso}")

    # 2) Cargar las preguntas (cada una con su `clave`: el texto que debe contener el fragmento correcto).
    tipos = set(args.tipos.split(","))
    preguntas = [p for p in json.loads(Path(args.preguntas).read_text(encoding="utf-8")) if p["tipo"] in tipos]
    print(f"\nPreguntas: {len(preguntas)} (tipos: {', '.join(sorted(tipos))}) · modo: {args.modo} · k={args.k}")
    print(f"Embeddings: {proveedor_embeddings()} · {modelo_embeddings()}")

    # 3) Probar cada configuración.
    estrategias = [e.strip() for e in args.estrategias.split(",") if e.strip()]
    hay_docling, msg_docling = docling_disponible()
    if DOCLING in estrategias and not hay_docling:
        print(f"\n❌ Pediste `{DOCLING}`, pero {msg_docling}")
        sys.exit(1)
    if args.estrategias == ap.get_default("estrategias"):
        # Sin pedir nada concreto: si Docling está instalado se añade la configuración nueva; si no, se avisa y sigue.
        if hay_docling:
            estrategias.append(DOCLING)
        else:
            print(f"\nℹ️ `{DOCLING}` omitida: {msg_docling}")
    tamanos = [int(t) for t in args.tamanos.split(",")]
    tokens = [int(t) for t in args.tokens.split(",")]
    resultados = []
    for estrategia, tamano in configuraciones(estrategias, tamanos, tokens):
        unidad = "tokens" if estrategia == DOCLING else "palabras"
        print(f"  probando {estrategia} · {tamano} {unidad}…", end="\r", flush=True)
        resultados.append(probar(documentos, estrategia, tamano, args.modo, preguntas, args.k, args.documentos))
    resultados.sort(key=lambda r: (-r["mrr"], r["contexto"]))

    # 4) La tabla.
    print(" " * 60)
    print(f"{'estrategia':<16}{'tam.':>9}{'fragm.':>8}{'hit@1':>8}{'hit@' + str(args.k):>8}{'MRR':>7}{'ctx':>6}")
    for r in resultados:
        tam = "—" if r["estrategia"] == "seccion" else (f"{r['tamano']} tok" if r["estrategia"] == DOCLING else r["tamano"])
        print(f"{r['estrategia']:<16}{tam:>9}{r['fragmentos']:>8}"
              f"{r['hit1']:>5}/{r['n']:<2}{r['hitk']:>5}/{r['n']:<2}{r['mrr']:>7.2f}{r['contexto']:>6.0f}")

    mejor = resultados[0]
    print(f"\n🏆 Mejor por MRR: {etiqueta(mejor)} (MRR {mejor['mrr']:.2f}, contexto ≈ {mejor['contexto']:.0f} palabras).")
    barato = min((r for r in resultados if r["mrr"] >= mejor["mrr"] - 0.05), key=lambda r: r["contexto"])
    if barato is not mejor:
        print(f"💡 Casi igual de bueno (MRR {barato['mrr']:.2f}) con menos contexto: {etiqueta(barato)} "
              f"(≈ {barato['contexto']:.0f} palabras por consulta).")
    print("⚠️ Con pocas preguntas, 1–2 aciertos de diferencia son ruido: compara también el contexto (ctx).")

    if args.detalle:
        print(f"\nPreguntas en las que la mejor configuración NO acertó en el primer lugar:")
        for f in mejor["fallos"]:
            donde = f"posición {f['posicion']}" if f["posicion"] else f"fuera de los {args.k} primeros"
            print(f"  {f['id']} ({donde}): {f['pregunta']}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["estrategia", "tamano (palabras; tokens en docling-hibrido)", "fragmentos", "hit1", f"hit{args.k}", "n", "mrr", "contexto"])
            for r in resultados:
                w.writerow([r["estrategia"], r["tamano"], r["fragmentos"], r["hit1"], r["hitk"], r["n"],
                            round(r["mrr"], 3), round(r["contexto"], 1)])
        print(f"\nTabla guardada en {args.csv}")


if __name__ == "__main__":
    main()
