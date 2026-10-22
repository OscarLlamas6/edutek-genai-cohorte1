"""Práctica S4 · Buscador de documentos con embeddings y una base de datos vectorial.

PREGUNTA QUE RESPONDE
    Tengo una carpeta de notas (manuales, políticas, runbooks). ¿Puedo encontrar el párrafo
    que responde a una pregunta aunque NO use las mismas palabras?

QUÉ HACE
    1. Lee los archivos .md de una carpeta y los parte en párrafos (cada párrafo es un fragmento).
    2. Convierte cada fragmento en un vector (embedding).
    3. Los guarda en el almacén que elijas: Neon (PostgreSQL + pgvector), Pinecone o memoria.
    4. Con --buscar, vectoriza tu pregunta y devuelve los fragmentos más parecidos, con su
       puntaje (similitud coseno) y su fuente.
    5. Con --comparar, hace la misma búsqueda en todos los almacenes configurados y compara tiempos.

CÓMO EJECUTARLO (desde la carpeta s04-embeddings-bases-vectoriales/)
    uv run python practica/buscador_documentos.py --almacen memoria --indexar --buscar "se venció el certificado"
    uv run python practica/buscador_documentos.py --almacen neon --indexar
    uv run python practica/buscador_documentos.py --almacen neon --buscar "no puedo entrar a mi cuenta"
    uv run python practica/buscador_documentos.py --almacen pinecone --indexar --buscar "¿cada cuánto cambian la clave?"
    uv run python practica/buscador_documentos.py --comparar "¿cuánto se conservan las copias de seguridad?"
    uv run python practica/buscador_documentos.py --almacen neon --vaciar        # borra los datos

💰 COSTO
    - Embeddings locales (LM Studio, por defecto): sin costo.
    - Embeddings con OpenAI: fracciones de centavo para estas notas.
    - Neon y Pinecone: planes gratuitos; consumen cuota, no dinero.

IDEAS PARA EXPERIMENTAR (ver practica/README.md)
    - Busca con palabras distintas a las del texto (sinónimos, otras formas de decirlo).
    - Compara el mismo texto indexado con --almacen neon y con --almacen pinecone.
    - Agrega un archivo .md a datos/notas/ y vuelve a indexar.
"""

import argparse
import sys
import time
from pathlib import Path

# Este script vive en practica/, pero los módulos están una carpeta más arriba.
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from dotenv import find_dotenv, load_dotenv  # noqa: E402 (el import va tras ajustar sys.path)

from almacenes import Fragmento, almacenes_configurados, crear_almacen  # noqa: E402
from embeddings import costo_embeddings_usd, modelo_embeddings, proveedor_embeddings, vector_consulta, vectorizar, USO  # noqa: E402


def leer_fragmentos(carpeta: Path) -> list[Fragmento]:
    """Convierte cada archivo .md en fragmentos: uno por párrafo (separados por líneas en blanco).

    Cada fragmento lleva metadatos: de qué archivo viene y cuál es el título del documento. Los metadatos
    permiten luego CITAR la fuente y filtrar la búsqueda.
    """
    fragmentos = []
    for archivo in sorted(carpeta.glob("*.md")):
        parrafos = [p.strip() for p in archivo.read_text(encoding="utf-8").split("\n\n") if p.strip()]
        titulo = parrafos[0].lstrip("# ").strip() if parrafos and parrafos[0].startswith("#") else archivo.stem
        # El primer párrafo es el título: se guarda como metadato y no como fragmento.
        cuerpo = parrafos[1:] if parrafos and parrafos[0].startswith("#") else parrafos
        for n, texto in enumerate(cuerpo):
            fragmentos.append(Fragmento(
                id=f"{archivo.stem}#{n}",          # id estable: reindexar actualiza en lugar de duplicar
                texto=texto,
                metadatos={"fuente": archivo.name, "titulo": titulo, "parrafo": n},
            ))
    return fragmentos


def indexar(almacen, carpeta: Path) -> None:
    fragmentos = leer_fragmentos(carpeta)
    print(f"Leídos {len(fragmentos)} fragmentos de {carpeta.name}/")
    t0 = time.perf_counter()
    vectores = vectorizar([f.texto for f in fragmentos], tipo="documento")
    for f, v in zip(fragmentos, vectores):
        f.vector = v
    print(f"Vectorizados en {time.perf_counter() - t0:.1f} s ({len(vectores[0])} dimensiones, modelo {modelo_embeddings()})")
    almacen.preparar(len(vectores[0]))
    t0 = time.perf_counter()
    almacen.insertar(fragmentos)
    print(f"Guardados en {almacen.nombre} en {time.perf_counter() - t0:.1f} s · total en el almacén: {almacen.contar()}")


def mostrar(resultados) -> None:
    for i, r in enumerate(resultados, start=1):
        print(f"  {i}. [{r.puntaje:.3f}] {r.metadatos.get('fuente', '?')}  →  {r.texto[:110]}{'…' if len(r.texto) > 110 else ''}")


def buscar(almacen, pregunta: str, k: int) -> float:
    """Busca y devuelve cuántos milisegundos tardó (sin contar el tiempo de vectorizar la pregunta)."""
    vector = vector_consulta(pregunta)
    t0 = time.perf_counter()
    resultados = almacen.buscar(vector, k=k)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\nPregunta: {pregunta!r}  ·  {almacen.nombre}  ·  {ms:.0f} ms")
    mostrar(resultados)
    return ms


def main() -> int:
    ap = argparse.ArgumentParser(description="Buscador de documentos con embeddings y base vectorial")
    ap.add_argument("--almacen", choices=("memoria", "neon", "pinecone"), default="memoria")
    ap.add_argument("--carpeta", default=str(RAIZ / "datos" / "notas"), help="Carpeta con archivos .md")
    ap.add_argument("--indexar", action="store_true", help="Lee, vectoriza y guarda los documentos")
    ap.add_argument("--buscar", metavar="PREGUNTA", help="Busca los fragmentos más parecidos a la pregunta")
    ap.add_argument("--comparar", metavar="PREGUNTA", help="Busca la misma pregunta en todos los almacenes configurados")
    ap.add_argument("--vaciar", action="store_true", help="Borra los datos del almacén elegido")
    ap.add_argument("-k", type=int, default=3, help="Cuántos resultados devolver")
    args = ap.parse_args()
    load_dotenv(find_dotenv(usecwd=True))

    print(f"Embeddings: {proveedor_embeddings()} · {modelo_embeddings()}")

    if args.comparar:
        # La memoria no persiste entre ejecuciones, así que aquí la reindexamos antes de buscar.
        for nombre, configurado in almacenes_configurados().items():
            if not configurado:
                print(f"\n(Se omite {nombre}: no está configurado en tu .env)")
                continue
            almacen = crear_almacen(nombre)
            if nombre == "memoria":
                indexar(almacen, Path(args.carpeta))
            else:
                almacen.preparar(len(vector_consulta("prueba")))
            buscar(almacen, args.comparar, args.k)
        _resumen_costo()
        return 0

    almacen = crear_almacen(args.almacen)
    if args.vaciar:
        almacen.preparar(len(vector_consulta("prueba")))
        almacen.vaciar()
        print(f"{args.almacen}: datos borrados.")
        return 0
    if args.indexar:
        indexar(almacen, Path(args.carpeta))
    elif args.almacen == "memoria" and args.buscar:
        print("La memoria no guarda nada entre ejecuciones: usa --indexar junto con --buscar.")
        return 1
    else:
        almacen.preparar(len(vector_consulta("prueba")))   # asegura la estructura y la dimensión correctas
    if args.buscar:
        buscar(almacen, args.buscar, args.k)
    _resumen_costo()
    return 0


def _resumen_costo() -> None:
    if proveedor_embeddings() == "openai":
        print(f"\nTokens vectorizados: {USO.tokens} · costo estimado de embeddings: US$ {costo_embeddings_usd():.6f}")


if __name__ == "__main__":
    sys.exit(main())
