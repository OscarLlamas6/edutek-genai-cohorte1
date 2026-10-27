"""Práctica S5 · Un RAG básico, fase por fase.

PREGUNTA QUE RESPONDE
    ¿Cuáles son las fases de un RAG y qué entra y qué sale de cada una?

    Este programa recorre el flujo completo UNA FASE A LA VEZ y muestra lo que produce cada una.
    No tiene opciones ni bases de datos externas: todo vive en memoria y cabe en una pantalla.
    (Cuando entiendas este, `rag_simple.py` es el mismo flujo con Neon, Pinecone, citas verificadas y evaluación.)

LAS FASES
    Preparación (se hace una vez, antes de que alguien pregunte):
        1. CARGAR      leer el documento              archivo  → texto con secciones
        2. FRAGMENTAR  partirlo en trozos pequeños    texto    → lista de fragmentos
        3. VECTORIZAR  convertir cada trozo en vector fragmentos → vectores (embeddings)
        4. GUARDAR     dejar los vectores a mano      vectores → almacén (aquí, una matriz en memoria)
    Consulta (se hace en cada pregunta):
        5. RECUPERAR   buscar los trozos más parecidos a la pregunta   pregunta → k fragmentos
        6. AUMENTAR    armar el prompt: instrucciones + fragmentos + pregunta
        7. GENERAR     el LLM redacta la respuesta usando SOLO esos fragmentos

    Las fases 1 a 4 se llaman "ingesta" (el equivalente a un pipeline de ETL). Las 5 a 7, "consulta".

CÓMO EJECUTARLO (desde la carpeta s05-fundamentos-rag/)
    uv run python practica/rag_basico.py "¿Cuánto cuesta el plan de 300 Mbps?"
    uv run python practica/rag_basico.py --sin-llm "¿Cuánto cuesta el plan de 300 Mbps?"
    uv run python practica/rag_basico.py --proveedor local "¿Ofrecen televisión por cable?"

💰 COSTO
    - Embeddings: gratis con el modelo local de LM Studio.
    - La fase 7 llama al LLM: ≈ US$ 0.0005 con Anthropic u OpenAI; gratis con --proveedor local.
    - Con --sin-llm se detiene en la fase 6 (muestra el prompt) y no gasta nada.

IDEAS PARA EXPERIMENTAR
    - Cambia ESTRATEGIA a "seccion" o "fijo" y mira cómo cambian los fragmentos (fase 2) y los puntajes (fase 5).
    - Cambia K a 1 y luego a 8: ¿qué pasa con la respuesta?
    - Pregunta algo que el manual NO contiene y observa los puntajes y la respuesta.
"""

import argparse
import sys
from pathlib import Path

# Este script vive en practica/, pero los módulos están una carpeta más arriba.
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from dotenv import find_dotenv, load_dotenv  # noqa: E402 (el import va tras ajustar sys.path)

from almacenes import crear_almacen  # noqa: E402
from embeddings import modelo_embeddings, proveedor_embeddings, vector_consulta, vectorizar  # noqa: E402
from rag import (  # noqa: E402
    construir_prompt, fragmentar, generar, leer_documento, proveedor_generacion, texto_a_vectorizar,
)

DOCUMENTO = RAIZ / "datos" / "manual_soporte.md"
ESTRATEGIA = "oraciones"   # "seccion" | "oraciones" | "fijo"
MAX_PALABRAS = 70          # tamaño aproximado de cada fragmento
K = 3                      # cuántos fragmentos se le entregan al LLM


def titulo(numero: int, nombre: str, entra: str, sale: str) -> None:
    print(f"\n━━━ Fase {numero} · {nombre} ━━━  ({entra} → {sale})")


def main() -> int:
    ap = argparse.ArgumentParser(description="RAG básico, fase por fase")
    ap.add_argument("pregunta", help="Pregunta sobre el manual")
    ap.add_argument("--sin-llm", action="store_true", help="Se detiene antes de llamar al LLM (no gasta nada)")
    ap.add_argument("--proveedor", choices=("anthropic", "openai", "local"), help="Quién redacta la respuesta")
    args = ap.parse_args()
    load_dotenv(find_dotenv(usecwd=True))
    print(f"Embeddings: {proveedor_embeddings()} · {modelo_embeddings()}")

    # ═════════ INGESTA (una vez) ═════════

    # 1. CARGAR: leemos el archivo y lo separamos en secciones por sus encabezados "## ".
    titulo(1, "CARGAR", "archivo", "texto con secciones")
    documento = leer_documento(DOCUMENTO)
    print(f"{documento.fuente}: {len(documento.secciones)} secciones")
    for nombre, texto in documento.secciones[:3]:
        print(f"  · {nombre} ({len(texto.split())} palabras)")
    print("  · …")

    # 2. FRAGMENTAR (chunking): el modelo de embeddings y el LLM trabajan mejor con trozos pequeños y
    #    enfocados que con un manual entero. Cada fragmento conserva metadatos (de qué sección viene)
    #    para poder CITAR la fuente después.
    titulo(2, "FRAGMENTAR", "texto", "fragmentos")
    fragmentos = fragmentar(documento, estrategia=ESTRATEGIA, max_palabras=MAX_PALABRAS)
    print(f"{len(fragmentos)} fragmentos ({ESTRATEGIA}, ~{MAX_PALABRAS} palabras). Ejemplo:")
    ejemplo = fragmentos[3]
    print(f"  id={ejemplo.id} · sección={ejemplo.metadatos['seccion']!r}\n  «{ejemplo.texto[:160]}…»")

    # 3. VECTORIZAR (embeddings): cada fragmento se convierte en una lista de números que representa su
    #    significado. Textos con significado parecido quedan con vectores cercanos (Sesión 4).
    titulo(3, "VECTORIZAR", "fragmentos", "vectores")
    vectores = vectorizar([texto_a_vectorizar(f) for f in fragmentos], tipo="documento")
    for f, v in zip(fragmentos, vectores):
        f.vector = v
    print(f"{len(vectores)} vectores de {len(vectores[0])} dimensiones. Primeros 4 números del primero: "
          f"{[round(float(x), 3) for x in vectores[0][:4]]}")

    # 4. GUARDAR: dejamos los fragmentos (texto + metadatos + vector) en un almacén. Aquí es la memoria;
    #    con Neon o Pinecone el resto del programa no cambiaría: es la misma interfaz (Sesión 4).
    titulo(4, "GUARDAR", "vectores", "almacén")
    almacen = crear_almacen("memoria")
    almacen.preparar(len(vectores[0]))
    almacen.insertar(fragmentos)
    print(f"Almacén en memoria con {almacen.contar()} fragmentos.")

    # ═════════ CONSULTA (en cada pregunta) ═════════

    # 5. RECUPERAR: la pregunta se vectoriza con el MISMO modelo y se buscan los k fragmentos más cercanos.
    titulo(5, "RECUPERAR", "pregunta", f"{K} fragmentos")
    print(f"Pregunta: {args.pregunta}")
    encontrados = almacen.buscar(vector_consulta(args.pregunta), k=K)
    for i, r in enumerate(encontrados, start=1):
        print(f"  [{i}] similitud {r.puntaje:.3f} · {r.metadatos['seccion']}")
    print("  (La similitud va de -1 a 1: mientras más alta, más parecido. Si el mejor es bajo, quizá el manual no responde.)")

    # 6. AUMENTAR: armamos el prompt = instrucciones (usa SOLO los fragmentos, cita con [1], di "no sé" si no
    #    está) + los fragmentos numerados + la pregunta. Esto es lo que hace "aumentada" a la generación.
    titulo(6, "AUMENTAR", "fragmentos + pregunta", "prompt")
    prompt = construir_prompt(args.pregunta, encontrados, version="v1")
    print(f"Prompt de {len(prompt.split())} palabras. Así termina:\n" + "\n".join("  | " + l for l in prompt.splitlines()[-9:]))
    if args.sin_llm:
        print("\n(--sin-llm: nos detenemos aquí; no se llamó al LLM.)")
        return 0

    # 7. GENERAR: el LLM lee el prompt y redacta la respuesta. Si el prompt está bien hecho, se apoya solo en
    #    los fragmentos y los cita; si no los contienen, dice que no lo sabe.
    titulo(7, "GENERAR", "prompt", "respuesta")
    llamada = generar(prompt, args.proveedor)
    print(f"({proveedor_generacion(args.proveedor)} · {llamada.modelo} · US$ {llamada.costo:.6f})\n")
    print(llamada.texto.strip())
    print("\nLos números [1], [2]… apuntan a los fragmentos de la fase 5: así se verifica de dónde salió cada dato.")
    print("Si la respuesta no trae ninguno, no se puede verificar: los modelos pequeños suelen olvidar citar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
