"""Elegir y optimizar embeddings: dimensiones (Matryoshka) y cuantización, MEDIDOS sobre el golden set.

Un vector de 768 números en `float32` ocupa 768 × 4 = 3 072 bytes. Con millones de fragmentos, eso es
memoria, disco y tiempo de búsqueda. Hay dos formas de achicarlo sin volver a entrenar nada:

    Matryoshka (MRL)   Algunos modelos se entrenan para que los PRIMEROS números del vector ya sean un buen
                       vector pequeño (como las muñecas rusas). Se TRUNCA a 256 o 128 dimensiones y se
                       RENORMALIZA (se divide entre la nueva longitud) para que el coseno siga funcionando.
    Cuantización       Se guarda cada número con menos bits: int8 (1 byte, 4 veces menos que float32) o 1 bit
                       (32 veces menos). Se pierde un poco de precisión numérica.

Un modelo que NO se entrenó con Matryoshka también se puede truncar, pero la calidad suele caer mucho más:
por eso se mide con TUS preguntas antes de decidir. Todo aquí es numpy y es gratis.
"""

from __future__ import annotations

import os

import numpy as np

from embeddings import vectorizar
from almacenes import Fragmento, Resultado
from evaluacion import con_respuesta, metricas_recuperacion


def normalizar_filas(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def truncar(m: np.ndarray, dimensiones: int) -> np.ndarray:
    """Matryoshka: quedarse con las primeras `dimensiones` columnas y RENORMALIZAR cada fila."""
    return normalizar_filas(np.asarray(m, dtype=np.float32)[:, :dimensiones])


def cuantizar_int8(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float32 → int8 con una escala por vector: valor_int8 = round(valor / escala), escala = máximo ÷ 127."""
    m = np.asarray(m, dtype=np.float32)
    escala = np.abs(m).max(axis=1, keepdims=True) / 127.0
    escala[escala == 0] = 1.0
    return np.round(m / escala).astype(np.int8), escala.astype(np.float32)


def descuantizar_int8(q: np.ndarray, escala: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * escala


def cuantizar_binario(m: np.ndarray) -> np.ndarray:
    """1 bit por dimensión (¿el número es positivo?), empaquetado de 8 en 8: 32 veces menos que float32."""
    return np.packbits(np.asarray(m) > 0, axis=1)


def puntajes_hamming(consulta_bits: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Similitud binaria: cuántos bits coinciden (más alto = más parecido)."""
    diferentes = np.unpackbits(np.bitwise_xor(bits, consulta_bits), axis=1).sum(axis=1)
    return bits.shape[1] * 8 - diferentes


def bytes_de(*arreglos: np.ndarray) -> int:
    return int(sum(a.nbytes for a in arreglos))


def evaluar_matriz(fragmentos: list[Fragmento], puntajes_por_pregunta: dict[str, np.ndarray], golden: list[dict], k: int = 4) -> dict:
    """Recall@k, Precision@k, MRR y hit@1 promedio, con los puntajes que dé cada variante de vector.

    `puntajes_por_pregunta[id]` es un arreglo con un puntaje por fragmento (más alto = más parecido). Usa las
    mismas métricas que el resto de la sesión (evaluacion.metricas_recuperacion), solo sobre preguntas con respuesta.
    """
    filas = []
    for p in con_respuesta(golden):
        orden = np.argsort(-puntajes_por_pregunta[p["id"]])[:k]
        top = [Resultado(fragmentos[i].id, fragmentos[i].texto, fragmentos[i].metadatos, float(puntajes_por_pregunta[p["id"]][i]))
               for i in orden]
        m = metricas_recuperacion(top, p, k)
        filas.append((m, m["primer_relevante"] == 1))
    n = len(filas)
    return {
        "n": n,
        "recall": sum(m["recall"] for m, _ in filas) / n,
        "precision": sum(m["precision"] for m, _ in filas) / n,
        "mrr": sum(m["mrr"] for m, _ in filas) / n,
        "hit1": sum(h for _, h in filas),
    }


# ───────────────────────── Ayudas para medir con VARIOS modelos y variantes ─────────────────────────


def vectores_con(modelo: str, textos_fragmentos: list[str], preguntas: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Vectoriza fragmentos y preguntas con un modelo de embeddings concreto (sin cambiar el del resto de la sesión)."""
    anterior = os.environ.get("EMBEDDINGS_MODEL")
    os.environ["EMBEDDINGS_MODEL"] = modelo
    try:
        m = np.array(vectorizar(textos_fragmentos, tipo="documento"), dtype=np.float32)
        q = {p["id"]: np.array(vectorizar([p["pregunta"]], tipo="consulta")[0], dtype=np.float32) for p in con_respuesta(preguntas)}
    finally:
        if anterior is None:
            os.environ.pop("EMBEDDINGS_MODEL", None)
        else:
            os.environ["EMBEDDINGS_MODEL"] = anterior
    return m, q


def puntajes(m: np.ndarray, consultas: dict[str, np.ndarray], dimensiones: int | None = None) -> dict[str, np.ndarray]:
    """Coseno de cada pregunta contra todos los fragmentos; con `dimensiones` se trunca (Matryoshka) y se renormaliza."""
    mt = truncar(m, dimensiones) if dimensiones else normalizar_filas(m)
    salida = {}
    for i, q in consultas.items():
        q = q[:dimensiones] if dimensiones else q
        salida[i] = mt @ (q / np.linalg.norm(q))
    return salida


def modelos_de_embeddings_locales() -> list[str]:
    """Los modelos de embeddings que LM Studio ofrece ahora (los que tienen 'embed' en el nombre). [] si no hay servidor."""
    import openai

    try:
        cliente = openai.OpenAI(base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1"),
                                api_key=os.getenv("LOCAL_LLM_API_KEY", "no-se-usa"), timeout=10, max_retries=0)
        return sorted(m.id for m in cliente.models.list().data if "embed" in m.id.lower())
    except Exception:  # noqa: BLE001 - sin servidor local, la comparación simplemente se omite
        return []


def maxsim(q: np.ndarray, d: np.ndarray) -> tuple[float, np.ndarray]:
    """Interacción tardía (ColBERT): para cada vector de la pregunta, su mejor coincidencia en el documento; se suman.

    q: (tokens de la pregunta × dim) · d: (tokens del documento × dim). Devuelve (puntaje, matriz de similitudes).
    """
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    sim = qn @ dn.T
    return float(sim.max(axis=1).sum()), sim
