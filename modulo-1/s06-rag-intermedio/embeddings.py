"""Embeddings: convertir texto en vectores.

Un EMBEDDING es una lista de números (un vector) que representa el significado de un
texto: textos con significado parecido quedan CERCA en el espacio vectorial. Es lo que
permite buscar por significado y no solo por palabras exactas (búsqueda semántica), y es
la base del RAG.

Este módulo ofrece dos proveedores de embeddings, elegibles con una variable de entorno:

    EMBEDDINGS_PROVIDER=local     # por defecto: un modelo en tu computadora (LM Studio). Gratis.
    EMBEDDINGS_PROVIDER=openai    # la API de OpenAI (de pago por uso, muy barata)

    EMBEDDINGS_MODEL=...          # opcional; si no, se usa el modelo por defecto del proveedor
    EMBEDDINGS_DIMENSIONS=...     # opcional; solo OpenAI: acorta el vector (p. ej. 768)

¿Por qué no hay una opción de Anthropic? Anthropic no ofrece un modelo de embeddings
propio. Quien use una llave de Anthropic para el chat usa los embeddings locales (gratis)
o los de OpenAI.

💰 COSTO
    local   → sin costo por token (usa tu CPU/RAM).
    openai  → de pago por uso; text-embedding-3-small cuesta unos US$ 0.02 por millón de
              tokens: indexar un manual pequeño cuesta fracciones de centavo.

IMPORTANTE: los vectores de DISTINTOS modelos no se pueden mezclar ni comparar entre sí.
Si cambias de modelo, debes volver a vectorizar y a indexar todo (y la base vectorial
debe crearse con la dimensión del nuevo modelo).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

PROVEEDORES_EMBEDDINGS = ("local", "openai")

# Modelo por defecto de cada proveedor. Los nombres cambian: puedes fijarlo con EMBEDDINGS_MODEL.
#   local  → el identificador que muestra LM Studio para el modelo de embeddings descargado.
#            Por defecto uno MULTILINGÜE (nomic-embed-text-v2-moe, 768 dim, ~344 MB): los modelos
#            centrados en inglés (p. ej. nomic v1.5) recuperan mal textos en español.
#   openai → text-embedding-3-small (1536 dimensiones por defecto).
MODELO_POR_DEFECTO = {
    "local": "text-embedding-nomic-embed-text-v2-moe",
    "openai": "text-embedding-3-small",
}

# USD por millón de tokens (referencia a octubre de 2026; confirma en la página de precios).
PRECIOS_EMBEDDINGS = {"text-embedding-3-small": 0.02}

# Algunos modelos de embeddings esperan un PREFIJO distinto para documentos y para consultas
# (se entrenaron así). Si no lo pones, la calidad de la búsqueda baja. Se busca por fragmento
# del nombre del modelo: (prefijo_documento, prefijo_consulta).
PREFIJOS = {
    "nomic": ("search_document: ", "search_query: "),
    "mxbai": ("", "Represent this sentence for searching relevant passages: "),
}


@dataclass
class UsoEmbeddings:
    """Acumula cuántos tokens se vectorizaron (para estimar el costo con OpenAI)."""

    tokens: int = 0
    llamadas: int = 0


USO = UsoEmbeddings()


def proveedor_embeddings() -> str:
    p = os.getenv("EMBEDDINGS_PROVIDER", "local").strip().lower()
    if p not in PROVEEDORES_EMBEDDINGS:
        raise ValueError(f"EMBEDDINGS_PROVIDER='{p}' no es válido. Usa: {', '.join(PROVEEDORES_EMBEDDINGS)}.")
    return p


def modelo_embeddings() -> str:
    return os.getenv("EMBEDDINGS_MODEL") or MODELO_POR_DEFECTO[proveedor_embeddings()]


def dimensiones_pedidas() -> int | None:
    """Dimensión que se le pide al modelo (solo OpenAI la admite); None = la del modelo."""
    valor = os.getenv("EMBEDDINGS_DIMENSIONS")
    return int(valor) if valor and proveedor_embeddings() == "openai" else None


def costo_embeddings_usd(tokens: int | None = None, modelo: str | None = None) -> float:
    """Costo estimado de los tokens vectorizados (0 para modelos locales)."""
    tokens = USO.tokens if tokens is None else tokens
    precio = PRECIOS_EMBEDDINGS.get(modelo or modelo_embeddings(), 0.0)
    return tokens * precio / 1_000_000


def _prefijos() -> tuple[str, str]:
    nombre = modelo_embeddings().lower()
    for clave, par in PREFIJOS.items():
        if clave in nombre:
            return par
    return ("", "")


def _cliente():
    """Cliente del SDK de OpenAI: apunta a OpenAI o al servidor local de LM Studio."""
    import openai

    if proveedor_embeddings() == "local":
        return openai.OpenAI(
            base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("LOCAL_LLM_API_KEY", "no-se-usa"),
            max_retries=1,
            timeout=300,
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Falta OPENAI_API_KEY (EMBEDDINGS_PROVIDER=openai). Defínela en tu .env o usa "
            "EMBEDDINGS_PROVIDER=local con LM Studio."
        )
    return openai.OpenAI(max_retries=2)


def embeddings_disponibles() -> tuple[bool, str]:
    """Comprueba si el proveedor de embeddings responde (sin gastar nada relevante)."""
    import openai

    try:
        v = vectorizar(["prueba"], tipo="consulta")
        return True, f"ok · {proveedor_embeddings()} · {modelo_embeddings()} · {len(v[0])} dimensiones"
    except openai.APIConnectionError:
        return False, (
            "No hay servidor local de embeddings. Abre LM Studio, activa el servidor local (Local API server) "
            "y descarga un modelo de embeddings (ver docs/LMSTUDIO.md)."
        )
    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001 - mensaje claro para cualquier otro fallo
        return False, f"El proveedor de embeddings respondió con un error ({type(exc).__name__}): {exc}"


def vectorizar(textos: list[str], tipo: str = "documento", lote: int = 32) -> list[list[float]]:
    """Convierte una lista de textos en una lista de vectores (uno por texto).

    `tipo` es "documento" (lo que vas a guardar) o "consulta" (lo que vas a buscar): algunos
    modelos usan un prefijo distinto para cada uno. Se envían en lotes para no exceder
    límites del servidor.
    """
    if tipo not in ("documento", "consulta"):
        raise ValueError("tipo debe ser 'documento' o 'consulta'")
    pref_doc, pref_cons = _prefijos()
    prefijo = pref_doc if tipo == "documento" else pref_cons
    cliente = _cliente()
    extra = {}
    dims = dimensiones_pedidas()
    if dims:
        extra["dimensions"] = dims

    salida: list[list[float]] = []
    for i in range(0, len(textos), lote):
        trozo = [prefijo + t for t in textos[i : i + lote]]
        resp = cliente.embeddings.create(model=modelo_embeddings(), input=trozo, **extra)
        # El servidor devuelve un vector por texto, en el mismo orden (por si acaso, ordenamos por índice).
        datos = sorted(resp.data, key=lambda d: d.index)
        salida.extend(d.embedding for d in datos)
        USO.llamadas += 1
        USO.tokens += getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
    return salida


def vector_consulta(texto: str) -> list[float]:
    """Vectoriza UNA consulta del usuario."""
    return vectorizar([texto], tipo="consulta")[0]


# ───────────────────────── Matemática de vectores ─────────────────────────


def norma(v: list[float]) -> float:
    """Longitud (norma euclídea) de un vector."""
    return math.sqrt(sum(x * x for x in v))


def similitud_coseno(a: list[float], b: list[float]) -> float:
    """Similitud coseno entre dos vectores, de -1 a 1 (1 = misma dirección = significado muy parecido).

    coseno(a, b) = (a · b) / (|a| × |b|)

    Mide el ÁNGULO entre los vectores, no su longitud: dos textos de significado parecido
    apuntan hacia la misma dirección aunque uno sea más largo que el otro.
    """
    producto = sum(x * y for x, y in zip(a, b))
    return producto / (norma(a) * norma(b))


def normalizar(matriz: np.ndarray) -> np.ndarray:
    """Divide cada fila entre su norma (longitud 1). Con vectores normalizados, coseno = producto punto."""
    n = np.linalg.norm(matriz, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return matriz / n
