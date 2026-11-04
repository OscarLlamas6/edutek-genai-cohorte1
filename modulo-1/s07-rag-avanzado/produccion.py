"""Del notebook a producción: caché semántica, actualización incremental del índice y trazas.

Tres piezas pequeñas, sin dependencias nuevas y gratis (solo embeddings locales):

    CacheSemantica       Si alguien ya hizo una pregunta MUY parecida, devuelve la respuesta guardada
                         sin llamar al LLM. "Muy parecida" = similitud coseno ≥ un UMBRAL que se calibra
                         midiendo (demasiado bajo: respuestas equivocadas; demasiado alto: nunca acierta).
    IndiceIncremental    Reindexar TODO cada vez que cambia una línea del manual es caro. Se guarda el
                         HASH del contenido de cada fragmento y solo se vuelve a vectorizar lo que cambió.
    Traza                Qué registrar de cada pregunta (tiempos por etapa, fragmentos, versiones, costo)
                         para poder depurar y medir en producción.

Los tres son versiones de juguete de lo que hacen los sistemas reales (Redis, GPTCache, trazas con
OpenTelemetry, Langfuse, Phoenix…): lo importante es entender QUÉ hacen y QUÉ pueden salir mal.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import numpy as np

from almacenes import Almacen, Fragmento
from embeddings import USO, vectorizar
from rag import texto_a_vectorizar


# ───────────────────────── 1. Caché semántica ─────────────────────────


@dataclass
class CacheSemantica:
    """Guarda (pregunta, respuesta); una pregunta nueva ACIERTA si se parece lo bastante a una guardada."""

    umbral: float = 0.90
    preguntas: list[str] = field(default_factory=list)
    respuestas: list[str] = field(default_factory=list)
    _matriz: np.ndarray | None = None
    aciertos: int = 0
    fallos: int = 0

    def guardar(self, pregunta: str, respuesta: str) -> None:
        self.guardar_varias([pregunta], [respuesta])

    def guardar_varias(self, preguntas: list[str], respuestas: list[str]) -> None:
        v = np.array(vectorizar(preguntas, tipo="consulta"), dtype=np.float32)   # preguntas con preguntas: mismo prefijo
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        self._matriz = v if self._matriz is None else np.vstack([self._matriz, v])
        self.preguntas += preguntas
        self.respuestas += respuestas

    def consultar(self, pregunta: str) -> tuple[bool, int | None, float]:
        """(¿acierto?, índice de la entrada más parecida, similitud). No cuenta estadísticas: úsalo para medir."""
        if self._matriz is None:
            return False, None, 0.0
        q = np.array(vectorizar([pregunta], tipo="consulta")[0], dtype=np.float32)
        s = self._matriz @ (q / np.linalg.norm(q))
        i = int(np.argmax(s))
        return bool(s[i] >= self.umbral), i, float(s[i])

    def obtener(self, pregunta: str) -> str | None:
        """La respuesta guardada si hay acierto; None si hay que llamar al LLM. Lleva la cuenta de aciertos y fallos."""
        ok, i, _ = self.consultar(pregunta)
        self.aciertos += ok
        self.fallos += not ok
        return self.respuestas[i] if ok else None


# ───────────────────────── 2. Actualización incremental del índice ─────────────────────────


def hash_contenido(f: Fragmento) -> str:
    """Huella del contenido que se vectoriza: si no cambia el texto (ni su encabezado), no hay que re-vectorizar."""
    return hashlib.sha256(texto_a_vectorizar(f).encode("utf-8")).hexdigest()[:16]


@dataclass
class IndiceIncremental:
    """Un índice que sabe qué fragmentos YA tiene vectorizados (por hash de contenido) y solo procesa lo nuevo.

    Un almacén real (Neon, Pinecone) borraría por id; aquí, con el almacén en memoria, se reconstruye el almacén
    con los vectores que ya teníamos: lo que cuesta dinero y tiempo es VECTORIZAR, y eso sí se evita.
    """

    almacen: Almacen
    vistos: dict[str, Fragmento] = field(default_factory=dict)      # hash → fragmento con su vector

    def sincronizar(self, fragmentos: list[Fragmento]) -> dict:
        nuevos_hash = {hash_contenido(f): f for f in fragmentos}
        a_vectorizar = [f for h, f in nuevos_hash.items() if h not in self.vistos]
        borrados = [h for h in self.vistos if h not in nuevos_hash]
        llamadas_antes = USO.llamadas
        if a_vectorizar:
            vectores = vectorizar([texto_a_vectorizar(f) for f in a_vectorizar], tipo="documento")
            for f, v in zip(a_vectorizar, vectores):
                f.vector = v
        for h in borrados:
            del self.vistos[h]
        for h, f in nuevos_hash.items():
            self.vistos.setdefault(h, f)
        activos = [self.vistos[h] for h in nuevos_hash]
        self.almacen.preparar(len(activos[0].vector))
        self.almacen.vaciar()
        self.almacen.insertar(activos)
        return {
            "total": len(fragmentos),
            "vectorizados": len(a_vectorizar),
            "reutilizados": len(fragmentos) - len(a_vectorizar),
            "borrados": len(borrados),
            "palabras_vectorizadas": sum(len(texto_a_vectorizar(f).split()) for f in a_vectorizar),
            "palabras_totales": sum(len(texto_a_vectorizar(f).split()) for f in fragmentos),
            "llamadas_embeddings": USO.llamadas - llamadas_antes,
        }


# ───────────────────────── 3. Trazas ─────────────────────────


@dataclass
class Traza:
    """Cronometra las etapas de UNA pregunta y junta lo que conviene registrar. `registro()` devuelve un dict listo para JSON."""

    datos: dict = field(default_factory=dict)
    tiempos_ms: dict = field(default_factory=dict)

    def etapa(self, nombre: str):
        traza = self

        class _Cronometro:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                traza.tiempos_ms[nombre] = round((time.perf_counter() - self.t0) * 1000)

        return _Cronometro()

    def registro(self) -> dict:
        return {**self.datos, "tiempos_ms": self.tiempos_ms, "total_ms": sum(self.tiempos_ms.values())}

    def __str__(self) -> str:
        return json.dumps(self.registro(), ensure_ascii=False, indent=2)
