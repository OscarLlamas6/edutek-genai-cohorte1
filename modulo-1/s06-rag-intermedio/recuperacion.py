"""Recuperación intermedia: búsqueda híbrida, filtros, reranking y reescritura de consultas.

En la Sesión 5 la recuperación era UNA sola cosa: vectorizar la pregunta y traer los k fragmentos más parecidos
(búsqueda SEMÁNTICA). Funciona bien con paráfrasis ("se me olvidó la clave" ≈ "restablecer contraseña") y mal
con lo que depende de una palabra EXACTA (un código de error, un modelo de equipo, un número de teléfono).
Esta sesión añade las piezas que usan los RAG en producción:

    ┌─────────────┐
    │ pregunta    │ ──► (¿conversación? ► REESCRIBIR) ──┐
    └─────────────┘                                      ▼
                         ┌───────────── BÚSQUEDA ─────────────┐
                         │ semántica (vectores)               │   FILTROS por metadatos
                         │ léxica   (BM25, palabras exactas)  │   (publico, fuente, tipo…)
                         │      └──── fusión RRF ────┘        │
                         └──────────────┬─────────────────────┘
                                        ▼  candidatos (p. ej. 8)
                                  RERANKING (reordenar con un modelo que lee pregunta + fragmento)
                                        ▼  los mejores k  ──►  prompt del LLM (Sesión 5)

Todo reutiliza lo anterior: embeddings.py y almacenes.py (Sesión 4), rag.py y cliente_llm.py (Sesiones 2 y 5),
plantillas.py y prompts versionados (Sesión 3).

💰 COSTO: la búsqueda (vectorial, BM25, RRF, filtros) es gratis con embeddings locales. `reescribir_consulta`,
   `expandir_consulta` y `rerank_llm` hacen 1 llamada al LLM por pregunta: ≈ US$ 0.0003 con un modelo económico
   de proveedor, 0 con el modelo local.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from almacenes import AlmacenMemoria, Fragmento, Resultado
from embeddings import modelo_embeddings, vector_consulta, vectorizar
from plantillas import renderizar
from rag import NO_SE, Respuesta, citas_en, construir_prompt, generar, texto_a_vectorizar

# ───────────────────────── Texto: normalizar y tokenizar ─────────────────────────


def sin_acentos(texto: str) -> str:
    """'Penalización' → 'Penalizacion'. Quita los acentos para que 'dólares' y 'dolares' sean la misma palabra."""
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def normalizar_texto(texto: str) -> str:
    """Minúsculas, sin acentos y con espacios simples. Sirve para comparar textos sin que importen las mayúsculas."""
    return " ".join(sin_acentos(texto).lower().split())


# Palabras que no ayudan a buscar ("el", "de", "que"…). Lista corta, solo para BM25.
PALABRAS_VACIAS = frozenset(
    "a al algo ante como con cual cuales cuando cuanto cuantos cuanta cuantas de del desde donde el ella ellos en entre es "
    "esta este esto estos fue ha han hay la las le les lo los me mi mis muy no nos o para pero por que se si sin sobre su sus "
    "te tiene tienen tu un una uno unos unas y ya yo hago hace puedo puede pueden debe deben es son ser estar".split()
)

_PATRON_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenizar(texto: str, prefijo: int | None = 5) -> list[str]:
    """Convierte un texto en la lista de 'términos' que usa BM25.

    - Minúsculas y sin acentos.
    - Conserva códigos con guion o punto COMPLETOS ('e-217', '192.168.1.1', '0800-26337') y también sus partes.
    - Quita palabras vacías.
    - `prefijo` (por defecto 5): recorta las palabras al inicio ('cancelar', 'cancelación' → 'cance'). Es un
      "stemming" ingenuo, pero junta las formas de una misma palabra. No se aplica a tokens con dígitos (códigos).
    """
    tokens: list[str] = []
    for t in _PATRON_TOKEN.findall(normalizar_texto(texto)):
        partes = [t] + (re.split(r"[-.]", t) if re.search(r"[-.]", t) else [])
        for p in partes:
            if not p or p in PALABRAS_VACIAS:
                continue
            if prefijo and p.isalpha() and len(p) > prefijo:
                p = p[:prefijo]
            tokens.append(p)
    return tokens


# ───────────────────────── Búsqueda léxica: BM25 ─────────────────────────


def _cumple(metadatos: dict, filtro: dict | None) -> bool:
    return not filtro or all(metadatos.get(c) == v for c, v in filtro.items())


class BM25:
    """BM25: el algoritmo clásico de búsqueda por PALABRAS (el de Elasticsearch, OpenSearch o Lucene).

    Idea: un fragmento puntúa más cuando contiene las palabras de la consulta, sobre todo si son RARAS en el
    corpus (IDF) y aparecen varias veces (frecuencia), pero con rendimientos decrecientes (k1) y sin favorecer
    a los fragmentos más largos (b). No entiende significado: 'contraseña' y 'clave' son palabras distintas.
    """

    def __init__(self, fragmentos: list[Fragmento], k1: float = 1.5, b: float = 0.75, prefijo: int | None = 5):
        self.fragmentos = fragmentos
        self.k1, self.b, self.prefijo = k1, b, prefijo
        self._tf: list[Counter] = []
        self._largo: list[int] = []
        df: Counter = Counter()
        for f in fragmentos:
            toks = tokenizar(texto_a_vectorizar(f), prefijo)
            self._tf.append(Counter(toks))
            self._largo.append(len(toks))
            df.update(set(toks))
        n = len(fragmentos)
        self._largo_medio = (sum(self._largo) / n) if n else 0.0
        # IDF "suavizado": nunca es negativo, aunque un término esté en casi todos los fragmentos.
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def buscar(self, consulta: str, k: int = 3, filtro: dict | None = None) -> list[Resultado]:
        terminos = tokenizar(consulta, self.prefijo)
        puntajes: list[tuple[float, int]] = []
        for i, f in enumerate(self.fragmentos):
            if not _cumple(f.metadatos, filtro):
                continue
            tf, largo = self._tf[i], self._largo[i]
            s = 0.0
            for t in terminos:
                if t in tf:
                    f_t = tf[t]
                    s += self._idf[t] * f_t * (self.k1 + 1) / (f_t + self.k1 * (1 - self.b + self.b * largo / self._largo_medio))
            if s > 0:
                puntajes.append((s, i))
        puntajes.sort(key=lambda x: -x[0])
        return [Resultado(self.fragmentos[i].id, self.fragmentos[i].texto, self.fragmentos[i].metadatos, s)
                for s, i in puntajes[:k]]


# ───────────────────────── Índice: vectores + BM25 sobre los mismos fragmentos ─────────────────────────

_CACHE_VECTORES: dict[tuple[str, str, str], list[float]] = {}


def vectorizar_cacheado(textos: list[str], tipo: str = "documento") -> list[list[float]]:
    """Como `vectorizar`, pero recuerda los textos ya vectorizados (en esta sesión de Python).

    En el laboratorio se prueban muchas configuraciones que comparten fragmentos: sin caché, se pagaría (en tiempo,
    o en dinero con OpenAI) la misma vectorización muchas veces.
    """
    modelo = modelo_embeddings()
    faltan = [t for t in dict.fromkeys(textos) if (modelo, tipo, t) not in _CACHE_VECTORES]
    if faltan:
        for t, v in zip(faltan, vectorizar(faltan, tipo=tipo)):
            _CACHE_VECTORES[(modelo, tipo, t)] = v
    return [_CACHE_VECTORES[(modelo, tipo, t)] for t in textos]


def vector_consulta_cacheado(texto: str) -> list[float]:
    return vectorizar_cacheado([texto], tipo="consulta")[0]


@dataclass
class Indice:
    """Los mismos fragmentos guardados de dos maneras: vectores (semántica) y BM25 (palabras)."""

    fragmentos: list[Fragmento]
    almacen: AlmacenMemoria
    bm25: BM25
    por_id: dict[str, Fragmento] = field(default_factory=dict)


def construir_indice(fragmentos: list[Fragmento], prefijo: int | None = 5) -> Indice:
    """Vectoriza (con caché) y construye los dos índices. Gratis con embeddings locales."""
    vectores = vectorizar_cacheado([texto_a_vectorizar(f) for f in fragmentos], tipo="documento")
    for f, v in zip(fragmentos, vectores):
        f.vector = v
    almacen = AlmacenMemoria()
    almacen.preparar(len(vectores[0]))
    almacen.insertar(fragmentos)
    return Indice(fragmentos, almacen, BM25(fragmentos, prefijo=prefijo), {f.id: f for f in fragmentos})


# ───────────────────────── Búsquedas ─────────────────────────


def buscar_vectorial(indice: Indice, consulta: str, k: int = 3, filtro: dict | None = None) -> list[Resultado]:
    """Búsqueda SEMÁNTICA (la de la Sesión 5)."""
    return indice.almacen.buscar(vector_consulta_cacheado(consulta), k=k, filtro=filtro)


def buscar_bm25(indice: Indice, consulta: str, k: int = 3, filtro: dict | None = None) -> list[Resultado]:
    """Búsqueda LÉXICA: palabras exactas."""
    return indice.bm25.buscar(consulta, k=k, filtro=filtro)


def fusion_rrf(listas: list[list[Resultado]], k: int = 3, c: int = 60, pesos: list[float] | None = None) -> list[Resultado]:
    """Reciprocal Rank Fusion: junta varias listas ordenadas SIN comparar sus puntajes.

    Cada lista aporta 1 / (c + posición) a cada fragmento. Se suman los aportes y se reordena. Como solo usa la
    POSICIÓN, no importa que el coseno vaya de 0 a 1 y BM25 de 0 a 20: son escalas incomparables, y RRF las
    evita. Un fragmento bien ubicado en AMBAS listas sube al primer lugar. `c` (60 es el valor clásico) suaviza
    la diferencia entre las primeras posiciones.
    """
    pesos = pesos or [1.0] * len(listas)
    total: dict[str, float] = defaultdict(float)
    ejemplo: dict[str, Resultado] = {}
    for lista, peso in zip(listas, pesos):
        for pos, r in enumerate(lista, start=1):
            total[r.id] += peso / (c + pos)
            ejemplo.setdefault(r.id, r)
    ordenados = sorted(total, key=lambda i: -total[i])[:k]
    return [Resultado(i, ejemplo[i].texto, ejemplo[i].metadatos, total[i]) for i in ordenados]


def buscar_hibrido(indice: Indice, consulta: str, k: int = 3, filtro: dict | None = None,
                   candidatos: int = 10, pesos: list[float] | None = None) -> list[Resultado]:
    """Búsqueda HÍBRIDA: semántica + BM25, fusionadas con RRF. Cada una aporta sus `candidatos` mejores."""
    return fusion_rrf(
        [buscar_vectorial(indice, consulta, candidatos, filtro), buscar_bm25(indice, consulta, candidatos, filtro)],
        k=k, pesos=pesos,
    )


MODOS = {"vectorial": buscar_vectorial, "bm25": buscar_bm25, "hibrido": buscar_hibrido}


# ───────────────────────── Padre-hijo ─────────────────────────


def expandir_a_padre(resultados: list[Resultado]) -> list[Resultado]:
    """Cambia cada fragmento HIJO por el texto de su PADRE (la sección entera), sin repetir padres y respetando el orden."""
    vistos: set[str] = set()
    salida: list[Resultado] = []
    for r in resultados:
        pid = r.metadatos.get("padre_id", r.id)
        if pid in vistos:
            continue
        vistos.add(pid)
        salida.append(Resultado(pid, r.metadatos.get("texto_padre", r.texto), r.metadatos, r.puntaje))
    return salida


# ───────────────────────── Reescritura y expansión de la consulta (LLM) ─────────────────────────


def _linea_limpia(texto: str) -> str:
    """Primera línea no vacía, sin comillas ni prefijos como 'Consulta reescrita:'."""
    for linea in texto.strip().splitlines():
        linea = linea.strip().strip('"“”«»')
        linea = re.sub(r"^(consulta( reescrita)?|respuesta)\s*:\s*", "", linea, flags=re.I)
        if linea:
            return linea
    return texto.strip()


def reescribir_consulta(pregunta: str, historial: list[dict], proveedor: str | None = None, version: str = "v1"):
    """💰 Reescribe una pregunta de seguimiento ("¿y los domingos?") como consulta AUTOCONTENIDA usando el historial.

    `historial`: lista de {"rol": "cliente" | "asistente", "texto": "..."}. Devuelve (consulta, ResultadoLLM).
    Sin historial no hace falta llamar al LLM: devuelve la pregunta tal cual.
    """
    if not historial:
        return pregunta, None
    prompt = renderizar("reescritura", version, historial=historial, pregunta=pregunta)
    r = generar(prompt, proveedor, max_tokens=80)
    return _linea_limpia(r.texto), r


def expandir_consulta(pregunta: str, n: int = 3, proveedor: str | None = None, version: str = "v1"):
    """💰 Pide al LLM `n` formas distintas de preguntar lo mismo (multi-consulta). Devuelve (variantes, ResultadoLLM)."""
    r = generar(renderizar("expansion", version, pregunta=pregunta, n=n), proveedor, max_tokens=160)
    variantes = []
    for linea in r.texto.splitlines():
        linea = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", linea).strip().strip('"“”«»')
        linea = re.sub(r"^pregunta\s*:\s*", "", linea, flags=re.I)      # modelos pequeños repiten la etiqueta
        if linea and not linea.endswith(":") and linea.lower() != pregunta.lower():
            variantes.append(linea)
    return variantes[:n], r


def buscar_multiconsulta(indice: Indice, consultas: list[str], k: int = 3, filtro: dict | None = None,
                         modo: str = "hibrido", candidatos: int = 10) -> list[Resultado]:
    """Busca con VARIAS consultas (la original y sus variantes) y fusiona las listas con RRF."""
    listas = [MODOS[modo](indice, q, candidatos, filtro) for q in consultas]
    return fusion_rrf(listas, k=k)


# ───────────────────────── Reranking ─────────────────────────


def rerank_llm(pregunta: str, resultados: list[Resultado], proveedor: str | None = None, k: int = 3,
               version: str = "v1", max_palabras: int = 90):
    """💰 Reordena los candidatos pidiéndole a un LLM que los ordene por utilidad. Devuelve (resultados, ResultadoLLM).

    Es la versión "casera" del reranking: un modelo que lee la pregunta Y cada fragmento JUNTOS (a diferencia de los
    embeddings, que codifican cada texto por separado). En producción se usa un modelo especializado (un
    *cross-encoder* como bge-reranker, o la API de Cohere/Voyage/Jina), más rápido, barato y fiable que un LLM general.
    Si el LLM responde algo ilegible, se conserva el orden original.
    """
    if len(resultados) <= 1:
        return resultados[:k], None
    recortados = [Resultado(r.id, " ".join(r.texto.split()[:max_palabras]), r.metadatos, r.puntaje) for r in resultados]
    r = generar(renderizar("rerank", version, pregunta=pregunta, fragmentos=recortados), proveedor, max_tokens=60)
    orden: list[int] = []
    for n in re.findall(r"\d+", r.texto):
        i = int(n) - 1
        if 0 <= i < len(resultados) and i not in orden:
            orden.append(i)
    orden += [i for i in range(len(resultados)) if i not in orden]       # los que el modelo olvidó, al final
    return [resultados[i] for i in orden][:k], r


# ───────────────────────── El pipeline completo ─────────────────────────


def responder_avanzado(indice: Indice, pregunta: str, historial: list[dict] | None = None, filtro: dict | None = None,
                       k: int = 3, candidatos: int = 8, reescribir: bool = False, rerank: bool = False,
                       version: str = "v1", proveedor: str | None = None) -> tuple[Respuesta, dict]:
    """💰 RAG intermedio completo: [reescribir] → búsqueda híbrida con filtro → [rerank] → prompt con citas → LLM → verificar citas.

    Es `rag.responder` de la Sesión 5 con las piezas de esta sesión conectadas. Los pasos entre corchetes son opcionales
    (cada uno añade una llamada al LLM). Devuelve la `Respuesta` y un diccionario con lo que pasó en cada paso, para poder
    inspeccionarlo: la consulta usada, los candidatos y el costo de cada llamada.
    """
    pasos: dict = {"consulta": pregunta, "llamadas": []}
    consulta = pregunta
    if reescribir and historial:
        consulta, r_reescritura = reescribir_consulta(pregunta, historial, proveedor)
        pasos["consulta"] = consulta
        pasos["llamadas"].append(("reescritura", r_reescritura))
    candidatos_res = buscar_hibrido(indice, consulta, candidatos if rerank else k, filtro)
    pasos["candidatos"] = candidatos_res
    if rerank:
        fragmentos, r_rerank = rerank_llm(consulta, candidatos_res, proveedor, k=k)
        pasos["llamadas"].append(("rerank", r_rerank))
    else:
        fragmentos = candidatos_res[:k]
    if not fragmentos:
        return Respuesta(pregunta, NO_SE, [], sin_respuesta=True), pasos
    llamada = generar(construir_prompt(pregunta, fragmentos, version), proveedor)
    pasos["llamadas"].append(("generacion", llamada))
    texto = llamada.texto.strip()
    citas = citas_en(texto)
    invalidas = {n for n in citas if not 1 <= n <= len(fragmentos)}
    return Respuesta(pregunta, texto, fragmentos, citas=citas, citas_invalidas=invalidas,
                     sin_respuesta=texto.startswith("No encuentro"), llamada=llamada), pasos


# ───────────────────────── Evaluación ─────────────────────────


def contiene_clave(texto: str, clave: str) -> bool:
    """¿El fragmento contiene el dato que responde la pregunta? (sin importar mayúsculas ni acentos)."""
    return normalizar_texto(clave) in normalizar_texto(texto)


def evaluar_busqueda(preguntas: list[dict], buscador, k: int = 3) -> dict:
    """Mide un buscador con un conjunto de preguntas. `buscador(pregunta_dict, k)` devuelve una lista de resultados.

    Cada pregunta trae `clave`: el texto que debe aparecer en un fragmento para que ESE fragmento responda.
    Métricas:
        hit@1   la respuesta está en el primer resultado
        hit@k   la respuesta está entre los k primeros
        MRR     promedio de 1 / posición del primer acierto (1.0 = siempre primero; 0 = nunca aparece)
        contexto  palabras promedio que se enviarían al LLM (los k primeros): más palabras = más tokens = más costo
    """
    h1 = hk = 0
    rr = 0.0
    palabras = 0
    fallos = []
    for p in preguntas:
        res = buscador(p, k)
        pos = next((i for i, r in enumerate(res, start=1) if contiene_clave(r.texto, p["clave"])), None)
        h1 += pos == 1
        hk += pos is not None
        rr += 1 / pos if pos else 0.0
        palabras += sum(len(r.texto.split()) for r in res)
        if pos != 1:
            fallos.append({"id": p["id"], "pregunta": p["pregunta"], "posicion": pos})
    n = len(preguntas)
    return {"n": n, "hit1": h1, "hitk": hk, "k": k, "mrr": rr / n if n else 0.0,
            "contexto": palabras / n if n else 0.0, "fallos": fallos}
