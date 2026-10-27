"""GraphRAG a escala de juguete: un grafo de conocimiento de servicios de TI y tres formas de consultarlo.

Un RAG con vectores busca TEXTOS PARECIDOS a la pregunta. Eso funciona cuando la respuesta está escrita en
un párrafo. Falla cuando la respuesta hay que CONSTRUIRLA siguiendo relaciones entre cosas:

    "Si cae la base de datos de clientes, ¿qué servicios dejan de funcionar?"

El catálogo (datos/servicios_ti.md) solo dice, en cada sección, DE QUÉ DEPENDE cada servicio. Nadie escribió
"de la base de datos dependen X, Y y Z": hay que recorrer las dependencias al revés, varios saltos. Es la
misma diferencia que entre buscar en un archivo de texto y consultar una CMDB (base de configuración).

GraphRAG añade al RAG un GRAFO DE CONOCIMIENTO:

    texto ──(LLM extrae)──► tripletas (sujeto, relación, objeto) ──► grafo ──► recorrer vecinos / calcular
                                                                                       │
                                                      pregunta ──► entidades ──────────┘ ──► contexto ──► LLM

Este módulo tiene lo mínimo para verlo funcionar con networkx (sin servicios externos):
    1. Construir el grafo (tripletas curadas a mano o extraídas con un LLM) y consultarlo con algoritmos.
    2. RecuperadorGrafo: de una pregunta a los hechos del grafo que la contestan ("búsqueda local").
    3. Extraer tripletas con un LLM y medirlas contra las curadas (casi siempre el paso más frágil).
    4. Evaluar preguntas multi-salto: vectorial vs grafo vs combinado.

Las implementaciones reales (Microsoft GraphRAG, LightRAG) añaden lo que aquí NO está: extraer entidades
con un LLM a gran escala, detectar comunidades (Leiden), generar resúmenes por comunidad (búsqueda global)
y guardar todo en un almacén. Se explican en el notebook, no se instalan.

💰 COSTO
    - Construir y consultar el grafo curado: gratis (networkx, sin LLM).
    - Extraer tripletas con un LLM: 1 llamada por sección del catálogo (≈ 16; gratis con el modelo local).
    - evaluar_multisalto(): 1 llamada al LLM por pregunta.
"""

from __future__ import annotations

import json
import re
import textwrap
from collections import defaultdict
from pathlib import Path

import networkx as nx
from pydantic import BaseModel

from almacenes import Resultado
from cliente_llm import ResultadoLLM, llamar_local
from evaluacion import normalizar, pedir_json_con_ejemplo, promedio, responder_con
from plantillas import renderizar
from rag import generar, leer_documento

RELACIONES_DE_IMPACTO = ("depende_de", "corre_en")   # las que propagan una falla; "administrado_por" no

# ───────────────────────── 1. El grafo y sus algoritmos ─────────────────────────


def construir_grafo(entidades: dict, tripletas: list) -> nx.DiGraph:
    """Grafo dirigido: un nodo por entidad y una arista (sujeto → objeto) por tripleta, con su `relacion`."""
    G = nx.DiGraph()
    for nombre, atributos in entidades.items():
        G.add_node(nombre, **atributos)
    for sujeto, relacion, objeto in tripletas:
        G.add_edge(sujeto, objeto, relacion=relacion)
    return G


def cargar_grafo(ruta: str | Path = "datos/servicios_ti_grafo.json") -> nx.DiGraph:
    """Carga el grafo CURADO a mano (el camino fiable: no depende de que un LLM extraiga bien)."""
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return construir_grafo(datos["entidades"], [tuple(t) for t in datos["tripletas"]])


def _impacto(G: nx.DiGraph) -> nx.DiGraph:
    """Subgrafo con solo las relaciones por las que se propaga una falla (A depende de B: arista A → B)."""
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    H.add_edges_from((a, b) for a, b, d in G.edges(data=True) if d["relacion"] in RELACIONES_DE_IMPACTO)
    return H


def afectados_por(G: nx.DiGraph, nodo: str) -> list[str]:
    """Si `nodo` falla, ¿qué servicios dejan de funcionar? Todos los que dependen de él, directa o indirectamente."""
    return sorted(nx.ancestors(_impacto(G), nodo))


def dependencias_de(G: nx.DiGraph, nodo: str) -> list[str]:
    """¿De qué depende `nodo`, directa o indirectamente?"""
    return sorted(nx.descendants(_impacto(G), nodo))


def administrador_de(G: nx.DiGraph, nodo: str) -> str | None:
    return next((b for _, b, d in G.out_edges(nodo, data=True) if d["relacion"] == "administrado_por"), None)


def equipos_de(G: nx.DiGraph, nodos: list[str]) -> list[str]:
    return sorted({e for n in nodos if (e := administrador_de(G, n))})


def mas_dependidos(G: nx.DiGraph, top: int = 3) -> list[tuple[str, int]]:
    """Servicios con más servicios que dependen DIRECTAMENTE de ellos. Una pregunta 'global': hay que contar."""
    H = _impacto(G)
    return sorted(((n, H.in_degree(n)) for n in H.nodes), key=lambda x: (-x[1], x[0]))[:top]


# ───────────────────────── 2. De la pregunta a los hechos del grafo ─────────────────────────


def entidades_en(G: nx.DiGraph, texto: str) -> list[str]:
    """Entidades del grafo mencionadas en un texto (por nombre o alias). Es el 'entity linking' más simple posible.

    Busca primero los nombres más largos y tapa lo ya encontrado, para que el alias "base de datos" no
    se cuente dentro de "Base de datos de clientes".
    """
    t = normalizar(texto)
    candidatos = []
    for nodo, atributos in G.nodes(data=True):
        for alias in [nodo, *atributos.get("alias", [])]:
            candidatos.append((normalizar(alias), nodo))
    candidatos.sort(key=lambda c: -len(c[0]))
    encontradas: list[tuple[int, str]] = []
    for alias, nodo in candidatos:
        m = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", t)
        if m:
            encontradas.append((m.start(), nodo))
            t = t[: m.start()] + " " * (m.end() - m.start()) + t[m.end():]
    return list(dict.fromkeys(n for _, n in sorted(encontradas)))


def frase_de(sujeto: str, relacion: str, objeto: str) -> str:
    plantilla = {"depende_de": "{s} depende de {o}.", "corre_en": "{s} se ejecuta en {o}.",
                 "administrado_por": "{s} es administrado por {o}."}.get(relacion, "{s} " + relacion + " {o}.")
    return plantilla.format(s=sujeto, o=objeto)


def vecindad(G: nx.DiGraph, nodos: list[str], profundidad: int = 2) -> list[tuple[str, str, str]]:
    """Aristas (sujeto, relación, objeto) a menos de `profundidad` saltos de los nodos, en AMBOS sentidos.

    Es la 'búsqueda local' de GraphRAG: partir de las entidades de la pregunta y mirar a sus vecinos.
    Con profundidad 1 se ve lo directo; cada salto más trae más hechos (y más ruido).
    """
    aristas: set[tuple[str, str, str]] = set()
    for n in nodos:
        sub = nx.ego_graph(G, n, radius=profundidad, undirected=True)
        aristas |= {(a, G.edges[a, b]["relacion"], b) for a, b in sub.edges}
    return sorted(aristas)


_INTENCION_IMPACTO = ("falla", "falle", "cae", "caiga", "caen", "apaga", "apague", "afecta", "afectan", "afectado",
                      "afectados", "deja de funcionar", "dejan de funcionar", "intervenir")


class RecuperadorGrafo:
    """Convierte una pregunta en 'fragmentos' construidos con hechos del grafo.

    modo="vecindad"  → solo los hechos cercanos a las entidades de la pregunta (búsqueda local).
    modo="recorrido" → además, un 'análisis' calculado con un algoritmo (impacto, dependencias, ranking),
                       elegido con reglas simples según la pregunta. En un sistema real, esa elección la hace
                       un LLM que traduce la pregunta a una consulta del grafo (texto → Cypher/SPARQL).
    """

    def __init__(self, G: nx.DiGraph, profundidad: int = 2, modo: str = "vecindad"):
        if modo not in ("vecindad", "recorrido"):
            raise ValueError("modo debe ser 'vecindad' o 'recorrido'")
        self.G, self.profundidad, self.modo = G, profundidad, modo
        self.nombre = f"grafo ({modo}, prof. {profundidad})"
        self.costo = 0.0

    def _analisis(self, pregunta: str, nodos: list[str]) -> list[str]:
        q, G = normalizar(pregunta), self.G
        if "mas servicios" in q and "depend" in q:                      # pregunta global: hay que contar
            return ["Análisis del grafo, servicios con más servicios que dependen directamente de ellos: "
                    + "; ".join(f"{n} ({c})" for n, c in mas_dependidos(G)) + "."]
        if nodos and any(p in q for p in _INTENCION_IMPACTO):           # "si falla X, ¿qué se afecta?"
            n = nodos[0]
            afectados = afectados_por(G, n)
            lineas = [f"Análisis del grafo: si falla «{n}», dejan de funcionar {len(afectados)} servicios: "
                      + (", ".join(afectados) or "ninguno") + "."]
            if "equipos" in q:
                lineas.append(f"Equipos que administran «{n}» y los servicios afectados: "
                              + ", ".join(equipos_de(G, [n, *afectados])) + ".")
            return lineas
        if nodos and "depende" in q:                                    # "¿de qué depende X?"
            n = nodos[0]
            deps = dependencias_de(G, n)
            return [f"Análisis del grafo: «{n}» depende, directa o indirectamente, de {len(deps)} elementos: "
                    + ", ".join(deps) + "."]
        return []

    def __call__(self, pregunta: str, k: int | None = None) -> list[Resultado]:
        """`k` se ignora: el tamaño del contexto lo decide la profundidad, no un número de fragmentos."""
        nodos = entidades_en(self.G, pregunta)
        analisis = self._analisis(pregunta, nodos) if self.modo == "recorrido" else []   # una pregunta global no nombra entidades
        resultados = [Resultado(id=f"analisis#{i}", texto=t, metadatos={"seccion": "análisis del grafo"}, puntaje=1.0)
                      for i, t in enumerate(analisis)]
        por_sujeto: dict[str, list[str]] = defaultdict(list)
        for s, r, o in vecindad(self.G, nodos, self.profundidad):
            por_sujeto[s].append(frase_de(s, r, o))
        for s in sorted(por_sujeto, key=lambda s: (s not in nodos, s)):          # primero los de la pregunta
            resultados.append(Resultado(id=f"grafo:{s}", texto=" ".join(por_sujeto[s]),
                                        metadatos={"seccion": f"grafo · {s}"}, puntaje=1.0))
        return resultados


class RecuperadorCombinado:
    """Une los fragmentos de varios recuperadores (p. ej. vectorial + grafo), sin repetir identificadores."""

    def __init__(self, *recuperadores, nombre: str = "vectorial + grafo"):
        self.recuperadores, self.nombre = recuperadores, nombre

    @property
    def costo(self) -> float:
        return sum(getattr(r, "costo", 0.0) for r in self.recuperadores)

    def __call__(self, pregunta: str, k: int = 4) -> list[Resultado]:
        vistos, salida = set(), []
        for r in self.recuperadores:
            for f in r(pregunta, k):
                if f.id not in vistos:
                    vistos.add(f.id)
                    salida.append(f)
        return salida


# ───────────────────────── 3. Extraer el grafo con un LLM (el paso frágil) ─────────────────────────


class Tripleta(BaseModel):
    sujeto: str
    relacion: str
    objeto: str


class ExtraccionTripletas(BaseModel):
    tripletas: list[Tripleta]


def extraer_tripletas(manual: str | Path, version: str = "v1", entidades: list[str] | None = None,
                      proveedor: str | None = None, modelo: str | None = None) -> tuple[list[tuple[str, str, str]], list[ResultadoLLM]]:
    """💰 Pide al LLM las tripletas de cada sección del catálogo (1 llamada por sección).

    version="v1" → nombres libres. version="v2" → se le da la lista de entidades conocidas y debe usar solo esas.
    `modelo` solo aplica al proveedor local (para probar un modelo más grande).
    """
    ejemplo = '{"tripletas": [{"sujeto": "nombre", "relacion": "depende_de", "objeto": "nombre"}]}'

    def llamar(texto: str) -> ResultadoLLM:
        if modelo:
            return llamar_local(texto, modelo=modelo, max_tokens=700, temperature=0)
        return generar(texto, proveedor, max_tokens=700)

    tripletas: list[tuple[str, str, str]] = []
    llamadas: list[ResultadoLLM] = []
    for titulo, texto in leer_documento(manual).secciones:
        if titulo == "Introducción":
            continue
        prompt = renderizar("extraer_tripletas", version, titulo=titulo, texto=texto, entidades=entidades or [])
        try:
            e, usadas = pedir_json_con_ejemplo(llamar, prompt, ExtraccionTripletas, ejemplo)
        except ValueError:
            continue
        llamadas += usadas
        tripletas += [(t.sujeto.strip(), t.relacion.strip(), t.objeto.strip()) for t in e.tripletas]
    return tripletas, llamadas


_ARTICULO = re.compile(r"^(el|la|los|las)\s+", re.IGNORECASE)


def nombre_canonico(G: nx.DiGraph, nombre: str) -> str | None:
    """Nombre oficial de una entidad escrita de cualquier forma (con alias, sin artículo, sin acentos), o None."""
    limpio = normalizar(_ARTICULO.sub("", nombre.strip()))
    for nodo, atributos in G.nodes(data=True):
        if limpio in {normalizar(nodo), *(normalizar(a) for a in atributos.get("alias", []))}:
            return nodo
    return None


def grafo_desde_tripletas(G_base: nx.DiGraph, tripletas: list[tuple[str, str, str]]) -> nx.DiGraph:
    """Grafo armado con tripletas extraídas: lleva cada nombre al oficial y descarta las que no reconoce.

    Conserva los nodos del grafo curado para poder comparar; solo cambian las aristas. Un error de
    extracción (una arista de más o de menos) cambia las respuestas de TODOS los recorridos que pasan por ella.
    """
    G = nx.DiGraph()
    G.add_nodes_from(G_base.nodes(data=True))
    for s, r, o in tripletas:
        cs, co = nombre_canonico(G_base, s), nombre_canonico(G_base, o)
        if cs and co and cs != co:
            G.add_edge(cs, co, relacion=normalizar(r).replace(" ", "_"))
    return G


def comparar_tripletas(G_curado: nx.DiGraph, extraidas: list[tuple[str, str, str]]) -> dict:
    """Mide las tripletas extraídas contra las curadas: precisión (¿qué fracción es correcta?) y recall (¿cuántas halló?).

    Una tripleta extraída es correcta si, tras llevar sus dos entidades al nombre oficial, existe igual en el grafo curado.
    """
    curadas = {(a, d["relacion"], b) for a, b, d in G_curado.edges(data=True)}
    canonicas, no_reconocidas, descartadas = set(), [], 0
    for s, r, o in extraidas:
        cs, co = nombre_canonico(G_curado, s), nombre_canonico(G_curado, o)
        if cs is None or co is None:
            no_reconocidas += [x for x, c in ((s, cs), (o, co)) if c is None]
            descartadas += 1
            continue
        canonicas.add((cs, normalizar(r).replace(" ", "_"), co))
    correctas = canonicas & curadas
    precision = len(correctas) / len(extraidas) if extraidas else 0.0
    recall = len(correctas) / len(curadas)
    return {"extraidas": len(extraidas), "correctas": len(correctas), "curadas": len(curadas),
            "con_entidad_no_reconocida": descartadas, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "nombres_no_reconocidos": sorted(set(no_reconocidas)), "faltan": sorted(curadas - canonicas),
            "sobran": sorted(canonicas - curadas)}


# ───────────────────────── 4. Evaluar preguntas multi-salto ─────────────────────────


def cobertura(G: nx.DiGraph, texto: str, esperadas: list[str], pregunta: str = "") -> dict:
    """Qué fracción de las entidades esperadas aparece en un texto (recall) y qué fracción de las entidades
    mencionadas es esperada (precisión). Las entidades que nombra la propia pregunta no cuentan."""
    mencionadas = set(entidades_en(G, texto)) - set(entidades_en(G, pregunta))
    esperadas = set(esperadas)
    aciertos = mencionadas & esperadas
    return {"recall": len(aciertos) / len(esperadas), "precision": len(aciertos) / len(mencionadas) if mencionadas else 0.0,
            "faltan": sorted(esperadas - mencionadas), "sobran": sorted(mencionadas - esperadas)}


def evaluar_multisalto(G: nx.DiGraph, preguntas: list[dict], recuperador, k: int = 4, version: str = "v2",
                       proveedor: str | None = None, con_generacion: bool = True) -> list[dict]:
    """💰 Evalúa un recuperador con las preguntas multi-salto (datos/preguntas_multisalto.json).

    Por pregunta mide dos cosas, por separado como siempre:
      contexto  → ¿las entidades de la respuesta llegaron al prompt?   (la recuperación)
      respuesta → ¿el modelo las nombró, y solo esas?                  (la generación)
    Con con_generacion=False solo se mide el contexto (gratis).
    """
    filas = []
    for p in preguntas:
        costo_antes = getattr(recuperador, "costo", 0.0)
        if con_generacion:
            r = responder_con(recuperador, p["pregunta"], k=k, version=version, proveedor=proveedor)
            fragmentos, texto = r.fragmentos, r.texto
        else:
            fragmentos, texto, r = recuperador(p["pregunta"], k), "", None
        contexto = cobertura(G, "\n".join(f.texto for f in fragmentos), p["esperadas"], p["pregunta"])
        fila = {"id": p["id"], "tipo": p["tipo"], "saltos": p["saltos"], "contexto_recall": contexto["recall"],
                "palabras_contexto": sum(len(f.texto.split()) for f in fragmentos), "respuesta": texto,
                "costo": getattr(recuperador, "costo", 0.0) - costo_antes + (r.llamada.costo if r and r.llamada else 0.0)}
        if r is not None:
            resp = cobertura(G, texto, p["esperadas"], p["pregunta"])
            fila.update({"respuesta_recall": resp["recall"], "respuesta_precision": resp["precision"],
                         "faltan": resp["faltan"], "sobran": resp["sobran"], "se_abstuvo": r.sin_respuesta})
        filas.append(fila)
    return filas


def resumen_multisalto(filas: list[dict]) -> dict:
    r = {"n": len(filas), "contexto_recall": promedio([f["contexto_recall"] for f in filas]),
         "palabras_contexto": promedio([float(f["palabras_contexto"]) for f in filas]),
         "costo": sum(f["costo"] for f in filas)}
    if filas and "respuesta_recall" in filas[0]:
        r["respuesta_recall"] = promedio([f["respuesta_recall"] for f in filas])
        r["respuesta_precision"] = promedio([f["respuesta_precision"] for f in filas])
        r["completas"] = sum(f["respuesta_recall"] == 1.0 and f["respuesta_precision"] == 1.0 for f in filas)
    return r


# ───────────────────────── 5. Dibujar el grafo ─────────────────────────

_COLORES = {"infraestructura": "#9aa5b1", "datos": "#6fa8dc", "seguridad": "#b4a7d6", "plataforma": "#93c47d",
            "externo": "#f6b26b", "negocio": "#ffd966", "soporte": "#ead1dc"}


def dibujar_grafo(G: nx.DiGraph, resaltar: list[str] | tuple[str, ...] = (), titulo: str = "", ax=None):
    """Dibuja los servicios por capas: arriba lo que usa el cliente, abajo la infraestructura. La flecha A → B es 'A depende de B'."""
    import matplotlib.pyplot as plt

    H = _impacto(G)
    H.remove_nodes_from([n for n, d in G.nodes(data=True) if d.get("tipo") == "equipo"])
    capas = {n: i for i, grupo in enumerate(nx.topological_generations(H)) for n in grupo}
    por_capa: dict[int, list[str]] = defaultdict(list)
    for n, c in capas.items():
        por_capa[c].append(n)
    pos = {n: (j - (len(g) - 1) / 2, -c * 1.4) for c, g in por_capa.items() for j, n in enumerate(sorted(g))}
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6.5))
    colores = ["#e06666" if n in resaltar else _COLORES.get(G.nodes[n].get("tipo"), "#cccccc") for n in H.nodes]
    nx.draw_networkx_edges(H, pos, ax=ax, arrows=True, arrowsize=12, edge_color="#888888", node_size=2600, width=1)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=colores, node_size=2600, edgecolors="#555555")
    nx.draw_networkx_labels(H, pos, ax=ax, labels={n: textwrap.fill(n, 14) for n in H.nodes}, font_size=6.5)
    ax.set_title(titulo or "Servicios de TI: A → B significa 'A depende de B'")
    ax.axis("off")
    return ax
