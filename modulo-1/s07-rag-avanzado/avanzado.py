"""Técnicas avanzadas de recuperación: small-to-big, encabezados contextuales, descomposición y HyDE.

El RAG básico (rag.py) recupera los k fragmentos más parecidos a la pregunta y se los da al modelo.
Falla de formas predecibles; cada técnica de este módulo ataca una:

    Falla del RAG básico                          Técnica
    ────────────────────────────────────────────  ───────────────────────────────────────────────
    Fragmento pequeño = buen vector, pero le      SMALL-TO-BIG (parent-document): buscar con el
    falta contexto para responder                 fragmento pequeño, entregar el grande
    Un fragmento suelto es ambiguo                ENCABEZADO CONTEXTUAL: anteponer la sección
    La pregunta necesita DOS temas (multi-salto)  DESCOMPOSICIÓN: separar en subpreguntas
    La pregunta no se parece al texto de la       HyDE: buscar con una respuesta hipotética
    respuesta

Qué NO está aquí, porque se vio en la Sesión 6: reescritura de la consulta, búsqueda híbrida
(palabras clave + semántica) y reranking. Y la técnica que no se resuelve con fragmentos, GraphRAG,
vive en grafo.py.

Todos los "recuperadores" de este módulo tienen la misma forma, para poder evaluarlos con las mismas
métricas (evaluacion.evaluar_rag(recuperador=...)):

    recuperador(pregunta, k) -> list[Resultado]        # y, opcionalmente, un atributo `costo`

💰 COSTO
    - small-to-big y los encabezados NO llaman al LLM (solo embeddings locales): gratis.
    - La descomposición y HyDE hacen 1 llamada corta al LLM por pregunta (gratis con el modelo local;
      ≈ US$ 0.0002 con un modelo de proveedor).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from almacenes import Almacen, Fragmento, Resultado
from cliente_llm import ResultadoLLM
from evaluacion import pedir_json_con_ejemplo
from embeddings import vectorizar
from plantillas import renderizar
from rag import fragmentar, generar, leer_documento, recuperar, texto_a_vectorizar

# ───────────────────────── 1. Small-to-big (parent-document) y encabezados contextuales ─────────────────────────
#
# El dilema del tamaño del fragmento: el fragmento PEQUEÑO tiene un vector preciso (habla de una sola
# cosa) pero poco contexto; el GRANDE tiene contexto pero un vector borroso. Small-to-big evita elegir:
# se BUSCA con el pequeño y se ENTREGA al modelo el grande al que pertenece (su "padre").
#
#     búsqueda (precisa)         entrega (con contexto)
#     fragmento pequeño  ──────►  sección completa      ("seccion")
#     fragmento pequeño  ──────►  él + sus vecinos      ("ventana")


def ingestar_pequenos(almacen: Almacen, ruta: str | Path, max_palabras: int = 35, solape: int = 1,
                      encabezado: bool = True) -> list[Fragmento]:
    """Fragmenta en trozos PEQUEÑOS, los vectoriza y los guarda (reemplaza lo que hubiera en el almacén).

    encabezado=True  → se vectoriza "sección. texto" (encabezado contextual): el título aporta el tema.
    encabezado=False → se vectoriza solo el texto del fragmento, sin saber a qué sección pertenece.
    """
    fragmentos = fragmentar(leer_documento(ruta), "oraciones", max_palabras, solape)
    textos = [texto_a_vectorizar(f) if encabezado else f.texto for f in fragmentos]
    for f, v in zip(fragmentos, vectorizar(textos, tipo="documento")):
        f.vector = v
    almacen.preparar(len(fragmentos[0].vector))
    almacen.vaciar()
    almacen.insertar(fragmentos)
    return fragmentos


def ingestar_con_metadatos(almacen: Almacen, ruta: str | Path, metadatos: dict, max_palabras: int = 70, solape: int = 1,
                           vaciar: bool = True) -> list[Fragmento]:
    """Como rag.ingestar(), pero añade `metadatos` (p. ej. {"vigencia": "actual"}) a cada fragmento.

    Sirve para filtrar al buscar: `almacen.buscar(vector, k, filtro={"vigencia": "actual"})` ignora lo que no cumple.
    """
    fragmentos = fragmentar(leer_documento(ruta), "oraciones", max_palabras, solape)
    for f, v in zip(fragmentos, vectorizar([texto_a_vectorizar(f) for f in fragmentos], tipo="documento")):
        f.vector = v
        f.metadatos.update(metadatos)
    almacen.preparar(len(fragmentos[0].vector))
    if vaciar:
        almacen.vaciar()
    almacen.insertar(fragmentos)
    return fragmentos


class RecuperadorSmallToBig:
    """Busca con fragmentos pequeños y entrega el contexto grande al que pertenecen.

    modo="seccion" → entrega la sección completa (el "documento padre").
    modo="ventana" → entrega el fragmento encontrado más el anterior y el siguiente de su sección.
    `k` es cuántos PADRES se entregan; se buscan `k_hijos` hijos y se agrupan por padre.
    """

    def __init__(self, almacen: Almacen, ruta: str | Path, fragmentos: list[Fragmento], modo: str = "seccion",
                 k_hijos: int = 8):
        if modo not in ("seccion", "ventana"):
            raise ValueError("modo debe ser 'seccion' o 'ventana'")
        self.almacen, self.modo, self.k_hijos = almacen, modo, k_hijos
        self.nombre = f"small-to-big ({modo})"
        self.costo = 0.0
        self.secciones = dict(leer_documento(ruta).secciones)
        self.por_seccion: dict[str, list[Fragmento]] = defaultdict(list)
        for f in fragmentos:
            self.por_seccion[f.metadatos["seccion"]].append(f)

    def __call__(self, pregunta: str, k: int = 2) -> list[Resultado]:
        hijos = recuperar(self.almacen, pregunta, k=self.k_hijos)
        entregados: dict[str, Resultado] = {}
        for h in hijos:                                   # ya vienen de mayor a menor similitud
            if self.modo == "seccion":
                clave, texto = h.metadatos["seccion"], self.secciones[h.metadatos["seccion"]]
            else:
                vecinos = self.por_seccion[h.metadatos["seccion"]]
                n = next(i for i, f in enumerate(vecinos) if f.id == h.id)
                clave = h.id
                texto = " ".join(f.texto for f in vecinos[max(0, n - 1): n + 2])
            if clave not in entregados:                   # el primer hijo de cada padre fija su puntaje
                entregados[clave] = Resultado(id=f"padre:{clave}", texto=" ".join(texto.split()),
                                              metadatos=dict(h.metadatos), puntaje=h.puntaje)
            if len(entregados) == k:
                break
        return list(entregados.values())


# ───────────────────────── 2. Descomposición de preguntas (multi-salto) ─────────────────────────
#
# "Si mi factura venció y quiero pagar en efectivo un sábado, ¿hasta qué hora puedo ir?" necesita DOS
# temas (facturación y agencias). Un solo vector de la pregunta completa se parece un poco a cada uno y
# mucho a ninguno. Separarla en subpreguntas permite buscar cada tema por separado y juntar los resultados.


class Descomposicion(BaseModel):
    subpreguntas: list[str] = Field(description="Subpreguntas sencillas, cada una sobre un solo tema.")


def descomponer(pregunta: str, proveedor: str | None = None, maximo: int = 3,
                version: str = "v2") -> tuple[list[str], list[ResultadoLLM]]:
    """💰 Pide al LLM separar la pregunta. Si no devuelve JSON válido, usa la pregunta original.

    La versión v1 del prompt traía un ejemplo resuelto y un modelo pequeño lo copiaba en la salida (subpreguntas sobre
    paquetes de mensajería en una pregunta sobre facturas); la v2 no lleva ejemplo. Ver prompts/CHANGELOG.md.
    """
    prompt = renderizar("descomponer", version, pregunta=pregunta, maximo=maximo)
    llamadas: list[ResultadoLLM] = []
    try:
        d, llamadas = pedir_json_con_ejemplo(lambda texto: generar(texto, proveedor, max_tokens=250), prompt,
                                             Descomposicion, '{"subpreguntas": ["subpregunta 1", "subpregunta 2"]}')
        subs = [s.strip() for s in d.subpreguntas if s.strip()][:maximo]
    except ValueError:
        subs = []
    return subs or [pregunta], llamadas


def fusionar_rrf(listas: list[list[Resultado]], k: int, constante: int = 60) -> list[Resultado]:
    """Fusión por rango recíproco (RRF): cada resultado suma 1 ÷ (constante + posición) en cada lista donde aparece.

    Combina varias búsquedas sin comparar sus puntajes (que no son comparables entre consultas):
    solo importa en qué posición quedó cada fragmento. Un fragmento que sale en varias listas sube.
    """
    puntos: dict[str, float] = defaultdict(float)
    por_id: dict[str, Resultado] = {}
    for lista in listas:
        for pos, r in enumerate(lista, start=1):
            puntos[r.id] += 1 / (constante + pos)
            por_id.setdefault(r.id, r)
    orden = sorted(puntos, key=lambda i: puntos[i], reverse=True)[:k]
    return [por_id[i] for i in orden]


class RecuperadorDescompuesto:
    """Separa la pregunta en subpreguntas, busca cada una y fusiona los resultados con RRF."""

    def __init__(self, almacen: Almacen, proveedor: str | None = None, k_sub: int = 3, incluir_original: bool = True):
        self.almacen, self.proveedor, self.k_sub, self.incluir_original = almacen, proveedor, k_sub, incluir_original
        self.nombre = "descomposición"
        self.costo = 0.0
        self.subpreguntas: dict[str, list[str]] = {}      # qué subpreguntas se generaron (para mostrarlas)

    def __call__(self, pregunta: str, k: int = 4) -> list[Resultado]:
        subs, llamadas = descomponer(pregunta, self.proveedor)
        self.costo += sum(c.costo for c in llamadas)
        self.subpreguntas[pregunta] = subs
        consultas = ([pregunta] if self.incluir_original else []) + [s for s in subs if s != pregunta]
        return fusionar_rrf([recuperar(self.almacen, q, k=self.k_sub) for q in consultas], k)


# ───────────────────────── 3. HyDE: buscar con una respuesta hipotética ─────────────────────────
#
# Una PREGUNTA ("¿Hasta cuándo puedo pagar?") y su RESPUESTA ("La factura vence el día 15…") se parecen
# menos de lo que parece. HyDE (Hypothetical Document Embeddings) pide al LLM que escriba una respuesta
# hipotética, aunque invente los datos, y busca con el vector de ESE texto: se parece más a los fragmentos.
# El texto inventado nunca llega al usuario: solo sirve para buscar.


class RecuperadorHyDE:
    """💰 Busca con el vector de una respuesta hipotética (1 llamada corta al LLM por pregunta)."""

    def __init__(self, almacen: Almacen, proveedor: str | None = None):
        self.almacen, self.proveedor = almacen, proveedor
        self.nombre = "HyDE"
        self.costo = 0.0
        self.hipotesis: dict[str, str] = {}

    def __call__(self, pregunta: str, k: int = 4) -> list[Resultado]:
        r = generar(renderizar("hyde", "v1", pregunta=pregunta), self.proveedor, max_tokens=150)
        self.costo += r.costo
        self.hipotesis[pregunta] = r.texto.strip()
        vector = vectorizar([r.texto.strip()], tipo="documento")[0]     # se trata como un documento, no como consulta
        return self.almacen.buscar(vector, k=k)
