"""Estrategias de fragmentado (chunking) para el laboratorio de la Sesión 6.

La Sesión 5 trajo tres estrategias (`seccion`, `oraciones`, `fijo`, en rag.py). Aquí las completamos con dos más
y con un patrón de recuperación:

    fijo         ventanas de N palabras, SIN mirar la gramática. Rápida y ciega: corta ideas por la mitad.
    oraciones    junta oraciones hasta N palabras y repite las últimas (solape).
    seccion      un fragmento por sección (o por página en un PDF).
    recursivo    corta por el separador MÁS GRANDE que quepa (párrafo → línea → oración → palabra) y luego
                 junta trozos vecinos hasta N palabras. Es la idea del "RecursiveCharacterTextSplitter" de LangChain.
    semantico    mide qué tan parecidas son las oraciones vecinas (embeddings) y corta donde el tema CAMBIA.
    padre-hijo   (patrón) se BUSCA con fragmentos pequeños y precisos, pero al modelo se le ENTREGA el fragmento
                 grande (la sección) al que pertenecen. Precisión al buscar + contexto al responder.

Todas devuelven la misma estructura `Fragmento` (id, texto, metadatos) de la Sesión 4, con estos metadatos:
    fuente, seccion, n     (los de la Sesión 5)
    tipo                   md | html | docx | pdf
    publico                residencial | negocio | general  (se infiere del título de la sección; sirve para FILTRAR)

💰 COSTO: gratis. Solo `semantico` llama al modelo de embeddings (local por defecto) para cada oración.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from almacenes import Fragmento
from embeddings import vectorizar
from ingesta import DocumentoExt
from rag import _oraciones, fragmentar as _fragmentar_s05

ESTRATEGIAS = ("fijo", "oraciones", "seccion", "recursivo", "semantico")


def _palabras(texto: str) -> int:
    return len(texto.split())


def inferir_publico(titulo_seccion: str) -> str:
    """Etiqueta de audiencia a partir del título: el DOCX separa 'Parte A · Clientes residenciales' y 'Parte B · Clientes Negocio'."""
    t = titulo_seccion.lower()
    if "negocio" in t:
        return "negocio"
    if "residencial" in t:
        return "residencial"
    return "general"


def _con_metadatos(fragmentos: list[Fragmento], doc: DocumentoExt) -> list[Fragmento]:
    for f in fragmentos:
        f.metadatos["tipo"] = doc.tipo
        f.metadatos["publico"] = inferir_publico(f.metadatos["seccion"])
    return fragmentos


# ───────────────────────── recursivo ─────────────────────────

_SEPARADORES = ("\n\n", "\n", ". ", " ")


def _partir_recursivo(texto: str, max_palabras: int, nivel: int = 0) -> list[str]:
    """Parte `texto` con el separador más grande que haga falta hasta que cada trozo quepa en `max_palabras`."""
    texto = texto.strip()
    if not texto:
        return []
    if _palabras(texto) <= max_palabras or nivel >= len(_SEPARADORES):
        return [texto]
    sep = _SEPARADORES[nivel]
    trozos = [t for t in texto.split(sep) if t.strip()]
    if len(trozos) == 1:                                   # este separador no sirve: probar el siguiente
        return _partir_recursivo(texto, max_palabras, nivel + 1)
    salida: list[str] = []
    for t in trozos:
        t = t if sep == " " else (t + sep.strip() if sep.strip() and not t.endswith(sep.strip()) else t)
        salida.extend(_partir_recursivo(t, max_palabras, nivel + 1))
    return salida


def _fragmentar_recursivo(doc: DocumentoExt, max_palabras: int, solape: int) -> list[Fragmento]:
    trozos: list[tuple[str, str]] = []
    for titulo, texto in doc.secciones:
        piezas = _partir_recursivo(texto, max_palabras)
        actual: list[str] = []
        for p in piezas:                                    # juntar piezas vecinas hasta llenar el tamaño
            if actual and sum(_palabras(x) for x in actual) + _palabras(p) > max_palabras:
                trozos.append((titulo, " ".join(actual)))
                actual = actual[-solape:] if solape else []
            actual.append(p)
        if actual and (not trozos or " ".join(actual) != trozos[-1][1]):
            trozos.append((titulo, " ".join(actual)))
    return [Fragmento(id=f"{Path(doc.fuente).stem}#{n}", texto=" ".join(t.split()),
                      metadatos={"fuente": doc.fuente, "seccion": s, "n": n})
            for n, (s, t) in enumerate(trozos)]


# ───────────────────────── semántico ─────────────────────────


def _fragmentar_semantico(doc: DocumentoExt, max_palabras: int, percentil: float = 25.0) -> list[Fragmento]:
    """Corta donde baja la similitud entre oraciones consecutivas (cambio de tema). `max_palabras` es un tope duro."""
    trozos: list[tuple[str, str]] = []
    for titulo, texto in doc.secciones:
        oraciones = _oraciones(texto)
        if len(oraciones) < 3:
            trozos.append((titulo, " ".join(oraciones)))
            continue
        v = np.array(vectorizar(oraciones, tipo="documento"), dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        sim = (v[:-1] * v[1:]).sum(axis=1)                  # coseno entre la oración i y la i+1
        corte = np.percentile(sim, percentil)               # las uniones más "flojas" de esta sección
        actual = [oraciones[0]]
        for i in range(1, len(oraciones)):
            llena = _palabras(" ".join(actual)) + _palabras(oraciones[i]) > max_palabras
            if sim[i - 1] <= corte or llena:
                trozos.append((titulo, " ".join(actual)))
                actual = []
            actual.append(oraciones[i])
        trozos.append((titulo, " ".join(actual)))
    return [Fragmento(id=f"{Path(doc.fuente).stem}#{n}", texto=t,
                      metadatos={"fuente": doc.fuente, "seccion": s, "n": n})
            for n, (s, t) in enumerate(trozos)]


# ───────────────────────── punto de entrada ─────────────────────────


def fragmentar_doc(doc: DocumentoExt, estrategia: str = "oraciones", max_palabras: int = 70, solape: int = 1) -> list[Fragmento]:
    """Divide un documento (de cualquier formato) con la estrategia elegida y añade los metadatos de la sesión."""
    if estrategia not in ESTRATEGIAS:
        raise ValueError(f"estrategia debe ser una de: {', '.join(ESTRATEGIAS)}")
    if estrategia == "recursivo":
        frs = _fragmentar_recursivo(doc, max_palabras, solape)
    elif estrategia == "semantico":
        frs = _fragmentar_semantico(doc, max_palabras)
    else:
        frs = _fragmentar_s05(doc, estrategia=estrategia, max_palabras=max_palabras, solape=solape)
    return _con_metadatos(frs, doc)


def fragmentar_corpus(documentos: list[DocumentoExt], **opciones) -> list[Fragmento]:
    """Fragmenta varios documentos con la misma configuración."""
    return [f for d in documentos for f in fragmentar_doc(d, **opciones)]


# ───────────────────────── padre-hijo ─────────────────────────


def fragmentar_padre_hijo(doc: DocumentoExt, max_palabras_hijo: int = 35) -> list[Fragmento]:
    """Fragmentos HIJOS pequeños (para buscar) que recuerdan el texto de su PADRE (la sección entera).

    Cada hijo lleva en `metadatos` el `padre_id` y el `texto_padre`. Se busca con los hijos y, al armar el
    prompt, se entrega el padre (sin repetir padres). Ver `recuperacion.expandir_a_padre`.
    """
    hijos: list[Fragmento] = []
    for p, (titulo, texto) in enumerate(doc.secciones):
        padre_id = f"{Path(doc.fuente).stem}§{p}"
        pequenos = _fragmentar_s05(
            type(doc)(fuente=doc.fuente, secciones=[(titulo, texto)]),
            estrategia="oraciones", max_palabras=max_palabras_hijo, solape=0)
        for j, h in enumerate(pequenos):
            h.id = f"{padre_id}.{j}"
            h.metadatos.update({"padre_id": padre_id, "texto_padre": " ".join(texto.split()), "n": len(hijos)})
            hijos.append(h)
    return _con_metadatos(hijos, doc)


def resumen_fragmentos(fragmentos: list[Fragmento]) -> dict:
    """Cantidad y tamaño (en palabras) de los fragmentos: lo primero que debes mirar al probar una estrategia."""
    tam = [_palabras(f.texto) for f in fragmentos]
    return {"n": len(tam), "min": min(tam), "media": round(sum(tam) / len(tam), 1), "max": max(tam)}
