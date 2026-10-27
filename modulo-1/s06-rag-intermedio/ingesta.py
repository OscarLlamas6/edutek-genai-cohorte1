"""Ingesta multiformato: leer Markdown, HTML, DOCX y PDF y convertirlos en SECCIONES de texto limpio.

Un RAG real casi nunca recibe un .md bien ordenado. Recibe páginas web, contratos en Word, boletines en PDF.
El primer paso (y el que más se subestima) es EXTRAER el texto y conservar su estructura (títulos, secciones,
páginas): si aquí entra basura, todo lo que sigue (fragmentar, vectorizar, recuperar) trabaja con basura.

    ┌─────────┐   ┌────────────┐   ┌─────────────────────┐
    │ archivo │ → │ extraer    │ → │ Documento           │ → fragmentar → vectorizar → guardar
    │ .md .html│   │ texto +    │   │   fuente, tipo,     │
    │ .docx .pdf│  │ estructura │   │   secciones[(título, texto)]
    └─────────┘   └────────────┘   └─────────────────────┘

Lo que hace cada lector:
    .md    separa por encabezados `## ` (el lector de la Sesión 5).
    .html  quita lo que NO es contenido (<script>, <style>, menús, pie de página) y separa por <h2>.
    .docx  respeta los estilos de título (Heading 1/2) y convierte las TABLAS en líneas de texto.
    .pdf   extrae una sección por PÁGINA. Si una página no tiene texto (un escaneo), lo avisa: hace falta OCR.

💰 COSTO: ninguno. Todo corre en tu equipo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rag import Documento, leer_documento as leer_markdown

EXTENSIONES = (".md", ".html", ".htm", ".docx", ".pdf")


@dataclass
class DocumentoExt(Documento):
    """Un Documento de la Sesión 5 más el tipo de archivo y los avisos del lector (p. ej. 'página sin texto')."""

    tipo: str = "md"
    avisos: list[str] = field(default_factory=list)


def _limpiar(texto: str) -> str:
    """Colapsa espacios y saltos de línea repetidos (sin tocar los saltos entre párrafos)."""
    texto = re.sub(r"[ \t ]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# ───────────────────────── Markdown ─────────────────────────


def _desde_markdown(ruta: Path) -> DocumentoExt:
    d = leer_markdown(ruta)
    return DocumentoExt(fuente=d.fuente, secciones=d.secciones, tipo="md")


# ───────────────────────── HTML ─────────────────────────


def _desde_html(ruta: Path) -> DocumentoExt:
    """Lee una página web: conserva el contenido principal y descarta menús, scripts y pie de página."""
    from bs4 import BeautifulSoup

    sopa = BeautifulSoup(ruta.read_text(encoding="utf-8"), "html.parser")
    # 1) Quitar el "ruido": no es contenido, y si se queda se vectoriza como si lo fuera.
    for etiqueta in sopa(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        etiqueta.decompose()
    cuerpo = sopa.find("main") or sopa.body or sopa

    # 2) Recorrer los elementos EN ORDEN y abrir una sección nueva en cada <h2>.
    secciones: list[tuple[str, str]] = []
    titulo, buffer = "Introducción", []

    def cerrar():
        if "".join(buffer).strip():
            secciones.append((titulo, "\n".join(buffer).strip()))

    for el in cuerpo.find_all(["h1", "h2", "h3", "p", "li"]):
        texto = " ".join(el.get_text().split())
        if not texto:
            continue
        if el.name == "h1":
            continue                                   # el título de la página no es una sección
        if el.name in ("h2", "h3"):
            cerrar()
            titulo, buffer = texto, []
        elif el.name == "li":
            buffer.append("- " + texto)
        else:
            buffer.append(texto)
    cerrar()
    return DocumentoExt(fuente=ruta.name, secciones=secciones, tipo="html")


# ───────────────────────── DOCX ─────────────────────────


def _desde_docx(ruta: Path) -> DocumentoExt:
    """Lee un Word. Los títulos 'Heading 1' y 'Heading 2' forman el nombre de la sección ('Parte A › Compensaciones')."""
    from docx import Document as Docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = Docx(str(ruta))
    h1 = ""
    titulo, buffer = "Introducción", []
    secciones: list[tuple[str, str]] = []

    def cerrar():
        if "".join(buffer).strip():
            secciones.append((titulo, "\n".join(buffer).strip()))

    for hijo in d.element.body.iterchildren():            # párrafos y tablas, en el orden del documento
        if hijo.tag.endswith("}p"):
            p = Paragraph(hijo, d)
            texto = p.text.strip()
            estilo = p.style.name if p.style is not None else ""
            if not texto:
                continue
            if estilo == "Title":
                continue
            if estilo == "Heading 1":
                cerrar()
                h1, titulo, buffer = texto, texto, []
            elif estilo == "Heading 2":
                cerrar()
                titulo, buffer = (f"{h1} › {texto}" if h1 else texto), []
            else:
                buffer.append(texto)
        elif hijo.tag.endswith("}tbl"):
            # Una tabla no es texto corrido: la convertimos en una línea por fila ("celda | celda | celda").
            for fila in Table(hijo, d).rows:
                buffer.append(" | ".join(c.text.strip() for c in fila.cells))
    cerrar()
    return DocumentoExt(fuente=ruta.name, secciones=secciones, tipo="docx")


# ───────────────────────── PDF ─────────────────────────


def _desde_pdf(ruta: Path) -> DocumentoExt:
    """Lee un PDF con capa de texto: una sección por página. Una página sin texto genera un aviso (hace falta OCR)."""
    from pypdf import PdfReader

    lector = PdfReader(str(ruta))
    secciones: list[tuple[str, str]] = []
    avisos: list[str] = []
    for n, pagina in enumerate(lector.pages, start=1):
        texto = _limpiar(pagina.extract_text() or "")
        if not texto:
            avisos.append(f"Página {n} sin texto extraíble: parece un escaneo (imagen). Necesita OCR.")
            continue
        # El PDF guarda líneas cortadas por el ancho de la página: unimos las líneas de un mismo párrafo.
        texto = re.sub(r"(?<![.:!?])\n(?!\n)", " ", texto)
        secciones.append((f"Página {n}", texto))
    return DocumentoExt(fuente=ruta.name, secciones=secciones, tipo="pdf", avisos=avisos)


# ───────────────────────── Punto de entrada ─────────────────────────


def leer_documento_multi(ruta: str | Path) -> DocumentoExt:
    """Lee cualquier formato soportado y devuelve un Documento con sus secciones."""
    ruta = Path(ruta)
    extension = ruta.suffix.lower()
    lectores = {".md": _desde_markdown, ".html": _desde_html, ".htm": _desde_html,
                ".docx": _desde_docx, ".pdf": _desde_pdf}
    if extension not in lectores:
        raise ValueError(f"Formato no soportado: {extension}. Usa: {', '.join(EXTENSIONES)}")
    return lectores[extension](ruta)


def leer_carpeta(carpeta: str | Path, excluir: tuple[str, ...] = ()) -> list[DocumentoExt]:
    """Lee todos los documentos soportados de una carpeta (ordenados por nombre)."""
    documentos = []
    for ruta in sorted(Path(carpeta).iterdir()):
        if ruta.suffix.lower() in EXTENSIONES and ruta.name not in excluir:
            documentos.append(leer_documento_multi(ruta))
    return documentos
