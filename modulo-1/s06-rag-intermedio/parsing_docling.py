"""Parsing moderno con Docling: layout, tablas, OCR y chunking híbrido (OPCIONAL).

`ingesta.py` trae un lector a mano por formato (Markdown, HTML, DOCX, PDF). Docling los reemplaza por UN solo
convertidor que además entiende el DISEÑO de la página (layout): reconoce títulos, párrafos, listas, tablas y figuras,
reconstruye el orden de lectura y, si la página es una imagen, le aplica OCR.

    archivo (md, html, docx, pdf, imagen…) ─► DocumentConverter ─► DoclingDocument ─► HybridChunker ─► fragmentos
                                              (layout + tablas       (jerarquía de        (texto + encabezados
                                               + OCR si hace falta)   títulos + tokens)     + páginas)

⚠️ OPCIONAL. Docling es pesado (PyTorch y varios cientos de MB de modelos que se descargan la primera vez), así que
NO forma parte del entorno base de la sesión. Se instala con:

    uv sync --group docling

Este módulo se puede importar sin Docling: solo falla al USAR sus funciones, con un mensaje claro
(ver `docling_disponible`). Así el resto del notebook sigue funcionando.

⚠️ NO CONFUNDIR: el "chunking híbrido" de Docling (jerarquía del documento + límite de tokens) NO es la "búsqueda
híbrida" de la Parte 6 (BM25 + vectores). Comparten la palabra "híbrido" y nada más.

💰 COSTO: ninguno. Todo corre en tu equipo (CPU). Lo que cuesta es TIEMPO: el primer PDF carga los modelos (≈ 15 a 20 s).
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path

from almacenes import Fragmento
from fragmentado import inferir_publico

# PyTorch avisa de un detalle interno de sus convoluciones al correr el OCR; no afecta al resultado.
warnings.filterwarnings("ignore", message=".*padding='same'.*", category=UserWarning)

COMANDO_INSTALAR = "uv sync --group docling"

# Cada modelo de embeddings se entrena con SU tokenizador. Para que "128 tokens" signifique lo mismo para el
# fragmentador y para el modelo, el HybridChunker debe contar con el tokenizador del modelo de embeddings.
# (Se busca por fragmento del nombre del modelo, como en `embeddings.PREFIJOS`.)
TOKENIZADORES = {
    "nomic-embed-text-v2": "nomic-ai/nomic-embed-text-v2-moe",
    "nomic-embed-text-v1": "nomic-ai/nomic-embed-text-v1.5",
    "mxbai": "mixedbread-ai/mxbai-embed-large-v1",
}
TOKENIZADOR_POR_DEFECTO = "sentence-transformers/all-MiniLM-L6-v2"


def docling_disponible() -> tuple[bool, str]:
    """(¿está instalado Docling?, mensaje). El mensaje trae el comando exacto si falta."""
    try:
        from importlib.metadata import version

        import docling  # noqa: F401

        return True, f"Docling {version('docling')} disponible"
    except ImportError:
        return False, f"Docling no está instalado. Instálalo con: {COMANDO_INSTALAR}  (si estás en un notebook, reinicia el kernel después)"


def tokenizador_para(modelo_embeddings: str | None = None) -> str:
    """Identificador (Hugging Face) del tokenizador que corresponde al modelo de embeddings en uso."""
    if modelo_embeddings is None:
        from embeddings import modelo_embeddings as _modelo

        modelo_embeddings = _modelo()
    nombre = modelo_embeddings.lower()
    for fragmento, tokenizador in TOKENIZADORES.items():
        if fragmento in nombre:
            return tokenizador
    return TOKENIZADOR_POR_DEFECTO


@lru_cache(maxsize=2)
def convertidor(ocr: bool = True):
    """Un `DocumentConverter` reutilizable (cargar los modelos es lo lento: se hace una vez por proceso)."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opciones = PdfPipelineOptions()
    opciones.do_ocr = ocr
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opciones)})


@lru_cache(maxsize=32)
def _convertir(ruta: str, ocr: bool, modificado: float):
    return convertidor(ocr).convert(ruta).document


def leer_con_docling(ruta: str | Path, ocr: bool = True):
    """Convierte cualquier archivo soportado en un `DoclingDocument`. Con `ocr=False` ignora las páginas-imagen.

    Recuerda los archivos ya convertidos (mientras no cambien): el laboratorio prueba varios tamaños con los
    mismos documentos y convertir un PDF es lo lento.
    """
    ruta = Path(ruta)
    return _convertir(str(ruta.resolve()), ocr, ruta.stat().st_mtime)


def leer_pdf_adaptativo(ruta: str | Path, minimo_caracteres: int = 20):
    """Estrategia práctica: primero SIN OCR (rápido) y, si el resultado casi no tiene texto, reintentar CON OCR.

    Cubre dos casos: el escaneo puro (no hay capa de texto) y los PDF de diseño tipo presentación, cuyas páginas
    Docling clasifica como imagen. Devuelve `(documento, uso_ocr)`: guarda `uso_ocr` en tus metadatos, porque un
    texto sacado con OCR hay que revisarlo con más cuidado.
    """
    documento = leer_con_docling(ruta, ocr=False)
    if len(documento.export_to_markdown().strip()) >= minimo_caracteres:
        return documento, False
    return leer_con_docling(ruta, ocr=True), True


def precargar(carpeta: str | Path = "datos") -> None:
    """Descarga y carga TODOS los modelos que usa el notebook (layout, tablas, OCR y tokenizador). Hazlo ANTES de clase.

        uv run python -c "from parsing_docling import precargar; precargar()"

    La primera vez baja varios cientos de MB de modelos desde Hugging Face; las siguientes usa el caché del equipo.
    """
    import time

    ok, mensaje = docling_disponible()
    print(("✅ " if ok else "❌ ") + mensaje)
    if not ok:
        return
    carpeta = Path(carpeta)
    for nombre, ocr in (("boletin_mantenimiento.pdf", False), ("acta_escaneada.pdf", True)):
        t0 = time.perf_counter()
        leer_con_docling(carpeta / nombre, ocr=ocr)
        print(f"   {nombre:<28} {'con OCR' if ocr else 'sin OCR':<8} listo en {time.perf_counter() - t0:.0f} s")
    t0 = time.perf_counter()
    crear_chunker()
    print(f"   tokenizador {tokenizador_para()}  listo en {time.perf_counter() - t0:.0f} s")
    print("Listo: los modelos quedaron en el caché de tu equipo.")


def crear_chunker(max_tokens: int = 128, tokenizador: str | None = None, tablas_markdown: bool = True):
    """`HybridChunker` con el tokenizador del modelo de embeddings.

    `tablas_markdown=True` escribe cada tabla como tabla Markdown (se conservan filas y columnas); por defecto,
    Docling escribe cada celda como una frase "columna = valor".
    """
    from docling.chunking import HybridChunker

    opciones = {"tokenizer": tokenizador or tokenizador_para(), "max_tokens": max_tokens}
    if tablas_markdown:
        from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer, ChunkingSerializerProvider
        from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

        class _ProveedorMarkdown(ChunkingSerializerProvider):
            def get_serializer(self, doc):
                return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())

        opciones["serializer_provider"] = _ProveedorMarkdown()
    return HybridChunker(**opciones)


def paginas_de(fragmento_docling) -> list[int]:
    """Páginas donde está el fragmento (vacío para formatos sin páginas: md, html, docx)."""
    return sorted({p.page_no for it in fragmento_docling.meta.doc_items for p in it.prov})


def _tiene_tabla(fragmento_docling) -> bool:
    return any(str(getattr(it, "label", "")).endswith("table") for it in fragmento_docling.meta.doc_items)


def fragmentar_hibrido(ruta: str | Path, max_tokens: int = 128, tokenizador: str | None = None,
                       tablas_markdown: bool = True, ocr: bool = True, chunker=None) -> list[Fragmento]:
    """Lee UN archivo con Docling y lo parte con `HybridChunker`, devolviendo los mismos `Fragmento` de la sesión.

    Metadatos, iguales a los del resto de estrategias (para que el laboratorio y los filtros funcionen igual):
        fuente    nombre del archivo
        seccion   los encabezados que contienen al fragmento, unidos con " › " (como el lector DOCX de `ingesta.py`)
        n         número del fragmento dentro del archivo
        tipo      md | html | docx | pdf
        publico   residencial | negocio | general (se infiere de la sección, como en `fragmentado.py`)
    y además, propios de Docling: `paginas` (lista), `tokens` (con encabezados) y `encabezados` (lista).

    `Fragmento.texto` es el texto del fragmento SIN los encabezados; al vectorizar, `rag.texto_a_vectorizar` antepone
    la sección ("sección. texto"), que equivale a `chunker.contextualize(fragmento)`.
    """
    ruta = Path(ruta)
    doc = leer_con_docling(ruta, ocr=ocr)
    chunker = chunker or crear_chunker(max_tokens, tokenizador, tablas_markdown)
    trozos = list(chunker.chunk(dl_doc=doc))

    # Casi todos los formatos traen un título del documento (el "# " inicial) que encabeza TODOS los fragmentos:
    # no distingue nada y se descarta (si no, "seccion" y "publico" no coincidirían con los de los otros lectores).
    jerarquias = [t.meta.headings for t in trozos if t.meta.headings]
    titulo_comun = None
    if jerarquias and len({h[0] for h in jerarquias}) == 1 and any(len(h) > 1 for h in jerarquias):
        titulo_comun = jerarquias[0][0]

    fragmentos = []
    for n, t in enumerate(trozos):
        encabezados = list(t.meta.headings or [])
        if titulo_comun and encabezados[:1] == [titulo_comun]:
            encabezados = encabezados[1:]          # el fragmento inicial queda sin encabezados: "Introducción"
        seccion = " › ".join(encabezados) if encabezados else "Introducción"
        fragmentos.append(Fragmento(
            id=f"{ruta.stem}#{n}",
            texto=t.text.strip() if _tiene_tabla(t) else " ".join(t.text.split()),   # las tablas conservan sus filas
            metadatos={"fuente": ruta.name, "seccion": seccion, "n": n, "tipo": {"htm": "html"}.get(ruta.suffix.lower().lstrip("."), ruta.suffix.lower().lstrip(".")),
                       "publico": inferir_publico(seccion), "paginas": paginas_de(t),
                       "tokens": chunker.tokenizer.count_tokens(chunker.contextualize(t)),
                       "encabezados": encabezados},
        ))
    return fragmentos


def fragmentar_carpeta_hibrido(carpeta: str | Path, max_tokens: int = 128, excluir: tuple[str, ...] = (),
                               **opciones) -> list[Fragmento]:
    """Lo mismo para todos los documentos de una carpeta (md, html, docx, pdf): un solo parser para todos."""
    from ingesta import EXTENSIONES

    chunker = crear_chunker(max_tokens, opciones.pop("tokenizador", None), opciones.pop("tablas_markdown", True))
    salida: list[Fragmento] = []
    for ruta in sorted(Path(carpeta).iterdir()):
        if ruta.suffix.lower() in EXTENSIONES and ruta.name not in excluir:
            salida.extend(fragmentar_hibrido(ruta, max_tokens, chunker=chunker, **opciones))
    return salida


if os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") is None:
    # Las barras de descarga de modelos ensucian la salida del notebook; el mensaje de error sigue viéndose.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
