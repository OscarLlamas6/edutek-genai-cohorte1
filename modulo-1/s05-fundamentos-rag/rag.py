"""Un RAG simple: ingesta, recuperación y generación con citas.

RAG = Retrieval-Augmented Generation (generación aumentada con recuperación). En vez de
confiar en lo que el modelo "recuerda", le damos, junto con la pregunta, los fragmentos de
TUS documentos que más se parecen a ella, y le pedimos que responda SOLO con eso y citando.

    ┌──────────── INGESTA (una vez) ────────────┐   ┌─────── CONSULTA (cada pregunta) ───────┐
    documento → fragmentar → vectorizar → guardar   pregunta → vectorizar → buscar k fragmentos
                                        (almacén)                              ↓
                                                    prompt = instrucciones + fragmentos + pregunta
                                                                               ↓
                                                              LLM → respuesta con citas [1] [2]

Este módulo reutiliza los de las sesiones anteriores:
    embeddings.py   vectores (Sesión 4)
    almacenes.py    memoria, Neon o Pinecone (Sesión 4)
    cliente_llm.py  el LLM de tu proveedor (Sesión 2)
    plantillas.py   prompts versionados (Sesión 3)

💰 COSTO: cada pregunta = 1 vectorización (gratis con el modelo local) + 1 llamada al LLM.
   Con el modelo económico del proveedor, ≈ US$ 0.0005 por pregunta; con el modelo local, 0.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from almacenes import Almacen, Fragmento, Resultado
from cliente_llm import ResultadoLLM, hay_api_key, llamar_llm, llamar_local, local_disponible, modelo_economico, modelo_local
from embeddings import vector_consulta, vectorizar
from plantillas import renderizar

NO_SE = "No encuentro esa información en el manual."
PROVEEDORES_GENERACION = ("anthropic", "openai", "local")


# ───────────────────────── 1. Fragmentar (chunking) ─────────────────────────


@dataclass
class Documento:
    fuente: str          # nombre del archivo
    secciones: list[tuple[str, str]]   # (título de la sección, texto)


def leer_documento(ruta: str | Path) -> Documento:
    """Lee un .md y lo separa en secciones por los encabezados `## `. El `# ` inicial es el título del manual."""
    ruta = Path(ruta)
    texto = ruta.read_text(encoding="utf-8")
    secciones: list[tuple[str, str]] = []
    titulo, buffer = "Introducción", []
    for linea in texto.splitlines():
        if linea.startswith("## "):
            if "".join(buffer).strip():
                secciones.append((titulo, "\n".join(buffer).strip()))
            titulo, buffer = linea[3:].strip(), []
        elif linea.startswith("# "):
            continue
        else:
            buffer.append(linea)
    if "".join(buffer).strip():
        secciones.append((titulo, "\n".join(buffer).strip()))
    return Documento(fuente=ruta.name, secciones=secciones)


def _oraciones(texto: str) -> list[str]:
    """Separa en oraciones de forma sencilla (por punto, ? y !, y por saltos de párrafo)."""
    partes = re.split(r"(?<=[.!?])\s+|\n{2,}", texto.strip())
    return [p.strip() for p in partes if p.strip()]


def fragmentar(doc: Documento, estrategia: str = "oraciones", max_palabras: int = 70, solape: int = 1) -> list[Fragmento]:
    """Divide el documento en fragmentos (chunks) y les pone metadatos para poder CITAR.

    Estrategias:
      "seccion"   → un fragmento por sección. Simple; las secciones largas diluyen el significado.
      "oraciones" → junta oraciones hasta `max_palabras`, repitiendo las últimas `solape` oraciones
                    en el siguiente fragmento (el SOLAPE evita cortar una idea justo por la mitad).
      "fijo"      → ventanas de `max_palabras` palabras con solape de `solape * 10` palabras (ignora la gramática).

    Cada fragmento lleva su sección como metadato; al vectorizar se antepone para dar contexto.
    """
    if estrategia not in ("seccion", "oraciones", "fijo"):
        raise ValueError("estrategia debe ser 'seccion', 'oraciones' o 'fijo'")
    trozos: list[tuple[str, str]] = []   # (sección, texto)
    for titulo, texto in doc.secciones:
        if estrategia == "seccion":
            trozos.append((titulo, " ".join(texto.split())))
        elif estrategia == "fijo":
            palabras = texto.split()
            paso = max(1, max_palabras - solape * 10)
            for i in range(0, len(palabras), paso):
                trozos.append((titulo, " ".join(palabras[i : i + max_palabras])))
                if i + max_palabras >= len(palabras):
                    break
        else:
            actual: list[str] = []
            for o in _oraciones(texto):
                if actual and sum(len(x.split()) for x in actual) + len(o.split()) > max_palabras:
                    trozos.append((titulo, " ".join(actual)))
                    actual = actual[-solape:] if solape else []
                actual.append(o)
            if actual and (not trozos or " ".join(actual) != trozos[-1][1]):
                trozos.append((titulo, " ".join(actual)))
    return [
        Fragmento(id=f"{Path(doc.fuente).stem}#{n}", texto=texto,
                  metadatos={"fuente": doc.fuente, "seccion": seccion, "n": n})
        for n, (seccion, texto) in enumerate(trozos)
    ]


# ───────────────────────── 2. Ingesta ─────────────────────────


def texto_a_vectorizar(f: Fragmento) -> str:
    """Se vectoriza 'sección + texto': el título aporta contexto (un fragmento suelto puede ser ambiguo)."""
    return f"{f.metadatos['seccion']}. {f.texto}"


def ingestar(almacen: Almacen, ruta: str | Path, **opciones_fragmentado) -> list[Fragmento]:
    """Lee, fragmenta, vectoriza y guarda un documento. Reingestar actualiza (mismos ids), no duplica."""
    fragmentos = fragmentar(leer_documento(ruta), **opciones_fragmentado)
    vectores = vectorizar([texto_a_vectorizar(f) for f in fragmentos], tipo="documento")
    for f, v in zip(fragmentos, vectores):
        f.vector = v
    almacen.preparar(len(vectores[0]))
    almacen.vaciar()                  # evita fragmentos viejos si cambiaste la estrategia de fragmentado
    almacen.insertar(fragmentos)
    return fragmentos


# ───────────────────────── 3. Recuperación ─────────────────────────


def recuperar(almacen: Almacen, pregunta: str, k: int = 4) -> list[Resultado]:
    """Los k fragmentos más parecidos a la pregunta (ordenados de mayor a menor similitud)."""
    return almacen.buscar(vector_consulta(pregunta), k=k)


# ───────────────────────── 4. Generación con citas ─────────────────────────


def construir_prompt(pregunta: str, resultados: list[Resultado], version: str = "v1") -> str:
    """Prompt = plantilla versionada + fragmentos numerados + pregunta."""
    return renderizar("rag_respuesta", version, fragmentos=resultados, pregunta=pregunta)


def proveedor_generacion(preferido: str | None = None) -> str:
    p = (preferido or os.getenv("RAG_PROVIDER") or os.getenv("LLM_PROVIDER", "anthropic")).strip().lower()
    if p not in PROVEEDORES_GENERACION:
        raise ValueError(f"Proveedor '{p}' no válido. Usa: {', '.join(PROVEEDORES_GENERACION)}.")
    return p


def proveedores_disponibles() -> dict[str, bool]:
    """Qué proveedores de generación puedes usar ahora: llave configurada (Anthropic, OpenAI) o servidor local activo."""
    return {
        "anthropic": hay_api_key("anthropic"),
        "openai": hay_api_key("openai"),
        "local": local_disponible(modelo_local())[0],
    }


def generar(prompt: str, proveedor: str | None = None, max_tokens: int = 500) -> ResultadoLLM:
    """💰 Llama al LLM (de pago con Anthropic/OpenAI; gratis con 'local')."""
    p = proveedor_generacion(proveedor)
    if p == "local":
        return llamar_local(prompt, modelo=modelo_local(), max_tokens=max_tokens, temperature=0)
    return llamar_llm(prompt, proveedor=p, modelo=modelo_economico(p), max_tokens=max_tokens,
                      streaming=False, esfuerzo="minimo")


def citas_en(texto: str) -> set[int]:
    """Números citados en el texto: [1], [2] o [1, 3]."""
    numeros: set[int] = set()
    for grupo in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", texto):
        numeros.update(int(n) for n in re.split(r"\s*,\s*", grupo))
    return numeros


@dataclass
class Respuesta:
    pregunta: str
    texto: str
    fragmentos: list[Resultado]
    citas: set[int] = field(default_factory=set)            # citas que el modelo escribió
    citas_invalidas: set[int] = field(default_factory=set)  # citas a fragmentos que NO existen
    sin_respuesta: bool = False
    llamada: ResultadoLLM | None = None                     # tokens y costo (None si no se llamó al LLM)

    @property
    def sin_citas(self) -> bool:
        """El modelo respondió con datos pero no citó ningún fragmento (no verificable)."""
        return not self.sin_respuesta and not self.citas

    @property
    def fuentes(self) -> list[Resultado]:
        """Fragmentos efectivamente citados por el modelo."""
        return [self.fragmentos[n - 1] for n in sorted(self.citas) if 1 <= n <= len(self.fragmentos)]


def responder(almacen: Almacen, pregunta: str, k: int = 4, version: str = "v1",
              proveedor: str | None = None, umbral: float | None = None) -> Respuesta:
    """Flujo completo: recuperar → (¿vale la pena?) → generar → verificar citas.

    `umbral`: si el mejor fragmento tiene similitud menor, ni siquiera se llama al LLM y se responde
    "no sé". Ahorra dinero y evita inventar, pero un umbral mal calibrado descarta buenas respuestas:
    calíbralo mirando los puntajes de tus propias preguntas.
    """
    fragmentos = recuperar(almacen, pregunta, k=k)
    if not fragmentos or (umbral is not None and fragmentos[0].puntaje < umbral):
        return Respuesta(pregunta, NO_SE, fragmentos, sin_respuesta=True)

    llamada = generar(construir_prompt(pregunta, fragmentos, version), proveedor)
    texto = llamada.texto.strip()
    citas = citas_en(texto)
    invalidas = {n for n in citas if not 1 <= n <= len(fragmentos)}
    return Respuesta(pregunta, texto, fragmentos, citas=citas, citas_invalidas=invalidas,
                     sin_respuesta=texto.startswith("No encuentro"), llamada=llamada)


# ───────────────────────── 5. Evaluación mínima ─────────────────────────


def evaluar_recuperacion(almacen: Almacen, preguntas: list[dict], k: int = 4) -> dict:
    """hit@k de la RECUPERACIÓN (sin LLM, sin costo): ¿la sección correcta está entre los k fragmentos?

    Evalúa la recuperación por separado de la generación: si el fragmento correcto no llega al prompt,
    ningún prompt puede arreglarlo.
    """
    con_respuesta = [p for p in preguntas if not p.get("fuera_de_alcance")]
    h1 = hk = 0
    fallos = []
    for p in con_respuesta:
        secciones = [r.metadatos["seccion"] for r in recuperar(almacen, p["pregunta"], k=k)]
        h1 += secciones[0] == p["seccion"]
        hk += p["seccion"] in secciones
        if p["seccion"] not in secciones:
            fallos.append((p["id"], p["pregunta"], p["seccion"], secciones[:2]))
    return {"total": len(con_respuesta), "hit1": h1, f"hit{k}": hk, "k": k, "fallos": fallos}
