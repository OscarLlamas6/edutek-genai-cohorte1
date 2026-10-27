"""Evaluación de un RAG: métricas de recuperación, comprobaciones de generación y un juez LLM.

Evaluar un RAG es como monitorear un servicio: defines qué medir (SLI), cuál es el mínimo aceptable
(SLO) y lo vuelves a medir cada vez que cambias algo (prueba de regresión). Un RAG tiene DOS capas
que se miden por separado, porque fallan por razones distintas:

    pregunta ─► RECUPERACIÓN ─► fragmentos ─► GENERACIÓN ─► respuesta
                 Recall@K                      dato clave, citas válidas,
                 Precision@K                   "no sé" bien dicho, cifras con respaldo,
                 MRR                           juez LLM (fidelidad, corrección)

Si la recuperación no trae el fragmento correcto, ningún prompt lo arregla; si lo trae y el modelo
responde mal, el problema está en la generación. Medirlas juntas esconde cuál de las dos falló.

Este módulo reutiliza los de las sesiones anteriores (rag.py, cliente_llm.py, plantillas.py…) y NO
modifica el RAG: lo trata como una caja que se evalúa.

💰 COSTO
    - Las métricas de recuperación NO llaman al LLM: son gratis (solo vectorizan las preguntas).
    - evaluar_rag(con_generacion=True) hace 1 llamada al LLM por pregunta (como responder()).
    - El juez LLM hace 1 llamada por criterio y por respuesta (más reintentos si el JSON viene mal).
    Con el modelo local (LM Studio) todo cuesta 0; con un proveedor, fracciones de centavo.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from almacenes import Almacen, Resultado
from cliente_llm import ResultadoLLM, extraer_json, llamar_llm, llamar_local, modelo_economico, modelo_local
from plantillas import renderizar
from rag import NO_SE, Respuesta, citas_en, construir_prompt, generar, leer_documento, proveedor_generacion, recuperar

# ───────────────────────── 1. El conjunto de oro (golden set) ─────────────────────────

TIPOS = ("directa", "multi_seccion", "multi_salto", "ambigua", "fuera_parecida", "fuera_ajena")


def cargar_golden(ruta: str | Path = "datos/golden_set.json", solo_rapido: bool = False) -> list[dict]:
    """Lee el conjunto de preguntas de evaluación (el 'golden set').

    Cada pregunta trae: `tipo`, `pregunta`, `respuesta_esperada`, `secciones` (las secciones del manual
    que SÍ contienen la respuesta; vacío si es fuera de alcance), `claves` (datos que deben aparecer) y
    `rapido` (si pertenece al subconjunto de 10 preguntas para evaluar en pocos segundos).
    """
    golden = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return [p for p in golden if p.get("rapido")] if solo_rapido else golden


def con_respuesta(golden: list[dict]) -> list[dict]:
    return [p for p in golden if not p.get("fuera_de_alcance")]


# ───────────────────────── 2. Métricas de RECUPERACIÓN (sin LLM, sin costo) ─────────────────────────
#
# Qué es "relevante": un fragmento es relevante si pertenece a una de las secciones que el golden set
# marca como correctas para esa pregunta. Se etiqueta por SECCIÓN (no por fragmento) para que la
# misma etiqueta sirva aunque cambies el tamaño de los fragmentos.


def es_relevante(r: Resultado, pregunta: dict) -> bool:
    return r.metadatos.get("seccion") in pregunta["secciones"]


def metricas_recuperacion(resultados: list[Resultado], pregunta: dict, k: int | None = None) -> dict | None:
    """Recall@K, Precision@K, MRR y hit@K de UNA pregunta, mirando solo los k primeros resultados.

    - Recall@K    = secciones relevantes que aparecen en el top-k ÷ secciones relevantes que existen.
                    "¿Traje todo lo que necesitaba?" (con 1 sola sección relevante equivale a hit@K).
    - Precision@K = fragmentos relevantes del top-k ÷ k.
                    "¿Cuánto de lo que traje sirve?" El resto es ruido que se paga en tokens.
    - MRR         = 1 ÷ posición del primer fragmento relevante (0 si no aparece).
                    "¿Qué tan arriba quedó el primero útil?"
    Devuelve None si la pregunta no tiene respuesta en el manual (no hay nada que recuperar).
    """
    esperadas = set(pregunta["secciones"])
    if not esperadas:
        return None
    top = resultados[:k] if k else resultados
    marcas = [es_relevante(r, pregunta) for r in top]
    cubiertas = {r.metadatos["seccion"] for r, m in zip(top, marcas) if m}
    primero = next((i for i, m in enumerate(marcas, start=1) if m), None)
    return {
        "recall": len(cubiertas) / len(esperadas),
        "precision": sum(marcas) / len(top) if top else 0.0,
        "mrr": 1 / primero if primero else 0.0,
        "hit": primero is not None,
        "primer_relevante": primero,
    }


def promedio(valores: list[float]) -> float:
    return sum(valores) / len(valores) if valores else 0.0


def evaluar_recuperacion_en_k(almacen: Almacen, golden: list[dict], ks: tuple[int, ...] = (1, 2, 3, 4, 6)) -> dict[int, dict]:
    """Promedios de las métricas para varios valores de k, buscando UNA sola vez con el k mayor.

    Solo cuenta las preguntas con respuesta en el manual. No llama al LLM: es gratis.
    """
    kmax = max(ks)
    buscadas = {p["id"]: recuperar(almacen, p["pregunta"], k=kmax) for p in con_respuesta(golden)}
    tabla = {}
    for k in ks:
        filas = [metricas_recuperacion(buscadas[p["id"]], p, k) for p in con_respuesta(golden)]
        tabla[k] = {
            "recall": promedio([f["recall"] for f in filas]),
            "precision": promedio([f["precision"] for f in filas]),
            "mrr": promedio([f["mrr"] for f in filas]),
            "hit": promedio([f["hit"] for f in filas]),
            "n": len(filas),
        }
    return tabla


def puntajes_mejor_fragmento(almacen: Almacen, golden: list[dict]) -> dict[str, list[tuple[str, float]]]:
    """Similitud del MEJOR fragmento de cada pregunta, separada en 'con respuesta' y 'fuera de alcance'.

    Sirve para calibrar un umbral de abstención: si las dos listas se solapan, ningún umbral las separa.
    """
    salida: dict[str, list[tuple[str, float]]] = {"con_respuesta": [], "fuera_de_alcance": []}
    for p in golden:
        mejor = recuperar(almacen, p["pregunta"], k=1)[0].puntaje
        salida["fuera_de_alcance" if p.get("fuera_de_alcance") else "con_respuesta"].append((p["id"], mejor))
    return salida


# ───────────────────────── 3. Comprobaciones de GENERACIÓN (código, sin juez) ─────────────────────────
#
# Son baratas, deterministas y repetibles: lo primero que debes automatizar. No sustituyen a una
# persona ni a un juez LLM, pero atrapan regresiones al instante.


def normalizar(texto: str) -> str:
    """Minúsculas y sin acentos, para comparar 'Instalación' con 'instalacion'."""
    sin_acentos = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in sin_acentos if unicodedata.category(c) != "Mn")


def contiene_claves(texto: str, claves: list[str]) -> bool:
    """¿Aparecen todos los datos clave? Cada clave puede tener alternativas separadas por '|'."""
    t = normalizar(texto)
    return all(any(normalizar(alt.strip()) in t for alt in clave.split("|")) for clave in claves)


_CITA = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")


def numeros_en(texto: str) -> set[str]:
    """Cifras que aparecen en un texto (ignora las citas como [1]; '1,50' y '1.50' son la misma)."""
    return {n.replace(",", ".") for n in _NUMERO.findall(_CITA.sub(" ", texto))}


def numeros_sin_respaldo(respuesta: str, fragmentos: list[Resultado], pregunta: str = "") -> list[str]:
    """Cifras de la respuesta que NO están en los fragmentos entregados ni en la pregunta.

    Es un 'checksum' de la alucinación: si el modelo escribe 35 y en el contexto no hay un 35, lo inventó.
    Limitaciones: no detecta una cifra correcta pero puesta en el concepto equivocado (el 15 del
    vencimiento usado como día de suspensión), ni datos inventados que no son números.
    """
    contexto = set()
    for f in fragmentos:
        # El título de la sección ("4. Equipos y garantía") también se muestra al modelo: sus números son legítimos.
        contexto |= numeros_en(f.texto) | numeros_en(f.metadatos.get("seccion", ""))
    return sorted((numeros_en(respuesta) - contexto - numeros_en(pregunta)), key=float)


def precision_de_citas(citas: set[int], fragmentos: list[Resultado], pregunta: dict) -> float | None:
    """De los fragmentos que el modelo CITÓ, ¿qué fracción es relevante? None si no citó nada válido."""
    validos = [fragmentos[n - 1] for n in sorted(citas) if 1 <= n <= len(fragmentos)]
    if not validos:
        return None
    return sum(es_relevante(f, pregunta) for f in validos) / len(validos)


# Diagnóstico: en cuál capa falló cada pregunta. Es la columna más útil de un informe.
CAPA_DEL_DIAGNOSTICO = {
    "ok": "—",
    "abstencion_por_recuperacion": "recuperación",
    "recuperacion_incompleta": "recuperación",
    "abstencion_indebida": "generación",
    "dato_incorrecto": "generación",
    "cifra_sin_respaldo": "generación (alucinación)",
    "alucinacion": "generación (alucinación)",
    "sin_citas_validas": "verificabilidad",
}


def diagnosticar(p: dict, *, se_abstuvo: bool, recall: float | None, dato_ok: bool | None,
                 sin_respaldo: list[str], citas_validas: bool | None) -> str:
    """Asigna a la pregunta una de las categorías de CAPA_DEL_DIAGNOSTICO (solo la primera que aplica)."""
    if p.get("fuera_de_alcance"):
        return "ok" if se_abstuvo else "alucinacion"          # debía decir "no sé" y contestó
    if se_abstuvo:
        # No contestó. Si ni siquiera llegó la sección completa, la culpa es de la recuperación.
        return "abstencion_por_recuperacion" if (recall or 0) < 1 else "abstencion_indebida"
    if not dato_ok:
        return "recuperacion_incompleta" if (recall or 0) < 1 else "dato_incorrecto"
    if sin_respaldo:
        return "cifra_sin_respaldo"
    if not citas_validas:
        return "sin_citas_validas"
    return "ok"


# ───────────────────────── 4. El juez LLM (LLM-as-judge) ─────────────────────────
#
# Un juez es OTRO modelo al que le das una rúbrica. Escala a miles de respuestas, pero es un
# instrumento que hay que CALIBRAR contra etiquetas humanas: puede ser indulgente, preferir respuestas
# largas o favorecer a modelos parecidos a él. Un juez sin calibrar es un medidor sin certificar.


class VeredictoFidelidad(BaseModel):
    """¿Todo lo que afirma la respuesta está respaldado por el contexto? (fidelidad = groundedness)"""

    afirmaciones_sin_respaldo: list[str] = Field(
        description="Afirmaciones de la respuesta que NO están en el contexto. Lista vacía si todas están respaldadas."
    )
    razon: str = Field(description="Una frase que explica el veredicto.")
    veredicto: Literal["fiel", "no_fiel"]

    @property
    def fiel(self) -> bool:
        return self.veredicto == "fiel"


class VeredictoCorreccion(BaseModel):
    """¿La respuesta coincide con la respuesta esperada escrita por una persona?"""

    razon: str = Field(description="Una frase que explica el veredicto.")
    veredicto: Literal["correcta", "incorrecta"]

    @property
    def correcta(self) -> bool:
        return self.veredicto == "correcta"


# Tope de tokens de salida del juez. Holgado a propósito: un modelo que "razona" (p. ej. gemma-4) gasta cientos de
# tokens pensando antes de escribir el JSON, y con un tope corto la respuesta llega vacía. El costo lo fija lo que
# el modelo realmente genera, no el tope.
TOKENS_JUEZ = int(os.getenv("JUEZ_MAX_TOKENS", "1500"))


def proveedor_juez() -> str:
    """Quién juzga: JUEZ_PROVIDER, o el mismo que genera. Ideal: un modelo distinto y más fuerte que el generador."""
    return (os.getenv("JUEZ_PROVIDER") or proveedor_generacion()).strip().lower()


def modelo_juez(proveedor: str | None = None) -> str:
    p = proveedor or proveedor_juez()
    return os.getenv("JUEZ_MODEL") or (modelo_local() if p == "local" else modelo_economico(p))


def _llamar_juez(prompt: str, proveedor: str | None = None, modelo: str | None = None) -> ResultadoLLM:
    """💰 Una llamada al juez. Temperatura 0 con el modelo local: el mismo caso debe dar el mismo veredicto."""
    p = proveedor or proveedor_juez()
    if p == "local":
        return llamar_local(prompt, modelo=modelo or modelo_juez(p), max_tokens=TOKENS_JUEZ, temperature=0)
    return llamar_llm(prompt, proveedor=p, modelo=modelo or modelo_juez(p), max_tokens=TOKENS_JUEZ, streaming=False, esfuerzo="minimo")


def juzgar_fidelidad(pregunta: str, respuesta: str, fragmentos: list[Resultado], version: str = "v2",
                     proveedor: str | None = None, modelo: str | None = None):
    """💰 Pide al juez un VeredictoFidelidad. Devuelve (veredicto | None, llamadas); None si no devolvió JSON válido."""
    prompt = renderizar("juez_fidelidad", version, pregunta=pregunta, respuesta=respuesta, fragmentos=fragmentos)
    return _pedir(prompt, VeredictoFidelidad,
                  '{"afirmaciones_sin_respaldo": ["afirmación que el contexto no respalda"], "razon": "una frase", '
                  '"veredicto": "fiel" o "no_fiel"}  (lista vacía si todas están respaldadas)', proveedor, modelo)


def juzgar_correccion(pregunta: str, respuesta: str, esperada: str, version: str = "v1",
                      proveedor: str | None = None, modelo: str | None = None):
    """💰 Pide al juez un VeredictoCorreccion. Devuelve (veredicto | None, llamadas)."""
    prompt = renderizar("juez_correccion", version, pregunta=pregunta, respuesta=respuesta, esperada=esperada)
    return _pedir(prompt, VeredictoCorreccion, '{"razon": "una frase", "veredicto": "correcta" o "incorrecta"}',
                  proveedor, modelo)


def pedir_json_con_ejemplo(llamar, prompt: str, modelo_datos, ejemplo: str, max_intentos: int = 2):
    """Pide JSON mostrando un EJEMPLO de la forma esperada y lo valida con pydantic (reintenta explicando el error).

    Es la variante de cliente_llm.pedir_json() para modelos pequeños: allí se envía el esquema JSON completo
    ("type", "properties", "required"…) y un modelo de 3B parámetros suele devolver el esquema en vez de los datos.
    Un ejemplo corto de la forma funciona mucho mejor. Devuelve (objeto, llamadas) o lanza ValueError.
    """
    instruccion = (f"{prompt}\n\nResponde ÚNICAMENTE con un objeto JSON con esta forma exacta, sin texto adicional "
                   f"ni cercas de código:\n{ejemplo}")
    llamadas: list[ResultadoLLM] = []
    error = ""
    for intento in range(max_intentos):
        r = llamar(instruccion if intento == 0 else f"{instruccion}\n\nTu respuesta anterior fue inválida ({error}). Corrígela.")
        llamadas.append(r)
        try:
            return modelo_datos.model_validate_json(extraer_json(r.texto)), llamadas
        except ValidationError as exc:
            error = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    raise ValueError(f"JSON inválido tras {max_intentos} intentos: {error}")


def _pedir(prompt: str, modelo_datos, ejemplo: str, proveedor, modelo):
    """Una consulta al juez con salida validada. Devuelve (veredicto | None, llamadas)."""
    llamadas: list[ResultadoLLM] = []

    def llamar(texto: str) -> ResultadoLLM:
        return _llamar_juez(texto, proveedor, modelo)

    try:
        veredicto, llamadas = pedir_json_con_ejemplo(llamar, prompt, modelo_datos, ejemplo)
        return veredicto, llamadas
    except ValueError:        # el juez no devolvió JSON válido ni tras reintentar
        return None, llamadas


def contexto_desde_secciones(manual: str | Path, titulos: list[str]) -> list[Resultado]:
    """Arma 'fragmentos' con el texto completo de las secciones indicadas (para probar al juez sin recuperar)."""
    doc = leer_documento(manual)
    por_titulo = dict(doc.secciones)
    return [Resultado(id=t, texto=por_titulo[t], metadatos={"seccion": t}, puntaje=1.0) for t in titulos]


def cohen_kappa(humanos: list[bool], juez: list[bool]) -> float:
    """Acuerdo entre dos evaluadores corregido por el azar (1 = perfecto, 0 = tanto como el azar).

    La exactitud sola engaña cuando las clases están desbalanceadas: un juez que dice siempre 'fiel'
    acierta la mitad si la mitad de las respuestas es fiel, y su kappa es 0.
    """
    n = len(humanos)
    if n == 0:
        return 0.0
    observado = sum(h == j for h, j in zip(humanos, juez)) / n
    ph, pj = sum(humanos) / n, sum(juez) / n
    azar = ph * pj + (1 - ph) * (1 - pj)
    return 1.0 if azar == 1 else (observado - azar) / (1 - azar)


def calibrar_juez(etiquetas: list[dict], golden: list[dict], manual: str | Path, criterio: str = "fidelidad",
                  version: str | None = None, proveedor: str | None = None, modelo: str | None = None,
                  progreso=None) -> dict:
    """💰 Mide al juez contra las etiquetas humanas de datos/etiquetas_juez.json.

    criterio = "fidelidad" (¿todo está respaldado por el contexto?) o "correccion" (¿coincide con lo esperado?).
    Devuelve exactitud, kappa y la matriz de confusión con nombres que se entienden sin estadística:
      aprobo_bien   el juez aprobó y la persona también
      aprobo_mal    el juez aprobó algo que la persona rechazó (INDULGENCIA: el error más caro)
      rechazo_mal   el juez rechazó algo que la persona aprobó (SEVERIDAD)
      rechazo_bien  el juez rechazó y la persona también
    """
    por_id = {p["id"]: p for p in golden}
    clave_humana = "fiel" if criterio == "fidelidad" else "correcta"
    filas, humanos, predichos, sin_veredicto, costo, seg = [], [], [], 0, 0.0, 0.0
    for e in etiquetas:
        p = por_id[e["pregunta_id"]]
        t0 = time.perf_counter()
        if criterio == "fidelidad":
            frs = contexto_desde_secciones(manual, e["contexto"])
            v, llamadas = juzgar_fidelidad(p["pregunta"], e["respuesta"], frs, version or "v2", proveedor, modelo)
            decision = None if v is None else v.fiel
        else:
            v, llamadas = juzgar_correccion(p["pregunta"], e["respuesta"], p["respuesta_esperada"], version or "v1", proveedor, modelo)
            decision = None if v is None else v.correcta
        seg += time.perf_counter() - t0
        costo += sum(c.costo for c in llamadas)
        if decision is None:
            sin_veredicto += 1
            decision = True          # sin veredicto, un sistema ingenuo "dejaría pasar" la respuesta
        humanos.append(e[clave_humana])
        predichos.append(decision)
        filas.append({"id": e["id"], "humano": e[clave_humana], "juez": decision, "nota": e["nota"],
                      "razon": getattr(v, "razon", "(sin JSON válido)")})
        if progreso:
            progreso(e["id"])
    n = len(etiquetas)
    vp = sum(h and j for h, j in zip(humanos, predichos))
    fp = sum((not h) and j for h, j in zip(humanos, predichos))
    fn = sum(h and (not j) for h, j in zip(humanos, predichos))
    vn = sum((not h) and (not j) for h, j in zip(humanos, predichos))
    return {
        "criterio": criterio, "n": n, "exactitud": (vp + vn) / n, "kappa": cohen_kappa(humanos, predichos),
        "aprobo_bien": vp, "aprobo_mal": fp, "rechazo_mal": fn, "rechazo_bien": vn,
        "indulgencia": fp / (fp + vn) if (fp + vn) else 0.0,     # de lo que la persona rechazó, cuánto aprobó el juez
        "severidad": fn / (fn + vp) if (fn + vp) else 0.0,       # de lo que la persona aprobó, cuánto rechazó el juez
        "sin_veredicto": sin_veredicto, "costo": costo, "segundos": seg, "filas": filas,
    }


# ───────────────────────── 5. Evaluar el RAG completo ─────────────────────────


def responder_con(recuperador, pregunta: str, k: int = 4, version: str = "v1", proveedor: str | None = None,
                  umbral: float | None = None) -> Respuesta:
    """Igual que rag.responder(), pero con CUALQUIER recuperador: una función (pregunta, k) → lista de Resultado.

    Así se pueden evaluar técnicas avanzadas (small-to-big, descomposición, grafo…) con las mismas
    métricas que el RAG básico: lo único que cambia es de dónde salen los fragmentos.
    """
    fragmentos = recuperador(pregunta, k)
    if not fragmentos or (umbral is not None and fragmentos[0].puntaje < umbral):
        return Respuesta(pregunta, NO_SE, fragmentos, sin_respuesta=True)
    llamada = generar(construir_prompt(pregunta, fragmentos, version), proveedor)
    texto = llamada.texto.strip()
    citas = citas_en(texto)
    invalidas = {n for n in citas if not 1 <= n <= len(fragmentos)}
    return Respuesta(pregunta, texto, fragmentos, citas=citas, citas_invalidas=invalidas,
                     sin_respuesta=texto.startswith("No encuentro"), llamada=llamada)


@dataclass
class FilaEvaluacion:
    id: str
    tipo: str
    pregunta: str
    respuesta: str
    fuera_de_alcance: bool
    # recuperación
    recall: float | None = None
    precision: float | None = None
    mrr: float | None = None
    hit: bool | None = None
    mejor_puntaje: float = 0.0
    secciones_recuperadas: list[str] = field(default_factory=list)
    palabras_contexto: int = 0          # tamaño del contexto entregado al modelo (más palabras = más tokens = más costo)
    # generación
    se_abstuvo: bool | None = None
    dato_ok: bool | None = None
    citas_validas: bool | None = None
    precision_citas: float | None = None
    numeros_sin_respaldo: list[str] = field(default_factory=list)
    diagnostico: str = "sin_generacion"
    # juez
    fiel: bool | None = None
    correcta: bool | None = None
    razon_juez: str = ""
    # costo
    costo: float = 0.0
    segundos: float = 0.0


@dataclass
class Informe:
    config: dict
    filas: list[FilaEvaluacion]
    segundos: float = 0.0

    def resumen(self, filas: list[FilaEvaluacion] | None = None) -> dict:
        """Los SLI del informe: promedios y tasas, listos para compararse con un SLO."""
        filas = self.filas if filas is None else filas
        con = [f for f in filas if not f.fuera_de_alcance]
        fuera = [f for f in filas if f.fuera_de_alcance]
        generadas = [f for f in filas if f.se_abstuvo is not None]
        contestadas = [f for f in con if f.se_abstuvo is False]
        juzgadas = [f for f in filas if f.fiel is not None]
        corregidas = [f for f in filas if f.correcta is not None]
        r = {
            "n": len(filas), "n_con_respuesta": len(con), "n_fuera_de_alcance": len(fuera),
            "recall": promedio([f.recall for f in con if f.recall is not None]),
            "precision": promedio([f.precision for f in con if f.precision is not None]),
            "mrr": promedio([f.mrr for f in con if f.mrr is not None]),
            "hit": promedio([float(f.hit) for f in con if f.hit is not None]),
        }
        if generadas:
            r.update({
                "datos_correctos": promedio([float(f.dato_ok) for f in con if f.dato_ok is not None]),
                "citas_validas": promedio([float(f.citas_validas) for f in contestadas if f.citas_validas is not None]),
                "abstencion_correcta": promedio([float(f.se_abstuvo) for f in fuera]),
                "abstencion_indebida": promedio([float(f.se_abstuvo) for f in con if f.se_abstuvo is not None]),
                "alucinaciones": sum(f.diagnostico in ("alucinacion", "cifra_sin_respaldo") for f in filas) / len(filas),
                "ok": sum(f.diagnostico == "ok" for f in filas) / len(filas),
            })
        if juzgadas:
            r["fidelidad_juez"] = promedio([float(f.fiel) for f in juzgadas])
        if corregidas:
            r["correccion_juez"] = promedio([float(f.correcta) for f in corregidas])
        r["palabras_contexto"] = promedio([float(f.palabras_contexto) for f in filas])
        r["costo_usd"] = sum(f.costo for f in filas)
        r["segundos_por_pregunta"] = promedio([f.segundos for f in generadas]) if generadas else 0.0
        return r

    def por_tipo(self) -> dict[str, dict]:
        return {t: self.resumen([f for f in self.filas if f.tipo == t]) for t in TIPOS if any(f.tipo == t for f in self.filas)}

    def diagnosticos(self) -> dict[str, list[str]]:
        """Qué preguntas cayeron en cada diagnóstico (sin los 'ok')."""
        salida: dict[str, list[str]] = {}
        for f in self.filas:
            if f.diagnostico not in ("ok", "sin_generacion"):
                salida.setdefault(f.diagnostico, []).append(f.id)
        return salida

    def a_dict(self) -> dict:
        return {"config": self.config, "segundos": self.segundos, "resumen": self.resumen(),
                "por_tipo": self.por_tipo(), "diagnosticos": self.diagnosticos(),
                "filas": [asdict(f) for f in self.filas]}


def evaluar_rag(almacen: Almacen | None, golden: list[dict], k: int = 4, version: str = "v1", proveedor: str | None = None,
                umbral: float | None = None, con_generacion: bool = True, juez: str | None = None,
                version_juez: str = "v2", progreso=None, recuperador=None, **config_extra) -> Informe:
    """💰 Evalúa el RAG de punta a punta con el golden set y devuelve un Informe.

    con_generacion=False → solo la recuperación (gratis, no llama al LLM).
    con_generacion=True  → 1 llamada al LLM por pregunta (gratis con proveedor 'local').
    juez                 → None, "fidelidad", "correccion" o "ambos": añade 1–2 llamadas al juez por respuesta.
    recuperador          → opcional: función (pregunta, k) → fragmentos, para evaluar una técnica avanzada. Si tiene un
                           atributo `costo` (llamadas al LLM que hace por dentro), se suma al costo de cada pregunta.
    """
    if recuperador is None:
        recuperador = lambda pregunta, kk: recuperar(almacen, pregunta, k=kk)   # noqa: E731 - el recuperador básico
    t_inicio = time.perf_counter()
    filas: list[FilaEvaluacion] = []
    for p in golden:
        fuera = bool(p.get("fuera_de_alcance"))
        t0 = time.perf_counter()
        costo_antes = getattr(recuperador, "costo", 0.0)
        if con_generacion:
            r = responder_con(recuperador, p["pregunta"], k=k, version=version, proveedor=proveedor, umbral=umbral)
            fragmentos, texto = r.fragmentos, r.texto
        else:
            r = None
            fragmentos, texto = recuperador(p["pregunta"], k), ""
        m = metricas_recuperacion(fragmentos, p)
        fila = FilaEvaluacion(
            id=p["id"], tipo=p["tipo"], pregunta=p["pregunta"], respuesta=texto, fuera_de_alcance=fuera,
            mejor_puntaje=fragmentos[0].puntaje if fragmentos else 0.0,
            secciones_recuperadas=[f.metadatos["seccion"] for f in fragmentos],
            palabras_contexto=sum(len(f.texto.split()) for f in fragmentos),
            costo=getattr(recuperador, "costo", 0.0) - costo_antes,
        )
        if m:
            fila.recall, fila.precision, fila.mrr, fila.hit = m["recall"], m["precision"], m["mrr"], m["hit"]
        if r is not None:
            fila.se_abstuvo = r.sin_respuesta
            if not fuera:
                fila.dato_ok = contiene_claves(texto, p["claves"]) if not r.sin_respuesta else False
                if not r.sin_respuesta:
                    fila.citas_validas = bool(r.citas) and not r.citas_invalidas
                    fila.precision_citas = precision_de_citas(r.citas, fragmentos, p)
            fila.numeros_sin_respaldo = [] if r.sin_respuesta else numeros_sin_respaldo(texto, fragmentos, p["pregunta"])
            fila.diagnostico = diagnosticar(p, se_abstuvo=r.sin_respuesta, recall=fila.recall, dato_ok=fila.dato_ok,
                                            sin_respaldo=fila.numeros_sin_respaldo, citas_validas=fila.citas_validas)
            if r.llamada:
                fila.costo += r.llamada.costo
            if juez in ("fidelidad", "ambos"):
                if r.sin_respuesta:
                    fila.fiel = True             # decir "no sé" no afirma nada: es fiel por definición
                else:
                    v, llamadas = juzgar_fidelidad(p["pregunta"], texto, fragmentos, version_juez)
                    fila.costo += sum(c.costo for c in llamadas)
                    fila.fiel = None if v is None else v.fiel
                    fila.razon_juez = getattr(v, "razon", "")
            if juez in ("correccion", "ambos"):
                v, llamadas = juzgar_correccion(p["pregunta"], texto, p["respuesta_esperada"])
                fila.costo += sum(c.costo for c in llamadas)
                fila.correcta = None if v is None else v.correcta
        fila.segundos = time.perf_counter() - t0
        filas.append(fila)
        if progreso:
            progreso(fila)
    config = {"k": k, "version_prompt": version, "proveedor": proveedor_generacion(proveedor) if con_generacion else None,
              "umbral": umbral, "con_generacion": con_generacion, "juez": juez, "n_preguntas": len(golden), **config_extra}
    return Informe(config=config, filas=filas, segundos=time.perf_counter() - t_inicio)


# ───────────────────────── 6. SLO: el umbral de aprobación ─────────────────────────
#
# Un SLO convierte el informe en una decisión: "¿puedo desplegar este cambio?". Cada entrada es
# (comparador, valor). Los comparadores son ">=" (más es mejor) o "<=" (menos es mejor).

SLO_POR_DEFECTO: dict[str, tuple[str, float]] = {
    "recall": (">=", 0.90),
    "datos_correctos": (">=", 0.80),
    "abstencion_correcta": (">=", 0.75),
    "alucinaciones": ("<=", 0.10),
}


def verificar_slo(resumen: dict, slo: dict[str, tuple[str, float]] | None = None) -> list[dict]:
    """Compara un resumen con el SLO. Cada fila: métrica, valor, regla y si cumple."""
    filas = []
    for metrica, (op, limite) in (slo or SLO_POR_DEFECTO).items():
        if metrica not in resumen:
            continue
        valor = resumen[metrica]
        filas.append({"metrica": metrica, "valor": valor, "regla": f"{op} {limite:.2f}",
                      "cumple": valor >= limite if op == ">=" else valor <= limite})
    return filas


# ───────────────────────── 7. Informes ─────────────────────────

ETIQUETAS_RESUMEN = {
    "recall": "Recall@K", "precision": "Precision@K", "mrr": "MRR", "hit": "hit@K",
    "datos_correctos": "Datos correctos", "citas_validas": "Citas válidas",
    "abstencion_correcta": '"No sé" bien dicho', "abstencion_indebida": '"No sé" indebido',
    "alucinaciones": "Alucinaciones", "ok": "Preguntas sin fallo",
    "fidelidad_juez": "Fidelidad (juez)", "correccion_juez": "Corrección (juez)",
}
# Métricas que no son porcentajes (se muestran aparte).
ETIQUETAS_CANTIDAD = {"palabras_contexto": "Palabras de contexto"}


def _pct(x: float) -> str:
    return f"{x * 100:.0f} %"


def imprimir_resumen(inf: Informe, titulo: str = "") -> None:
    """Resumen de una línea por métrica, en consola."""
    res = inf.resumen()
    cfg = inf.config
    print(f"{titulo or 'Informe'} · k={cfg['k']} · prompt {cfg['version_prompt']} · proveedor {cfg['proveedor'] or '—'} · "
          f"{res['n']} preguntas ({res['n_con_respuesta']} con respuesta, {res['n_fuera_de_alcance']} fuera de alcance)")
    for clave, nombre in ETIQUETAS_RESUMEN.items():
        if clave in res:
            print(f"  {nombre:<22} {_pct(res[clave]):>6}")
    print(f"  {'Palabras de contexto':<22} {res['palabras_contexto']:>6.0f}  (promedio por pregunta)")
    print(f"  {'Costo':<22} US$ {res['costo_usd']:.5f}   ·   {inf.segundos:.0f} s en total")
    diag = inf.diagnosticos()
    if diag:
        print("  Fallos por diagnóstico:", "; ".join(f"{d} ({', '.join(ids)})" for d, ids in diag.items()))


def tabla_comparativa(informes: dict[str, Informe], metricas: tuple[str, ...] | None = None) -> str:
    """Tabla de texto: una columna por configuración, una fila por métrica (para 'medir → cambiar → medir')."""
    resumenes = {nombre: inf.resumen() for nombre, inf in informes.items()}
    metricas = metricas or tuple(m for m in ETIQUETAS_RESUMEN if all(m in r for r in resumenes.values()))
    ancho = max(12, *(len(n) for n in informes))
    lineas = [f"{'':<22}" + "".join(f"{n:>{ancho + 2}}" for n in informes)]
    for m in metricas:
        lineas.append(f"{ETIQUETAS_RESUMEN.get(m, m):<22}" + "".join(f"{_pct(r[m]):>{ancho + 2}}" for r in resumenes.values()))
    lineas.append(f"{'Palabras de contexto':<22}" + "".join(f"{r['palabras_contexto']:>{ancho + 2}.0f}" for r in resumenes.values()))
    lineas.append(f"{'Costo (US$)':<22}" + "".join(f"{r['costo_usd']:>{ancho + 2}.5f}" for r in resumenes.values()))
    return "\n".join(lineas)


def tabla_por_tipo(inf: Informe) -> str:
    """Una fila por tipo de pregunta: cuántas hay y cómo le fue a cada tipo (los promedios globales esconden esto)."""
    con_gen = inf.config.get("con_generacion", True)
    cab = f"{'tipo':<16}{'n':>3}{'Recall@K':>10}{'Prec.@K':>9}"
    if con_gen:
        cab += f"{'datos ok':>10}{'no sé ok':>10}{'alucin.':>9}"
    lineas = [cab]
    for tipo, r in inf.por_tipo().items():
        con = r["n_con_respuesta"] > 0
        fila = f"{tipo:<16}{r['n']:>3}{_pct(r['recall']) if con else '—':>10}{_pct(r['precision']) if con else '—':>9}"
        if con_gen:
            fila += (f"{_pct(r['datos_correctos']) if con else '—':>10}"
                     f"{_pct(r['abstencion_correcta']) if r['n_fuera_de_alcance'] else '—':>10}{_pct(r['alucinaciones']):>9}")
        lineas.append(fila)
    return "\n".join(lineas)


def informe_markdown(inf: Informe, slo: dict[str, tuple[str, float]] | None = None) -> str:
    """Informe legible para revisar en un pull request: configuración, resumen, SLO, por tipo y fallos."""
    res, cfg = inf.resumen(), inf.config
    out = ["# Informe de calidad del RAG", "",
           f"- **Configuración:** k = {cfg['k']} · prompt `{cfg['version_prompt']}` · proveedor `{cfg['proveedor']}` · umbral {cfg['umbral']}",
           f"- **Preguntas:** {res['n']} ({res['n_con_respuesta']} con respuesta, {res['n_fuera_de_alcance']} fuera de alcance)",
           f"- **Contexto:** {res['palabras_contexto']:.0f} palabras por pregunta, en promedio",
           f"- **Costo:** US$ {res['costo_usd']:.5f} · {inf.segundos:.0f} s", "", "## Resumen", "",
           "| Métrica | Valor |", "| --- | --- |"]
    out += [f"| {ETIQUETAS_RESUMEN[c]} | {_pct(res[c])} |" for c in ETIQUETAS_RESUMEN if c in res]
    if "datos_correctos" in res:
        out += ["", "## ¿Cumple el SLO?", "", "| Métrica | Valor | Regla | Cumple |", "| --- | --- | --- | --- |"]
        for f in verificar_slo(res, slo):
            out.append(f"| {ETIQUETAS_RESUMEN.get(f['metrica'], f['metrica'])} | {_pct(f['valor'])} | {f['regla']} | {'✅' if f['cumple'] else '❌'} |")
    out += ["", "## Por tipo de pregunta", "", "| Tipo | n | Recall@K | Precision@K | Datos correctos | Alucinaciones |", "| --- | --- | --- | --- | --- | --- |"]
    for tipo, r in inf.por_tipo().items():
        out.append(f"| {tipo} | {r['n']} | {_pct(r['recall']) if r['n_con_respuesta'] else '—'} | "
                   f"{_pct(r['precision']) if r['n_con_respuesta'] else '—'} | "
                   f"{_pct(r['datos_correctos']) if 'datos_correctos' in r and r['n_con_respuesta'] else '—'} | "
                   f"{_pct(r['alucinaciones']) if 'alucinaciones' in r else '—'} |")
    out += ["", "## Preguntas con fallo", ""]
    fallos = [f for f in inf.filas if f.diagnostico not in ("ok", "sin_generacion")]
    if fallos:
        out += ["| Id | Diagnóstico | Capa | Pregunta |", "| --- | --- | --- | --- |"]
        out += [f"| {f.id} | {f.diagnostico} | {CAPA_DEL_DIAGNOSTICO[f.diagnostico]} | {f.pregunta} |" for f in fallos]
    else:
        out.append("Ninguna.")
    return "\n".join(out) + "\n"


def guardar_informe(inf: Informe, carpeta: str | Path = "informes", nombre: str | None = None,
                    slo: dict[str, tuple[str, float]] | None = None) -> tuple[Path, Path]:
    """Guarda el informe como JSON (para compararlo con el siguiente) y como Markdown (para leerlo)."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    nombre = nombre or time.strftime("%Y%m%d-%H%M%S")
    ruta_json, ruta_md = carpeta / f"{nombre}.json", carpeta / f"{nombre}.md"
    ruta_json.write_text(json.dumps(inf.a_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    ruta_md.write_text(informe_markdown(inf, slo), encoding="utf-8")
    return ruta_json, ruta_md
