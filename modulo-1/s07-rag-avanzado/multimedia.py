"""RAG multimodal (audio y video): de un archivo de medios a fragmentos con TIEMPO.

Un RAG de audio o de video es el mismo RAG de siempre, con un paso de ingesta distinto: antes de
fragmentar, hay que convertir el sonido en TEXTO. Lo importante es no perder el tiempo: cada
fragmento recuerda en qué segundo empieza y termina, y así la respuesta puede citar "minuto 03:00".

    archivo (mp3/mp4) ─► [ffmpeg: solo audio, 16 kHz, mono] ─► Whisper (ASR) ─► segmentos {inicio, fin, texto}
                                                                                   │
                       ventanas de ≈ 50 s ◄───────────────────────────────────────┘
                              │
              embeddings (el embeddings.py del curso) ─► almacén ─► búsqueda ─► respuesta con cita mm:ss

    Video: se añade (opcional) una segunda vía, los FOTOGRAMAS clave, que se describen con un modelo con
    visión o con OCR y se alinean por tiempo con la transcripción.

Dependencias (OPCIONALES: el resto de la sesión funciona sin ellas):

    uv sync --group media        # instala faster-whisper y av<16 (NO hace falta torch)
    ffmpeg                       # programa del sistema; solo para extraer audio o fotogramas

`av` debe ser < 16: con faster-whisper 1.2.1, `av` 19 falla con
"TypeError: open() got an unexpected keyword argument 'metadata_errors'".

💰 COSTO: todo corre en tu computadora (CPU). No llama a ningún servicio de pago. La primera vez,
Whisper descarga el modelo (base ≈ 140 MB, small ≈ 460 MB) y lo guarda en la caché de Hugging Face.

Los medios son contenido de terceros: viven en `datos/media/`, una carpeta ignorada por Git. NO los
subas al repositorio.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from almacenes import Fragmento, Resultado

CARPETA_MEDIA = Path("datos/media")
EXTENSIONES_AUDIO = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
EXTENSIONES_VIDEO = (".mp4", ".mkv", ".mov", ".webm")


class MediaNoDisponible(RuntimeError):
    """Falta una dependencia opcional o un archivo; el mensaje dice cómo arreglarlo."""


# ───────────────────────── Piezas de la transcripción ─────────────────────────


@dataclass
class Segmento:
    """Un tramo de habla que Whisper reconoció: de `inicio` a `fin` (segundos) se dijo `texto`."""

    inicio: float
    fin: float
    texto: str


@dataclass
class Transcripcion:
    fuente: str                  # nombre del archivo
    modelo: str                  # tamaño de Whisper (base, small…)
    idioma: str
    duracion: float              # segundos de audio
    segundos: float              # cuánto tardó transcribir (tiempo de reloj)
    segmentos: list[Segmento]
    desde_cache: bool = False

    @property
    def palabras(self) -> int:
        return sum(len(s.texto.split()) for s in self.segmentos)

    @property
    def veces_tiempo_real(self) -> float:
        return self.duracion / self.segundos if self.segundos else 0.0


def mmss(segundos: float) -> str:
    """Segundos → 'mm:ss' (o 'h:mm:ss' si pasa de una hora)."""
    s = int(segundos)
    h, resto = divmod(s, 3600)
    m, s = divmod(resto, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def medios_disponibles(carpeta: str | Path = CARPETA_MEDIA) -> dict[str, Path]:
    """Los archivos de audio y video que el instructor dejó en `datos/media/` (nombre → ruta)."""
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return {}
    permitidas = EXTENSIONES_AUDIO + EXTENSIONES_VIDEO
    return {p.name: p for p in sorted(carpeta.iterdir()) if p.suffix.lower() in permitidas}


def hay_faster_whisper() -> tuple[bool, str]:
    """¿Está instalada la dependencia opcional? Devuelve (sí/no, mensaje con instrucciones si no)."""
    try:
        import faster_whisper  # noqa: F401
        import av  # noqa: F401
    except ImportError:
        return False, ("Falta la dependencia opcional del RAG multimodal. Instálala con `uv sync --group media` "
                       "(faster-whisper y av<16; no hace falta torch) y reinicia el kernel.")
    return True, "faster-whisper disponible"


def hay_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


# ───────────────────────── 1. Audio: extraer y transcribir ─────────────────────────


def extraer_audio(video: str | Path, destino: str | Path | None = None) -> Path:
    """ffmpeg: quita la imagen y deja solo el audio, a 16 kHz y mono (el formato que Whisper usa por dentro)."""
    if not hay_ffmpeg():
        raise MediaNoDisponible("No encuentro `ffmpeg` en el PATH. Instálalo (por ejemplo `sudo apt install ffmpeg`).")
    video = Path(video)
    destino = Path(destino) if destino else video.with_suffix(".16k.wav")
    destino.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(destino)],
                   check=True)
    return destino


def _clave_cache(ruta: Path, modelo: str, idioma: str | None, prompt: str | None) -> str:
    st = ruta.stat()
    huella = f"{ruta.name}|{st.st_size}|{int(st.st_mtime)}|{modelo}|{idioma}|{prompt or ''}"
    return hashlib.sha1(huella.encode()).hexdigest()[:12]


def transcribir(ruta: str | Path, modelo: str = "small", idioma: str | None = None, initial_prompt: str | None = None,
                usar_cache: bool = True, hilos: int | None = None) -> Transcripcion:
    """Whisper (faster-whisper) en CPU: devuelve los segmentos con `inicio`, `fin` y `texto`.

    - `modelo`: "base" (≈ 140 MB, rápido) o "small" (≈ 460 MB, más preciso). Con GPU puedes usar uno mayor.
    - `idioma`: "en", "es"… (None = que Whisper lo detecte).
    - `initial_prompt`: vocabulario que Whisper debe esperar ("ChatGPT, GitHub Copilot…"): mitiga los
      errores con términos técnicos.
    - La primera vez tarda (minutos de audio ÷ la velocidad del equipo); el resultado se guarda en
      `datos/media/.cache/` y las siguientes veces es instantáneo (`usar_cache=False` lo fuerza de nuevo).
    """
    ok, msg = hay_faster_whisper()
    if not ok:
        raise MediaNoDisponible(msg)
    ruta = Path(ruta)
    if not ruta.is_file():
        raise MediaNoDisponible(f"No encuentro {ruta}. Coloca el archivo en {CARPETA_MEDIA}/ (ver README).")
    cache = ruta.parent / ".cache" / f"{ruta.stem[:40]}.{_clave_cache(ruta, modelo, idioma, initial_prompt)}.json"
    if usar_cache and cache.is_file():
        d = json.loads(cache.read_text(encoding="utf-8"))
        d["segmentos"] = [Segmento(**s) for s in d["segmentos"]]
        d["desde_cache"] = True
        return Transcripcion(**d)

    from faster_whisper import WhisperModel

    hilos = hilos or min(16, os.cpu_count() or 4)
    t0 = time.perf_counter()
    whisper = WhisperModel(modelo, device="cpu", compute_type="int8", cpu_threads=hilos)
    segs, info = whisper.transcribe(str(ruta), language=idioma, vad_filter=True, beam_size=1, initial_prompt=initial_prompt)
    segmentos = [Segmento(s.start, s.end, s.text.strip()) for s in segs if s.text.strip()]
    t = Transcripcion(fuente=ruta.name, modelo=modelo, idioma=info.language, duracion=info.duration,
                      segundos=time.perf_counter() - t0, segmentos=segmentos)
    if usar_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(asdict(t), ensure_ascii=False), encoding="utf-8")
    return t


# ───────────────────────── 2. Fragmentar CON tiempo ─────────────────────────


def ventanas_de_tiempo(segmentos: list[Segmento], ventana_s: float = 50, solape_s: float = 10) -> list[dict]:
    """Junta segmentos consecutivos hasta completar ≈ `ventana_s` segundos; cada ventana recuerda inicio y fin.

    El solape repite los últimos `solape_s` segundos al comienzo de la ventana siguiente: una idea cortada
    justo en el borde queda completa en alguna de las dos.
    """
    ventanas: list[dict] = []
    i = 0
    while i < len(segmentos):
        j, inicio = i, segmentos[i].inicio
        while j < len(segmentos) and (segmentos[j].fin - inicio < ventana_s or j == i):
            j += 1
        trozo = segmentos[i:j]
        ventanas.append({"inicio": trozo[0].inicio, "fin": trozo[-1].fin, "texto": " ".join(s.texto for s in trozo)})
        if j >= len(segmentos):
            break
        # La siguiente ventana arranca en el primer segmento que empieza dentro del solape (pero siempre avanza).
        k = j
        while k - 1 > i and segmentos[k - 1].inicio >= trozo[-1].fin - solape_s:
            k -= 1
        i = max(k, i + 1)
    return ventanas


def a_fragmentos(ventanas: list[dict], fuente: str) -> list[Fragmento]:
    """Ventanas → `Fragmento`s del curso. La sección es el rango de tiempo ('03:00–03:55'): sirve para citar."""
    return [
        Fragmento(id=f"{Path(fuente).stem[:24]}@{mmss(v['inicio'])}", texto=v["texto"],
                  metadatos={"fuente": fuente, "inicio": v["inicio"], "fin": v["fin"],
                             "seccion": f"{mmss(v['inicio'])}–{mmss(v['fin'])}"})
        for v in ventanas
    ]


def cita_de_tiempo(r: Fragmento | Resultado) -> str:
    """'[nombre 03:00–03:55]': la cita que un humano puede ir a verificar en el reproductor."""
    m = r.metadatos
    return f"[{Path(m['fuente']).stem[:30]} {mmss(m['inicio'])}–{mmss(m['fin'])}]"


# ───────────────────────── 3. Medir la recuperación por TIEMPO ─────────────────────────


def acierta_en_tiempo(r: Resultado, esperado: tuple[float, float]) -> bool:
    """¿La ventana recuperada se solapa con el tramo donde está la respuesta?"""
    return r.metadatos["inicio"] <= esperado[1] and r.metadatos["fin"] >= esperado[0]


def evaluar_por_tiempo(buscar, preguntas: list[dict], ks: tuple[int, ...] = (1, 3)) -> dict:
    """hit@k y MRR de una lista de preguntas con tramo esperado. `buscar(pregunta, k)` devuelve resultados."""
    kmax = max(ks)
    filas, hits = [], {k: 0 for k in ks}
    mrr = 0.0
    for p in preguntas:
        top = buscar(p["pregunta"], kmax)
        pos = next((i for i, r in enumerate(top, start=1) if acierta_en_tiempo(r, tuple(p["esperado"]))), None)
        for k in ks:
            hits[k] += pos is not None and pos <= k
        mrr += 1 / pos if pos else 0.0
        filas.append({"id": p["id"], "pregunta": p["pregunta"], "posicion": pos,
                      "primero": mmss(top[0].metadatos["inicio"]) if top else "—",
                      "esperado": f"{mmss(p['esperado'][0])}–{mmss(p['esperado'][1])}"})
    n = len(preguntas) or 1
    return {"n": len(preguntas), "mrr": mrr / n, **{f"hit@{k}": hits[k] for k in ks}, "filas": filas}


# ───────────────────────── 4. Video: fotogramas clave (opcional) ─────────────────────────


def extraer_fotogramas(video: str | Path, cada_s: int = 120, carpeta: str | Path | None = None, ancho: int = 640) -> list[tuple[float, Path]]:
    """ffmpeg: un fotograma cada `cada_s` segundos. Devuelve [(segundo, ruta_jpg), …] para alinear con la transcripción."""
    if not hay_ffmpeg():
        raise MediaNoDisponible("No encuentro `ffmpeg` en el PATH.")
    video = Path(video)
    carpeta = Path(carpeta) if carpeta else video.parent / ".cache" / f"{video.stem[:30]}_fotogramas"
    carpeta.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", f"fps=1/{cada_s},scale={ancho}:-2",
                    str(carpeta / "f_%04d.jpg")], check=True)
    return [((n) * cada_s, p) for n, p in enumerate(sorted(carpeta.glob("f_*.jpg")))]


def fotograma_mas_cercano(fotogramas: list[tuple[float, Path]], segundo: float) -> tuple[float, Path]:
    """Alineación por tiempo: de todos los fotogramas, el más cercano al segundo pedido."""
    return min(fotogramas, key=lambda f: abs(f[0] - segundo))


def describir_fotograma(ruta: str | Path, modelo: str | None = None, pregunta: str = "Describe en español, en dos frases, qué se ve y qué texto aparece.",
                        max_tokens: int = 1500) -> str:
    """Pide a un modelo CON VISIÓN (LM Studio) que describa una imagen. Lento en CPU (≈ 16 s por fotograma con un 12B).

    Honestidad: a 640×360 un modelo local lee de forma IRREGULAR la escritura a mano: en una prueba convirtió "↑ 0-CLICKS" en
    "↓ CLICKS" (la flecha invertida cambia el sentido) y en otra leyó "AGENTS!" como "PACIENTS". Para video de pizarra o de
    notas manuscritas, la transcripción del audio es lo que sostiene el RAG.
    """
    import openai

    cliente = openai.OpenAI(base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1"),
                            api_key=os.getenv("LOCAL_LLM_API_KEY", "no-se-usa"), timeout=300, max_retries=0)
    datos = base64.b64encode(Path(ruta).read_bytes()).decode()
    r = cliente.chat.completions.create(
        model=modelo or os.getenv("MODELO_VISION", "google/gemma-4-12b-qat"), max_tokens=max_tokens, temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": pregunta},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{datos}"}},
        ]}],
    )
    return (r.choices[0].message.content or "").strip()
