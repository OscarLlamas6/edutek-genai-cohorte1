"""Cliente LLM reutilizable del curso (Anthropic y OpenAI, con el mismo código).

Reúne lo esencial de un cliente "serio":

- DOS proveedores nativos, Anthropic y OpenAI, intercambiables con una variable de entorno,
- un modelo open-weight local (LM Studio u otro servidor compatible con OpenAI),
- streaming con medición de TTFT (tiempo hasta el primer token) y tokens/segundo,
- reintentos con espera exponencial y jitter, solo para errores transitorios,
- medición de uso de tokens y costo en cada llamada,
- salida estructurada: pedir JSON, validar con pydantic y reintentar si falla.

CÓMO ELEGIR EL PROVEEDOR (en tu archivo .env, en la raíz del repositorio)
    LLM_PROVIDER=anthropic        # por defecto
    LLM_PROVIDER=openai
    ANTHROPIC_API_KEY=...         # si usas anthropic
    OPENAI_API_KEY=...            # si usas openai
    LLM_MODEL=...                 # opcional: modelo económico a usar (si no, el de la tabla)
    LLM_MODEL_SAMPLING=...        # opcional: modelo para experimentar con temperature/top_p/top_k
    OPENAI_BASE_URL=...           # opcional: otro servicio compatible con OpenAI (Groq, OpenRouter…)

La función que usa casi todo el curso es `llamar_llm()`: recibe el prompt y devuelve un
`ResultadoLLM` (texto, tokens, tiempos, costo) sin importar el proveedor.

💰 COSTO
    llamar_llm() / llamar_claude() / llamar_openai() CONSUMEN DINERO en tu cuenta del proveedor.
    llamar_local() corre en tu computadora y no cuesta por token.

Este módulo se copia al repositorio del proyecto integrador y se reutiliza durante todo el
curso. La carpeta de cada sesión trae una copia idéntica para que funcione sola.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel, ValidationError

# Los dos SDK están en las dependencias de cada sesión. Se importan con tolerancia para que
# un error de instalación dé un mensaje claro en lugar de romper todo el módulo.
try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None
try:
    import openai
except ImportError:  # pragma: no cover
    openai = None

# ───────────────────────── Configuración: proveedor y modelos ─────────────────────────

PROVEEDORES = ("anthropic", "openai")

# Modelo "económico" (el que usamos para practicar) y modelo para experimentar con
# parámetros de muestreo (temperature, top_p, top_k), por proveedor.
#   - Anthropic: claude-haiku-5-5 no acepta temperature; claude-haiku-4-5 sí.
#   - OpenAI: gpt-6-luna acepta temperature/top_p solo con el razonamiento apagado
#     (esfuerzo "minimo"), así que sirve el mismo modelo.
# Los nombres cambian cada pocos meses: puedes sobrescribirlos con LLM_MODEL y LLM_MODEL_SAMPLING.
MODELOS_POR_PROVEEDOR = {
    "anthropic": {"economico": "claude-haiku-5-5", "muestreo": "claude-haiku-4-5"},
    "openai": {"economico": "gpt-6-luna", "muestreo": "gpt-6-luna"},
}

# USD por millón de tokens: (entrada, salida). Referencia a octubre de 2026.
# Los precios cambian: confirma en la página oficial de tu proveedor antes de presupuestar.
PRECIOS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-1": (10.00, 50.00),
    # OpenAI
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-6-sol": (2.00, 10.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
}


def proveedor_activo() -> str:
    """Proveedor elegido con la variable de entorno LLM_PROVIDER (por defecto, anthropic)."""
    p = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    if p not in PROVEEDORES:
        raise ValueError(f"LLM_PROVIDER='{p}' no es válido. Usa uno de: {', '.join(PROVEEDORES)}.")
    return p


def modelo_economico(proveedor: str | None = None) -> str:
    """Modelo económico del proveedor (LLM_MODEL tiene prioridad si está definida)."""
    p = proveedor or proveedor_activo()
    return os.getenv("LLM_MODEL") or MODELOS_POR_PROVEEDOR[p]["economico"]


def modelo_muestreo(proveedor: str | None = None) -> str:
    """Modelo para experimentar con temperature/top_p/top_k (LLM_MODEL_SAMPLING tiene prioridad)."""
    p = proveedor or proveedor_activo()
    return os.getenv("LLM_MODEL_SAMPLING") or MODELOS_POR_PROVEEDOR[p]["muestreo"]


def variable_api_key(proveedor: str | None = None) -> str:
    """Nombre de la variable de entorno que contiene la llave del proveedor."""
    return "ANTHROPIC_API_KEY" if (proveedor or proveedor_activo()) == "anthropic" else "OPENAI_API_KEY"


def hay_api_key(proveedor: str | None = None) -> bool:
    """¿Está definida la llave del proveedor?"""
    return bool(os.getenv(variable_api_key(proveedor)))


def _exigir_api_key(proveedor: str) -> None:
    if not hay_api_key(proveedor):
        raise RuntimeError(
            f"Falta {variable_api_key(proveedor)} (proveedor: {proveedor}). Copia .env.example a .env en la "
            "raíz del repositorio y pega tu llave (ver README de la sesión)."
        )


def costo_usd(modelo: str, tokens_entrada: int, tokens_salida: int) -> float:
    """Costo en USD. Un modelo que no está en PRECIOS (por ejemplo, uno local) cuesta 0."""
    if modelo not in PRECIOS:
        return 0.0
    precio_in, precio_out = PRECIOS[modelo]
    return (tokens_entrada * precio_in + tokens_salida * precio_out) / 1_000_000


# ───────────────────────── Resultado y contador de gasto ─────────────────────────


@dataclass
class ResultadoLLM:
    """Todo lo que importa de una llamada, en un solo objeto (igual para cualquier proveedor)."""

    texto: str
    modelo: str
    tokens_entrada: int                # todos los tokens de entrada (incluye los leídos de caché)
    tokens_salida: int                 # incluye el razonamiento interno, que también se cobra
    segundos_total: float
    ttft: float | None = None          # segundos hasta el primer token (solo con streaming)
    stop_reason: str | None = None     # normalizado: "end_turn" o "max_tokens" (u otro motivo)
    intentos: int = 1
    costo: float = 0.0
    tokens_cache_leidos: int = 0       # entrada servida desde la caché de prompt
    tokens_cache_escritos: int = 0     # entrada escrita en la caché (si el proveedor lo informa)
    proveedor: str = ""

    @property
    def tokens_por_segundo(self) -> float | None:
        """Velocidad de generación, sin contar la espera inicial."""
        if self.ttft is None or self.tokens_salida == 0:
            return None
        generacion = self.segundos_total - self.ttft
        return self.tokens_salida / generacion if generacion > 0 else None


@dataclass
class Gasto:
    """Acumula el costo real de todas las llamadas de una sesión."""

    resultados: list[ResultadoLLM] = field(default_factory=list)

    def registrar(self, r: ResultadoLLM) -> ResultadoLLM:
        self.resultados.append(r)
        return r

    @property
    def total(self) -> float:
        return sum(r.costo for r in self.resultados)

    def resumen(self) -> None:
        sin_precio = sorted({r.modelo for r in self.resultados if r.modelo not in PRECIOS and not r.modelo.startswith("local:")})
        print(f"Llamadas: {len(self.resultados)} · gasto acumulado: US$ {self.total:.6f}")
        if sin_precio:
            print(f"(Sin precio en PRECIOS, costo contado como 0: {', '.join(sin_precio)})")


# ───────────────────────── Reintentos ─────────────────────────

# Errores transitorios: vale la pena volver a intentar. Se reconocen los de ambos SDK.
_REINTENTABLES = tuple(
    clase
    for modulo in (anthropic, openai)
    if modulo is not None
    for clase in (
        modulo.RateLimitError,        # 429: límite de tasa
        modulo.InternalServerError,   # 5xx: falla del servicio
        modulo.APIConnectionError,    # red caída o intermitente
        modulo.APITimeoutError,       # timeout
    )
)


def es_reintentable(exc: Exception) -> bool:
    """429, 5xx, red y timeouts se reintentan. 400/401/403/404 son culpa nuestra: se corrigen."""
    if isinstance(exc, _REINTENTABLES):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):   # también aplica a clientes locales
        return True
    return False


T = TypeVar("T")


def con_reintentos(
    fn: Callable[[], T],
    max_reintentos: int = 4,
    base: float = 1.0,
    tope: float = 30.0,
    dormir: Callable[[float], None] = time.sleep,
    al_reintentar: Callable[[int, float, Exception], None] | None = None,
) -> tuple[T, int]:
    """Ejecuta `fn`; si falla con un error transitorio, espera y reintenta.

    La espera crece de forma exponencial (base, 2·base, 4·base…) hasta `tope`, y se
    multiplica por un número al azar entre 0.5 y 1 ("jitter") para que muchos clientes
    no reintenten todos al mismo tiempo.

    Devuelve (resultado, intentos_usados). Si se agotan los reintentos, relanza el error.
    """
    intento = 0
    while True:
        intento += 1
        try:
            return fn(), intento
        except Exception as exc:  # decidimos abajo si se reintenta
            if not es_reintentable(exc) or intento > max_reintentos:
                raise
            espera = min(tope, base * 2 ** (intento - 1)) * random.uniform(0.5, 1.0)
            if al_reintentar:
                al_reintentar(intento, espera, exc)
            dormir(espera)


# ───────────────────────── Esfuerzo de razonamiento (común a los dos proveedores) ─────────────────────────
#
# Los modelos actuales "piensan" antes de responder; esos tokens se cobran como salida. Cada
# proveedor lo controla con un parámetro distinto. Aquí usamos cuatro niveles comunes:
#
#   nivel      Anthropic (output_config.effort)   OpenAI (reasoning.effort)
#   "minimo"   low                                none   (razonamiento apagado)
#   "bajo"     low                                low
#   "medio"    medium                             medium
#   "alto"     high                               high
#
# En OpenAI, temperature y top_p solo se aceptan con el nivel "minimo" (razonamiento apagado).

ESFUERZO_ANTHROPIC = {"minimo": "low", "bajo": "low", "medio": "medium", "alto": "high"}
ESFUERZO_OPENAI = {"minimo": "none", "bajo": "low", "medio": "medium", "alto": "high"}

# Modelos de Anthropic que NO aceptan el parámetro effort.
_CLAUDE_SIN_ESFUERZO = ("claude-haiku-4-5", "claude-sonnet-4", "claude-opus-4-5", "claude-opus-4-1", "claude-3")


def _claude_admite_esfuerzo(modelo: str) -> bool:
    return not modelo.startswith(_CLAUDE_SIN_ESFUERZO)


def _openai_es_razonador(modelo: str) -> bool:
    """Los modelos gpt-5 / gpt-6 y la serie o aceptan el parámetro de razonamiento; gpt-4.x no."""
    return modelo.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))


# ───────────────────────── Proveedor: Anthropic ─────────────────────────


def llamar_claude(
    prompt: str | list[dict],
    modelo: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    streaming: bool = True,
    al_recibir_texto: Callable[[str], None] | None = None,
    max_reintentos: int = 4,
    cliente=None,
    api_key: str | None = None,
    esfuerzo: str | None = "minimo",
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    cache: bool = False,
    **extra,
) -> ResultadoLLM:
    """💰 CONSUME DINERO. Llama a Claude, con streaming opcional y reintentos.

    `prompt` puede ser un texto (un solo mensaje de usuario) o la lista completa de
    mensajes, porque la API no tiene estado: tú reenvías el historial.
    `esfuerzo` ("minimo", "bajo", "medio", "alto") controla el razonamiento interno; por defecto
    "minimo" (tareas simples y baratas). Usa None para no enviar el parámetro.
    `temperature`, `top_p` y `top_k` solo los aceptan algunos modelos (ver Sesión 1, Parte 6):
    el SDK 1.x ya no tiene esos argumentos, así que viajan en el cuerpo de la petición.
    `cache=True` pide cachear automáticamente el prefijo del prompt.
    `extra` se pasa tal cual a la API.
    """
    if anthropic is None:
        raise RuntimeError("Falta instalar el paquete 'anthropic' (uv sync).")
    modelo = modelo or modelo_economico("anthropic")
    if cliente is None:
        if api_key is None:
            _exigir_api_key("anthropic")
        # max_retries=0: los reintentos los hace con_reintentos() para que sean visibles.
        # (El SDK ya reintenta por su cuenta; en producción puedes confiar en eso.)
        cliente = anthropic.Anthropic(api_key=api_key, max_retries=0) if api_key else anthropic.Anthropic(max_retries=0)

    mensajes = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    params: dict = {"model": modelo, "max_tokens": max_tokens, "messages": mensajes, **extra}
    if system:
        params["system"] = system
    if esfuerzo and _claude_admite_esfuerzo(modelo):
        params["output_config"] = {**params.get("output_config", {}), "effort": ESFUERZO_ANTHROPIC[esfuerzo]}
    if cache:
        params["cache_control"] = {"type": "ephemeral"}
    muestreo = {k: v for k, v in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k)) if v is not None}
    if muestreo:
        params["extra_body"] = {**params.get("extra_body", {}), **muestreo}

    def una_llamada() -> ResultadoLLM:
        t0 = time.perf_counter()
        ttft = None
        if streaming:
            partes: list[str] = []
            with cliente.messages.stream(**params) as stream:
                for trozo in stream.text_stream:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    partes.append(trozo)
                    if al_recibir_texto:
                        al_recibir_texto(trozo)
                final = stream.get_final_message()
            texto = "".join(partes)
        else:
            final = cliente.messages.create(**params)
            texto = "".join(b.text for b in final.content if b.type == "text")
        total = time.perf_counter() - t0
        u = final.usage
        leidos = getattr(u, "cache_read_input_tokens", 0) or 0
        escritos = getattr(u, "cache_creation_input_tokens", 0) or 0
        t_in = u.input_tokens + leidos + escritos   # en Anthropic input_tokens NO incluye la caché
        return ResultadoLLM(
            texto=texto,
            modelo=modelo,
            tokens_entrada=t_in,
            tokens_salida=u.output_tokens,
            segundos_total=total,
            ttft=ttft,
            stop_reason=final.stop_reason,
            costo=costo_usd(modelo, t_in, u.output_tokens),
            tokens_cache_leidos=leidos,
            tokens_cache_escritos=escritos,
            proveedor="anthropic",
        )

    resultado, intentos = con_reintentos(una_llamada, max_reintentos=max_reintentos)
    resultado.intentos = intentos
    return resultado


# ───────────────────────── Proveedor: OpenAI ─────────────────────────


def llamar_openai(
    prompt: str | list[dict],
    modelo: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    streaming: bool = True,
    al_recibir_texto: Callable[[str], None] | None = None,
    max_reintentos: int = 4,
    cliente=None,
    api_key: str | None = None,
    esfuerzo: str | None = "minimo",
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    cache: bool = False,
    **extra,
) -> ResultadoLLM:
    """💰 CONSUME DINERO. Llama a un modelo de OpenAI con la Responses API, con streaming y reintentos.

    Mismos parámetros que llamar_claude(), para poder cambiar de proveedor sin tocar el resto del código.
    Diferencias a tener en cuenta:
      - `max_tokens` se envía como `max_output_tokens` (OpenAI exige un mínimo de 16) y INCLUYE el
        razonamiento interno del modelo.
      - `temperature` y `top_p` solo los aceptan los modelos con el razonamiento apagado
        (esfuerzo="minimo"). OpenAI no tiene `top_k`: se ignora.
      - La caché de prompt es automática, así que `cache` no hace nada.
    Para usar otro servicio compatible con OpenAI (Groq, OpenRouter…) define OPENAI_BASE_URL.
    """
    if openai is None:
        raise RuntimeError("Falta instalar el paquete 'openai' (uv sync).")
    modelo = modelo or modelo_economico("openai")
    if cliente is None:
        if api_key is None:
            _exigir_api_key("openai")
        cliente = openai.OpenAI(api_key=api_key, max_retries=0) if api_key else openai.OpenAI(max_retries=0)

    entrada = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    params: dict = {"model": modelo, "input": entrada, "max_output_tokens": max(16, max_tokens), **extra}
    if system:
        params["instructions"] = system
    if esfuerzo and _openai_es_razonador(modelo):
        params["reasoning"] = {**params.get("reasoning", {}), "effort": ESFUERZO_OPENAI[esfuerzo]}
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p

    def una_llamada() -> ResultadoLLM:
        t0 = time.perf_counter()
        ttft = None
        if streaming:
            partes: list[str] = []
            with cliente.responses.stream(**params) as stream:
                for evento in stream:
                    if evento.type == "response.output_text.delta":
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        partes.append(evento.delta)
                        if al_recibir_texto:
                            al_recibir_texto(evento.delta)
                final = stream.get_final_response()
            texto = "".join(partes)
        else:
            final = cliente.responses.create(**params)
            texto = final.output_text
        total = time.perf_counter() - t0
        u = final.usage
        detalle = getattr(u, "input_tokens_details", None)
        leidos = getattr(detalle, "cached_tokens", 0) or 0
        escritos = getattr(detalle, "cache_write_tokens", 0) or 0
        # Normalizamos el motivo de parada al vocabulario común.
        if final.status == "incomplete":
            motivo = getattr(final.incomplete_details, "reason", None) or "incomplete"
            parada = "max_tokens" if motivo == "max_output_tokens" else motivo
        else:
            parada = "end_turn"
        return ResultadoLLM(
            texto=texto,
            modelo=modelo,
            tokens_entrada=u.input_tokens,
            tokens_salida=u.output_tokens,
            segundos_total=total,
            ttft=ttft,
            stop_reason=parada,
            costo=costo_usd(modelo, u.input_tokens, u.output_tokens),
            tokens_cache_leidos=leidos,
            tokens_cache_escritos=escritos,
            proveedor="openai",
        )

    resultado, intentos = con_reintentos(una_llamada, max_reintentos=max_reintentos)
    resultado.intentos = intentos
    return resultado


# ───────────────────────── Función común: elige el proveedor por ti ─────────────────────────


def llamar_llm(prompt: str | list[dict], proveedor: str | None = None, modelo: str | None = None, **kwargs) -> ResultadoLLM:
    """💰 CONSUME DINERO. Llama al proveedor activo (LLM_PROVIDER) o al indicado en `proveedor`.

    Es la función que usa casi todo el curso. Acepta los mismos parámetros que llamar_claude() y
    llamar_openai(): system, max_tokens, streaming, al_recibir_texto, esfuerzo, temperature,
    top_p, top_k, cache, max_reintentos, api_key…
    """
    p = proveedor or proveedor_activo()
    if p == "anthropic":
        return llamar_claude(prompt, modelo=modelo, **kwargs)
    if p == "openai":
        return llamar_openai(prompt, modelo=modelo, **kwargs)
    raise ValueError(f"Proveedor desconocido: {p}")


def contar_tokens_exacto(texto: str, proveedor: str | None = None, modelo: str | None = None) -> int:
    """Cuenta los tokens de entrada de `texto` con el servicio del proveedor (sin generar texto).

    - Anthropic: `messages.count_tokens`, un endpoint GRATUITO (con límite de peticiones).
    - OpenAI: `responses.input_tokens.count`. No genera texto; confirma en la página de precios
      de OpenAI si tiene algún costo.
    Incluye unos pocos tokens de estructura del mensaje (roles y delimitadores).
    """
    p = proveedor or proveedor_activo()
    _exigir_api_key(p)
    modelo = modelo or modelo_economico(p)
    if p == "anthropic":
        r = anthropic.Anthropic().messages.count_tokens(model=modelo, messages=[{"role": "user", "content": texto}])
        return r.input_tokens
    return openai.OpenAI().responses.input_tokens.count(model=modelo, input=texto).input_tokens


# ───────────────────── Salida estructurada nativa (cada proveedor a su manera) ─────────────────────

M = TypeVar("M", bound=BaseModel)


def pedir_estructurado_nativo(
    prompt: str, modelo_datos: type[M], proveedor: str | None = None, modelo: str | None = None,
    max_tokens: int = 1024, esfuerzo: str | None = "minimo",
) -> tuple[M, ResultadoLLM]:
    """💰 CONSUME DINERO. Salida estructurada NATIVA: el proveedor garantiza el esquema.

    - Anthropic: `messages.parse(output_format=Modelo)` → `parsed_output`.
    - OpenAI: `responses.parse(text_format=Modelo)` → `output_parsed`.
    Depende del proveedor y del modelo. Aun así, valida lo que importe para tu negocio.
    Devuelve (objeto_validado, ResultadoLLM).
    """
    p = proveedor or proveedor_activo()
    _exigir_api_key(p)
    modelo = modelo or modelo_economico(p)
    t0 = time.perf_counter()
    if p == "anthropic":
        extra = {"output_config": {"effort": ESFUERZO_ANTHROPIC[esfuerzo]}} if esfuerzo and _claude_admite_esfuerzo(modelo) else {}
        r = anthropic.Anthropic(max_retries=0).messages.parse(
            model=modelo, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}],
            output_format=modelo_datos, **extra,
        )
        obj, t_in, t_out = r.parsed_output, r.usage.input_tokens, r.usage.output_tokens
    else:
        extra = {"reasoning": {"effort": ESFUERZO_OPENAI[esfuerzo]}} if esfuerzo and _openai_es_razonador(modelo) else {}
        r = openai.OpenAI(max_retries=0).responses.parse(
            model=modelo, input=[{"role": "user", "content": prompt}], text_format=modelo_datos,
            max_output_tokens=max(16, max_tokens), **extra,
        )
        obj, t_in, t_out = r.output_parsed, r.usage.input_tokens, r.usage.output_tokens
    resultado = ResultadoLLM(
        texto=obj.model_dump_json() if obj is not None else "", modelo=modelo, tokens_entrada=t_in, tokens_salida=t_out,
        segundos_total=time.perf_counter() - t0, costo=costo_usd(modelo, t_in, t_out), proveedor=p,
    )
    if obj is None:
        raise ValueError("El modelo no devolvió un objeto que cumpla el esquema.")
    return obj, resultado


# ───────────────────── Modelo open-weight local: LM Studio ─────────────────────
#
# Un modelo open-weight corre en TU computadora y se consume por HTTP en localhost.
# LM Studio (el programa que usamos en el curso) expone una API compatible con el
# formato de OpenAI (Chat Completions), así que se usa con el SDK `openai` apuntando a una URL local.
#
# Eso significa que llamar_local() sirve también para otros programas que hablen ese
# mismo formato: Ollama (http://localhost:11434/v1), llama-server de llama.cpp
# (http://localhost:8080/v1), etc. Solo hay que cambiar LOCAL_LLM_URL en el .env.

URL_LOCAL_POR_DEFECTO = "http://localhost:1234/v1"      # servidor local de LM Studio (el puerto puede variar)
MODELO_LOCAL_POR_DEFECTO = "llama-3.2-3b-instruct"      # nombre tal como lo muestra tu programa


def url_local() -> str:
    """URL del servidor local (variable de entorno LOCAL_LLM_URL, o la de LM Studio)."""
    return os.getenv("LOCAL_LLM_URL", URL_LOCAL_POR_DEFECTO)


def modelo_local() -> str:
    """Modelo local por defecto (variable de entorno LOCAL_LLM_MODEL)."""
    return os.getenv("LOCAL_LLM_MODEL", MODELO_LOCAL_POR_DEFECTO)


def _cliente_local(url: str | None = None):
    """Cliente OpenAI apuntando al servidor local."""
    return openai.OpenAI(
        base_url=url or url_local(),
        # El servidor local no valida la llave, pero el SDK exige un texto no vacío.
        api_key=os.getenv("LOCAL_LLM_API_KEY", "no-se-usa"),
        max_retries=0,   # los reintentos los hace con_reintentos(), como con los otros proveedores
        timeout=300,     # un modelo local sin GPU puede tardar bastante
    )


def llamar_local(
    prompt: str,
    modelo: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    al_recibir_texto: Callable[[str], None] | None = None,
    max_reintentos: int = 1,
    url: str | None = None,
    **parametros,
) -> ResultadoLLM:
    """Llama a un modelo open-weight que corre en TU computadora (LM Studio por defecto).

    No cuesta por token (el costo es tu hardware y tu electricidad), pero con 16 GB de
    RAM y sin GPU solo caben modelos pequeños. `parametros` se pasa tal cual a la API
    (temperature, top_p, seed…); al ser un modelo local, tú decides cuáles usar.

    Funciona con cualquier servidor compatible con OpenAI cambiando `url`
    (o la variable LOCAL_LLM_URL): LM Studio, Ollama, llama-server…
    """
    if openai is None:
        raise RuntimeError("Falta instalar el paquete 'openai' (uv sync).")
    cliente = _cliente_local(url)
    modelo = modelo or modelo_local()
    mensajes = []
    if system:
        mensajes.append({"role": "system", "content": system})
    mensajes.append({"role": "user", "content": prompt})

    def una_llamada() -> ResultadoLLM:
        t0 = time.perf_counter()
        ttft = None
        partes: list[str] = []
        tin = tout = 0
        parada = None
        try:
            flujo = cliente.chat.completions.create(
                model=modelo,
                messages=mensajes,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},   # pide el conteo de tokens al final del flujo
                **parametros,
            )
            for trozo in flujo:
                if trozo.choices:
                    texto = trozo.choices[0].delta.content or ""
                    if texto:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        partes.append(texto)
                        if al_recibir_texto:
                            al_recibir_texto(texto)
                    parada = trozo.choices[0].finish_reason or parada
                if getattr(trozo, "usage", None):   # último trozo: uso de tokens
                    tin = trozo.usage.prompt_tokens or 0
                    tout = trozo.usage.completion_tokens or 0
        except openai.APIConnectionError as exc:
            # Servidor apagado o URL equivocada: lo tratamos como error de conexión (reintentable).
            raise ConnectionError(f"No hay servidor local en {url or url_local()}") from exc
        return ResultadoLLM(
            texto="".join(partes),
            modelo=f"local:{modelo}",
            tokens_entrada=tin,   # 0 si tu servidor no informa el uso
            tokens_salida=tout,
            segundos_total=time.perf_counter() - t0,
            ttft=ttft,
            stop_reason={"length": "max_tokens", "stop": "end_turn"}.get(parada, parada),
            costo=0.0,
            proveedor="local",
        )

    resultado, intentos = con_reintentos(una_llamada, max_reintentos=max_reintentos)
    resultado.intentos = intentos
    return resultado


def local_disponible(modelo: str | None = None, url: str | None = None) -> tuple[bool, str]:
    """Revisa si el servidor local responde y (opcionalmente) si ofrece el modelo pedido."""
    destino = url or url_local()
    try:
        ids = [m.id for m in _cliente_local(url).models.list().data]
    except openai.APIConnectionError:
        return False, (
            f"No hay servidor local en {destino}. Abre LM Studio, ve a la pantalla Local Model API "
            "y activa Local API server (ver docs/LMSTUDIO.md)."
        )
    except Exception as exc:  # noqa: BLE001 - mensaje claro para cualquier otro fallo
        return False, f"El servidor local respondió con un error ({type(exc).__name__}): {exc}"
    if modelo and modelo not in ids:
        return False, (
            f"El modelo '{modelo}' no aparece en el servidor local. Disponibles: {', '.join(ids) or 'ninguno'}. "
            "Descárgalo en LM Studio o ajusta LOCAL_LLM_MODEL en tu .env."
        )
    return True, "ok"


# ───────────────────────── Salida estructurada portable (funciona con cualquier modelo) ─────────────────────────


def extraer_json(texto: str) -> str:
    """Quita cercas de código (```json … ```) y texto alrededor del primer objeto JSON."""
    limpio = texto.strip()
    cerca = re.search(r"```(?:json)?\s*(.*?)```", limpio, re.DOTALL)
    if cerca:
        limpio = cerca.group(1).strip()
    inicio, fin = limpio.find("{"), limpio.rfind("}")
    return limpio[inicio : fin + 1] if inicio != -1 and fin > inicio else limpio


def pedir_json(
    llamar: Callable[[str], ResultadoLLM],
    prompt: str,
    modelo_datos: type[M],
    max_intentos: int = 3,
) -> tuple[M, list[ResultadoLLM]]:
    """Pide JSON, lo valida contra un modelo pydantic y, si falla, reintenta explicando el error.

    Los tres pasos de la salida estructurada:
      1. pides JSON (o un esquema),
      2. validas al recibirlo,
      3. si falla, reintentas con el error o fallas con un mensaje claro.

    `llamar` es cualquier función texto → ResultadoLLM, así que sirve igual con Claude,
    OpenAI o un modelo local. Devuelve (objeto_validado, todas_las_llamadas_hechas).
    """
    esquema = json.dumps(modelo_datos.model_json_schema(), ensure_ascii=False)
    instruccion = (
        f"{prompt}\n\nResponde ÚNICAMENTE con un objeto JSON válido que cumpla este esquema, "
        f"sin texto adicional ni cercas de código:\n{esquema}"
    )
    llamadas: list[ResultadoLLM] = []
    ultimo_error = ""
    for intento in range(1, max_intentos + 1):
        texto_prompt = instruccion if intento == 1 else (
            f"{instruccion}\n\nTu respuesta anterior fue inválida: {ultimo_error}\nCorrígela."
        )
        r = llamar(texto_prompt)
        llamadas.append(r)
        try:
            return modelo_datos.model_validate_json(extraer_json(r.texto)), llamadas
        except ValidationError as exc:
            ultimo_error = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    raise ValueError(f"No se obtuvo JSON válido tras {max_intentos} intentos. Último error: {ultimo_error}")
