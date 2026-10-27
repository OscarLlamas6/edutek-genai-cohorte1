"""El "sistema de monitoreo" simulado de Andes Fibra: el mundo que el agente puede consultar y tocar.

En un caso real, estas funciones llamarían a una API de estado (Statuspage, Prometheus, el CMDB de
tu empresa…). Aquí leen un archivo JSON fijo (datos/servicios.json), así que las respuestas son
siempre las mismas: eso permite comparar modelos y repetir experimentos.

Este módulo NO sabe nada de LLM ni de LangChain. Son funciones normales de Python. Las "herramientas"
del agente (herramientas.py) y el bucle hecho a mano (agente_manual.py) las envuelven; así ves que una
herramienta no es magia: es una función tuya que el modelo puede pedir que ejecutes.

Tres ideas de diseño que verás en la sesión:
    1. Los errores esperados (un servicio que no existe) se devuelven como TEXTO legible que dice
       qué hacer, en vez de lanzar una excepción: el modelo puede leerlo y corregirse.
    2. Las funciones que LEEN (listar, estado) no tienen efectos; la que ESCRIBE (crear_ticket)
       es idempotente: pedirla dos veces con lo mismo no crea dos tickets.
    3. Nada aquí cuesta dinero.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

RUTA_SERVICIOS = Path(__file__).resolve().parent / "datos" / "servicios.json"

ESTADOS = ("operativo", "degradado", "caido", "mantenimiento")

# Los tickets "creados" viven en memoria mientras dura el programa (es una simulación).
TICKETS: dict[str, dict] = {}


def _normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y con guiones: 'App móvil' → 'app-movil'."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return "-".join(sin_tildes.lower().replace("·", " ").replace("-", " ").split())


def cargar_servicios() -> list[dict]:
    """Lee el archivo de servicios (siempre fresco: así puedes editarlo y volver a preguntar)."""
    return json.loads(RUTA_SERVICIOS.read_text(encoding="utf-8"))["servicios"]


def ids_validos() -> list[str]:
    return [s["id"] for s in cargar_servicios()]


def buscar_servicio(texto: str) -> dict | None:
    """Encuentra un servicio por su id o por su nombre, tolerando mayúsculas y tildes.

    Se aceptan coincidencias parciales ("app" encuentra "app-movil") solo si son únicas: si hay
    ambigüedad devolvemos None y la herramienta le pide al modelo que sea más específico.
    """
    buscado = _normalizar(texto)
    servicios = cargar_servicios()
    for s in servicios:
        if buscado in (s["id"], _normalizar(s["nombre"])):
            return s
    parciales = [s for s in servicios if buscado and (buscado in s["id"] or buscado in _normalizar(s["nombre"]))]
    return parciales[0] if len(parciales) == 1 else None


# ───────────────────────── Lectura (sin efectos) ─────────────────────────


def listar() -> str:
    """Una línea por servicio: identificador, nombre y estado."""
    return "\n".join(f"- {s['id']} ({s['nombre']}): {s['estado']}" for s in cargar_servicios())


def estado(servicio: str) -> str:
    """Estado detallado de UN servicio, como JSON; o un mensaje de error que explica cómo corregir."""
    s = buscar_servicio(servicio)
    if s is None:
        return (
            f"Error: no encontré un servicio llamado '{servicio}' (o el nombre es ambiguo). "
            f"Identificadores válidos: {', '.join(ids_validos())}. "
            "Vuelve a llamar con uno de ellos."
        )
    return json.dumps(s, ensure_ascii=False)


# ───────────────────────── Escritura (con efectos) ─────────────────────────


def crear_ticket(servicio: str, descripcion: str) -> str:
    """Registra un ticket de falla (SIMULADO: solo se guarda en memoria).

    Es IDEMPOTENTE: el id del ticket sale de un hash del servicio y la descripción, así que si el
    modelo repite la llamada (los agentes reintentan, y los modelos se repiten) no se duplican tickets.
    """
    s = buscar_servicio(servicio)
    if s is None:
        return f"Error: no encontré el servicio '{servicio}'. Identificadores válidos: {', '.join(ids_validos())}."
    if len(descripcion.strip()) < 10:
        return "Error: la descripción es demasiado corta. Describe qué ve el cliente y desde cuándo."
    ticket_id = "TCK-" + hashlib.sha1(f"{s['id']}|{descripcion.strip().lower()}".encode()).hexdigest()[:6].upper()
    if ticket_id in TICKETS:
        return f"Ya existe el ticket {ticket_id} para este reporte; no se creó otro."
    TICKETS[ticket_id] = {"servicio": s["id"], "descripcion": descripcion.strip()}
    return f"Ticket {ticket_id} creado para '{s['nombre']}'."


def reiniciar_tickets() -> None:
    """Borra los tickets simulados (útil entre experimentos)."""
    TICKETS.clear()
