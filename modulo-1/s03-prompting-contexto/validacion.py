"""Validación de la salida del modelo.

El modelo devuelve texto; tu programa necesita datos. Este módulo convierte lo uno en
lo otro y deja claro cuándo la respuesta no es utilizable.

JSON válido no significa JSON correcto: aquí también se valida el CONTENIDO
(categorías y prioridades permitidas), no solo la sintaxis.
"""

from typing import Literal

from pydantic import BaseModel, ValidationError

from cliente_llm import extraer_json

CATEGORIAS = ["red", "hardware", "software", "accesos", "otro"]
PRIORIDADES = ["baja", "media", "alta", "critica"]


class ClasificacionTicket(BaseModel):
    categoria: Literal["red", "hardware", "software", "accesos", "otro"]
    prioridad: Literal["baja", "media", "alta", "critica"]
    resumen: str


class SalidaInvalida(ValueError):
    """La respuesta del modelo no cumple el esquema esperado."""


def validar_clasificacion(texto: str) -> ClasificacionTicket:
    """Convierte la respuesta del modelo en un objeto validado o lanza SalidaInvalida."""
    try:
        return ClasificacionTicket.model_validate_json(extraer_json(texto))
    except ValidationError as exc:
        detalle = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'respuesta'}: {e['msg']}" for e in exc.errors())
        raise SalidaInvalida(detalle) from exc
