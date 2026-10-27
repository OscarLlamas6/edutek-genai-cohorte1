"""Plantillas de prompt versionadas.

Los prompts viven en archivos (`prompts/<nombre>/<version>.j2`), no en el código ni
en un chat. Así se versionan en Git, se revisan en un pull request y se prueban.

    from plantillas import renderizar, versiones
    versiones("clasificador_tickets")                      # ['v1', 'v2', 'v3']
    renderizar("clasificador_tickets", "v2", ticket="...", categorias=[...])

Usa Jinja2 con StrictUndefined: si falta una variable, falla en voz alta en lugar de
enviar un prompt a medias al modelo (y pagar por él).
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

RAIZ_PROMPTS = Path(__file__).resolve().parent / "prompts"

_entorno = Environment(
    loader=FileSystemLoader(RAIZ_PROMPTS),
    undefined=StrictUndefined,
    keep_trailing_newline=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def versiones(nombre: str) -> list[str]:
    """Versiones disponibles de un prompt, ordenadas."""
    return sorted(p.stem for p in (RAIZ_PROMPTS / nombre).glob("v*.j2"))


def renderizar(nombre: str, version: str, **variables) -> str:
    """Carga `prompts/<nombre>/<version>.j2` y rellena sus variables."""
    return _entorno.get_template(f"{nombre}/{version}.j2").render(**variables).strip()
