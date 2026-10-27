"""Las herramientas del bot de Andes Fibra, escritas con el decorador `@tool` de LangChain.

Una HERRAMIENTA es una función de Python que el modelo puede pedir que ejecutes. El modelo nunca
ejecuta nada: solo ve el NOMBRE, la DESCRIPCIÓN (el docstring) y los PARÁMETROS (los tipos) de cada
herramienta, y decide si pedir una llamada y con qué argumentos. Tu código la ejecuta y le devuelve
el resultado. Por eso estas tres cosas son, en la práctica, "el prompt" de la herramienta:

    nombre       verbo + objeto, en snake_case:  estado_servicio, no  api2  ni  herramienta_1
    docstring    QUÉ hace y CUÁNDO usarla (y cuándo no). Es lo que el modelo lee para elegir.
    tipos        str, int, Literal[...]: se convierten en el esquema JSON de los argumentos.

Tres herramientas LEEN y una ESCRIBE:

    listar_servicios      leer   estado de todos los servicios (una línea por servicio)
    estado_servicio       leer   detalle de UN servicio
    consultar_manual      leer   RAG: busca en el manual de soporte (reutiliza la Sesión 5)
    reportar_falla        ESCRIBE crea un ticket (simulado). Tiene efectos: se diseña con más cuidado

💰 COSTO: las herramientas no llaman a ningún LLM. `consultar_manual` usa embeddings (gratis con el
modelo local). Quien consume dinero es el modelo que decide usarlas, no las herramientas.
"""

from __future__ import annotations

from pathlib import Path

from langchain.tools import tool

import servicios
from almacenes import AlmacenMemoria
from rag import ingestar, recuperar

RUTA_MANUAL = Path(__file__).resolve().parent / "datos" / "manual_soporte.md"

_almacen_manual: AlmacenMemoria | None = None


def preparar_manual() -> int:
    """Ingesta el manual en un almacén en memoria (una sola vez por programa). Devuelve cuántos fragmentos hay.

    Es la misma ingesta de la Sesión 5 (fragmentar → vectorizar → guardar). Se llama de forma
    perezosa: el primer `consultar_manual` la dispara. Llámala antes si quieres que la primera
    pregunta no espere.
    """
    global _almacen_manual
    if _almacen_manual is None:
        almacen = AlmacenMemoria()
        ingestar(almacen, RUTA_MANUAL, estrategia="oraciones", max_palabras=70, solape=1)
        _almacen_manual = almacen
    return _almacen_manual.contar()


@tool(parse_docstring=True)
def listar_servicios() -> str:
    """Lista TODOS los servicios de Andes Fibra con su estado actual (operativo, degradado, caido o mantenimiento).

    Úsala cuando el usuario pregunte por el estado en general ("¿hay fallas?", "¿qué está caído?")
    o cuando no diga de qué servicio habla. Para el detalle de uno solo, usa estado_servicio.
    """
    return servicios.listar()


@tool(parse_docstring=True)
def estado_servicio(servicio: str) -> str:
    """Consulta el estado actual y el detalle de UN servicio de Andes Fibra.

    Úsala cuando el usuario pregunte por un servicio concreto: portal Mi Andes, app móvil, pagos en
    línea, chat de soporte, línea telefónica o la red de fibra de una zona.

    Args:
        servicio: identificador o nombre del servicio. Identificadores válidos: portal-mi-andes,
            app-movil, pagos-en-linea, chat-soporte, telefonia-soporte, red-zona-norte, red-zona-sur.
    """
    return servicios.estado(servicio)


@tool(parse_docstring=True)
def consultar_manual(pregunta: str) -> str:
    """Busca en el manual de soporte de Andes Fibra la respuesta a una pregunta sobre políticas y procedimientos.

    Úsala para preguntas sobre planes y precios, facturación y pagos, instalación, garantía,
    compensaciones, cambios de plan, cancelación, horarios de atención y seguridad. NO sirve para
    saber si un servicio está funcionando ahora mismo: para eso usa estado_servicio.

    Args:
        pregunta: la pregunta en lenguaje natural, con el mayor contexto posible.
    """
    try:
        preparar_manual()
        resultados = recuperar(_almacen_manual, pregunta, k=3)
    except Exception as error:   # embeddings apagados, modelo sin cargar…
        return f"Error: el manual no está disponible ahora ({type(error).__name__}). Dile al usuario que lo intente más tarde."
    if not resultados:
        return "El manual no contiene información sobre esa pregunta."
    return "\n\n".join(f"[Sección: {r.metadatos['seccion']}] {r.texto}" for r in resultados)


@tool(parse_docstring=True)
def reportar_falla(servicio: str, descripcion: str) -> str:
    """Crea un ticket de falla en el sistema de soporte (ACCIÓN QUE ESCRIBE: deja un registro real).

    Úsala SOLO si el usuario pide explícitamente reportar o abrir un ticket de una falla. No la uses
    para consultar el estado. Si el usuario no dio detalles suficientes, pídeselos antes de llamarla.

    Args:
        servicio: identificador del servicio afectado (por ejemplo: pagos-en-linea).
        descripcion: qué ve el cliente y desde cuándo (mínimo una frase completa).
    """
    return servicios.crear_ticket(servicio, descripcion)


# Conjuntos listos para usar. `SOLO_LECTURA` es el mínimo privilegio: sin herramientas que escriban.
SOLO_LECTURA = [listar_servicios, estado_servicio, consultar_manual]
TODAS = SOLO_LECTURA + [reportar_falla]
