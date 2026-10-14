"""Práctica S1 · Calculadora de tokens y costo.

PREGUNTA QUE RESPONDE
    ¿Cuántos tokens tiene este texto y cuánto costaría enviarlo a un modelo?

QUÉ HACE
    1. Lee un texto (o un archivo de texto).
    2. Cuenta sus tokens con dos o más tokenizadores distintos.
    3. Estima el costo de una llamada a la API con los precios del modelo elegido.
    4. Opcionalmente, pide a tu proveedor (Anthropic u OpenAI, según LLM_PROVIDER en tu .env)
       el conteo EXACTO de tokens.

CÓMO EJECUTARLO (desde la carpeta s01-fundamentos-llm/)
    uv run python practica/calculadora_tokens.py datos/ticket_es.txt
    uv run python practica/calculadora_tokens.py "Hola mundo" --salida 500
    uv run python practica/calculadora_tokens.py datos/ticket_en.txt --modelo claude-opus-5-5
    uv run python practica/calculadora_tokens.py datos/ticket_en.txt --modelo gpt-6-luna
    uv run python practica/calculadora_tokens.py datos/ticket_es.txt --tokenizadores o200k cl100k qwen
    uv run python practica/calculadora_tokens.py datos/ticket_es.txt --exacto

💰 COSTO
    Por defecto NO consume dinero: todo se calcula en tu computadora.
    La opción --exacto llama al endpoint de conteo de tokens de tu proveedor. No genera
    texto. En Anthropic es GRATUITO (con límite de peticiones por minuto); en OpenAI,
    confirma en su página de precios si se factura. Requiere tu API key.

IDEAS PARA EXPERIMENTAR (ver practica/README.md)
    - Compara el ticket en español con el de inglés: ¿cuánto más cuesta?
    - Prueba con un log, un JSON o código tuyo (datos de ejemplo, nunca reales).
"""

import argparse
import sys
from pathlib import Path

import tiktoken
from dotenv import find_dotenv, load_dotenv

# El módulo del curso que habla con los proveedores (está en esta misma carpeta de sesión).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cliente_llm import PRECIOS, contar_tokens_exacto, modelo_economico, proveedor_activo  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Precios
# ─────────────────────────────────────────────────────────────────────────────
# La tabla PRECIOS (USD por MILLÓN de tokens, como (entrada, salida)) vive en cliente_llm.py y
# incluye modelos de Anthropic y de OpenAI. Referencia a octubre de 2026: los precios cambian,
# confirma siempre en la página oficial del proveedor antes de presupuestar.
# Observa que los tokens de SALIDA cuestan varias veces más que los de entrada.

# Tokenizadores disponibles. Cada uno parte el texto de forma distinta.
#   o200k  → tiktoken, familia de modelos recientes de OpenAI
#   cl100k → tiktoken, generación anterior de OpenAI
#   qwen   → Hugging Face `tokenizers`, modelo open-weight (Qwen2.5)
TOKENIZADORES = ("o200k", "cl100k", "qwen")


# ─────────────────────────────────────────────────────────────────────────────
# Funciones
# ─────────────────────────────────────────────────────────────────────────────
def leer_texto(entrada: str) -> str:
    """Devuelve el texto a medir.

    Si `entrada` es la ruta de un archivo que existe, lee su contenido; si no,
    asume que `entrada` ya es el texto. Así el mismo argumento sirve para ambos.
    """
    ruta = Path(entrada)
    return ruta.read_text(encoding="utf-8") if ruta.is_file() else entrada


def contar_tokens(nombre: str, texto: str) -> int:
    """Cuenta los tokens de `texto` con el tokenizador `nombre`.

    Todo esto se ejecuta en local. La PRIMERA vez, cada librería descarga la tabla
    de su tokenizador (unos MB) y la guarda en caché.
    """
    if nombre == "o200k":
        # get_encoding devuelve el tokenizador; .encode() convierte texto → lista de ids.
        # La cantidad de tokens es simplemente el largo de esa lista.
        return len(tiktoken.get_encoding("o200k_base").encode(texto))

    if nombre == "cl100k":
        return len(tiktoken.get_encoding("cl100k_base").encode(texto))

    if nombre == "qwen":
        # Se importa aquí (y no arriba) para no pagar su tiempo de carga si no se usa.
        # Es normal ver un aviso sobre "unauthenticated requests to the HF Hub":
        # es inofensivo, solo indica que no usamos un token de Hugging Face.
        from tokenizers import Tokenizer

        return len(Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B").encode(texto).ids)

    raise ValueError(f"Tokenizador desconocido: {nombre}")


def contar_tokens_proveedor(modelo: str, texto: str) -> int:
    """Conteo EXACTO de tokens de entrada con el servicio de tu proveedor (no genera texto).

    Anthropic no publica su tokenizador como librería local, así que tiktoken y Qwen son solo
    APROXIMACIONES para Claude. Este endpoint da la cifra real: es el mismo número que luego verás
    como tokens de entrada en una respuesta. Para modelos de OpenAI tiktoken es el tokenizador
    oficial, pero el servicio de conteo evita dudas sobre el encoding de cada modelo.
    """
    try:
        return contar_tokens_exacto(texto, modelo=modelo)
    except RuntimeError as error:   # falta la API key del proveedor
        sys.exit(str(error))


def estimar_costo(modelo: str, tokens_entrada: int, tokens_salida: int) -> float:
    """Costo estimado de UNA llamada, en dólares.

    costo = (tokens_entrada × precio_entrada + tokens_salida × precio_salida) / 1 000 000

    Se divide entre un millón porque los precios están expresados por millón de tokens.
    """
    precio_entrada, precio_salida = PRECIOS[modelo]
    return (tokens_entrada * precio_entrada + tokens_salida * precio_salida) / 1_000_000


def imprimir_tabla(texto: str, conteos: dict[str, int], modelo: str, tokens_salida: int) -> None:
    """Muestra una fila por tokenizador con tokens, caracteres por token y costo."""
    print(f"\nTexto: {len(texto)} caracteres · {len(texto.split())} palabras")
    print(f"Modelo para el costo: {modelo} · salida estimada: {tokens_salida} tokens\n")

    print(f"{'Tokenizador':<28}{'Tokens':>8}{'Chars/token':>13}{'Costo USD':>14}")
    print("-" * 63)
    for nombre, n_tokens in conteos.items():
        # "Caracteres por token" es la métrica de eficiencia: cuanto MÁS BAJO, más
        # tokens necesita el tokenizador para el mismo texto (y más cuesta).
        chars_por_token = len(texto) / n_tokens
        costo = estimar_costo(modelo, n_tokens, tokens_salida)
        print(f"{nombre:<28}{n_tokens:>8}{chars_por_token:>13.2f}{costo:>14.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Programa principal
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Calculadora de tokens y costo")
    ap.add_argument("texto", help="Texto a medir, o ruta a un archivo de texto")
    ap.add_argument("--modelo", default=None, choices=sorted(PRECIOS),
                    help="Modelo cuyos precios se usan para estimar el costo (por defecto, el económico de tu proveedor)")
    ap.add_argument("--salida", type=int, default=300,
                    help="Tokens de salida que esperas recibir (la respuesta también se cobra)")
    ap.add_argument("--tokenizadores", nargs="+", default=["o200k", "cl100k"], choices=TOKENIZADORES,
                    help="Tokenizadores locales con los que comparar")
    ap.add_argument("--exacto", action="store_true",
                    help="Añade el conteo exacto del servicio de tu proveedor (requiere API key)")
    args = ap.parse_args()

    # Carga el .env de la raíz del repo (LLM_PROVIDER y la API key; la llave solo se necesita con --exacto).
    load_dotenv(find_dotenv(usecwd=True))

    modelo = args.modelo or modelo_economico()
    texto = leer_texto(args.texto)

    # Un conteo por tokenizador. Usamos un diccionario {nombre: tokens} para imprimirlo luego.
    conteos = {nombre: contar_tokens(nombre, texto) for nombre in args.tokenizadores}

    if args.exacto:
        conteos[f"{proveedor_activo()} ({modelo})"] = contar_tokens_proveedor(modelo, texto)

    imprimir_tabla(texto, conteos, modelo, args.salida)

    print(
        "\nNota: el conteo con tiktoken/Qwen es una APROXIMACIÓN (para Claude, siempre). "
        "Para el número exacto de tu proveedor usa --exacto."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
