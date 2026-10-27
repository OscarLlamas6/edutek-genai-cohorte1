"""Almacenes vectoriales: tres formas de guardar y buscar embeddings con la MISMA interfaz.

Un ALMACÉN (o base de datos vectorial) guarda vectores junto con su texto y sus metadatos,
y responde a la pregunta "¿cuáles son los K vectores más parecidos a este?".

    memoria   Una matriz de numpy en la RAM. Sin cuenta, sin red, sin persistencia. Sirve para
              aprender y como plan B. Compara el vector de la consulta con TODOS (búsqueda exacta).
    neon      PostgreSQL en la nube (Neon) con la extensión pgvector. Es una base relacional de
              siempre + una columna de tipo `vector`. Se consulta con SQL.
    pinecone  Una base de datos vectorial administrada (Pinecone), creada solo para esto.
              Se usa con su SDK: índices, namespaces, upsert y query.

Todos exponen lo mismo, así que el resto del código no cambia al cambiar de almacén:

    almacen = crear_almacen("neon")            # o "pinecone" o "memoria"
    almacen.preparar(dimension)                # crea la tabla / el índice si no existe
    almacen.insertar([Fragmento(...), ...])    # guarda (o actualiza) fragmentos
    almacen.buscar(vector_de_la_consulta, k=3) # devuelve los K más parecidos
    almacen.contar()                           # cuántos fragmentos hay
    almacen.vaciar()                           # borra los datos, conserva la estructura
    almacen.eliminar()                         # borra también la tabla / el índice

El puntaje de cada resultado es la SIMILITUD COSENO (de -1 a 1; mientras más alto, más parecido),
igual en los tres almacenes.

CONFIGURACIÓN (variables de entorno, en tu .env):
    NEON_DATABASE_URL     cadena de conexión de Neon (postgresql://usuario:clave@host/base?sslmode=require)
    NEON_TABLA            nombre de la tabla (por defecto "fragmentos")
    PINECONE_API_KEY      tu API key de Pinecone
    PINECONE_INDEX        nombre del índice (por defecto "edutek-cohorte1")
    PINECONE_NAMESPACE    namespace dentro del índice (por defecto "edutek")
    PINECONE_CLOUD / PINECONE_REGION   donde se crea el índice (por defecto aws / us-east-1,
                                       la única región del plan gratuito)

💰 COSTO: con los planes gratuitos de Neon y Pinecone no se cobra dinero, pero SÍ se consume cuota
(almacenamiento, horas de cómputo, unidades de lectura y escritura). Mira docs/CUENTA_NEON.md y
docs/CUENTA_PINECONE.md.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import numpy as np

NOMBRES_ALMACEN = ("memoria", "neon", "pinecone")


@dataclass
class Fragmento:
    """Un trozo de texto con su vector y sus metadatos (de dónde viene, qué sección es…)."""

    id: str
    texto: str
    metadatos: dict = field(default_factory=dict)
    vector: list[float] | None = None


@dataclass
class Resultado:
    """Un resultado de búsqueda: el fragmento y qué tan parecido es a la consulta (coseno)."""

    id: str
    texto: str
    metadatos: dict
    puntaje: float


class Almacen:
    """Interfaz común. Cada almacén concreto implementa estos métodos."""

    nombre = "base"

    def preparar(self, dimension: int) -> None:
        raise NotImplementedError

    def insertar(self, fragmentos: list[Fragmento]) -> None:
        raise NotImplementedError

    def buscar(self, vector: list[float], k: int = 3, filtro: dict | None = None) -> list[Resultado]:
        raise NotImplementedError

    def contar(self) -> int:
        raise NotImplementedError

    def vaciar(self) -> None:
        raise NotImplementedError

    def eliminar(self) -> None:
        raise NotImplementedError


# ═════════════════════════ 1. Memoria (numpy) ═════════════════════════


class AlmacenMemoria(Almacen):
    """Todo en la RAM con numpy. Búsqueda EXACTA: compara la consulta contra todos los vectores."""

    nombre = "memoria"

    def __init__(self):
        self._fragmentos: list[Fragmento] = []
        self._matriz: np.ndarray | None = None   # vectores normalizados, una fila por fragmento

    def preparar(self, dimension: int) -> None:
        self._dimension = dimension

    def insertar(self, fragmentos: list[Fragmento]) -> None:
        por_id = {f.id: i for i, f in enumerate(self._fragmentos)}
        for f in fragmentos:
            if f.id in por_id:                    # mismo id: se actualiza (como un upsert)
                self._fragmentos[por_id[f.id]] = f
            else:
                self._fragmentos.append(f)
        m = np.array([f.vector for f in self._fragmentos], dtype=np.float32)
        norma = np.linalg.norm(m, axis=1, keepdims=True)
        norma[norma == 0] = 1.0
        self._matriz = m / norma                  # con vectores de longitud 1, coseno = producto punto

    def buscar(self, vector: list[float], k: int = 3, filtro: dict | None = None) -> list[Resultado]:
        if self._matriz is None:
            return []
        q = np.array(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        puntajes = self._matriz @ q               # una multiplicación matriz × vector: todos los cosenos de una vez
        orden = np.argsort(-puntajes)
        salida = []
        for i in orden:
            f = self._fragmentos[i]
            if filtro and any(f.metadatos.get(c) != v for c, v in filtro.items()):
                continue
            salida.append(Resultado(f.id, f.texto, f.metadatos, float(puntajes[i])))
            if len(salida) == k:
                break
        return salida

    def contar(self) -> int:
        return len(self._fragmentos)

    def vaciar(self) -> None:
        self._fragmentos, self._matriz = [], None

    def eliminar(self) -> None:
        self.vaciar()


# ═════════════════════════ 2. Neon (PostgreSQL + pgvector) ═════════════════════════


def _identificador_seguro(nombre: str) -> str:
    """Los nombres de tabla no se pueden pasar como parámetros SQL: los validamos para evitar inyección SQL."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", nombre):
        raise ValueError(f"Nombre de tabla no válido: {nombre!r} (usa letras, números y guion bajo)")
    return nombre


class AlmacenNeon(Almacen):
    """PostgreSQL de Neon con pgvector. Los fragmentos viven en una tabla con una columna `vector(N)`."""

    nombre = "neon"

    def __init__(self, url: str | None = None, tabla: str | None = None):
        self.url = url or os.getenv("NEON_DATABASE_URL")
        if not self.url:
            raise RuntimeError(
                "Falta NEON_DATABASE_URL. Crea tu proyecto en Neon y copia la cadena de conexión en tu .env "
                "(ver docs/CUENTA_NEON.md)."
            )
        self.tabla = _identificador_seguro(tabla or os.getenv("NEON_TABLA", "fragmentos"))
        self._conn = None

    # ---- conexión ----
    def _conexion(self):
        """Abre la conexión (una sola, reutilizable). Neon suspende el cómputo tras 5 min sin uso: la
        primera conexión después de una pausa puede tardar unos segundos mientras se despierta."""
        if self._conn is None or self._conn.closed:
            import psycopg
            from pgvector.psycopg import register_vector

            self._conn = psycopg.connect(self.url, autocommit=True, connect_timeout=30)
            # La extensión debe existir ANTES de registrar el tipo `vector` en el cliente.
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self._conn)
        return self._conn

    def cerrar(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

    # ---- interfaz ----
    def preparar(self, dimension: int) -> None:
        c = self._conexion()
        c.execute(
            f"CREATE TABLE IF NOT EXISTS {self.tabla} ("
            "  id text PRIMARY KEY,"
            "  texto text NOT NULL,"
            "  metadatos jsonb NOT NULL DEFAULT '{}',"
            f"  embedding vector({int(dimension)}) NOT NULL"
            ")"
        )
        # Si la tabla ya existía con otra dimensión (otro modelo de embeddings), avisamos con claridad.
        existente = c.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding'",
            (self.tabla,),
        ).fetchone()[0]
        if existente != dimension:
            raise ValueError(
                f"La tabla '{self.tabla}' ya existe con vectores de {existente} dimensiones, pero tu modelo "
                f"de embeddings produce {dimension}. Usa otra tabla (NEON_TABLA) o bórrala con almacen.eliminar()."
            )

    def insertar(self, fragmentos: list[Fragmento]) -> None:
        from psycopg.types.json import Jsonb

        filas = [
            (f.id, f.texto, Jsonb(f.metadatos), np.array(f.vector, dtype=np.float32)) for f in fragmentos
        ]
        # ON CONFLICT ... DO UPDATE: si el id ya existe, se actualiza (por eso reindexar es seguro).
        with self._conexion().cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.tabla} (id, texto, metadatos, embedding) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET texto = EXCLUDED.texto, "
                "metadatos = EXCLUDED.metadatos, embedding = EXCLUDED.embedding",
                filas,
            )

    def buscar(self, vector: list[float], k: int = 3, filtro: dict | None = None) -> list[Resultado]:
        from psycopg.types.json import Jsonb

        q = np.array(vector, dtype=np.float32)
        donde, params = "", [q]
        if filtro:
            donde = "WHERE metadatos @> %s"          # @> = "contiene": los metadatos incluyen estos pares clave/valor
            params.append(Jsonb(filtro))
        params += [q, k]
        # <=> es la DISTANCIA coseno de pgvector (0 = idénticos). La similitud es 1 - distancia.
        filas = self._conexion().execute(
            f"SELECT id, texto, metadatos, 1 - (embedding <=> %s) AS similitud FROM {self.tabla} "
            f"{donde} ORDER BY embedding <=> %s LIMIT %s",
            params,
        ).fetchall()
        return [Resultado(i, t, m, float(s)) for i, t, m, s in filas]

    def contar(self) -> int:
        return self._conexion().execute(f"SELECT count(*) FROM {self.tabla}").fetchone()[0]

    def vaciar(self) -> None:
        self._conexion().execute(f"TRUNCATE {self.tabla}")

    def eliminar(self) -> None:
        self._conexion().execute(f"DROP TABLE IF EXISTS {self.tabla}")

    # ---- extras didácticos ----
    def crear_indice_hnsw(self) -> None:
        """Crea un índice HNSW (búsqueda aproximada, mucho más rápida a gran escala) para la distancia coseno."""
        self._conexion().execute(
            f"CREATE INDEX IF NOT EXISTS {self.tabla}_hnsw ON {self.tabla} USING hnsw (embedding vector_cosine_ops)"
        )

    def eliminar_indice_hnsw(self) -> None:
        self._conexion().execute(f"DROP INDEX IF EXISTS {self.tabla}_hnsw")

    def explicar(self, vector: list[float], k: int = 3, forzar_indice: bool = False) -> str:
        """Devuelve el plan de ejecución (EXPLAIN) de una búsqueda: ¿usa el índice o recorre toda la tabla?

        Con tablas muy pequeñas Postgres prefiere recorrerlas completas aunque exista un índice (es más
        barato). `forzar_indice=True` desactiva temporalmente el recorrido secuencial para MOSTRAR el índice.
        """
        q = np.array(vector, dtype=np.float32)
        c = self._conexion()
        if forzar_indice:
            c.execute("SET enable_seqscan = off")
        try:
            filas = c.execute(
                f"EXPLAIN SELECT id FROM {self.tabla} ORDER BY embedding <=> %s LIMIT %s", (q, k)
            ).fetchall()
        finally:
            if forzar_indice:
                c.execute("RESET enable_seqscan")
        # El vector de la consulta aparece escrito en el plan: recortamos las líneas largas para poder leerlas.
        return "\n".join(f[0] if len(f[0]) <= 100 else f[0][:97] + "..." for f in filas)


# ═════════════════════════ 3. Pinecone ═════════════════════════


class AlmacenPinecone(Almacen):
    """Pinecone (base vectorial administrada). Un ÍNDICE contiene NAMESPACES, y estos contienen registros."""

    nombre = "pinecone"

    def __init__(self, api_key: str | None = None, indice: str | None = None, namespace: str | None = None):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Falta PINECONE_API_KEY. Crea tu cuenta en Pinecone y tu API key (ver docs/CUENTA_PINECONE.md)."
            )
        self.nombre_indice = indice or os.getenv("PINECONE_INDEX", "edutek-cohorte1")
        self.namespace = namespace or os.getenv("PINECONE_NAMESPACE", "edutek")
        self.nube = os.getenv("PINECONE_CLOUD", "aws")
        self.region = os.getenv("PINECONE_REGION", "us-east-1")
        self._pc = None
        self._indice = None

    def _cliente(self):
        if self._pc is None:
            from pinecone import Pinecone

            self._pc = Pinecone(api_key=self.api_key)
        return self._pc

    def preparar(self, dimension: int) -> None:
        from pinecone import ServerlessSpec

        pc = self._cliente()
        if not pc.has_index(self.nombre_indice):
            # Serverless: no administras servidores; Pinecone escala y cobra por uso (unidades de lectura/escritura).
            # La dimensión y la métrica NO se pueden cambiar después de crear el índice.
            pc.create_index(
                name=self.nombre_indice,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.nube, region=self.region),
            )
        # Un índice recién creado tarda unos segundos en estar listo.
        for _ in range(60):
            if pc.describe_index(self.nombre_indice).status.ready:
                break
            time.sleep(2)
        self._indice = pc.Index(host=pc.describe_index(self.nombre_indice).host)
        # Si el índice ya existía con otra dimensión (otro modelo de embeddings), avisamos con claridad.
        existente = self._indice.describe_index_stats().dimension
        if existente and existente != dimension:
            raise ValueError(
                f"El índice '{self.nombre_indice}' tiene {existente} dimensiones, pero tu modelo de embeddings "
                f"produce {dimension}. Usa otro índice (PINECONE_INDEX) o bórralo con almacen.eliminar()."
            )

    def _idx(self):
        if self._indice is None:
            raise RuntimeError("Llama primero a almacen.preparar(dimension).")
        return self._indice

    @staticmethod
    def _metadatos_validos(metadatos: dict) -> dict:
        """Pinecone solo admite texto, números, booleanos y listas de texto como metadatos."""
        limpio = {}
        for clave, valor in metadatos.items():
            if isinstance(valor, (str, int, float, bool)) or (isinstance(valor, list) and all(isinstance(x, str) for x in valor)):
                limpio[clave] = valor
            else:
                limpio[clave] = str(valor)
        return limpio

    def insertar(self, fragmentos: list[Fragmento]) -> None:
        registros = [
            {
                "id": f.id,
                "values": [float(x) for x in f.vector],
                # En Pinecone el texto se guarda como un metadato más, para poder devolverlo con los resultados.
                "metadata": {**self._metadatos_validos(f.metadatos), "texto": f.texto},
            }
            for f in fragmentos
        ]
        # Un upsert admite hasta 1 000 registros y 2 MB por petición: enviamos en lotes pequeños.
        for i in range(0, len(registros), 100):
            self._idx().upsert(vectors=registros[i : i + 100], namespace=self.namespace)
        self._esperar(len(fragmentos))

    def _esperar(self, minimo: int, segundos: int = 30) -> None:
        """Pinecone es de consistencia EVENTUAL: los datos recién escritos pueden tardar unos segundos en
        aparecer en las búsquedas. Esperamos (con tope) a que el conteo del namespace alcance lo esperado."""
        limite = time.time() + segundos
        while time.time() < limite:
            if self.contar() >= minimo:
                return
            time.sleep(1)

    def buscar(self, vector: list[float], k: int = 3, filtro: dict | None = None) -> list[Resultado]:
        filtro_pinecone = {c: {"$eq": v} for c, v in filtro.items()} if filtro else None
        resp = self._idx().query(
            vector=[float(x) for x in vector],
            top_k=k,
            include_metadata=True,
            namespace=self.namespace,
            filter=filtro_pinecone,
        )
        salida = []
        for m in resp.matches:
            meta = dict(m.metadata or {})
            texto = meta.pop("texto", "")
            salida.append(Resultado(m.id, texto, meta, float(m.score)))
        return salida

    def contar(self) -> int:
        stats = self._idx().describe_index_stats()
        resumen = (stats.namespaces or {}).get(self.namespace)
        return int(getattr(resumen, "vector_count", 0) or 0) if resumen is not None else 0

    def vaciar(self) -> None:
        try:
            self._idx().delete(delete_all=True, namespace=self.namespace)
        except Exception as exc:  # noqa: BLE001 - un namespace vacío o inexistente no es un error para nosotros
            if "not found" not in str(exc).lower():
                raise

    def eliminar(self) -> None:
        pc = self._cliente()
        if pc.has_index(self.nombre_indice):
            pc.delete_index(self.nombre_indice)
        self._indice = None


# ═════════════════════════ Fábrica ═════════════════════════


def crear_almacen(nombre: str) -> Almacen:
    """Devuelve el almacén pedido: "memoria", "neon" o "pinecone"."""
    nombre = nombre.strip().lower()
    if nombre == "memoria":
        return AlmacenMemoria()
    if nombre == "neon":
        return AlmacenNeon()
    if nombre == "pinecone":
        return AlmacenPinecone()
    raise ValueError(f"Almacén desconocido: {nombre!r}. Usa uno de: {', '.join(NOMBRES_ALMACEN)}.")


def almacenes_configurados() -> dict[str, bool]:
    """Qué almacenes tienen su configuración en el entorno (memoria siempre está disponible)."""
    return {
        "memoria": True,
        "neon": bool(os.getenv("NEON_DATABASE_URL")),
        "pinecone": bool(os.getenv("PINECONE_API_KEY")),
    }
