"""
Pool de Conexiones SQLite Thread-Safe
=====================================
Gestiona conexiones a bases de datos SQLite de forma eficiente.
"""
import sqlite3
import threading
from contextlib import contextmanager
from queue import Queue, Empty, Full
from pathlib import Path
from typing import Generator, Optional, Dict, Any
import time

from src.utils.logger import get_logger
from src.utils.exceptions import DatabaseConnectionError

logger = get_logger(__name__)


class SQLiteConnectionPool:
    """
    Pool de conexiones SQLite thread-safe.

    Cada thread obtiene su propia conexión del pool,
    evitando problemas de concurrencia.

    Example:
        >>> pool = SQLiteConnectionPool(Path("data/db.sqlite"))
        >>> with pool.get_connection() as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT * FROM materiales")
    """

    def __init__(
        self,
        db_path: Path,
        pool_size: int = 5,
        timeout: float = 30.0,
        check_same_thread: bool = True
    ):
        """
        Inicializa el pool de conexiones.

        Args:
            db_path: Ruta a la base de datos SQLite
            pool_size: Número de conexiones en el pool
            timeout: Timeout para obtener conexión (segundos)
            check_same_thread: Verificar que la conexión se use en el mismo thread
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout
        self.check_same_thread = check_same_thread
        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._initialized = False
        self._connections_created = 0

        # Inicializar pool
        self._init_pool()

    def _init_pool(self):
        """Inicializa el pool con conexiones"""
        with self._lock:
            if self._initialized:
                return

            # Asegurar que el directorio existe
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Pre-crear conexiones
            for _ in range(self.pool_size):
                try:
                    conn = self._create_connection()
                    self._pool.put(conn)
                    self._connections_created += 1
                except Exception as e:
                    logger.error(f"Error creando conexión inicial: {e}")

            self._initialized = True
            logger.debug(f"Pool inicializado con {self._connections_created} conexiones para {self.db_path}")

    def _create_connection(self) -> sqlite3.Connection:
        """
        Crea una nueva conexión con configuración optimizada.

        Returns:
            Conexión SQLite configurada
        """
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.timeout,
            check_same_thread=self.check_same_thread,
            isolation_level='DEFERRED'
        )
        conn.row_factory = sqlite3.Row

        # Configuraciones de rendimiento y seguridad
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA temp_store=MEMORY")

        return conn

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager para obtener una conexión del pool.

        Usage:
            >>> with pool.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM tabla")

        Yields:
            Conexión SQLite del pool

        Raises:
            DatabaseConnectionError: Si no hay conexiones disponibles
        """
        conn = None
        try:
            conn = self._pool.get(timeout=self.timeout)
            yield conn
            conn.commit()
        except Empty:
            logger.error(f"Timeout esperando conexión del pool ({self.timeout}s)")
            raise DatabaseConnectionError(
                f"No hay conexiones disponibles en el pool después de {self.timeout}s"
            )
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass  # Ignorar errores de rollback
            logger.exception(f"Error en transacción de base de datos: {e}")
            raise
        finally:
            if conn:
                try:
                    self._pool.put(conn)
                except (Full, Exception):
                    # Si falla al devolver, crear nueva conexión
                    try:
                        conn.close()
                        new_conn = self._create_connection()
                        self._pool.put(new_conn)
                    except sqlite3.Error:
                        pass  # No hay nada más que hacer si falla la reconexión

    def execute_query(self, query: str, params: tuple = None) -> list:
        """
        Ejecuta una query SELECT y retorna resultados.

        Args:
            query: Query SQL
            params: Parámetros de la query

        Returns:
            Lista de diccionarios con resultados
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def execute_write(self, query: str, params: tuple = None) -> int:
        """
        Ejecuta una query INSERT/UPDATE/DELETE.

        Args:
            query: Query SQL
            params: Parámetros de la query

        Returns:
            lastrowid para INSERT, rowcount para UPDATE/DELETE
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.lastrowid or cursor.rowcount

    def execute_many(self, query: str, params_list: list) -> int:
        """
        Ejecuta múltiples operaciones en batch.

        Args:
            query: Query SQL con placeholders
            params_list: Lista de tuplas de parámetros

        Returns:
            Número de filas afectadas
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del pool.

        Returns:
            Dict con estadísticas
        """
        return {
            'db_path': str(self.db_path),
            'pool_size': self.pool_size,
            'connections_created': self._connections_created,
            'connections_available': self._pool.qsize(),
            'connections_in_use': self.pool_size - self._pool.qsize(),
        }

    def close_all(self):
        """Cierra todas las conexiones del pool"""
        with self._lock:
            closed = 0
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                    closed += 1
                except Empty:
                    break
                except Exception as e:
                    logger.debug(f"Error cerrando conexión: {e}")

            self._initialized = False
            logger.debug(f"Pool cerrado. {closed} conexiones cerradas.")


# ============================================================
# Pools globales para las bases de datos del proyecto
# ============================================================

_pools: Dict[str, SQLiteConnectionPool] = {}
_pools_lock = threading.Lock()


def get_pool(db_path: Path, pool_size: int = 5) -> SQLiteConnectionPool:
    """
    Obtiene o crea un pool para una base de datos.

    Args:
        db_path: Ruta a la base de datos
        pool_size: Tamaño del pool (solo se usa al crear)

    Returns:
        Pool de conexiones para la base de datos
    """
    key = str(db_path.resolve())

    with _pools_lock:
        if key not in _pools:
            _pools[key] = SQLiteConnectionPool(db_path, pool_size=pool_size)
            logger.debug(f"Nuevo pool creado para: {db_path}")

        return _pools[key]


def close_all_pools():
    """Cierra todos los pools globales"""
    with _pools_lock:
        for key, pool in _pools.items():
            try:
                pool.close_all()
            except Exception as e:
                logger.error(f"Error cerrando pool {key}: {e}")
        _pools.clear()
        logger.info("Todos los pools cerrados")


# ============================================================
# Paths predefinidos para el proyecto
# ============================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Paths de las bases de datos
DB_PATHS = {
    'mrp_analytics': DATA_DIR / "mrp_analytics.db",
    'sap_data': DATA_DIR / "sap_data.db",
    'catalogo': DATA_DIR / "catalogo_materiales.db",
    'equivalentes': DATA_DIR / "equivalentes.db",
}


def get_mrp_pool() -> SQLiteConnectionPool:
    """Obtiene pool para mrp_analytics.db"""
    return get_pool(DB_PATHS['mrp_analytics'])


def get_sap_pool() -> SQLiteConnectionPool:
    """Obtiene pool para sap_data.db"""
    return get_pool(DB_PATHS['sap_data'])


def get_catalogo_pool() -> SQLiteConnectionPool:
    """Obtiene pool para catalogo_materiales.db"""
    return get_pool(DB_PATHS['catalogo'])


def get_equivalentes_pool() -> SQLiteConnectionPool:
    """Obtiene pool para equivalentes.db"""
    return get_pool(DB_PATHS['equivalentes'])
