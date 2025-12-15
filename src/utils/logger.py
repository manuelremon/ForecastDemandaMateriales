"""
Sistema de Logging Estructurado para MRP Analytics
Reemplaza los print() por logging estructurado con rotacion de archivos.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

# Directorio de logs (relativo a la raiz del proyecto)
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Formato de log
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Obtiene un logger configurado con handlers de consola y archivo.

    Args:
        name: Nombre del logger (usar __name__ del modulo)
        level: Nivel de logging (default: INFO)

    Returns:
        Logger configurado

    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Mensaje informativo")
        >>> logger.warning("Advertencia")
        >>> logger.error("Error critico")
    """
    logger = logging.getLogger(name)

    # Evitar configuracion duplicada si ya tiene handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Captura todo, los handlers filtran

    # Formatter compartido
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # Handler para consola (solo INFO y superiores)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Handler para archivo con rotacion (DEBUG y superiores)
    log_file = LOG_DIR / "mrp_analytics.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10_000_000,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Handler separado para errores
    error_file = LOG_DIR / "mrp_analytics_errors.log"
    error_handler = RotatingFileHandler(
        error_file,
        maxBytes=5_000_000,  # 5 MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger


class LoggerMixin:
    """
    Mixin para agregar logging a clases.

    Example:
        >>> class MyService(LoggerMixin):
        ...     def do_something(self):
        ...         self.logger.info("Haciendo algo")
    """

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_execution_time(func):
    """
    Decorador para loggear tiempo de ejecucion de funciones.

    Example:
        >>> @log_execution_time
        ... def funcion_lenta():
        ...     time.sleep(1)
    """
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} ejecutado en {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"{func.__name__} fallo despues de {elapsed:.3f}s: {e}")
            raise

    return wrapper


# Logger global para uso rapido en toda la aplicacion
app_logger = get_logger("mrp_analytics")


# Funciones de conveniencia
def log_info(message: str, logger_name: str = "mrp_analytics"):
    """Log de nivel INFO"""
    get_logger(logger_name).info(message)


def log_warning(message: str, logger_name: str = "mrp_analytics"):
    """Log de nivel WARNING"""
    get_logger(logger_name).warning(message)


def log_error(message: str, logger_name: str = "mrp_analytics", exc_info: bool = False):
    """Log de nivel ERROR"""
    get_logger(logger_name).error(message, exc_info=exc_info)


def log_debug(message: str, logger_name: str = "mrp_analytics"):
    """Log de nivel DEBUG"""
    get_logger(logger_name).debug(message)
