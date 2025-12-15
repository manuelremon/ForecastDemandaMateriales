"""
Estrategias de Forecasting
==========================
Módulo que implementa el patrón Strategy para modelos de forecasting.

Permite agregar nuevos modelos fácilmente sin modificar el código existente.
"""
from typing import Dict, Type, Optional, List

from src.ml.strategies.base import ForecastStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Registro de estrategias disponibles
_STRATEGIES: Dict[str, Type[ForecastStrategy]] = {}


def registrar_estrategia(nombre: str, clase: Type[ForecastStrategy]):
    """
    Registra una estrategia de forecasting.

    Args:
        nombre: Identificador único del modelo
        clase: Clase que implementa ForecastStrategy
    """
    _STRATEGIES[nombre] = clase
    logger.debug(f"Estrategia registrada: {nombre}")


def obtener_estrategia(nombre: str, **kwargs) -> Optional[ForecastStrategy]:
    """
    Factory para obtener una estrategia de forecasting.

    Args:
        nombre: Nombre del modelo (random_forest, xgboost, prophet, arima, etc.)
        **kwargs: Parámetros del modelo

    Returns:
        Instancia de ForecastStrategy o None si no existe/no está disponible
    """
    if nombre not in _STRATEGIES:
        logger.warning(f"Estrategia '{nombre}' no encontrada")
        return None

    try:
        return _STRATEGIES[nombre](**kwargs)
    except ImportError as e:
        logger.warning(f"Estrategia '{nombre}' no disponible: {e}")
        return None
    except Exception as e:
        logger.error(f"Error creando estrategia '{nombre}': {e}")
        return None


def listar_estrategias() -> Dict[str, bool]:
    """
    Lista todas las estrategias registradas y su disponibilidad.

    Returns:
        Dict {nombre: disponible}
    """
    disponibles = {}
    for nombre, clase in _STRATEGIES.items():
        try:
            clase()
            disponibles[nombre] = True
        except ImportError:
            disponibles[nombre] = False
        except Exception:
            disponibles[nombre] = False
    return disponibles


def obtener_estrategias_disponibles() -> List[str]:
    """
    Retorna lista de nombres de estrategias disponibles (instaladas).

    Returns:
        Lista de nombres de estrategias que se pueden usar
    """
    return [nombre for nombre, disponible in listar_estrategias().items() if disponible]


def obtener_nombres_modelos() -> Dict[str, str]:
    """
    Retorna diccionario de identificador -> nombre legible.

    Útil para mostrar en UI.
    """
    nombres = {}
    for nombre in obtener_estrategias_disponibles():
        try:
            estrategia = obtener_estrategia(nombre)
            if estrategia:
                nombres[nombre] = estrategia.nombre_modelo
        except Exception:
            pass
    return nombres


# =============================================================================
# Registro de Estrategias
# =============================================================================

# Estrategias sklearn (siempre disponibles)
from src.ml.strategies.sklearn_models import (
    RandomForestStrategy,
    GradientBoostingStrategy,
    RidgeStrategy
)

registrar_estrategia('random_forest', RandomForestStrategy)
registrar_estrategia('gradient_boosting', GradientBoostingStrategy)
registrar_estrategia('linear', RidgeStrategy)

# XGBoost (opcional)
try:
    from src.ml.strategies.xgboost_model import XGBoostStrategy
    registrar_estrategia('xgboost', XGBoostStrategy)
except ImportError:
    logger.info("XGBoost no disponible")

# Prophet (opcional)
try:
    from src.ml.strategies.prophet_model import ProphetStrategy
    registrar_estrategia('prophet', ProphetStrategy)
except ImportError:
    logger.info("Prophet no disponible")

# ARIMA (opcional)
try:
    from src.ml.strategies.arima_model import ARIMAStrategy
    registrar_estrategia('arima', ARIMAStrategy)
except ImportError:
    logger.info("ARIMA no disponible")


# Exports públicos
__all__ = [
    'ForecastStrategy',
    'obtener_estrategia',
    'listar_estrategias',
    'obtener_estrategias_disponibles',
    'obtener_nombres_modelos',
    'registrar_estrategia',
    # Estrategias
    'RandomForestStrategy',
    'GradientBoostingStrategy',
    'RidgeStrategy',
]
