"""
Constantes centralizadas del proyecto Forecast
===============================================
Evita duplicacion de valores en multiples archivos.
"""

# ============================================================================
# Modelos de Machine Learning disponibles
# ============================================================================

MODELOS_ML = {
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
    'linear': 'Regresion Lineal',
    'xgboost': 'XGBoost',
    'prophet': 'Prophet (Facebook)',
    'arima': 'ARIMA/SARIMAX'
}

# Modelo por defecto
MODELO_DEFAULT = 'random_forest'


def obtener_opciones_modelos(incluir_avanzados: bool = True) -> list:
    """
    Genera opciones para dropdown de modelos ML.

    Args:
        incluir_avanzados: Si incluir modelos que requieren instalacion extra

    Returns:
        Lista de dicts con label y value para dropdown
    """
    modelos_base = ['random_forest', 'gradient_boosting', 'linear']
    modelos_avanzados = ['xgboost', 'prophet', 'arima']

    modelos = modelos_base + (modelos_avanzados if incluir_avanzados else [])

    return [{"label": MODELOS_ML[m], "value": m} for m in modelos]


def obtener_nombre_modelo(codigo: str) -> str:
    """Obtiene el nombre legible de un modelo por su codigo"""
    return MODELOS_ML.get(codigo, codigo)


# ============================================================================
# Horizontes de prediccion
# ============================================================================

HORIZONTES_PREDICCION = [
    {"label": "7 dias", "value": 7},
    {"label": "30 dias", "value": 30},
    {"label": "90 dias", "value": 90},
]

HORIZONTE_DEFAULT = 30


# ============================================================================
# Niveles de confianza para intervalos de prediccion
# ============================================================================

NIVELES_CONFIANZA = [
    {"label": "90%", "value": 0.90},
    {"label": "95%", "value": 0.95},
]

CONFIANZA_DEFAULT = 0.95


# ============================================================================
# Configuracion de datos
# ============================================================================

# Numero minimo de muestras para entrenar modelo ML completo
MIN_SAMPLES_ML = 10

# Limite de materiales en forecast masivo (para rendimiento)
# El consumo historico NO tiene limite, solo la cantidad de materiales a procesar
MAX_MATERIALES_MASIVO = 500
