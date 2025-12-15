"""
Constantes centralizadas del proyecto Forecast
===============================================
Evita duplicacion de valores en multiples archivos.
"""

# ============================================================================
# Modelos de Machine Learning disponibles
# ============================================================================

MODELOS_ML = {
    'random_forest': {
        'nombre': 'Random Forest',
        'tooltip': 'Combina múltiples árboles de decisión. Robusto, estable y excelente para datos con ruido. Recomendado para la mayoría de casos.'
    },
    'gradient_boosting': {
        'nombre': 'Gradient Boosting',
        'tooltip': 'Construye árboles secuencialmente corrigiendo errores. Alta precisión pero más lento. Bueno para patrones complejos.'
    },
    'linear': {
        'nombre': 'Regresión Lineal',
        'tooltip': 'Modelo simple y rápido. Ideal para tendencias lineales claras. Menos preciso con datos no lineales.'
    },
    'xgboost': {
        'nombre': 'XGBoost',
        'tooltip': 'Versión optimizada de Gradient Boosting. Muy rápido y preciso. Requiere instalación adicional.'
    },
    'prophet': {
        'nombre': 'Prophet (Meta)',
        'tooltip': 'Diseñado para series temporales con estacionalidad. Excelente para datos con patrones anuales/semanales.'
    },
    'arima': {
        'nombre': 'ARIMA/SARIMAX',
        'tooltip': 'Modelo estadístico clásico para series temporales. Captura tendencias y estacionalidad. Requiere datos bien estructurados.'
    }
}

# Modelo por defecto
MODELO_DEFAULT = 'random_forest'


def obtener_opciones_modelos(incluir_avanzados: bool = True) -> list:
    """
    Genera opciones para dropdown de modelos ML con tooltips.

    Args:
        incluir_avanzados: Si incluir modelos que requieren instalacion extra

    Returns:
        Lista de dicts con label, value y title (tooltip) para dropdown
    """
    modelos_base = ['random_forest', 'gradient_boosting', 'linear']
    modelos_avanzados = ['xgboost', 'prophet', 'arima']

    modelos = modelos_base + (modelos_avanzados if incluir_avanzados else [])

    return [
        {
            "label": MODELOS_ML[m]['nombre'],
            "value": m,
            "title": MODELOS_ML[m]['tooltip']
        }
        for m in modelos
    ]


def obtener_nombre_modelo(codigo: str) -> str:
    """Obtiene el nombre legible de un modelo por su codigo"""
    modelo = MODELOS_ML.get(codigo)
    return modelo['nombre'] if modelo else codigo


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
