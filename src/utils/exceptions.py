"""
Excepciones Personalizadas para MRP Analytics
Define excepciones especificas para mejorar el manejo de errores.
"""


class MRPAnalyticsError(Exception):
    """
    Excepcion base para la aplicacion MRP Analytics.
    Todas las excepciones personalizadas heredan de esta.
    """

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} | Detalles: {self.details}"
        return self.message


# ============================================================================
# Excepciones de Base de Datos
# ============================================================================

class DatabaseError(MRPAnalyticsError):
    """Excepcion base para errores de base de datos"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Error al conectar con la base de datos"""
    pass


class QueryExecutionError(DatabaseError):
    """Error al ejecutar una query SQL"""

    def __init__(self, message: str, query: str = None, params: tuple = None):
        details = {}
        if query:
            details['query'] = query[:200]  # Truncar para seguridad
        if params:
            details['params_count'] = len(params)
        super().__init__(message, details)


class DataIntegrityError(DatabaseError):
    """Error de integridad de datos (foreign key, unique, etc.)"""
    pass


# ============================================================================
# Excepciones de Validacion
# ============================================================================

class ValidationError(MRPAnalyticsError):
    """Excepcion base para errores de validacion"""
    pass


class DataValidationError(ValidationError):
    """Error al validar datos de entrada"""

    def __init__(self, message: str, field: str = None, value=None):
        details = {}
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)[:100]  # Truncar
        super().__init__(message, details)


class FileValidationError(ValidationError):
    """Error al validar un archivo"""

    def __init__(self, message: str, filename: str = None, expected_format: str = None):
        details = {}
        if filename:
            details['filename'] = filename
        if expected_format:
            details['expected_format'] = expected_format
        super().__init__(message, details)


class ConfigurationError(ValidationError):
    """Error en la configuracion del sistema"""
    pass


# ============================================================================
# Excepciones de Procesamiento
# ============================================================================

class ProcessingError(MRPAnalyticsError):
    """Excepcion base para errores de procesamiento"""
    pass


class FileProcessingError(ProcessingError):
    """Error al procesar un archivo"""

    def __init__(self, message: str, filename: str = None, line_number: int = None):
        details = {}
        if filename:
            details['filename'] = filename
        if line_number:
            details['line_number'] = line_number
        super().__init__(message, details)


class DataTransformationError(ProcessingError):
    """Error al transformar datos"""
    pass


class CalculationError(ProcessingError):
    """Error en calculos MRP/EOQ/ROP"""

    def __init__(self, message: str, calculation_type: str = None, material_code: str = None):
        details = {}
        if calculation_type:
            details['calculation_type'] = calculation_type
        if material_code:
            details['material_code'] = material_code
        super().__init__(message, details)


# ============================================================================
# Excepciones de Conexion Externa
# ============================================================================

class ExternalConnectionError(MRPAnalyticsError):
    """Excepcion base para errores de conexion externa"""
    pass


class SAPConnectionError(ExternalConnectionError):
    """Error al conectar con SAP HANA"""

    def __init__(self, message: str, host: str = None, port: int = None):
        details = {}
        if host:
            details['host'] = host
        if port:
            details['port'] = port
        super().__init__(message, details)


class APIError(ExternalConnectionError):
    """Error en llamadas a APIs externas"""

    def __init__(self, message: str, endpoint: str = None, status_code: int = None):
        details = {}
        if endpoint:
            details['endpoint'] = endpoint
        if status_code:
            details['status_code'] = status_code
        super().__init__(message, details)


# ============================================================================
# Excepciones de ML/Prediccion
# ============================================================================

class MLError(MRPAnalyticsError):
    """Excepcion base para errores de Machine Learning"""
    pass


class ModelTrainingError(MLError):
    """Error al entrenar un modelo"""

    def __init__(self, message: str, model_type: str = None, samples: int = None):
        details = {}
        if model_type:
            details['model_type'] = model_type
        if samples:
            details['samples'] = samples
        super().__init__(message, details)


class PredictionError(MLError):
    """Error al realizar predicciones"""
    pass


class InsufficientDataError(MLError):
    """No hay suficientes datos para entrenar/predecir"""

    def __init__(self, message: str, required: int = None, available: int = None):
        details = {}
        if required:
            details['required'] = required
        if available:
            details['available'] = available
        super().__init__(message, details)


# ============================================================================
# Excepciones de Cache
# ============================================================================

class CacheError(MRPAnalyticsError):
    """Error relacionado con el sistema de cache"""
    pass


class CacheExpiredError(CacheError):
    """El cache ha expirado"""
    pass


class CacheMissError(CacheError):
    """No se encontro el dato en cache"""
    pass
