"""
Validadores de Entrada para MRP Analytics
Funciones para validar y sanitizar datos de entrada.
"""
import re
from typing import Any, Optional, List, Dict, Union, Callable
import pandas as pd
import numpy as np

from src.utils.exceptions import DataValidationError, FileValidationError


# ============================================================================
# Validadores de Tipos Basicos
# ============================================================================

def validate_not_none(value: Any, field_name: str) -> Any:
    """
    Valida que el valor no sea None.

    Args:
        value: Valor a validar
        field_name: Nombre del campo (para mensajes de error)

    Returns:
        El valor si es valido

    Raises:
        DataValidationError: Si el valor es None
    """
    if value is None:
        raise DataValidationError(f"{field_name} no puede ser None", field=field_name)
    return value


def validate_not_empty(value: Any, field_name: str) -> Any:
    """
    Valida que el valor no este vacio.

    Args:
        value: Valor a validar
        field_name: Nombre del campo

    Returns:
        El valor si es valido

    Raises:
        DataValidationError: Si el valor es None o vacio
    """
    if value is None:
        raise DataValidationError(f"{field_name} no puede ser None", field=field_name, value=value)

    if isinstance(value, str) and not value.strip():
        raise DataValidationError(f"{field_name} no puede estar vacio", field=field_name, value=value)

    if isinstance(value, (list, dict, tuple)) and len(value) == 0:
        raise DataValidationError(f"{field_name} no puede estar vacio", field=field_name)

    return value


def validate_numeric(
    value: Any,
    field_name: str,
    min_value: float = None,
    max_value: float = None,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> float:
    """
    Valida y convierte un valor a numerico.

    Args:
        value: Valor a validar
        field_name: Nombre del campo
        min_value: Valor minimo permitido (opcional)
        max_value: Valor maximo permitido (opcional)
        allow_nan: Permitir NaN (default: False)
        allow_inf: Permitir infinito (default: False)

    Returns:
        Valor como float

    Raises:
        DataValidationError: Si la validacion falla
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise DataValidationError(
            f"{field_name} debe ser numerico",
            field=field_name,
            value=value
        )

    if not allow_nan and np.isnan(num):
        raise DataValidationError(
            f"{field_name} no puede ser NaN",
            field=field_name,
            value=value
        )

    if not allow_inf and np.isinf(num):
        raise DataValidationError(
            f"{field_name} no puede ser infinito",
            field=field_name,
            value=value
        )

    if min_value is not None and num < min_value:
        raise DataValidationError(
            f"{field_name} debe ser >= {min_value}",
            field=field_name,
            value=value
        )

    if max_value is not None and num > max_value:
        raise DataValidationError(
            f"{field_name} debe ser <= {max_value}",
            field=field_name,
            value=value
        )

    return num


def validate_positive(value: Any, field_name: str, allow_zero: bool = True) -> float:
    """
    Valida que el valor sea positivo.

    Args:
        value: Valor a validar
        field_name: Nombre del campo
        allow_zero: Permitir cero (default: True)

    Returns:
        Valor como float positivo
    """
    min_val = 0 if allow_zero else 0.0001
    return validate_numeric(value, field_name, min_value=min_val)


def validate_integer(value: Any, field_name: str, min_value: int = None, max_value: int = None) -> int:
    """
    Valida y convierte un valor a entero.

    Args:
        value: Valor a validar
        field_name: Nombre del campo
        min_value: Valor minimo permitido
        max_value: Valor maximo permitido

    Returns:
        Valor como int
    """
    num = validate_numeric(value, field_name, min_value=min_value, max_value=max_value)
    return int(num)


# ============================================================================
# Validadores de Strings
# ============================================================================

def sanitize_string(
    value: Any,
    max_length: int = 255,
    strip: bool = True,
    remove_control_chars: bool = True
) -> str:
    """
    Sanitiza una cadena de texto.

    Args:
        value: Valor a sanitizar
        max_length: Longitud maxima permitida
        strip: Eliminar espacios al inicio/fin
        remove_control_chars: Eliminar caracteres de control

    Returns:
        String sanitizado
    """
    if value is None:
        return ""

    result = str(value)

    if strip:
        result = result.strip()

    if remove_control_chars:
        # Eliminar caracteres de control ASCII (0x00-0x1F y 0x7F-0x9F)
        result = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', result)

    if len(result) > max_length:
        result = result[:max_length]

    return result


def validate_string_pattern(
    value: str,
    pattern: str,
    field_name: str,
    error_message: str = None
) -> str:
    """
    Valida que un string cumpla con un patron regex.

    Args:
        value: Valor a validar
        pattern: Patron regex
        field_name: Nombre del campo
        error_message: Mensaje de error personalizado

    Returns:
        El string si es valido

    Raises:
        DataValidationError: Si no cumple el patron
    """
    if not re.match(pattern, value):
        msg = error_message or f"{field_name} no cumple con el formato requerido"
        raise DataValidationError(msg, field=field_name, value=value)
    return value


def validate_material_code(code: str) -> str:
    """
    Valida y sanitiza un codigo de material.

    Args:
        code: Codigo de material

    Returns:
        Codigo sanitizado

    Raises:
        DataValidationError: Si el codigo es invalido
    """
    code = sanitize_string(code, max_length=50)

    if not code:
        raise DataValidationError("Codigo de material vacio", field="codigo")

    # Solo alfanumericos, guiones y guiones bajos
    if not re.match(r'^[A-Za-z0-9\-_]+$', code):
        raise DataValidationError(
            "Codigo de material invalido. Solo se permiten letras, numeros, guiones y guiones bajos",
            field="codigo",
            value=code
        )

    return code


def validate_centro(centro: str) -> str:
    """
    Valida codigo de centro (4 caracteres alfanumericos).

    Args:
        centro: Codigo de centro

    Returns:
        Codigo sanitizado
    """
    centro = sanitize_string(centro, max_length=10)

    if not centro:
        raise DataValidationError("Centro vacio", field="centro")

    return centro


def validate_almacen(almacen: str) -> str:
    """
    Valida codigo de almacen (4 digitos).

    Args:
        almacen: Codigo de almacen

    Returns:
        Codigo sanitizado
    """
    almacen = sanitize_string(almacen, max_length=10)

    if not almacen:
        raise DataValidationError("Almacen vacio", field="almacen")

    return almacen


# ============================================================================
# Validadores de DataFrames
# ============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    numeric_columns: List[str] = None,
    min_rows: int = 0
) -> pd.DataFrame:
    """
    Valida estructura de un DataFrame.

    Args:
        df: DataFrame a validar
        required_columns: Columnas requeridas
        numeric_columns: Columnas que deben ser numericas
        min_rows: Numero minimo de filas

    Returns:
        DataFrame validado

    Raises:
        DataValidationError: Si la validacion falla
    """
    if df is None:
        raise DataValidationError("DataFrame es None")

    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Se esperaba DataFrame, recibido {type(df)}")

    if len(df) < min_rows:
        raise DataValidationError(
            f"DataFrame debe tener al menos {min_rows} filas, tiene {len(df)}"
        )

    # Verificar columnas requeridas
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise DataValidationError(f"Columnas faltantes: {missing}")

    # Validar y convertir columnas numericas
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df


def validate_dataframe_not_empty(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    """
    Valida que el DataFrame no este vacio.

    Args:
        df: DataFrame a validar
        name: Nombre para mensajes de error

    Returns:
        DataFrame si es valido
    """
    if df is None or len(df) == 0:
        raise DataValidationError(f"{name} esta vacio")
    return df


def clean_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
    fill_value: float = 0,
    clip_negative: bool = False
) -> pd.DataFrame:
    """
    Limpia columnas numericas en un DataFrame.

    Args:
        df: DataFrame a procesar
        columns: Columnas a limpiar
        fill_value: Valor para reemplazar NaN
        clip_negative: Convertir negativos a 0

    Returns:
        DataFrame con columnas limpias
    """
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_value)

            if clip_negative:
                df[col] = df[col].clip(lower=0)

    return df


# ============================================================================
# Validadores de Archivos
# ============================================================================

ALLOWED_EXTENSIONS = {'.xlsx', '.xls', '.csv'}
MAX_FILE_SIZE_MB = 50


def validate_file_extension(filename: str, allowed: set = None) -> str:
    """
    Valida extension de archivo.

    Args:
        filename: Nombre del archivo
        allowed: Extensiones permitidas (default: xlsx, xls, csv)

    Returns:
        Nombre del archivo si es valido

    Raises:
        FileValidationError: Si la extension no es permitida
    """
    allowed = allowed or ALLOWED_EXTENSIONS

    if not filename:
        raise FileValidationError("Nombre de archivo vacio")

    ext = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    if ext not in allowed:
        raise FileValidationError(
            f"Extension no permitida: {ext}. Permitidas: {allowed}",
            filename=filename,
            expected_format=', '.join(allowed)
        )

    return filename


def validate_file_content(
    content: bytes,
    filename: str,
    max_size_mb: float = MAX_FILE_SIZE_MB
) -> bytes:
    """
    Valida contenido de archivo.

    Args:
        content: Contenido del archivo en bytes
        filename: Nombre del archivo
        max_size_mb: Tamano maximo en MB

    Returns:
        Contenido si es valido

    Raises:
        FileValidationError: Si el contenido es invalido
    """
    if not content:
        raise FileValidationError("Archivo vacio", filename=filename)

    size_mb = len(content) / (1024 * 1024)

    if size_mb > max_size_mb:
        raise FileValidationError(
            f"Archivo demasiado grande: {size_mb:.1f}MB. Maximo: {max_size_mb}MB",
            filename=filename
        )

    return content


# ============================================================================
# Validadores de SQL (para prevenir injection)
# ============================================================================

# Whitelist de tablas permitidas
ALLOWED_TABLES = frozenset([
    'materiales', 'stock_historico', 'consumos', 'predicciones',
    'modelos_ml', 'entrenamientos_log', 'alertas', 'configuracion',
    'agente_consultas', 'stock', 'consumo_historico', 'pedidos_sap',
    'materiales_bbdd', 'equivalencias'
])


def validate_table_name(table: str) -> str:
    """
    Valida que el nombre de tabla este en la whitelist.

    Args:
        table: Nombre de la tabla

    Returns:
        Nombre de la tabla si es valido

    Raises:
        DataValidationError: Si la tabla no esta permitida
    """
    table = sanitize_string(table, max_length=50).lower()

    if table not in ALLOWED_TABLES:
        raise DataValidationError(
            f"Tabla no permitida: {table}",
            field="table_name",
            value=table
        )

    return table


def escape_like_pattern(pattern: str) -> str:
    """
    Escapa caracteres especiales para patron LIKE.

    Args:
        pattern: Patron de busqueda

    Returns:
        Patron escapado
    """
    # Escapar caracteres especiales de LIKE
    pattern = pattern.replace('\\', '\\\\')
    pattern = pattern.replace('%', r'\%')
    pattern = pattern.replace('_', r'\_')
    return pattern


def validate_search_term(term: str, max_length: int = 100) -> str:
    """
    Valida y sanitiza un termino de busqueda.

    Args:
        term: Termino de busqueda
        max_length: Longitud maxima

    Returns:
        Termino sanitizado
    """
    if not term:
        return ""

    term = sanitize_string(term, max_length=max_length)

    # Eliminar caracteres potencialmente peligrosos
    term = re.sub(r'[;\'"\\]', '', term)

    return term
