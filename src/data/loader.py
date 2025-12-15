"""
Cargador de datos Excel/CSV para MRP Analytics
"""
import pandas as pd
import io
import base64
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# Cache para conversiones Dict <-> DataFrame
# ============================================================================
_df_cache: Dict[str, pd.DataFrame] = {}
_df_cache_max_size = 10  # Máximo número de DataFrames en cache


# Esquema esperado para datos de stock
SCHEMA_STOCK = {
    "codigo": str,
    "descripcion": str,
    "centro": str,
    "almacen": str,
    "stock_actual": float,
    "stock_seguridad": float,
    "punto_pedido": float,
    "stock_maximo": float,
    "costo_unitario": float,
}

# Columnas requeridas
COLUMNAS_REQUERIDAS = ["codigo", "descripcion", "stock_actual"]

# Columnas opcionales con valores por defecto
COLUMNAS_OPCIONALES = {
    "centro": "GENERAL",
    "almacen": "PRINCIPAL",
    "sector": "",
    "stock_seguridad": 0,
    "punto_pedido": 0,
    "stock_maximo": 0,
    "costo_unitario": 0,
    "ubicacion": "",
    "proveedor": "",
    "lead_time": 14,
    "consumo_mensual": 0,
}


def decodificar_contenido(contents: str) -> bytes:
    """Decodifica el contenido base64 del upload"""
    content_type, content_string = contents.split(',')
    return base64.b64decode(content_string)


def cargar_excel(contents: str, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Carga un archivo Excel

    Args:
        contents: Contenido base64 del archivo
        filename: Nombre del archivo

    Returns:
        Tuple (DataFrame, mensaje_error)
    """
    try:
        decoded = decodificar_contenido(contents)
        df = pd.read_excel(io.BytesIO(decoded))
        return normalizar_dataframe(df)
    except Exception as e:
        return None, f"Error al cargar Excel: {str(e)}"


def cargar_csv(contents: str, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Carga un archivo CSV

    Args:
        contents: Contenido base64 del archivo
        filename: Nombre del archivo

    Returns:
        Tuple (DataFrame, mensaje_error)
    """
    try:
        decoded = decodificar_contenido(contents)

        # Intentar diferentes encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(io.StringIO(decoded.decode(encoding)))
                return normalizar_dataframe(df)
            except UnicodeDecodeError:
                continue

        return None, "No se pudo decodificar el archivo CSV"
    except Exception as e:
        return None, f"Error al cargar CSV: {str(e)}"


def cargar_datos(contents: str, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Carga datos desde un archivo (detecta tipo automáticamente)

    Args:
        contents: Contenido base64 del archivo
        filename: Nombre del archivo

    Returns:
        Tuple (DataFrame, mensaje_error)
    """
    filename_lower = filename.lower()

    if filename_lower.endswith(('.xlsx', '.xls')):
        return cargar_excel(contents, filename)
    elif filename_lower.endswith('.csv'):
        return cargar_csv(contents, filename)
    else:
        return None, f"Formato no soportado: {filename}. Use Excel (.xlsx) o CSV."


def normalizar_dataframe(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Normaliza un DataFrame a la estructura esperada

    - Convierte nombres de columnas a minúsculas
    - Agrega columnas faltantes con valores por defecto
    - Valida columnas requeridas
    """
    # Normalizar nombres de columnas
    df.columns = df.columns.str.lower().str.strip()
    df.columns = df.columns.str.replace(' ', '_')

    # Verificar columnas requeridas
    columnas_faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
    if columnas_faltantes:
        return None, f"Columnas requeridas faltantes: {', '.join(columnas_faltantes)}"

    # Agregar columnas opcionales faltantes
    for col, default in COLUMNAS_OPCIONALES.items():
        if col not in df.columns:
            df[col] = default

    # Convertir tipos numéricos
    columnas_numericas = [
        'stock_actual', 'stock_seguridad', 'punto_pedido',
        'stock_maximo', 'costo_unitario', 'lead_time', 'consumo_mensual'
    ]

    for col in columnas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Limpiar strings
    for col in ['codigo', 'descripcion', 'centro', 'almacen', 'sector', 'ubicacion', 'proveedor']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Eliminar filas sin código
    df = df[df['codigo'].notna() & (df['codigo'] != '') & (df['codigo'] != 'nan')]

    return df, None


def obtener_filtros_unicos(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Obtiene valores únicos para filtros

    Returns:
        Dict con listas de valores únicos por columna
    """
    filtros = {}

    for col in ['centro', 'almacen', 'sector', 'proveedor']:
        if col in df.columns:
            valores = df[col].dropna().unique().tolist()
            # Filtrar valores vacíos, 'nan', 'None', etc.
            valores = [str(v).strip() for v in valores
                      if v and str(v).strip() and str(v).strip().lower() not in ('nan', 'none', '')]
            if valores:
                filtros[col] = sorted(valores)

    return filtros


def calcular_resumen(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula un resumen de los datos cargados
    """
    return {
        "total_materiales": len(df),
        "total_centros": df['centro'].nunique() if 'centro' in df.columns else 0,
        "total_almacenes": df['almacen'].nunique() if 'almacen' in df.columns else 0,
        "valor_inventario": (df['stock_actual'] * df['costo_unitario']).sum() if 'costo_unitario' in df.columns else 0,
        "stock_total": df['stock_actual'].sum() if 'stock_actual' in df.columns else 0,
    }


def df_a_dict(df: pd.DataFrame) -> List[Dict]:
    """Convierte DataFrame a lista de diccionarios para JSON"""
    return df.to_dict('records')


def _compute_data_hash(data: List[Dict]) -> str:
    """Calcula hash único para una lista de diccionarios"""
    if not data:
        return "empty"
    # Hash basado en longitud y primeros/últimos elementos
    sample = str(len(data))
    if data:
        sample += str(data[0]) + str(data[-1])
    return hashlib.md5(sample.encode()).hexdigest()[:16]


def dict_a_df(data: List[Dict], use_cache: bool = True) -> pd.DataFrame:
    """
    Convierte lista de diccionarios a DataFrame con cache.

    Args:
        data: Lista de diccionarios
        use_cache: Usar cache (default: True)

    Returns:
        DataFrame con los datos
    """
    global _df_cache

    if not data:
        return pd.DataFrame()

    if not use_cache:
        return pd.DataFrame(data)

    # Calcular hash de los datos
    data_hash = _compute_data_hash(data)

    # Verificar cache
    if data_hash in _df_cache:
        logger.debug(f"Cache hit para dict_a_df (hash: {data_hash})")
        return _df_cache[data_hash].copy()

    # Cache miss - crear DataFrame
    df = pd.DataFrame(data)

    # Limpiar cache si está lleno
    if len(_df_cache) >= _df_cache_max_size:
        # Eliminar entrada más antigua (FIFO)
        oldest_key = next(iter(_df_cache))
        del _df_cache[oldest_key]
        logger.debug(f"Cache lleno, eliminando entrada: {oldest_key}")

    # Guardar en cache
    _df_cache[data_hash] = df.copy()
    logger.debug(f"Cache miss para dict_a_df, guardando (hash: {data_hash})")

    return df


def clear_df_cache():
    """Limpia el cache de DataFrames"""
    global _df_cache
    _df_cache.clear()
    logger.debug("Cache de DataFrames limpiado")
