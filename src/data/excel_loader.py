"""
Cargador de archivos Excel para Forecasting
============================================
Procesa y valida archivos Excel con datos de consumo historico.
"""
import pandas as pd
import io
import base64
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# Cache para conversiones Dict <-> DataFrame
# ============================================================================
_df_cache: Dict[str, pd.DataFrame] = {}
_df_cache_max_size = 10

# Columnas requeridas para cada solapa
COLUMNAS_CONSUMO_REQUERIDAS = ['fecha', 'codigosap', 'centro', 'almacen', 'cantidad']
COLUMNAS_CONSUMO_OPCIONALES = ['descripcion']
COLUMNAS_MATERIALES = ['codigosap', 'descripcion', 'unidad_medida', 'grupo']


def decodificar_archivo(contents: str, filename: str) -> Optional[bytes]:
    """
    Decodifica el contenido del archivo subido desde base64.

    Args:
        contents: Contenido en formato base64 (data:application/...;base64,...)
        filename: Nombre del archivo

    Returns:
        Bytes del archivo o None si hay error
    """
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return decoded
    except Exception as e:
        logger.error(f"Error decodificando archivo {filename}: {e}")
        return None


def validar_columnas(df: pd.DataFrame, requeridas: List[str], nombre_solapa: str) -> List[str]:
    """
    Valida que el DataFrame tenga las columnas requeridas.

    Returns:
        Lista de errores (vacia si todo OK)
    """
    errores = []
    columnas_df = [col.lower().strip() for col in df.columns]

    for col in requeridas:
        if col.lower() not in columnas_df:
            errores.append(f"Solapa '{nombre_solapa}': Falta columna requerida '{col}'")

    return errores


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas a minusculas y sin espacios"""
    df.columns = [col.lower().strip() for col in df.columns]
    return df


def parsear_fechas(df: pd.DataFrame, columna: str = 'fecha') -> Tuple[pd.DataFrame, List[str]]:
    """
    Parsea y valida la columna de fechas.

    Returns:
        Tuple (DataFrame con fechas parseadas, lista de errores)
    """
    errores = []
    df = df.copy()

    try:
        # Intentar varios formatos de fecha
        df[columna] = pd.to_datetime(df[columna], dayfirst=True, errors='coerce')

        # Verificar fechas invalidas
        fechas_invalidas = df[df[columna].isna()].index.tolist()
        if fechas_invalidas:
            n_invalidas = len(fechas_invalidas)
            if n_invalidas <= 5:
                errores.append(f"Fechas invalidas en filas: {fechas_invalidas[:5]}")
            else:
                errores.append(f"{n_invalidas} fechas invalidas encontradas")

    except Exception as e:
        errores.append(f"Error parseando fechas: {e}")

    return df, errores


def validar_cantidades(df: pd.DataFrame, columna: str = 'cantidad') -> List[str]:
    """Valida que las cantidades sean numericas y positivas"""
    errores = []

    # Convertir a numerico
    df[columna] = pd.to_numeric(df[columna], errors='coerce')

    # Verificar NaN
    nan_count = df[columna].isna().sum()
    if nan_count > 0:
        errores.append(f"{nan_count} cantidades no numericas encontradas")

    # Verificar negativos
    negativos = (df[columna] < 0).sum()
    if negativos > 0:
        errores.append(f"{negativos} cantidades negativas encontradas")

    return errores


def cargar_excel_forecast(
    contents: str,
    filename: str
) -> Dict[str, Any]:
    """
    Carga y valida archivo Excel de forecast.

    Args:
        contents: Contenido del archivo en base64
        filename: Nombre del archivo

    Returns:
        {
            'success': bool,
            'consumo': DataFrame con consumo historico (o None),
            'materiales': DataFrame con catalogo (o None),
            'errores': lista de errores de validacion,
            'advertencias': lista de advertencias,
            'resumen': dict con estadisticas del archivo
        }
    """
    resultado = {
        'success': False,
        'consumo': None,
        'materiales': None,
        'errores': [],
        'advertencias': [],
        'resumen': {}
    }

    # Decodificar archivo
    decoded = decodificar_archivo(contents, filename)
    if decoded is None:
        resultado['errores'].append("No se pudo decodificar el archivo")
        return resultado

    # Verificar extension
    if not filename.lower().endswith(('.xlsx', '.xls')):
        resultado['errores'].append("El archivo debe ser Excel (.xlsx o .xls)")
        return resultado

    try:
        # Leer archivo Excel
        excel_file = pd.ExcelFile(io.BytesIO(decoded))
        hojas_disponibles = excel_file.sheet_names
        logger.info(f"Hojas encontradas: {hojas_disponibles}")

        # =================================================================
        # Procesar solapa Consumo_Historico (REQUERIDA)
        # =================================================================
        solapa_consumo = None
        for nombre in ['Consumo_Historico', 'consumo_historico', 'Consumo', 'consumo', 'Historico', 'historico']:
            if nombre in hojas_disponibles:
                solapa_consumo = nombre
                break

        if solapa_consumo is None:
            # Usar la primera hoja si no encuentra la especifica
            solapa_consumo = hojas_disponibles[0]
            resultado['advertencias'].append(
                f"No se encontro solapa 'Consumo_Historico', usando '{solapa_consumo}'"
            )

        df_consumo = pd.read_excel(excel_file, sheet_name=solapa_consumo)
        df_consumo = normalizar_columnas(df_consumo)

        # Validar columnas requeridas
        errores_cols = validar_columnas(df_consumo, COLUMNAS_CONSUMO_REQUERIDAS, solapa_consumo)
        resultado['errores'].extend(errores_cols)

        if errores_cols:
            return resultado

        # Parsear fechas
        df_consumo, errores_fechas = parsear_fechas(df_consumo)
        resultado['errores'].extend(errores_fechas)

        # Validar cantidades
        errores_cant = validar_cantidades(df_consumo)
        resultado['errores'].extend(errores_cant)

        # Eliminar filas con datos criticos faltantes
        filas_antes = len(df_consumo)
        df_consumo = df_consumo.dropna(subset=['fecha', 'codigosap', 'cantidad'])
        filas_despues = len(df_consumo)

        if filas_antes - filas_despues > 0:
            resultado['advertencias'].append(
                f"Se eliminaron {filas_antes - filas_despues} filas con datos faltantes"
            )

        # Asegurar tipos de datos
        df_consumo['codigosap'] = df_consumo['codigosap'].astype(str).str.strip()
        df_consumo['centro'] = df_consumo['centro'].astype(str).str.strip().str.zfill(4)
        df_consumo['almacen'] = df_consumo['almacen'].astype(str).str.strip().str.zfill(4)

        if 'descripcion' not in df_consumo.columns:
            df_consumo['descripcion'] = ''
        else:
            df_consumo['descripcion'] = df_consumo['descripcion'].fillna('').astype(str)

        # Renombrar codigosap a codigo para uso interno
        df_consumo = df_consumo.rename(columns={'codigosap': 'codigo'})

        resultado['consumo'] = df_consumo

        # =================================================================
        # Procesar solapa Materiales (OPCIONAL)
        # =================================================================
        solapa_materiales = None
        for nombre in ['Materiales', 'materiales', 'Catalogo', 'catalogo']:
            if nombre in hojas_disponibles:
                solapa_materiales = nombre
                break

        if solapa_materiales:
            try:
                df_materiales = pd.read_excel(excel_file, sheet_name=solapa_materiales)
                df_materiales = normalizar_columnas(df_materiales)

                if 'codigosap' in df_materiales.columns and 'descripcion' in df_materiales.columns:
                    df_materiales['codigosap'] = df_materiales['codigosap'].astype(str).str.strip()
                    df_materiales = df_materiales.rename(columns={'codigosap': 'codigo'})
                    resultado['materiales'] = df_materiales
                else:
                    resultado['advertencias'].append(
                        "Solapa 'Materiales' ignorada: faltan columnas codigoSAP/descripcion"
                    )
            except Exception as e:
                resultado['advertencias'].append(f"Error leyendo solapa Materiales: {e}")

        # =================================================================
        # Generar resumen
        # =================================================================
        materiales_unicos = df_consumo['codigo'].nunique()
        centros_unicos = df_consumo['centro'].nunique()
        almacenes_unicos = df_consumo['almacen'].nunique()
        fecha_min = df_consumo['fecha'].min()
        fecha_max = df_consumo['fecha'].max()
        total_registros = len(df_consumo)

        resultado['resumen'] = {
            'total_registros': total_registros,
            'materiales_unicos': materiales_unicos,
            'centros_unicos': centros_unicos,
            'almacenes_unicos': almacenes_unicos,
            'fecha_inicio': fecha_min.strftime('%d/%m/%Y') if pd.notna(fecha_min) else 'N/A',
            'fecha_fin': fecha_max.strftime('%d/%m/%Y') if pd.notna(fecha_max) else 'N/A',
            'dias_datos': (fecha_max - fecha_min).days if pd.notna(fecha_min) and pd.notna(fecha_max) else 0
        }

        # Verificar datos suficientes
        if total_registros < 10:
            resultado['advertencias'].append(
                f"Pocos datos ({total_registros} registros). Se recomiendan al menos 30 por material."
            )

        # Si no hay errores criticos, marcar como exitoso
        if not resultado['errores']:
            resultado['success'] = True
            logger.info(f"Excel cargado exitosamente: {materiales_unicos} materiales, {total_registros} registros")

    except Exception as e:
        logger.error(f"Error procesando Excel: {e}")
        resultado['errores'].append(f"Error procesando archivo: {str(e)}")

    return resultado


def obtener_materiales_desde_excel(df_consumo: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Obtiene lista de materiales unicos del DataFrame de consumo.

    Args:
        df_consumo: DataFrame con columnas codigo, descripcion

    Returns:
        Lista de dicts con label y value para dropdown
    """
    if df_consumo is None or len(df_consumo) == 0:
        return []

    # Obtener materiales unicos con su descripcion mas reciente
    materiales = df_consumo.groupby('codigo').agg({
        'descripcion': 'first',
        'cantidad': 'sum'
    }).reset_index()

    # Ordenar por cantidad total (mas consumidos primero)
    materiales = materiales.sort_values('cantidad', ascending=False)

    opciones = []
    for _, row in materiales.iterrows():
        codigo = row['codigo']
        desc = row['descripcion'][:40] if row['descripcion'] else ''
        label = f"{codigo} - {desc}" if desc else codigo
        opciones.append({
            'label': label,
            'value': codigo
        })

    return opciones


def filtrar_consumo_por_material(
    df_consumo: pd.DataFrame,
    codigo: str,
    centro: Optional[str] = None,
    almacen: Optional[str] = None
) -> pd.DataFrame:
    """
    Filtra el DataFrame de consumo por material, centro y almacen.

    Args:
        df_consumo: DataFrame completo de consumo
        codigo: Codigo del material a filtrar
        centro: Centro (opcional)
        almacen: Almacen (opcional)

    Returns:
        DataFrame filtrado y agregado por fecha
    """
    if df_consumo is None or len(df_consumo) == 0:
        return pd.DataFrame()

    # Filtrar por codigo
    df = df_consumo[df_consumo['codigo'] == codigo].copy()

    # Filtrar por centro si se especifica
    if centro and centro != 'Todos':
        df = df[df['centro'] == centro]

    # Filtrar por almacen si se especifica
    if almacen and almacen != 'Todos':
        df = df[df['almacen'] == almacen]

    if len(df) == 0:
        return pd.DataFrame()

    # Agregar por fecha (sumar cantidades del mismo dia)
    df_agregado = df.groupby('fecha').agg({
        'codigo': 'first',
        'descripcion': 'first',
        'centro': 'first',
        'almacen': 'first',
        'cantidad': 'sum'
    }).reset_index()

    # Ordenar por fecha
    df_agregado = df_agregado.sort_values('fecha')

    return df_agregado


def obtener_centros_desde_excel(df_consumo: pd.DataFrame) -> List[str]:
    """Obtiene lista de centros unicos del DataFrame"""
    if df_consumo is None or len(df_consumo) == 0:
        return []
    return sorted(df_consumo['centro'].unique().tolist())


def obtener_almacenes_desde_excel(
    df_consumo: pd.DataFrame,
    centro: Optional[str] = None
) -> List[str]:
    """Obtiene lista de almacenes, opcionalmente filtrados por centro"""
    if df_consumo is None or len(df_consumo) == 0:
        return []

    df = df_consumo
    if centro and centro != 'Todos':
        df = df[df['centro'] == centro]

    return sorted(df['almacen'].unique().tolist())


# ============================================================================
# Funciones de conversion Dict <-> DataFrame (migradas de loader.py)
# ============================================================================

def _compute_data_hash(data: List[Dict]) -> str:
    """Calcula hash unico para una lista de diccionarios"""
    if not data:
        return "empty"
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

    data_hash = _compute_data_hash(data)

    if data_hash in _df_cache:
        return _df_cache[data_hash].copy()

    df = pd.DataFrame(data)

    if len(_df_cache) >= _df_cache_max_size:
        oldest_key = next(iter(_df_cache))
        del _df_cache[oldest_key]

    _df_cache[data_hash] = df.copy()
    return df


def obtener_filtros_unicos(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Obtiene valores unicos para filtros de dropdowns.

    Returns:
        Dict con listas de valores unicos por columna
    """
    filtros = {}

    for col in ['centro', 'almacen', 'sector', 'proveedor']:
        if col in df.columns:
            valores = df[col].dropna().unique().tolist()
            valores = [str(v).strip() for v in valores
                      if v and str(v).strip() and str(v).strip().lower() not in ('nan', 'none', '')]
            if valores:
                filtros[col] = sorted(valores)

    return filtros


def clear_df_cache():
    """Limpia el cache de DataFrames"""
    global _df_cache
    _df_cache.clear()
