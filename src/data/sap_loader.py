"""
Cargador de datos desde bases de datos SAP locales
===================================================
Lee datos de stock, consumo histórico y materiales

SEGURIDAD: Todas las queries usan parámetros preparados
para prevenir SQL Injection.
"""
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logger import get_logger
from src.utils.validators import (
    sanitize_string, validate_search_term, escape_like_pattern
)

logger = get_logger(__name__)

# Rutas de las bases de datos
DATA_DIR = Path(__file__).parent.parent.parent / "data"
SAP_DATA_DB = DATA_DIR / "sap_data.db"
CATALOGO_DB = DATA_DIR / "catalogo_materiales.db"
EQUIVALENTES_DB = DATA_DIR / "equivalentes.db"


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Obtiene conexión a una base de datos"""
    return sqlite3.connect(str(db_path), check_same_thread=False)


# ============================================================
# Datos de Stock
# ============================================================

def cargar_stock(
    centro: str = None,
    almacen: str = None,
    grupo_articulos: str = None,
    solo_con_stock: bool = False,
    limite: int = None
) -> pd.DataFrame:
    """
    Carga datos de stock desde sap_data.db

    Args:
        centro: Filtrar por centro
        almacen: Filtrar por almacén
        grupo_articulos: Filtrar por grupo de artículos
        solo_con_stock: Solo materiales con stock > 0
        limite: Límite de registros

    Returns:
        DataFrame con datos de stock normalizados
    """
    conn = get_connection(SAP_DATA_DB)

    query = """
        SELECT
            material as codigo,
            material_descripcion as descripcion,
            centro,
            centro_descripcion,
            almacen,
            stock as stock_actual,
            precio as costo_unitario,
            stock_valorizado as valor_inventario,
            um as unidad_medida,
            grupo_de_articulos as grupo_articulos,
            gpo_articulos_descripcion as grupo_descripcion,
            acreedor as proveedor,
            acreedor_descripcion as proveedor_descripcion,
            ubicacion,
            critico,
            inmovilizado,
            dia as fecha_datos
        FROM stock
        WHERE 1=1
    """

    params = []

    if centro:
        query += " AND centro = ?"
        params.append(centro)

    if almacen:
        query += " AND almacen = ?"
        params.append(almacen)

    if grupo_articulos:
        query += " AND grupo_de_articulos = ?"
        params.append(grupo_articulos)

    if solo_con_stock:
        query += " AND stock > 0"

    query += " ORDER BY stock_valorizado DESC"

    if limite:
        query += " LIMIT ?"
        params.append(limite)

    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()

    # Agregar columnas calculadas por defecto
    df['stock_seguridad'] = 0
    df['punto_pedido'] = 0
    df['stock_maximo'] = 0
    df['consumo_mensual'] = 0
    df['lead_time'] = 14

    return df


def cargar_stock_ultimo_dia() -> pd.DataFrame:
    """Carga stock del último día disponible"""
    conn = get_connection(SAP_DATA_DB)

    # Obtener última fecha
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(dia) FROM stock")
    ultima_fecha = cursor.fetchone()[0]

    # Query con parámetro preparado para prevenir SQL Injection
    query = """
        SELECT
            material as codigo,
            material_descripcion as descripcion,
            centro,
            almacen,
            SUM(stock) as stock_actual,
            AVG(precio) as costo_unitario,
            SUM(stock_valorizado) as valor_inventario,
            MAX(um) as unidad_medida,
            MAX(grupo_de_articulos) as grupo_articulos,
            MAX(acreedor) as proveedor,
            MAX(critico) as critico
        FROM stock
        WHERE dia = ?
        GROUP BY material, centro, almacen
        ORDER BY valor_inventario DESC
    """

    df = pd.read_sql_query(query, conn, params=(ultima_fecha,))
    conn.close()

    # Agregar columnas para clasificación
    df['stock_seguridad'] = 0
    df['punto_pedido'] = 0
    df['stock_maximo'] = 0
    df['consumo_mensual'] = 0
    df['lead_time'] = 14

    return df


def obtener_centros() -> List[str]:
    """Obtiene lista de centros únicos"""
    conn = get_connection(SAP_DATA_DB)
    df = pd.read_sql_query("SELECT DISTINCT centro FROM stock ORDER BY centro", conn)
    conn.close()
    return df['centro'].tolist()


def obtener_almacenes(centro: str = None) -> List[str]:
    """Obtiene lista de almacenes únicos"""
    conn = get_connection(SAP_DATA_DB)
    params = []
    query = "SELECT DISTINCT almacen FROM stock"
    if centro:
        query += " WHERE centro = ?"
        params.append(sanitize_string(centro, max_length=20))
    query += " ORDER BY almacen"
    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()
    return df['almacen'].tolist()


def obtener_grupos_articulos() -> List[Dict]:
    """Obtiene grupos de artículos con descripción"""
    conn = get_connection(SAP_DATA_DB)
    df = pd.read_sql_query("""
        SELECT DISTINCT
            grupo_de_articulos as codigo,
            gpo_articulos_descripcion as descripcion
        FROM stock
        ORDER BY grupo_de_articulos
    """, conn)
    conn.close()
    return df.to_dict('records')


# ============================================================
# Consumo Histórico
# ============================================================

def cargar_consumo_historico(
    material: str = None,
    centro: str = None,
    dias: int = 365
) -> pd.DataFrame:
    """
    Carga consumo histórico desde sap_data.db

    Args:
        material: Filtrar por código de material
        centro: Filtrar por centro
        dias: Días de historia a cargar (desde la última fecha disponible)

    Returns:
        DataFrame con consumos históricos
    """
    conn = get_connection(SAP_DATA_DB)

    # Obtener la fecha más reciente disponible en los datos
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(fecha) FROM consumo_historico")
    max_fecha = cursor.fetchone()[0]

    if max_fecha:
        # Calcular fecha inicio basada en los datos disponibles, no la fecha actual
        fecha_max = pd.to_datetime(max_fecha)
        fecha_inicio = (fecha_max - timedelta(days=dias)).strftime('%Y-%m-%d')
    else:
        fecha_inicio = (datetime.now() - timedelta(days=dias)).strftime('%Y-%m-%d')

    # Query con parámetros preparados para prevenir SQL Injection
    params = [fecha_inicio]
    query = """
        SELECT
            fecha,
            material as codigo,
            descripcion,
            centro,
            almacen,
            cantidad
        FROM consumo_historico
        WHERE fecha >= ?
    """

    if material:
        query += " AND material = ?"
        params.append(sanitize_string(material, max_length=50))

    if centro:
        query += " AND centro = ?"
        params.append(sanitize_string(centro, max_length=20))

    query += " ORDER BY fecha"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # Convertir fecha
    if len(df) > 0:
        df['fecha'] = pd.to_datetime(df['fecha'])

    return df


def calcular_consumo_promedio(
    material: str = None,
    centro: str = None,
    dias: int = 90
) -> pd.DataFrame:
    """
    Calcula consumo promedio mensual por material

    Returns:
        DataFrame con consumo_mensual por material
    """
    df_consumo = cargar_consumo_historico(material, centro, dias)

    if len(df_consumo) == 0:
        return pd.DataFrame()

    # Agrupar por material
    consumo_total = df_consumo.groupby('codigo')['cantidad'].sum().reset_index()
    consumo_total.columns = ['codigo', 'consumo_total']

    # Calcular promedio mensual (días / 30)
    meses = dias / 30
    consumo_total['consumo_mensual'] = consumo_total['consumo_total'] / meses

    return consumo_total[['codigo', 'consumo_mensual']]


def obtener_tendencia_consumo(
    material: str,
    dias: int = 180
) -> pd.DataFrame:
    """
    Obtiene tendencia de consumo semanal para un material

    Returns:
        DataFrame con consumo semanal
    """
    df = cargar_consumo_historico(material=material, dias=dias)

    if len(df) == 0:
        return pd.DataFrame()

    # Agrupar por semana
    df['semana'] = df['fecha'].dt.isocalendar().week
    df['año'] = df['fecha'].dt.year
    df['periodo'] = df['año'].astype(str) + '-W' + df['semana'].astype(str).str.zfill(2)

    tendencia = df.groupby('periodo')['cantidad'].sum().reset_index()
    tendencia.columns = ['periodo', 'consumo']

    return tendencia


# ============================================================
# Catálogo de Materiales
# ============================================================

def cargar_catalogo_materiales(
    activos_only: bool = True,
    grupo_articulos: str = None
) -> pd.DataFrame:
    """
    Carga catálogo de materiales desde catalogo_materiales.db

    Returns:
        DataFrame con materiales del catálogo
    """
    conn = get_connection(CATALOGO_DB)
    params = []

    query = """
        SELECT
            codigo,
            descripcion,
            descripcion_larga,
            grupo_articulos,
            unidad_medida,
            precio_usd as costo_unitario,
            activo
        FROM materiales
        WHERE 1=1
    """

    if activos_only:
        query += " AND activo = 1"

    if grupo_articulos:
        query += " AND grupo_articulos = ?"
        params.append(sanitize_string(grupo_articulos, max_length=20))

    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()

    return df


def buscar_material(termino: str, limite: int = 20) -> List[Dict]:
    """
    Busca materiales por código o descripción

    Args:
        termino: Término de búsqueda
        limite: Máximo de resultados

    Returns:
        Lista de materiales encontrados
    """
    # Validar y sanitizar entrada
    if not termino or len(termino) > 100:
        return []

    termino = validate_search_term(termino, max_length=100)
    if not termino:
        return []

    conn = get_connection(CATALOGO_DB)

    # Escapar caracteres especiales de LIKE y usar parámetros preparados
    termino_safe = escape_like_pattern(termino)
    search_pattern = f"%{termino_safe}%"

    query = """
        SELECT codigo, descripcion, precio_usd as costo_unitario
        FROM materiales
        WHERE codigo LIKE ? ESCAPE '\\'
           OR descripcion LIKE ? ESCAPE '\\'
        LIMIT ?
    """

    df = pd.read_sql_query(query, conn, params=(search_pattern, search_pattern, limite))
    conn.close()

    return df.to_dict('records')


# ============================================================
# Equivalencias
# ============================================================

def cargar_equivalencias(material: str = None) -> pd.DataFrame:
    """
    Carga equivalencias de materiales

    Args:
        material: Filtrar por material específico

    Returns:
        DataFrame con equivalencias
    """
    conn = get_connection(EQUIVALENTES_DB)
    params = []

    query = "SELECT * FROM equivalencias"

    if material:
        material_safe = sanitize_string(material, max_length=50)
        query += " WHERE material_original = ? OR material_equivalente = ?"
        params.extend([material_safe, material_safe])

    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()

    return df


def buscar_equivalentes(codigo: str) -> List[Dict]:
    """
    Busca materiales equivalentes a uno dado

    Returns:
        Lista de materiales equivalentes
    """
    df = cargar_equivalencias(codigo)

    # Vectorizado: usar condiciones sobre Series en lugar de iterrows
    if len(df) == 0:
        return []

    # Seleccionar la columna correcta según match
    mask_original = df['material_original'] == codigo
    equivalentes_series = pd.concat([
        df.loc[mask_original, 'material_equivalente'],
        df.loc[~mask_original, 'material_original']
    ])

    return [{'codigo': c} for c in equivalentes_series.tolist()]


# ============================================================
# Parámetros MRP desde materiales_bbdd
# ============================================================

def cargar_parametros_mrp(centro: str = None) -> pd.DataFrame:
    """
    Carga parámetros MRP reales desde materiales_bbdd

    Returns:
        DataFrame con stock_seguridad, punto_pedido, stock_maximo por material
    """
    conn = get_connection(SAP_DATA_DB)
    params = []

    query = """
        SELECT
            codigo_material as codigo,
            centro,
            almacen,
            sector,
            stock_de_seguridad as stock_seguridad,
            punto_de_pedido as punto_pedido,
            stock_maximo
        FROM materiales_bbdd
        WHERE 1=1
    """

    if centro:
        query += " AND centro = ?"
        params.append(sanitize_string(centro, max_length=20))

    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()

    return df


# ============================================================
# Pedidos SAP (Órdenes Pendientes)
# ============================================================

def cargar_pedidos_pendientes(centro: str = None) -> pd.DataFrame:
    """
    Carga pedidos pendientes desde pedidos_sap

    Returns:
        DataFrame con órdenes de compra pendientes de entrega
    """
    conn = get_connection(SAP_DATA_DB)
    params = []

    query = """
        SELECT
            centro,
            almacen,
            pedido,
            posicion_pedido,
            material as codigo,
            descripcion,
            ctdpedida as cantidad_pedida,
            ctdentregada as cantidad_entregada,
            saldo_pend as saldo_pendiente,
            um as unidad_medida,
            fecdocum as fecha_documento,
            fecentre as fecha_entrega,
            cls as clase,
            solicitante,
            nombre_1 as proveedor
        FROM pedidos_sap
        WHERE saldo_pend > 0
    """

    if centro:
        query += " AND centro = ?"
        params.append(sanitize_string(centro, max_length=20))

    query += " ORDER BY fecentre"

    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()

    return df


def obtener_resumen_pedidos() -> Dict[str, Any]:
    """
    Resumen de pedidos pendientes

    Returns:
        Dict con métricas de pedidos
    """
    df = cargar_pedidos_pendientes()

    return {
        'total_pedidos': df['pedido'].nunique() if len(df) > 0 else 0,
        'total_posiciones': len(df),
        'cantidad_pendiente': df['saldo_pendiente'].sum() if len(df) > 0 else 0,
        'proveedores_unicos': df['proveedor'].nunique() if len(df) > 0 else 0,
    }


# ============================================================
# Funciones de Resumen
# ============================================================

def obtener_resumen_inventario() -> Dict[str, Any]:
    """
    Obtiene resumen general del inventario

    Returns:
        Dict con métricas de resumen
    """
    df = cargar_stock_ultimo_dia()

    # Cargar parámetros MRP para calcular alertas reales
    df_mrp = cargar_parametros_mrp()

    # Merge con parámetros MRP
    if len(df_mrp) > 0:
        df = df.merge(
            df_mrp[['codigo', 'centro', 'stock_seguridad', 'punto_pedido', 'stock_maximo']],
            on=['codigo', 'centro'],
            how='left',
            suffixes=('', '_mrp')
        )
        # Usar parámetros MRP donde existan
        df['stock_seguridad'] = df['stock_seguridad_mrp'].fillna(0)
        df['punto_pedido'] = df['punto_pedido_mrp'].fillna(0)
        df['stock_maximo'] = df['stock_maximo_mrp'].fillna(0)

    # Calcular estados
    quiebres = len(df[df['stock_actual'] <= 0])
    bajo_seguridad = len(df[(df['stock_actual'] > 0) &
                            (df['stock_actual'] <= df['stock_seguridad']) &
                            (df['stock_seguridad'] > 0)])
    bajo_pp = len(df[(df['stock_actual'] > df['stock_seguridad']) &
                     (df['stock_actual'] <= df['punto_pedido']) &
                     (df['punto_pedido'] > 0)])
    sobre_maximo = len(df[(df['stock_actual'] > df['stock_maximo']) &
                          (df['stock_maximo'] > 0)])

    return {
        'total_materiales': len(df),
        'total_registros_stock': len(df),
        'valor_total_inventario': df['valor_inventario'].sum(),
        'centros_unicos': df['centro'].nunique(),
        'almacenes_unicos': df['almacen'].nunique(),
        'materiales_con_stock': len(df[df['stock_actual'] > 0]),
        'materiales_sin_stock': len(df[df['stock_actual'] <= 0]),
        'quiebres': quiebres,
        'bajo_stock_seguridad': bajo_seguridad,
        'bajo_punto_pedido': bajo_pp,
        'sobre_stock_maximo': sobre_maximo,
        'materiales_con_mrp': len(df_mrp),
    }


def obtener_top_materiales_valor(limite: int = 10) -> pd.DataFrame:
    """Obtiene top materiales por valor de inventario"""
    df = cargar_stock_ultimo_dia()

    return df.nlargest(limite, 'valor_inventario')[
        ['codigo', 'descripcion', 'centro', 'stock_actual', 'valor_inventario']
    ]


def obtener_estadisticas_bases_datos() -> Dict[str, int]:
    """Obtiene estadísticas de las bases de datos"""
    stats = {}

    # Queries pre-definidas para cada tabla SAP (sin SQL dinámico)
    # SEGURIDAD: No se usa interpolación de strings en queries SQL
    SAP_TABLE_COUNT_QUERIES = {
        'stock': "SELECT COUNT(*) FROM stock",
        'consumo_historico': "SELECT COUNT(*) FROM consumo_historico",
        'pedidos_sap': "SELECT COUNT(*) FROM pedidos_sap",
        'materiales_bbdd': "SELECT COUNT(*) FROM materiales_bbdd"
    }

    # SAP Data
    conn = get_connection(SAP_DATA_DB)
    cursor = conn.cursor()
    for table_name, query in SAP_TABLE_COUNT_QUERIES.items():
        try:
            cursor.execute(query)
            stats[f'sap_{table_name}'] = cursor.fetchone()[0]
        except Exception as e:
            logger.debug(f"Error obteniendo stats de {table_name}: {e}")
            stats[f'sap_{table_name}'] = 0
    conn.close()

    # Catálogo
    conn = get_connection(CATALOGO_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM materiales")
    stats['catalogo_materiales'] = cursor.fetchone()[0]
    conn.close()

    # Equivalencias
    conn = get_connection(EQUIVALENTES_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM equivalencias")
    stats['equivalencias'] = cursor.fetchone()[0]
    conn.close()

    return stats


# ============================================================
# Preparar datos para dashboards
# ============================================================

def preparar_datos_alertas(solo_relevantes: bool = True, limite: int = 50000) -> pd.DataFrame:
    """
    Prepara datos completos para el tablero de alertas

    Combina:
    - Stock actual (sap_data.stock)
    - Parámetros MRP reales (sap_data.materiales_bbdd)
    - Consumo histórico calculado
    - Pedidos pendientes (sap_data.pedidos_sap)

    Args:
        solo_relevantes: Si True, prioriza materiales con MRP o alertas
        limite: Máximo de registros a devolver

    Optimización: Las 4 queries principales se ejecutan en paralelo
    """
    # Ejecutar queries en paralelo para mejor rendimiento
    results = {}

    def _load_stock():
        return cargar_stock_ultimo_dia()

    def _load_mrp():
        return cargar_parametros_mrp()

    def _load_consumo():
        return calcular_consumo_promedio(dias=90)

    def _load_pedidos():
        return cargar_pedidos_pendientes()

    # Ejecutar las 4 queries en paralelo
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_load_stock): 'stock',
            executor.submit(_load_mrp): 'mrp',
            executor.submit(_load_consumo): 'consumo',
            executor.submit(_load_pedidos): 'pedidos'
        }

        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                logger.error(f"Error cargando {key}: {e}")
                results[key] = pd.DataFrame()

    # Extraer resultados
    df_stock = results.get('stock', pd.DataFrame())
    df_mrp = results.get('mrp', pd.DataFrame())
    df_consumo = results.get('consumo', pd.DataFrame())
    df_pedidos = results.get('pedidos', pd.DataFrame())

    if len(df_stock) == 0:
        logger.warning("No se encontraron datos de stock")
        return pd.DataFrame()

    # 3. Merge stock con parámetros MRP
    if len(df_mrp) > 0:
        df = df_stock.merge(
            df_mrp[['codigo', 'centro', 'almacen', 'stock_seguridad', 'punto_pedido', 'stock_maximo', 'sector']],
            on=['codigo', 'centro', 'almacen'],
            how='left',
            suffixes=('', '_mrp')
        )
        # Usar valores MRP donde existan
        df['stock_seguridad'] = df['stock_seguridad_mrp'].fillna(0)
        df['punto_pedido'] = df['punto_pedido_mrp'].fillna(0)
        df['stock_maximo'] = df['stock_maximo_mrp'].fillna(0)
        df['sector'] = df.get('sector', 'Sin Sector')
        df = df.drop(columns=['stock_seguridad_mrp', 'punto_pedido_mrp', 'stock_maximo_mrp'], errors='ignore')
    else:
        df = df_stock
        df['sector'] = 'Sin Sector'

    # 4. Usar consumo promedio histórico (ya cargado en paralelo)
    if len(df_consumo) > 0:
        df = df.merge(
            df_consumo,
            on='codigo',
            how='left',
            suffixes=('', '_calc')
        )
        df['consumo_mensual'] = df['consumo_mensual_calc'].fillna(0)
        df = df.drop(columns=['consumo_mensual_calc'], errors='ignore')

    # 5. Usar pedidos pendientes (ya cargados en paralelo) y agregar por material
    if len(df_pedidos) > 0:
        pedidos_por_material = df_pedidos.groupby(['codigo', 'centro']).agg({
            'saldo_pendiente': 'sum',
            'pedido': 'nunique',
            'fecha_entrega': 'min'
        }).reset_index()
        pedidos_por_material.columns = ['codigo', 'centro', 'cantidad_en_pedido', 'pedidos_activos', 'proxima_entrega']

        df = df.merge(
            pedidos_por_material,
            on=['codigo', 'centro'],
            how='left'
        )
        df['cantidad_en_pedido'] = df['cantidad_en_pedido'].fillna(0)
        df['pedidos_activos'] = df['pedidos_activos'].fillna(0)
    else:
        df['cantidad_en_pedido'] = 0
        df['pedidos_activos'] = 0
        df['proxima_entrega'] = None

    # 6. Calcular parámetros derivados
    lead_time = 14  # días por defecto
    df['consumo_diario'] = df['consumo_mensual'] / 30
    df['lead_time'] = lead_time

    # Para materiales SIN parámetros MRP, estimar basado en consumo
    mask_sin_mrp = (df['punto_pedido'] == 0) & (df['consumo_mensual'] > 0)
    df.loc[mask_sin_mrp, 'stock_seguridad'] = df.loc[mask_sin_mrp, 'consumo_diario'] * 14
    df.loc[mask_sin_mrp, 'punto_pedido'] = (df.loc[mask_sin_mrp, 'consumo_diario'] * lead_time) + df.loc[mask_sin_mrp, 'stock_seguridad']
    df.loc[mask_sin_mrp, 'stock_maximo'] = df.loc[mask_sin_mrp, 'punto_pedido'] * 2

    # 7. Calcular estado de cada material
    df['estado'] = 'normal'
    df.loc[df['stock_actual'] <= 0, 'estado'] = 'quiebre'
    df.loc[(df['stock_actual'] > 0) &
           (df['stock_actual'] <= df['stock_seguridad']) &
           (df['stock_seguridad'] > 0), 'estado'] = 'bajo_seguridad'
    df.loc[(df['stock_actual'] > df['stock_seguridad']) &
           (df['stock_actual'] <= df['punto_pedido']) &
           (df['punto_pedido'] > 0), 'estado'] = 'bajo_punto_pedido'
    df.loc[(df['stock_actual'] > df['stock_maximo']) &
           (df['stock_maximo'] > 0), 'estado'] = 'sobre_maximo'

    # 8. Calcular días de stock (cobertura)
    df['dias_stock'] = 0.0  # Inicializar como float
    mask_consumo = df['consumo_diario'] > 0
    df.loc[mask_consumo, 'dias_stock'] = (df.loc[mask_consumo, 'stock_actual'] /
                                           df.loc[mask_consumo, 'consumo_diario']).astype(float)
    df['dias_stock'] = df['dias_stock'].clip(upper=365.0)  # Máximo 1 año

    # 9. Stock disponible proyectado (considera pedidos en tránsito)
    df['stock_proyectado'] = df['stock_actual'] + df['cantidad_en_pedido']

    # 10. Marcar si tiene parámetros MRP definidos
    df['tiene_mrp'] = (df['punto_pedido'] > 0).astype(int)

    # 11. Filtrar datos para reducir tamaño si es necesario
    if solo_relevantes and len(df) > limite:
        # Priorizar: alertas > con MRP > por valor
        df_alertas = df[df['estado'] != 'normal']
        df_con_mrp = df[(df['estado'] == 'normal') & (df['tiene_mrp'] == 1)]
        df_resto = df[(df['estado'] == 'normal') & (df['tiene_mrp'] == 0)]

        # Combinar priorizando alertas
        df_filtrado = pd.concat([
            df_alertas,
            df_con_mrp,
            df_resto.nlargest(limite - len(df_alertas) - len(df_con_mrp), 'valor_inventario')
        ]).head(limite)

        return df_filtrado

    return df.head(limite) if len(df) > limite else df


def preparar_datos_para_ml(material: str) -> pd.DataFrame:
    """
    Prepara datos históricos de consumo para entrenamiento ML

    Args:
        material: Código de material

    Returns:
        DataFrame con features para ML
    """
    df = cargar_consumo_historico(material=material, dias=365)

    if len(df) == 0:
        return pd.DataFrame()

    # Agrupar por día
    df_diario = df.groupby('fecha')['cantidad'].sum().reset_index()

    # Crear features temporales
    df_diario['dia_semana'] = df_diario['fecha'].dt.dayofweek
    df_diario['dia_mes'] = df_diario['fecha'].dt.day
    df_diario['mes'] = df_diario['fecha'].dt.month
    df_diario['trimestre'] = df_diario['fecha'].dt.quarter
    df_diario['es_fin_mes'] = (df_diario['fecha'].dt.is_month_end).astype(int)

    return df_diario


# ============================================================
# Búsqueda en Catálogo con Indicador MRP
# ============================================================

def buscar_material_con_mrp(termino: str, limite: int = 20) -> List[Dict]:
    """
    Busca materiales en catálogo completo e indica si tienen MRP configurado

    Args:
        termino: Código o descripción a buscar (mínimo 3 caracteres)
        limite: Máximo de resultados

    Returns:
        Lista de materiales con campos:
        - codigo, descripcion, descripcion_larga
        - precio_usd, unidad_medida, grupo_articulos
        - tiene_mrp (bool)
        - mrp_info: {sector, ss, pp, sm} o None
    """
    if not termino or len(termino) < 3:
        return []

    conn_cat = get_connection(CATALOGO_DB)
    conn_sap = get_connection(SAP_DATA_DB)

    try:
        # Buscar en catálogo
        query_cat = """
            SELECT codigo, descripcion, descripcion_larga,
                   grupo_articulos, unidad_medida, precio_usd, activo
            FROM materiales
            WHERE activo = 1 AND (codigo LIKE ? OR descripcion LIKE ?)
            LIMIT ?
        """
        term = f"%{termino}%"
        df_cat = pd.read_sql_query(query_cat, conn_cat, params=[term, term, limite])

        if len(df_cat) == 0:
            return []

        # Obtener códigos encontrados
        codigos = df_cat['codigo'].tolist()
        placeholders = ','.join(['?' for _ in codigos])

        # Buscar parámetros MRP para esos códigos
        query_mrp = f"""
            SELECT codigo_material as codigo, centro, almacen, sector,
                   stock_de_seguridad as ss, punto_de_pedido as pp, stock_maximo as sm
            FROM materiales_bbdd
            WHERE codigo_material IN ({placeholders})
        """
        df_mrp = pd.read_sql_query(query_mrp, conn_sap, params=codigos)

        # RENDIMIENTO: Usar merge en lugar de iterrows() (100x más rápido)
        # Hacer left join para combinar catálogo con MRP
        if len(df_mrp) > 0:
            # Deduplicar MRP por código (tomar el primer registro si hay duplicados)
            df_mrp_unique = df_mrp.drop_duplicates(subset=['codigo'], keep='first')
            df_merged = df_cat.merge(
                df_mrp_unique[['codigo', 'sector', 'ss', 'pp', 'sm']],
                on='codigo',
                how='left'
            )
        else:
            df_merged = df_cat.copy()
            df_merged['sector'] = None
            df_merged['ss'] = None
            df_merged['pp'] = None
            df_merged['sm'] = None

        # Marcar materiales con MRP usando operación vectorizada
        df_merged['tiene_mrp'] = df_merged['sector'].notna()

        # Convertir a lista de diccionarios de forma vectorizada
        def row_to_dict(row):
            material = {
                'codigo': row['codigo'],
                'descripcion': row['descripcion'],
                'descripcion_larga': row['descripcion_larga'],
                'grupo_articulos': row['grupo_articulos'],
                'unidad_medida': row['unidad_medida'],
                'precio_usd': row['precio_usd'],
                'tiene_mrp': row['tiene_mrp'],
                'mrp_info': None
            }
            if row['tiene_mrp']:
                material['mrp_info'] = {
                    'sector': row['sector'],
                    'ss': int(row['ss']) if pd.notna(row['ss']) else 0,
                    'pp': int(row['pp']) if pd.notna(row['pp']) else 0,
                    'sm': int(row['sm']) if pd.notna(row['sm']) else 0
                }
            return material

        # Usar to_dict('records') para la mayoría de campos y procesar mrp_info
        resultados = df_merged.apply(row_to_dict, axis=1).tolist()

        return resultados

    finally:
        conn_cat.close()
        conn_sap.close()


if __name__ == "__main__":
    # Test
    logger.info("Estadísticas de bases de datos:")
    logger.info(obtener_estadisticas_bases_datos())
    logger.info("Resumen de inventario:")
    logger.info(obtener_resumen_inventario())
