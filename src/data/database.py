"""
Base de datos SQLite para MRP Analytics
========================================
Persistencia de datos históricos, modelos y configuración

SEGURIDAD: Todas las queries usan parámetros preparados
para prevenir SQL Injection.
"""
import sqlite3
import pandas as pd
import pickle
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import os

from src.utils.logger import get_logger
from src.utils.validators import sanitize_string

logger = get_logger(__name__)

# Ruta de la base de datos
DB_PATH = Path(__file__).parent.parent.parent / "data" / "mrp_analytics.db"


def get_connection() -> sqlite3.Connection:
    """Obtiene conexión a la base de datos"""
    os.makedirs(DB_PATH.parent, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Inicializa las tablas de la base de datos"""
    conn = get_connection()
    cursor = conn.cursor()

    # Tabla de materiales (maestro)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS materiales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            codigo TEXT UNIQUE NOT NULL,
            descripcion TEXT,
            centro TEXT,
            almacen TEXT,
            unidad_medida TEXT DEFAULT 'UN',
            grupo_material TEXT,
            proveedor TEXT,
            lead_time_dias INTEGER DEFAULT 14,
            costo_unitario REAL DEFAULT 0,
            activo INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabla de stock histórico
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_historico (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id INTEGER NOT NULL,
            fecha DATE NOT NULL,
            stock_actual REAL,
            stock_seguridad REAL,
            punto_pedido REAL,
            stock_maximo REAL,
            valor_inventario REAL,
            FOREIGN KEY (material_id) REFERENCES materiales(id),
            UNIQUE(material_id, fecha)
        )
    """)

    # Tabla de consumos
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consumos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id INTEGER NOT NULL,
            fecha DATE NOT NULL,
            cantidad REAL NOT NULL,
            tipo_movimiento TEXT,
            documento TEXT,
            FOREIGN KEY (material_id) REFERENCES materiales(id)
        )
    """)

    # Tabla de predicciones
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predicciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id INTEGER NOT NULL,
            fecha_prediccion DATE NOT NULL,
            fecha_target DATE NOT NULL,
            valor_predicho REAL,
            limite_inferior REAL,
            limite_superior REAL,
            modelo_usado TEXT,
            confianza REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (material_id) REFERENCES materiales(id)
        )
    """)

    # Tabla de modelos entrenados
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS modelos_ml (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            tipo TEXT NOT NULL,
            material_id INTEGER,
            modelo_blob BLOB,
            scaler_blob BLOB,
            metricas TEXT,
            parametros TEXT,
            version INTEGER DEFAULT 1,
            activo INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (material_id) REFERENCES materiales(id)
        )
    """)

    # Tabla de entrenamientos (log)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entrenamientos_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modelo_id INTEGER,
            fecha_inicio TIMESTAMP,
            fecha_fin TIMESTAMP,
            registros_usados INTEGER,
            mae REAL,
            rmse REAL,
            r2 REAL,
            mape REAL,
            estado TEXT DEFAULT 'completado',
            error_mensaje TEXT,
            FOREIGN KEY (modelo_id) REFERENCES modelos_ml(id)
        )
    """)

    # Tabla de alertas generadas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alertas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id INTEGER NOT NULL,
            tipo_alerta TEXT NOT NULL,
            severidad TEXT DEFAULT 'media',
            mensaje TEXT,
            valor_actual REAL,
            valor_umbral REAL,
            resuelta INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            FOREIGN KEY (material_id) REFERENCES materiales(id)
        )
    """)

    # Tabla de configuración
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS configuracion (
            clave TEXT PRIMARY KEY,
            valor TEXT,
            descripcion TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tabla de consultas del agente IA
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agente_consultas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            consulta TEXT NOT NULL,
            respuesta TEXT,
            contexto TEXT,
            tokens_usados INTEGER,
            tiempo_respuesta_ms INTEGER,
            feedback INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insertar configuración por defecto
    config_defaults = [
        ('costo_orden', '50', 'Costo por orden de compra (USD)'),
        ('costo_mantener_pct', '20', 'Porcentaje de costo de mantenimiento'),
        ('nivel_servicio', '95', 'Nivel de servicio objetivo (%)'),
        ('lead_time_default', '14', 'Lead time por defecto (días)'),
        ('modelo_default', 'random_forest', 'Modelo ML por defecto'),
        ('horizonte_forecast', '30', 'Horizonte de predicción (días)'),
        ('retrain_interval_days', '7', 'Intervalo de re-entrenamiento (días)'),
    ]

    for clave, valor, desc in config_defaults:
        cursor.execute("""
            INSERT OR IGNORE INTO configuracion (clave, valor, descripcion)
            VALUES (?, ?, ?)
        """, (clave, valor, desc))

    # Crear índices para mejor performance
    # Índices existentes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_material_fecha ON stock_historico(material_id, fecha)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_consumos_material_fecha ON consumos(material_id, fecha)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alertas_material ON alertas(material_id, resuelta)")

    # Nuevos índices para optimización
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_materiales_codigo ON materiales(codigo)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_materiales_centro ON materiales(centro)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_materiales_activo ON materiales(activo)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predicciones_material ON predicciones(material_id, fecha_prediccion)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_modelos_tipo_activo ON modelos_ml(tipo, activo)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alertas_created ON alertas(created_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alertas_severidad ON alertas(severidad, resuelta)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_consumos_fecha ON consumos(fecha)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agente_created ON agente_consultas(created_at DESC)")

    logger.debug("Índices de base de datos verificados/creados")

    conn.commit()
    conn.close()

    logger.info(f"Base de datos inicializada en: {DB_PATH}")


# ============================================================
# CRUD Materiales
# ============================================================

def insertar_material(
    codigo: str,
    descripcion: str,
    centro: str = None,
    almacen: str = None,
    **kwargs
) -> int:
    """Inserta o actualiza un material"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO materiales (codigo, descripcion, centro, almacen,
                               unidad_medida, grupo_material, proveedor,
                               lead_time_dias, costo_unitario)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(codigo) DO UPDATE SET
            descripcion = excluded.descripcion,
            centro = excluded.centro,
            almacen = excluded.almacen,
            updated_at = CURRENT_TIMESTAMP
    """, (
        codigo, descripcion, centro, almacen,
        kwargs.get('unidad_medida', 'UN'),
        kwargs.get('grupo_material'),
        kwargs.get('proveedor'),
        kwargs.get('lead_time_dias', 14),
        kwargs.get('costo_unitario', 0)
    ))

    material_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return material_id


def obtener_material(codigo: str) -> Optional[Dict]:
    """Obtiene un material por código"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM materiales WHERE codigo = ?", (codigo,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def listar_materiales(activos_only: bool = True) -> List[Dict]:
    """Lista todos los materiales"""
    conn = get_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM materiales"
    if activos_only:
        query += " WHERE activo = 1"
    query += " ORDER BY codigo"

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ============================================================
# Stock Histórico
# ============================================================

def registrar_stock(
    material_id: int,
    fecha: str,
    stock_actual: float,
    stock_seguridad: float = 0,
    punto_pedido: float = 0,
    stock_maximo: float = 0,
    valor_inventario: float = 0
):
    """Registra stock histórico de un material"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO stock_historico
        (material_id, fecha, stock_actual, stock_seguridad, punto_pedido,
         stock_maximo, valor_inventario)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(material_id, fecha) DO UPDATE SET
            stock_actual = excluded.stock_actual,
            stock_seguridad = excluded.stock_seguridad,
            punto_pedido = excluded.punto_pedido,
            stock_maximo = excluded.stock_maximo,
            valor_inventario = excluded.valor_inventario
    """, (material_id, fecha, stock_actual, stock_seguridad,
          punto_pedido, stock_maximo, valor_inventario))

    conn.commit()
    conn.close()


def obtener_stock_historico(
    material_id: int,
    dias: int = 90
) -> pd.DataFrame:
    """Obtiene el histórico de stock de un material"""
    conn = get_connection()

    query = """
        SELECT fecha, stock_actual, stock_seguridad, punto_pedido,
               stock_maximo, valor_inventario
        FROM stock_historico
        WHERE material_id = ?
        AND fecha >= date('now', ?)
        ORDER BY fecha
    """

    df = pd.read_sql_query(query, conn, params=(material_id, f'-{dias} days'))
    conn.close()
    return df


# ============================================================
# Consumos
# ============================================================

def registrar_consumo(
    material_id: int,
    fecha: str,
    cantidad: float,
    tipo_movimiento: str = None,
    documento: str = None
):
    """Registra un consumo de material"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO consumos (material_id, fecha, cantidad, tipo_movimiento, documento)
        VALUES (?, ?, ?, ?, ?)
    """, (material_id, fecha, cantidad, tipo_movimiento, documento))

    conn.commit()
    conn.close()


def obtener_consumos(
    material_id: int,
    dias: int = 180
) -> pd.DataFrame:
    """Obtiene el histórico de consumos de un material"""
    conn = get_connection()

    query = """
        SELECT fecha, cantidad, tipo_movimiento
        FROM consumos
        WHERE material_id = ?
        AND fecha >= date('now', ?)
        ORDER BY fecha
    """

    df = pd.read_sql_query(query, conn, params=(material_id, f'-{dias} days'))
    conn.close()
    return df


def obtener_consumo_agregado(
    material_id: int = None,
    agrupacion: str = 'dia'
) -> pd.DataFrame:
    """Obtiene consumos agregados por día/semana/mes"""
    conn = get_connection()
    params = []

    # Validar agrupación con whitelist
    if agrupacion == 'semana':
        date_format = "strftime('%Y-%W', fecha)"
    elif agrupacion == 'mes':
        date_format = "strftime('%Y-%m', fecha)"
    else:
        date_format = "fecha"

    query = f"""
        SELECT {date_format} as periodo,
               material_id,
               SUM(cantidad) as cantidad_total,
               COUNT(*) as num_movimientos
        FROM consumos
        WHERE fecha >= date('now', '-365 days')
    """

    if material_id:
        query += " AND material_id = ?"
        params.append(int(material_id))

    query += f" GROUP BY {date_format}, material_id ORDER BY periodo"

    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()
    return df


# ============================================================
# Modelos ML
# ============================================================

def guardar_modelo(
    nombre: str,
    tipo: str,
    modelo: Any,
    scaler: Any = None,
    metricas: Dict = None,
    parametros: Dict = None,
    material_id: int = None
) -> int:
    """Guarda un modelo entrenado en la base de datos"""
    conn = get_connection()
    cursor = conn.cursor()

    # Serializar modelo y scaler
    modelo_blob = pickle.dumps(modelo)
    scaler_blob = pickle.dumps(scaler) if scaler else None

    # Desactivar modelos anteriores del mismo tipo/material
    if material_id:
        cursor.execute("""
            UPDATE modelos_ml SET activo = 0
            WHERE tipo = ? AND material_id = ?
        """, (tipo, material_id))
    else:
        cursor.execute("""
            UPDATE modelos_ml SET activo = 0
            WHERE tipo = ? AND material_id IS NULL
        """, (tipo,))

    # Insertar nuevo modelo
    cursor.execute("""
        INSERT INTO modelos_ml
        (nombre, tipo, material_id, modelo_blob, scaler_blob, metricas, parametros, activo)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1)
    """, (
        nombre, tipo, material_id, modelo_blob, scaler_blob,
        json.dumps(metricas) if metricas else None,
        json.dumps(parametros) if parametros else None
    ))

    modelo_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return modelo_id


def cargar_modelo(
    tipo: str,
    material_id: int = None
) -> Tuple[Any, Any, Dict]:
    """Carga el modelo activo de un tipo específico"""
    conn = get_connection()
    cursor = conn.cursor()

    if material_id:
        cursor.execute("""
            SELECT modelo_blob, scaler_blob, metricas
            FROM modelos_ml
            WHERE tipo = ? AND material_id = ? AND activo = 1
            ORDER BY created_at DESC LIMIT 1
        """, (tipo, material_id))
    else:
        cursor.execute("""
            SELECT modelo_blob, scaler_blob, metricas
            FROM modelos_ml
            WHERE tipo = ? AND material_id IS NULL AND activo = 1
            ORDER BY created_at DESC LIMIT 1
        """, (tipo,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None, None, {}

    modelo = pickle.loads(row['modelo_blob'])
    scaler = pickle.loads(row['scaler_blob']) if row['scaler_blob'] else None
    metricas = json.loads(row['metricas']) if row['metricas'] else {}

    return modelo, scaler, metricas


def registrar_entrenamiento(
    modelo_id: int,
    registros_usados: int,
    metricas: Dict,
    duracion_segundos: float,
    estado: str = 'completado',
    error: str = None
):
    """Registra un evento de entrenamiento"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO entrenamientos_log
        (modelo_id, fecha_inicio, fecha_fin, registros_usados,
         mae, rmse, r2, mape, estado, error_mensaje)
        VALUES (?, datetime('now', ?), datetime('now'), ?, ?, ?, ?, ?, ?, ?)
    """, (
        modelo_id,
        f'-{duracion_segundos} seconds',
        registros_usados,
        metricas.get('mae'),
        metricas.get('rmse'),
        metricas.get('r2'),
        metricas.get('mape'),
        estado,
        error
    ))

    conn.commit()
    conn.close()


# ============================================================
# Alertas
# ============================================================

def crear_alerta(
    material_id: int,
    tipo_alerta: str,
    severidad: str,
    mensaje: str,
    valor_actual: float = None,
    valor_umbral: float = None
) -> int:
    """Crea una nueva alerta"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO alertas
        (material_id, tipo_alerta, severidad, mensaje, valor_actual, valor_umbral)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (material_id, tipo_alerta, severidad, mensaje, valor_actual, valor_umbral))

    alerta_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return alerta_id


def obtener_alertas_activas(severidad: str = None) -> List[Dict]:
    """Obtiene alertas no resueltas"""
    conn = get_connection()
    cursor = conn.cursor()
    params = []

    query = """
        SELECT a.*, m.codigo, m.descripcion
        FROM alertas a
        JOIN materiales m ON a.material_id = m.id
        WHERE a.resuelta = 0
    """

    if severidad:
        # Validar severidad con whitelist
        allowed_severities = ('alta', 'media', 'baja', 'critica')
        severidad_safe = sanitize_string(severidad, max_length=20).lower()
        if severidad_safe in allowed_severities:
            query += " AND a.severidad = ?"
            params.append(severidad_safe)

    query += " ORDER BY a.created_at DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def resolver_alerta(alerta_id: int):
    """Marca una alerta como resuelta"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE alertas SET resuelta = 1, resolved_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (alerta_id,))
    conn.commit()
    conn.close()


# ============================================================
# Configuración
# ============================================================

def get_config(clave: str) -> Optional[str]:
    """Obtiene un valor de configuración"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT valor FROM configuracion WHERE clave = ?", (clave,))
    row = cursor.fetchone()
    conn.close()
    return row['valor'] if row else None


def set_config(clave: str, valor: str):
    """Establece un valor de configuración"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO configuracion (clave, valor, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(clave) DO UPDATE SET
            valor = excluded.valor,
            updated_at = CURRENT_TIMESTAMP
    """, (clave, valor))
    conn.commit()
    conn.close()


# ============================================================
# Consultas del Agente
# ============================================================

def registrar_consulta_agente(
    consulta: str,
    respuesta: str,
    contexto: Dict = None,
    tokens_usados: int = None,
    tiempo_ms: int = None
) -> int:
    """Registra una consulta al agente IA"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO agente_consultas
        (consulta, respuesta, contexto, tokens_usados, tiempo_respuesta_ms)
        VALUES (?, ?, ?, ?, ?)
    """, (consulta, respuesta, json.dumps(contexto) if contexto else None,
          tokens_usados, tiempo_ms))

    consulta_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return consulta_id


# ============================================================
# Utilidades
# ============================================================

def importar_desde_dataframe(df: pd.DataFrame) -> Dict[str, int]:
    """
    Importa datos desde un DataFrame a la base de datos.

    Optimizado con bulk insert para mejor rendimiento.

    Returns:
        Dict con contadores de registros importados
    """
    if df is None or len(df) == 0:
        return {'materiales': 0, 'stock': 0}

    conn = get_connection()
    cursor = conn.cursor()
    fecha_hoy = datetime.now().strftime('%Y-%m-%d')

    # Preparar datos para bulk insert de materiales (vectorizado)
    # Asegurar columnas con valores por defecto
    df_copy = df.copy()
    df_copy['codigo'] = df_copy.get('codigo', pd.Series([''] * len(df_copy))).astype(str)
    df_copy['descripcion'] = df_copy.get('descripcion', pd.Series([''] * len(df_copy))).astype(str)
    df_copy['centro'] = df_copy.get('centro', pd.Series([''] * len(df_copy))).astype(str)
    df_copy['almacen'] = df_copy.get('almacen', pd.Series([''] * len(df_copy))).astype(str)
    df_copy['proveedor'] = df_copy.get('proveedor', pd.Series([''] * len(df_copy))).astype(str)
    df_copy['lead_time'] = df_copy.get('lead_time', pd.Series([14] * len(df_copy))).fillna(14).astype(int)
    df_copy['costo_unitario'] = df_copy.get('costo_unitario', pd.Series([0.0] * len(df_copy))).fillna(0).astype(float)

    materiales_data = list(df_copy[['codigo', 'descripcion', 'centro', 'almacen',
                                     'proveedor', 'lead_time', 'costo_unitario']].itertuples(index=False, name=None))

    # Bulk insert de materiales con executemany
    cursor.executemany("""
        INSERT INTO materiales (codigo, descripcion, centro, almacen,
                               proveedor, lead_time_dias, costo_unitario)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(codigo) DO UPDATE SET
            descripcion = excluded.descripcion,
            updated_at = CURRENT_TIMESTAMP
    """, materiales_data)

    materiales_importados = len(materiales_data)

    # Obtener IDs de materiales insertados/actualizados
    codigos = [d[0] for d in materiales_data]
    placeholders = ','.join(['?' for _ in codigos])
    cursor.execute(f"SELECT id, codigo FROM materiales WHERE codigo IN ({placeholders})", codigos)
    codigo_to_id = {row[1]: row[0] for row in cursor.fetchall()}

    # Preparar datos para bulk insert de stock_historico (vectorizado)
    df_copy['material_id'] = df_copy['codigo'].map(codigo_to_id)
    df_copy['stock_actual'] = df_copy.get('stock_actual', pd.Series([0.0] * len(df_copy))).fillna(0).astype(float)
    df_copy['stock_seguridad'] = df_copy.get('stock_seguridad', pd.Series([0.0] * len(df_copy))).fillna(0).astype(float)
    df_copy['punto_pedido'] = df_copy.get('punto_pedido', pd.Series([0.0] * len(df_copy))).fillna(0).astype(float)
    df_copy['stock_maximo'] = df_copy.get('stock_maximo', pd.Series([0.0] * len(df_copy))).fillna(0).astype(float)
    df_copy['valor_inventario'] = df_copy['stock_actual'] * df_copy['costo_unitario']
    df_copy['fecha'] = fecha_hoy

    # Filtrar solo registros con material_id válido
    df_stock = df_copy[df_copy['material_id'].notna()].copy()
    df_stock['material_id'] = df_stock['material_id'].astype(int)

    stock_data = list(df_stock[['material_id', 'fecha', 'stock_actual', 'stock_seguridad',
                                 'punto_pedido', 'stock_maximo', 'valor_inventario']].itertuples(index=False, name=None))

    # Bulk insert de stock_historico
    if stock_data:
        cursor.executemany("""
            INSERT INTO stock_historico
            (material_id, fecha, stock_actual, stock_seguridad,
             punto_pedido, stock_maximo, valor_inventario)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(material_id, fecha) DO UPDATE SET
                stock_actual = excluded.stock_actual,
                stock_seguridad = excluded.stock_seguridad,
                punto_pedido = excluded.punto_pedido,
                stock_maximo = excluded.stock_maximo,
                valor_inventario = excluded.valor_inventario
        """, stock_data)

    conn.commit()
    conn.close()

    logger.info(f"Importación bulk completada: {materiales_importados} materiales, {len(stock_data)} registros de stock")

    return {
        'materiales': materiales_importados,
        'stock': len(stock_data)
    }


def get_estadisticas_db() -> Dict[str, Any]:
    """Obtiene estadísticas de la base de datos"""
    conn = get_connection()
    cursor = conn.cursor()

    stats = {}

    # Queries pre-definidas para cada tabla (sin SQL dinámico)
    # SEGURIDAD: No se usa interpolación de strings en queries SQL
    TABLE_COUNT_QUERIES = {
        'materiales': "SELECT COUNT(*) FROM materiales",
        'stock_historico': "SELECT COUNT(*) FROM stock_historico",
        'consumos': "SELECT COUNT(*) FROM consumos",
        'predicciones': "SELECT COUNT(*) FROM predicciones",
        'modelos_ml': "SELECT COUNT(*) FROM modelos_ml",
        'alertas': "SELECT COUNT(*) FROM alertas",
        'agente_consultas': "SELECT COUNT(*) FROM agente_consultas"
    }

    # Conteo de registros usando queries pre-definidas
    for table_name, query in TABLE_COUNT_QUERIES.items():
        try:
            cursor.execute(query)
            stats[table_name] = cursor.fetchone()[0]
        except Exception as e:
            logger.debug(f"Error obteniendo stats de {table_name}: {e}")
            stats[table_name] = 0

    # Alertas activas
    cursor.execute("SELECT COUNT(*) FROM alertas WHERE resuelta = 0")
    stats['alertas_activas'] = cursor.fetchone()[0]

    # Modelos activos
    cursor.execute("SELECT COUNT(*) FROM modelos_ml WHERE activo = 1")
    stats['modelos_activos'] = cursor.fetchone()[0]

    conn.close()
    return stats


# Inicializar la base de datos al importar el módulo
if __name__ == "__main__":
    init_database()
    logger.info("Base de datos inicializada correctamente")
    logger.info(get_estadisticas_db())
