"""
Conexión y consultas a SAP HANA
================================
Módulo para conectar con SAP HANA y extraer datos MRP

SEGURIDAD: Todas las queries usan parámetros preparados
para prevenir SQL Injection.
"""
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
import os
from dotenv import load_dotenv

from src.utils.logger import get_logger
from src.utils.validators import sanitize_string

logger = get_logger(__name__)

# Cargar variables de entorno
load_dotenv()

# Intentar importar hdbcli
try:
    from hdbcli import dbapi
    HANA_AVAILABLE = True
except ImportError:
    HANA_AVAILABLE = False
    logger.warning("hdbcli no instalado. Conexión SAP HANA no disponible.")


class SAPHanaConnection:
    """
    Gestiona la conexión con SAP HANA

    Ejemplo de uso:
        with SAPHanaConnection() as conn:
            df = conn.get_stock_data()
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
        encrypt: bool = True
    ):
        """
        Inicializa la conexión con SAP HANA

        Los parámetros pueden venir de variables de entorno:
        - HANA_HOST, HANA_PORT, HANA_USER, HANA_PASSWORD
        """
        self.host = host or os.getenv('HANA_HOST', 'localhost')
        self.port = port or int(os.getenv('HANA_PORT', 30015))
        self.user = user or os.getenv('HANA_USER', '')
        self.password = password or os.getenv('HANA_PASSWORD', '')
        self.encrypt = encrypt
        self.connection = None

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def connect(self) -> bool:
        """Establece conexión con SAP HANA"""
        if not HANA_AVAILABLE:
            raise RuntimeError("hdbcli no está instalado. Ejecute: pip install hdbcli")

        try:
            self.connection = dbapi.connect(
                address=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                encrypt=self.encrypt
            )
            return True
        except Exception as e:
            raise ConnectionError(f"Error conectando a SAP HANA: {str(e)}")

    def disconnect(self):
        """Cierra la conexión"""
        if self.connection:
            self.connection.close()
            self.connection = None

    def is_connected(self) -> bool:
        """Verifica si hay conexión activa"""
        return self.connection is not None and self.connection.isconnected()

    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """
        Ejecuta una consulta SQL y retorna un DataFrame

        Args:
            query: Consulta SQL
            params: Parámetros para la consulta (opcional)

        Returns:
            DataFrame con los resultados
        """
        if not self.is_connected():
            raise ConnectionError("No hay conexión activa con SAP HANA")

        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        finally:
            cursor.close()

    # ============================================================
    # Consultas MRP predefinidas
    # ============================================================

    def get_stock_data(
        self,
        centro: str = None,
        almacen: str = None
    ) -> pd.DataFrame:
        """
        Obtiene datos de stock desde SAP HANA

        Query típica para tabla MARD/MARC (Stock por almacén/centro)
        """
        query = """
        SELECT
            MATNR as codigo,
            MAKTX as descripcion,
            WERKS as centro,
            LGORT as almacen,
            LABST as stock_actual,
            EISBE as stock_seguridad,
            MINBE as punto_pedido,
            MABST as stock_maximo,
            STPRS as costo_unitario,
            PLIFZ as lead_time
        FROM SAPABAP1.MARD
        LEFT JOIN SAPABAP1.MAKT ON MARD.MATNR = MAKT.MATNR
        LEFT JOIN SAPABAP1.MARC ON MARD.MATNR = MARC.MATNR AND MARD.WERKS = MARC.WERKS
        LEFT JOIN SAPABAP1.MBEW ON MARD.MATNR = MBEW.MATNR AND MARD.WERKS = MBEW.BWKEY
        WHERE 1=1
        """

        params = []
        if centro:
            query += " AND WERKS = ?"
            params.append(centro)
        if almacen:
            query += " AND LGORT = ?"
            params.append(almacen)

        return self.execute_query(query, tuple(params) if params else None)

    def get_consumo_historico(
        self,
        meses: int = 12,
        centro: str = None
    ) -> pd.DataFrame:
        """
        Obtiene consumo histórico de materiales

        Query típica para tabla MSEG (Movimientos de material)
        """
        # Validar meses como entero positivo
        meses = max(1, min(int(meses), 120))  # Máximo 10 años

        params = []
        query = f"""
        SELECT
            MATNR as codigo,
            WERKS as centro,
            LGORT as almacen,
            BUDAT_MKPF as fecha,
            BWART as tipo_movimiento,
            MENGE as cantidad,
            DMBTR as valor
        FROM SAPABAP1.MSEG
        WHERE BUDAT_MKPF >= ADD_MONTHS(CURRENT_DATE, -{meses})
        AND BWART IN ('201', '202', '261', '262')
        """

        if centro:
            query += " AND WERKS = ?"
            params.append(sanitize_string(centro, max_length=10))

        return self.execute_query(query, tuple(params) if params else None)

    def get_pedidos_pendientes(self, centro: str = None) -> pd.DataFrame:
        """
        Obtiene pedidos de compra pendientes

        Query típica para tabla EKPO/EKKO (Pedidos de compra)
        """
        params = []
        query = """
        SELECT
            EKPO.MATNR as codigo,
            EKPO.WERKS as centro,
            EKPO.MENGE as cantidad_pedida,
            EKPO.NETPR as precio_unitario,
            EKKO.BEDAT as fecha_pedido,
            EKET.EINDT as fecha_entrega_prevista,
            EKKO.LIFNR as proveedor
        FROM SAPABAP1.EKPO
        JOIN SAPABAP1.EKKO ON EKPO.EBELN = EKKO.EBELN
        LEFT JOIN SAPABAP1.EKET ON EKPO.EBELN = EKET.EBELN AND EKPO.EBELP = EKET.EBELP
        WHERE EKPO.LOEKZ = ''
        AND EKPO.ELIKZ = ''
        """

        if centro:
            query += " AND EKPO.WERKS = ?"
            params.append(sanitize_string(centro, max_length=10))

        return self.execute_query(query, tuple(params) if params else None)

    def get_materiales_maestros(self) -> pd.DataFrame:
        """
        Obtiene datos maestros de materiales

        Query típica para tabla MARA/MAKT (Maestro de materiales)
        """
        query = """
        SELECT
            MARA.MATNR as codigo,
            MAKT.MAKTX as descripcion,
            MARA.MTART as tipo_material,
            MARA.MATKL as grupo_material,
            MARA.MEINS as unidad_medida,
            MARA.BRGEW as peso_bruto,
            MARA.NTGEW as peso_neto
        FROM SAPABAP1.MARA
        LEFT JOIN SAPABAP1.MAKT ON MARA.MATNR = MAKT.MATNR
        WHERE MAKT.SPRAS = 'S'  -- Español
        """
        return self.execute_query(query)


def test_connection(
    host: str = None,
    port: int = None,
    user: str = None,
    password: str = None
) -> Tuple[bool, str]:
    """
    Prueba la conexión con SAP HANA

    Returns:
        Tuple (success: bool, message: str)
    """
    if not HANA_AVAILABLE:
        return False, "hdbcli no instalado. Ejecute: pip install hdbcli"

    try:
        conn = SAPHanaConnection(host, port, user, password)
        conn.connect()

        # Verificar con query simple
        df = conn.execute_query("SELECT 1 FROM DUMMY")
        conn.disconnect()

        return True, "Conexión exitosa con SAP HANA"
    except Exception as e:
        return False, f"Error de conexión: {str(e)}"


def get_connection_status() -> Dict[str, Any]:
    """
    Retorna el estado de la configuración de conexión

    Returns:
        Dict con información del estado
    """
    return {
        "hdbcli_installed": HANA_AVAILABLE,
        "host_configured": bool(os.getenv('HANA_HOST')),
        "user_configured": bool(os.getenv('HANA_USER')),
        "password_configured": bool(os.getenv('HANA_PASSWORD')),
        "ready": HANA_AVAILABLE and all([
            os.getenv('HANA_HOST'),
            os.getenv('HANA_USER'),
            os.getenv('HANA_PASSWORD')
        ])
    }
