"""
Icon Registry - Single Source of Truth
======================================
Registro central de iconos y mapeo Font Awesome a Lucide
"""
from enum import Enum
from typing import Dict


class IconName(Enum):
    """Enum con todos los iconos disponibles en la aplicacion"""
    # Navegacion
    MENU = "menu"
    ALERTS = "alert-triangle"
    KPIS = "trending-up"
    FORECAST = "area-chart"
    MASIVO = "layers"
    PLANIFICACION = "list-checks"
    ASISTENTE = "bot"

    # Acciones
    DATABASE = "database"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    EXPORT = "file-down"
    FILE_EXCEL = "file-spreadsheet"
    FILE_PDF = "file-text"
    REFRESH = "refresh-cw"
    PLAY = "play"
    CHECK = "check"
    CLOSE = "x"
    SEARCH = "search"
    FILTER = "filter"
    CALCULATOR = "calculator"
    SEND = "send"

    # Estados MRP
    QUIEBRE = "x-circle"
    BAJO_PUNTO_PEDIDO = "alert-triangle"
    BAJO_SEGURIDAD = "shield-alert"
    SOBRESTOCK = "trending-up"
    SOBRESTOCK_CRITICO = "arrow-up-circle"
    BAJO_CONSUMO = "clock"
    NORMAL = "check-circle"

    # KPIs y Metricas
    CHART_LINE = "line-chart"
    CHART_BAR = "bar-chart-3"
    CHART_PIE = "pie-chart"
    CHART_AREA = "area-chart"
    DOLLAR = "dollar-sign"
    CALENDAR = "calendar"
    CALENDAR_CHECK = "calendar-check"
    TRUCK = "truck"
    PIGGY_BANK = "piggy-bank"
    SHOPPING_CART = "shopping-cart"
    BOX = "box"
    BOXES = "boxes"
    PACKAGE = "package"
    PACKAGE_OPEN = "package-open"
    ARCHIVE = "archive"
    WAREHOUSE = "warehouse"
    TARGET = "target"

    # UI General
    INFO = "info"
    WARNING = "alert-circle"
    ERROR = "alert-octagon"
    SUCCESS = "check-circle-2"
    ARROW_UP = "arrow-up"
    ARROW_DOWN = "arrow-down"
    MINUS = "minus"
    CLOUD_UPLOAD = "cloud-upload"
    MAGIC = "sparkles"
    LIGHTBULB = "lightbulb"
    MOUSE_POINTER = "mouse-pointer"
    ROCKET = "rocket"
    USER = "user"
    BUILDING = "building"
    INBOX = "inbox"
    TABLE = "table-2"
    SYNC = "refresh-cw"
    CUBE = "box"
    CUBES = "boxes"
    CIRCLE = "circle"
    HAND_POINTER = "pointer"


# Mapeo Font Awesome -> Lucide (compatibilidad hacia atras)
FA_TO_LUCIDE_MAP: Dict[str, str] = {
    # Navegacion/UI
    "fa-bars": "menu",
    "fa-search": "search",
    "fa-times": "x",
    "fa-check": "check",
    "fa-play": "play",
    "fa-redo": "refresh-cw",
    "fa-filter": "filter",

    # Alertas/Estados
    "fa-exclamation-triangle": "alert-triangle",
    "fa-exclamation-circle": "alert-circle",
    "fa-times-circle": "x-circle",
    "fa-check-circle": "check-circle",
    "fa-info-circle": "info",

    # Graficos/KPIs
    "fa-chart-line": "line-chart",
    "fa-chart-area": "area-chart",
    "fa-chart-bar": "bar-chart-3",
    "fa-chart-pie": "pie-chart",
    "fa-bullseye": "target",

    # Inventario/MRP
    "fa-box": "box",
    "fa-boxes": "boxes",
    "fa-box-open": "package-open",
    "fa-archive": "archive",
    "fa-warehouse": "warehouse",
    "fa-database": "database",
    "fa-shield-alt": "shield-alert",

    # Acciones
    "fa-upload": "upload",
    "fa-download": "download",
    "fa-file-upload": "file-up",
    "fa-file-excel": "file-spreadsheet",
    "fa-file-pdf": "file-text",
    "fa-paper-plane": "send",

    # Finanzas
    "fa-dollar-sign": "dollar-sign",
    "fa-piggy-bank": "piggy-bank",
    "fa-shopping-cart": "shopping-cart",

    # Tiempo/Calendario
    "fa-calendar": "calendar",
    "fa-calendar-alt": "calendar-days",
    "fa-calendar-check": "calendar-check",
    "fa-clock": "clock",

    # Transporte
    "fa-truck": "truck",

    # Tendencias
    "fa-arrow-up": "arrow-up",
    "fa-arrow-down": "arrow-down",
    "fa-arrow-trend-up": "trending-up",
    "fa-minus": "minus",
    "fa-sync-alt": "refresh-cw",

    # Miscelaneos
    "fa-robot": "bot",
    "fa-magic": "sparkles",
    "fa-lightbulb": "lightbulb",
    "fa-user": "user",
    "fa-building": "building",
    "fa-layer-group": "layers",
    "fa-tasks": "list-checks",
    "fa-rocket": "rocket",
    "fa-hand-pointer": "pointer",
    "fa-mouse-pointer": "mouse-pointer",
    "fa-calculator": "calculator",
    "fa-cube": "box",
    "fa-cubes": "boxes",
    "fa-circle": "circle",
    "fa-inbox": "inbox",
    "fa-table": "table-2",
    "fa-cloud-upload-alt": "cloud-upload",
}


# Configuracion de tamanos
ICON_SIZES: Dict[str, int] = {
    "xs": 12,
    "sm": 14,
    "md": 16,
    "lg": 20,
    "xl": 24,
    "2x": 32,
    "3x": 48,
    "4x": 64,
    "5x": 80,
}
