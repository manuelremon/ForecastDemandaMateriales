"""
Tema visual y configuracion de colores para MRP Analytics
Paleta: Apple iOS - Glassmorphism Design System
"""

# Colores del sistema - iOS Palette
COLORS = {
    # Fondos iOS (NO usar #FFFFFF puro)
    "bg_primary": "#F7F7F7",        # Fondo principal (claro)
    "bg_secondary": "#F7F7F7",      # Cards/superficies (CORREGIDO)
    "bg_dark": "#1F1F21",           # Sidebar/elementos oscuros
    "bg_glass": "rgba(247, 247, 247, 0.72)",  # Glass effect (CORREGIDO)

    # Colores primarios iOS
    "primary": "#007AFF",           # iOS Blue
    "purple": "#5856D6",            # iOS Purple
    "pink": "#FF2D55",              # iOS Pink
    "teal": "#5AC8FA",              # iOS Light Blue

    # Estados semanticos
    "success": "#4CD964",           # iOS Green
    "warning": "#FF9500",           # iOS Orange
    "danger": "#FF3B30",            # iOS Red
    "info": "#5AC8FA",              # iOS Light Blue
    "yellow": "#FFCC00",            # iOS Yellow

    # Grises iOS
    "text_primary": "#1F1F21",      # Texto principal
    "text_secondary": "#8E8E93",    # Texto secundario
    "text_muted": "#BDBEC2",        # Texto deshabilitado
    "border": "#C7C7CC",            # Bordes/separadores
    "border_light": "#D6CEC3",      # Bordes sutiles
    "grid_color": "rgba(199, 199, 204, 0.3)",  # Grid para graficos

    # Estados especificos MRP
    "quiebre": "#FF3B30",           # Critico - iOS Red
    "bajo_punto_pedido": "#FF9500", # Advertencia - iOS Orange
    "bajo_seguridad": "#FFCC00",    # Precaucion - iOS Yellow
    "sobrestock": "#5AC8FA",        # Info - iOS Teal (CORREGIDO)
    "sobrestock_critico": "#5856D6", # Atencion - iOS Purple
    "bajo_consumo": "#8E8E93",      # Neutro - iOS Gray
    "normal": "#4CD964",            # OK - iOS Green
}


def color_con_alpha(color_key: str, alpha: float = 0.2) -> str:
    """
    Convierte un color HEX de COLORS a RGBA con alpha especificado.
    Util para fillcolor de graficos Plotly.

    Args:
        color_key: Clave del color en COLORS dict
        alpha: Valor de opacidad (0.0 - 1.0)

    Returns:
        String RGBA, ej: "rgba(0, 122, 255, 0.2)"
    """
    hex_color = COLORS.get(color_key, '#007AFF')
    # Remover # si existe
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

# Gradientes para cards (glassmorphism)
GRADIENTS = {
    "primary": "linear-gradient(135deg, rgba(0, 122, 255, 0.9) 0%, rgba(0, 102, 214, 0.9) 100%)",
    "success": "linear-gradient(135deg, rgba(76, 217, 100, 0.9) 0%, rgba(60, 180, 80, 0.9) 100%)",
    "warning": "linear-gradient(135deg, rgba(255, 149, 0, 0.9) 0%, rgba(230, 130, 0, 0.9) 100%)",
    "danger": "linear-gradient(135deg, rgba(255, 59, 48, 0.9) 0%, rgba(220, 50, 40, 0.9) 100%)",
    "glass": "linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%)",
    "dark": "linear-gradient(135deg, rgba(31, 31, 33, 0.95) 0%, rgba(20, 20, 22, 0.95) 100%)",
}

# Template de Plotly para tema iOS 17/18
PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
        "font": {
            "color": "#1F1F21",
            "family": "-apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif",
            "size": 13
        },
        "title": {
            "font": {
                "color": "#1F1F21",
                "size": 17,
                "family": "-apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif"
            }
        },
        "xaxis": {
            "gridcolor": "rgba(60, 60, 67, 0.08)",
            "linecolor": "rgba(60, 60, 67, 0.12)",
            "tickcolor": "rgba(0, 0, 0, 0)",
            "zerolinecolor": "rgba(60, 60, 67, 0.12)",
            "tickfont": {"color": "#8E8E93", "size": 11}
        },
        "yaxis": {
            "gridcolor": "rgba(60, 60, 67, 0.08)",
            "linecolor": "rgba(60, 60, 67, 0.12)",
            "tickcolor": "rgba(0, 0, 0, 0)",
            "zerolinecolor": "rgba(60, 60, 67, 0.12)",
            "tickfont": {"color": "#8E8E93", "size": 11}
        },
        "legend": {
            "bgcolor": "rgba(255, 255, 255, 0.92)",
            "bordercolor": "rgba(60, 60, 67, 0.12)",
            "borderwidth": 1,
            "font": {"color": "#1F1F21", "family": "-apple-system, system-ui, sans-serif", "size": 12}
        },
        "colorway": [
            "#007AFF",  # iOS Blue
            "#34C759",  # iOS Green
            "#FF9500",  # iOS Orange
            "#FF3B30",  # iOS Red
            "#AF52DE",  # iOS Purple
            "#5AC8FA",  # iOS Teal
            "#FF2D55",  # iOS Pink
            "#FFCC00",  # iOS Yellow
        ],
        "hoverlabel": {
            "bgcolor": "rgba(28, 28, 30, 0.95)",
            "bordercolor": "rgba(255, 255, 255, 0.1)",
            "font": {"color": "#FFFFFF", "family": "-apple-system, system-ui, sans-serif", "size": 13}
        },
        "margin": {"l": 48, "r": 16, "t": 48, "b": 48}
    }
}

# Colores para graficos de estados MRP
ESTADO_COLORS = {
    "quiebre": "#FF3B30",
    "bajo_punto_pedido": "#FF9500",
    "bajo_seguridad": "#FFCC00",
    "sobrestock_critico": "#5856D6",
    "sobrestock": "#5AC8FA",        # iOS Teal (CORREGIDO)
    "bajo_consumo": "#8E8E93",
    "normal": "#4CD964"
}

# Secuencia de colores para graficos generales
COLOR_SEQUENCE = [
    "#FF3B30",  # Danger - Red
    "#FF9500",  # Warning - Orange
    "#4CD964",  # Success - Green
    "#007AFF",  # Info - Blue
    "#5856D6",  # Purple
    "#5AC8FA",  # Teal
]

# Colores para badges de estado
BADGE_COLORS = {
    "quiebre": {"bg": "#FF3B30", "text": "#FFFFFF"},
    "bajo_punto_pedido": {"bg": "#FF9500", "text": "#FFFFFF"},
    "bajo_seguridad": {"bg": "#FFCC00", "text": "#1F1F21"},
    "sobrestock_critico": {"bg": "#5856D6", "text": "#FFFFFF"},
    "sobrestock": {"bg": "#007AFF", "text": "#FFFFFF"},
    "bajo_consumo": {"bg": "#8E8E93", "text": "#FFFFFF"},
    "normal": {"bg": "#4CD964", "text": "#FFFFFF"},
}

# Estilos para KPI cards
KPI_STYLES = {
    "danger": {
        "accent": "#FF3B30",
        "icon_color": "#FF3B30",
        "class": "kpi-danger"
    },
    "warning": {
        "accent": "#FF9500",
        "icon_color": "#FF9500",
        "class": "kpi-warning"
    },
    "success": {
        "accent": "#4CD964",
        "icon_color": "#4CD964",
        "class": "kpi-success"
    },
    "primary": {
        "accent": "#007AFF",
        "icon_color": "#007AFF",
        "class": "kpi-primary"
    },
    "info": {
        "accent": "#5AC8FA",
        "icon_color": "#5AC8FA",
        "class": "kpi-info"
    },
    "purple": {
        "accent": "#5856D6",
        "icon_color": "#5856D6",
        "class": "kpi-purple"
    },
}
