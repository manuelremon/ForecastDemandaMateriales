"""
AG Grid Cell Renderers
======================
Renderers con iconos Lucide para AG Grid
"""
from typing import Optional

# SVG paths para iconos comunes (inline para mejor rendimiento en AG Grid)
LUCIDE_PATHS = {
    "check-circle": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "x-circle": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    "alert-triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "alert-circle": '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>',
    "shield-alert": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
    "trending-up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "arrow-up-circle": '<circle cx="12" cy="12" r="10"/><path d="m16 12-4-4-4 4"/><path d="M12 16V8"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "arrow-up": '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
    "arrow-down": '<path d="m19 12-7 7-7-7"/><path d="M12 5v14"/>',
    "minus": '<path d="M5 12h14"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
}


def get_lucide_svg(icon_name: str, size: int = 16, color: str = "currentColor") -> str:
    """
    Genera SVG inline de Lucide para AG Grid

    Args:
        icon_name: Nombre del icono Lucide
        size: Tamano en pixeles
        color: Color del icono

    Returns:
        str: SVG como string HTML
    """
    path = LUCIDE_PATHS.get(icon_name, '<circle cx="12" cy="12" r="10"/>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle;">{path}</svg>'''


# Configuracion de estados MRP
STATUS_CONFIG = {
    "quiebre": {
        "icon": "x-circle",
        "color": "#FF3B30",
        "bg": "rgba(255,59,48,0.15)",
        "label": "Quiebre"
    },
    "bajo_punto_pedido": {
        "icon": "alert-triangle",
        "color": "#FF9500",
        "bg": "rgba(255,149,0,0.15)",
        "label": "Bajo PP"
    },
    "bajo_seguridad": {
        "icon": "shield-alert",
        "color": "#FFCC00",
        "bg": "rgba(255,204,0,0.15)",
        "label": "Bajo Seg."
    },
    "sobrestock_critico": {
        "icon": "arrow-up-circle",
        "color": "#5856D6",
        "bg": "rgba(88,86,214,0.15)",
        "label": "Sobre. Crit."
    },
    "sobrestock": {
        "icon": "trending-up",
        "color": "#007AFF",
        "bg": "rgba(0,122,255,0.15)",
        "label": "Sobrestock"
    },
    "bajo_consumo": {
        "icon": "clock",
        "color": "#8E8E93",
        "bg": "rgba(142,142,147,0.15)",
        "label": "Bajo Cons."
    },
    "normal": {
        "icon": "check-circle",
        "color": "#4CD964",
        "bg": "rgba(76,217,100,0.15)",
        "label": "Normal"
    },
}


def format_status_cell(estado: str, texto: Optional[str] = None) -> str:
    """
    Formatea una celda de estado con icono y badge para AG Grid

    Args:
        estado: Clave del estado (quiebre, bajo_punto_pedido, etc.)
        texto: Texto a mostrar (opcional, usa label por defecto)

    Returns:
        str: HTML formateado para AG Grid
    """
    config = STATUS_CONFIG.get(estado.lower().replace(" ", "_"), STATUS_CONFIG["normal"])
    icon_svg = get_lucide_svg(config["icon"], size=14, color=config["color"])
    display_text = texto or config["label"]

    return f'''<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:6px;background:{config['bg']};font-size:11px;font-weight:600;color:{config['color']};">{icon_svg}{display_text}</span>'''


def format_trend_cell(valor: float, formato: str = "percent") -> str:
    """
    Formatea una celda de tendencia con icono direccional

    Args:
        valor: Valor numerico de la tendencia
        formato: Tipo de formato ('percent' o 'number')

    Returns:
        str: HTML formateado para AG Grid
    """
    if valor > 0:
        icon_name = "arrow-up"
        color = "#4CD964"
        prefix = "+"
    elif valor < 0:
        icon_name = "arrow-down"
        color = "#FF3B30"
        prefix = ""
    else:
        icon_name = "minus"
        color = "#8E8E93"
        prefix = ""

    icon_svg = get_lucide_svg(icon_name, size=12, color=color)

    if formato == "percent":
        text = f"{prefix}{valor:.1f}%"
    else:
        text = f"{prefix}{valor:,.0f}"

    return f'<span style="display:inline-flex;align-items:center;gap:4px;color:{color};font-weight:500;">{icon_svg}{text}</span>'


def format_boolean_cell(valor: bool) -> str:
    """
    Formatea una celda booleana con icono check/x

    Args:
        valor: Valor booleano

    Returns:
        str: HTML formateado para AG Grid
    """
    if valor:
        icon_svg = get_lucide_svg("check", size=16, color="#4CD964")
    else:
        icon_svg = get_lucide_svg("x", size=16, color="#FF3B30")

    return f'<span style="display:flex;justify-content:center;">{icon_svg}</span>'


def format_priority_cell(priority: str) -> str:
    """
    Formatea una celda de prioridad con color

    Args:
        priority: Nivel de prioridad ('alta', 'media', 'baja')

    Returns:
        str: HTML formateado para AG Grid
    """
    colors = {
        "alta": ("#FF3B30", "rgba(255,59,48,0.15)"),
        "media": ("#FF9500", "rgba(255,149,0,0.15)"),
        "baja": ("#4CD964", "rgba(76,217,100,0.15)"),
    }

    color, bg = colors.get(priority.lower(), ("#8E8E93", "rgba(142,142,147,0.15)"))

    return f'''<span style="display:inline-flex;padding:4px 10px;border-radius:6px;background:{bg};font-size:11px;font-weight:600;color:{color};">{priority.title()}</span>'''
