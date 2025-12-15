"""
Helpers para graficos Plotly
============================
Funciones utilitarias para crear graficos consistentes.
"""
import plotly.graph_objects as go
from src.utils.theme import PLOTLY_TEMPLATE, COLORS


def crear_figura_vacia(mensaje: str = "Sin datos", color_texto: str = None) -> go.Figure:
    """
    Crea una figura Plotly vacia con un mensaje centrado.

    Args:
        mensaje: Texto a mostrar en el centro del grafico
        color_texto: Color del texto (default: text_secondary del theme)

    Returns:
        go.Figure con el mensaje centrado
    """
    if color_texto is None:
        color_texto = COLORS.get('text_secondary', '#8e8e93')

    # Filtrar margin del template para evitar duplicado
    layout_base = {k: v for k, v in PLOTLY_TEMPLATE["layout"].items() if k != "margin"}

    fig = go.Figure()
    fig.update_layout(
        **layout_base,
        annotations=[{
            "text": mensaje,
            "showarrow": False,
            "font": {"size": 12, "color": color_texto},
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": 0.5
        }],
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


def crear_figura_error(mensaje: str = "Error al cargar datos") -> go.Figure:
    """
    Crea una figura Plotly indicando un error.

    Args:
        mensaje: Texto de error a mostrar

    Returns:
        go.Figure con el mensaje de error
    """
    return crear_figura_vacia(mensaje, color_texto=COLORS.get('danger', '#ff3b30'))


def crear_figura_warning(mensaje: str = "Advertencia") -> go.Figure:
    """
    Crea una figura Plotly indicando una advertencia.

    Args:
        mensaje: Texto de advertencia a mostrar

    Returns:
        go.Figure con el mensaje de advertencia
    """
    return crear_figura_vacia(mensaje, color_texto=COLORS.get('warning', '#ff9500'))
