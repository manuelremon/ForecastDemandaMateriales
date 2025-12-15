"""
Forecast - Aplicacion de Prediccion de Demanda
===============================================
Aplicacion Dash para forecasting de materiales con ML
"""
import os
import sys
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
from loguru import logger

# Configuracion de logging
logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level="INFO",
    colorize=True
)

# Inicializar la aplicacion Dash
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True,
    title="Forecast ML",
    update_title="Cargando..."
)

server = app.server

# Importar paginas
from src.pages import demanda, forecast_masivo

# Importar callbacks
from src.callbacks import demanda_callbacks, forecast_masivo_callbacks

# Estilos del sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "220px",
    "padding": "1rem",
    "backgroundColor": "#1e293b",
    "color": "white",
    "zIndex": 1000,
}

CONTENT_STYLE = {
    "marginLeft": "240px",
    "padding": "2rem",
    "minHeight": "100vh",
}


def create_sidebar():
    """Crea el sidebar de navegacion"""
    from src.components.icons import lucide_icon

    return html.Div([
        # Logo/Titulo
        html.Div([
            html.Div([
                lucide_icon("brain-circuit", size="2x", style={"color": "#3b82f6"}),
                html.Div([
                    html.H4("Forecast", className="mb-0", style={"fontWeight": "700"}),
                    html.Small("ML Predictions", className="text-muted")
                ], className="ms-2")
            ], className="d-flex align-items-center")
        ], className="mb-4 pb-3", style={"borderBottom": "1px solid rgba(255,255,255,0.1)"}),

        # Navegacion
        dbc.Nav([
            dbc.NavLink([
                lucide_icon("line-chart", size="sm", className="me-2"),
                "Forecast Individual"
            ], href="/", active="exact", className="nav-link-sidebar"),
            dbc.NavLink([
                lucide_icon("layers", size="sm", className="me-2"),
                "Forecast Masivo"
            ], href="/masivo", active="exact", className="nav-link-sidebar"),
        ], vertical=True, pills=True),

        # Footer
        html.Div([
            html.Hr(style={"borderColor": "rgba(255,255,255,0.1)"}),
            html.Small([
                lucide_icon("info", size="xs", className="me-1"),
                "v1.0.0"
            ], className="text-muted")
        ], style={"position": "absolute", "bottom": "1rem", "left": "1rem"})
    ], style=SIDEBAR_STYLE)


def create_layout():
    """Crea el layout principal de la aplicacion"""
    return html.Div([
        dcc.Location(id="url", refresh=False),
        create_sidebar(),
        html.Div(id="page-content", style=CONTENT_STYLE)
    ])


# Asignar layout
app.layout = create_layout()


# Callback para navegacion
@callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    """Renderiza la pagina segun la URL"""
    if pathname == "/" or pathname == "/forecast":
        return demanda.layout
    elif pathname == "/masivo":
        return forecast_masivo.layout
    else:
        return html.Div([
            html.H1("404 - Pagina no encontrada", className="text-danger"),
            html.P(f"La ruta '{pathname}' no existe."),
            dbc.Button("Ir al inicio", href="/", color="primary")
        ], className="text-center py-5")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8051))
    debug = os.environ.get("FLASK_ENV", "development") == "development"

    if debug:
        logger.info("Modo desarrollo: Cache de assets deshabilitado")
        app.config.suppress_callback_exceptions = True

    logger.info(f"Iniciando Forecast en puerto {port}")
    app.run(debug=debug, host="0.0.0.0", port=port)
