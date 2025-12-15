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
    "width": "120px",
    "padding": "0.75rem",
    "backgroundColor": "#1e293b",
    "color": "white",
    "zIndex": 1000,
    "display": "flex",
    "flexDirection": "column",
}

CONTENT_STYLE = {
    "marginLeft": "140px",
    "padding": "2rem",
    "minHeight": "100vh",
    "overflow": "visible",
}


def create_sidebar():
    """Crea el sidebar de navegacion compacto (120px)"""
    from src.components.icons import lucide_icon

    return html.Div([
        # Logo/Titulo centrado
        html.Div([
            html.Img(src="/assets/logo.png", style={"height": "36px", "width": "auto", "marginBottom": "4px"}),
            html.Div("Forecast", style={"fontWeight": "700", "fontSize": "0.85rem"}),
        ], className="text-center mb-3 pb-2", style={"borderBottom": "1px solid rgba(255,255,255,0.1)"}),

        # Navegacion compacta
        dbc.Nav([
            dbc.NavLink([
                lucide_icon("line-chart", size="sm"),
                html.Div("Individual", style={"fontSize": "0.65rem", "marginTop": "2px"})
            ], href="/", active="exact", className="nav-link-sidebar text-center flex-column py-2"),
            dbc.NavLink([
                lucide_icon("layers", size="sm"),
                html.Div("Masivo", style={"fontSize": "0.65rem", "marginTop": "2px"})
            ], href="/masivo", active="exact", className="nav-link-sidebar text-center flex-column py-2"),
        ], vertical=True, pills=True),

        # Seccion Datos compacta
        html.Div([
            html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "0.5rem 0"}),

            # Estado de Datos
            html.Div([
                lucide_icon("database", size="xs"),
                html.Span(id="indicador-estado-datos", className="ms-1")
            ], className="text-center mb-2", style={"fontSize": "0.65rem"}),

            # Boton descargar plantilla (solo icono)
            dbc.Button(
                lucide_icon("download", size="sm"),
                id="btn-descargar-plantilla-sidebar", color="secondary",
                outline=True, size="sm", className="w-100 mb-2",
                title="Descargar plantilla Excel"),

            # Upload de archivo (solo icono)
            dcc.Upload(
                id="upload-excel-sidebar",
                children=dbc.Button(
                    lucide_icon("arrow-up", size="sm"),
                    color="primary", outline=True, size="sm", className="w-100",
                    title="Importar datos Excel"),
                accept=".xlsx,.xls",
                style={"width": "100%"}
            ),

            # Indicador de archivo cargado
            html.Div(id="info-archivo-sidebar", className="mt-2 text-center",
                    style={"fontSize": "0.6rem"})

        ], className="mt-2"),

        # Espaciador flexible
        html.Div(style={"flex": "1"}),

        # Footer con botón Información
        html.Div([
            html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "0.5rem 0"}),
            dbc.Button([
                lucide_icon("info", size="sm"),
                html.Div("Info", style={"fontSize": "0.65rem", "marginTop": "2px"})
            ], id="btn-info-modal", color="link", className="w-100 text-white d-flex flex-column align-items-center py-2"),
            html.Small("v1.0", className="text-muted d-block text-center", style={"fontSize": "0.6rem"})
        ])
    ], style=SIDEBAR_STYLE)


def create_info_modal():
    """Crea el modal de información de la aplicación"""
    from src.components.icons import lucide_icon

    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            lucide_icon("info", size="sm", className="me-2"),
            "Información de la Aplicación"
        ]), close_button=True),
        dbc.ModalBody([
            # Sección: Qué es
            html.H5([lucide_icon("target", size="sm", className="me-2"), "¿Qué es Forecast ML?"],
                   className="text-primary mb-3"),
            html.P([
                "Forecast ML es una aplicación de ", html.Strong("predicción de demanda"),
                " que utiliza algoritmos de Machine Learning para estimar el consumo futuro de materiales ",
                "basándose en datos históricos."
            ], className="mb-4"),

            # Sección: Modelos ML
            html.H5([lucide_icon("cpu", size="sm", className="me-2"), "Modelos de Predicción"],
                   className="text-primary mb-3"),
            html.Ul([
                html.Li([html.Strong("Random Forest: "), "Modelo ensemble que combina múltiples árboles de decisión. Robusto ante outliers y datos ruidosos."]),
                html.Li([html.Strong("Gradient Boosting: "), "Construye árboles secuencialmente, corrigiendo errores previos. Alta precisión."]),
                html.Li([html.Strong("Linear Regression: "), "Modelo simple y rápido. Ideal para tendencias lineales."]),
                html.Li([html.Strong("Ridge/Lasso: "), "Regresión con regularización para evitar overfitting."]),
                html.Li([html.Strong("SVR: "), "Support Vector Regression. Efectivo con datos no lineales."]),
            ], className="mb-4"),

            # Sección: Datos
            html.H5([lucide_icon("database", size="sm", className="me-2"), "Datos Utilizados"],
                   className="text-primary mb-3"),
            html.P("La aplicación analiza:", className="mb-2"),
            html.Ul([
                html.Li("Consumo histórico por material, centro y almacén"),
                html.Li("Patrones semanales (día de la semana)"),
                html.Li("Estacionalidad mensual y anual"),
                html.Li("Tendencias a largo plazo"),
            ], className="mb-4"),

            # Sección: Tips
            html.H5([lucide_icon("lightbulb", size="sm", className="me-2"), "Tips para Mejores Resultados"],
                   className="text-warning mb-3"),
            dbc.Alert([
                html.Ul([
                    html.Li("Usa al menos 6 meses de datos históricos para mejores predicciones"),
                    html.Li("Datos más recientes tienen mayor peso en el modelo"),
                    html.Li("Random Forest es el modelo más estable para la mayoría de casos"),
                    html.Li("Ajusta el horizonte según tu necesidad (7-365 días)"),
                    html.Li("El nivel de confianza 95% ofrece buen balance precisión/rango"),
                ], className="mb-0")
            ], color="warning", className="mb-4"),

            # Sección: Métricas
            html.H5([lucide_icon("chart-bar", size="sm", className="me-2"), "Interpretación de Métricas"],
                   className="text-primary mb-3"),
            html.Ul([
                html.Li([html.Strong("R² (Coeficiente de determinación): "), "Valores cercanos a 1.0 indican mejor ajuste."]),
                html.Li([html.Strong("MAE (Error Absoluto Medio): "), "Promedio de errores. Menor es mejor."]),
                html.Li([html.Strong("RMSE: "), "Similar a MAE pero penaliza más los errores grandes."]),
            ], className="mb-4"),

            # Sección: Uso
            html.H5([lucide_icon("rocket", size="sm", className="me-2"), "¿Cómo usar la aplicación?"],
                   className="text-success mb-3"),
            html.Ol([
                html.Li("Descarga la plantilla Excel desde el sidebar"),
                html.Li("Llena los datos de consumo histórico"),
                html.Li("Importa el archivo Excel"),
                html.Li("Selecciona el material, centro y almacén"),
                html.Li("Configura el modelo, horizonte y confianza"),
                html.Li("Haz clic en 'Generar' para ver la predicción"),
            ]),
        ]),
        dbc.ModalFooter(
            dbc.Button("Cerrar", id="btn-cerrar-info-modal", className="ms-auto")
        ),
    ], id="modal-info", size="lg", scrollable=True)


def create_layout():
    """Crea el layout principal de la aplicacion"""
    return html.Div([
        dcc.Location(id="url", refresh=False),
        # Store global para datos de Excel (session storage)
        dcc.Store(id="store-excel-data", storage_type="session"),
        # Store legacy para compatibilidad con callbacks
        dcc.Store(id="data-store", storage_type="memory"),
        # Download para plantilla
        dcc.Download(id="download-plantilla-sidebar"),
        create_sidebar(),
        create_info_modal(),
        html.Div(id="page-content", style=CONTENT_STYLE)
    ])


# Asignar layout
app.layout = create_layout()

# Importar funciones de procesamiento de Excel
from src.data.excel_loader import cargar_excel_forecast
from src.components.icons import lucide_icon
from pathlib import Path

PLANTILLA_PATH = Path(__file__).parent / "data" / "plantilla_forecast.xlsx"


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


# Callback para descargar plantilla desde sidebar
@callback(
    Output("download-plantilla-sidebar", "data"),
    Input("btn-descargar-plantilla-sidebar", "n_clicks"),
    prevent_initial_call=True
)
def descargar_plantilla_sidebar(n_clicks):
    """Descarga la plantilla Excel desde el sidebar"""
    if n_clicks and PLANTILLA_PATH.exists():
        return dcc.send_file(str(PLANTILLA_PATH), filename="plantilla_forecast.xlsx")
    return None


# Callback para procesar Excel subido desde sidebar
@callback(
    Output("store-excel-data", "data"),
    Output("info-archivo-sidebar", "children"),
    Input("upload-excel-sidebar", "contents"),
    State("upload-excel-sidebar", "filename"),
    prevent_initial_call=True
)
def procesar_excel_sidebar(contents, filename):
    """Procesa el archivo Excel subido desde el sidebar"""
    from dash import no_update

    if contents is None:
        return no_update, no_update

    # Procesar Excel
    resultado = cargar_excel_forecast(contents, filename)

    if not resultado['success']:
        # Mostrar errores
        errores_html = html.Div([
            lucide_icon("alert-circle", size="xs", style={"color": "#ef4444"}),
            html.Small(" Error", className="text-danger")
        ], title="; ".join(resultado['errores']))
        return None, errores_html

    # Preparar datos para guardar en store
    resumen = resultado['resumen']
    df_consumo = resultado['consumo']

    # Convertir DataFrame a dict para el store (fechas a string)
    df_consumo['fecha'] = df_consumo['fecha'].dt.strftime('%Y-%m-%d')
    datos_store = {
        'consumo': df_consumo.to_dict('records'),
        'resumen': resumen,
        'filename': filename
    }

    # Crear indicador compacto de exito
    info_html = html.Div([
        lucide_icon("check-circle", size="xs", style={"color": "#22c55e"}),
        html.Small([
            f" {resumen['materiales_unicos']} mat. / {resumen['total_registros']} reg."
        ], className="text-success", style={"fontSize": "0.7rem"})
    ], title=f"Archivo: {filename}\nPeriodo: {resumen['fecha_inicio']} - {resumen['fecha_fin']}")

    return datos_store, info_html


# Callback para actualizar indicador de estado de datos
@callback(
    Output("indicador-estado-datos", "children"),
    Input("store-excel-data", "data")
)
def actualizar_indicador_estado(excel_data):
    """Muestra tilde verde si hay datos, X roja si no"""
    if excel_data and 'consumo' in excel_data:
        return lucide_icon("check", size="xs", style={"color": "#22c55e"})
    else:
        return lucide_icon("x", size="xs", style={"color": "#ef4444"})


# Callback para abrir/cerrar modal de información
@callback(
    Output("modal-info", "is_open"),
    Input("btn-info-modal", "n_clicks"),
    Input("btn-cerrar-info-modal", "n_clicks"),
    State("modal-info", "is_open"),
    prevent_initial_call=True
)
def toggle_info_modal(n_open, n_close, is_open):
    """Abre o cierra el modal de información"""
    from dash import ctx
    if ctx.triggered_id == "btn-info-modal":
        return True
    elif ctx.triggered_id == "btn-cerrar-info-modal":
        return False
    return is_open


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8051))
    debug = os.environ.get("FLASK_ENV", "development") == "development"

    if debug:
        logger.info("Modo desarrollo: Cache de assets deshabilitado")
        app.config.suppress_callback_exceptions = True

    logger.info(f"Iniciando Forecast en puerto {port}")
    # use_reloader=False evita doble ejecucion de callbacks en modo debug
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=False)
