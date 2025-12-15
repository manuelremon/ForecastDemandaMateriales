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
    title="Forecast MR",
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
        # Logo de la aplicación
        html.Div([
            html.Img(src="/assets/logo.png", style={"width": "90px", "height": "auto", "margin": "0 auto"}),
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
    """Crea el modal de información de la aplicación - Mejorado"""
    from src.components.icons import lucide_icon

    # Función helper para crear tarjetas de modelo
    def modelo_card(nombre, descripcion, recomendado=False, color="primary"):
        badge = dbc.Badge("Recomendado", color="success", className="ms-2") if recomendado else None
        return html.Div([
            html.Div([
                html.Strong(nombre),
                badge
            ], className="mb-1"),
            html.Small(descripcion, className="text-muted")
        ], className=f"p-2 mb-2 border-start border-{color} border-3 bg-light rounded-end")

    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            lucide_icon("sparkles", size="sm", className="me-2"),
            "Forecast MR - Guía de Usuario"
        ]), close_button=True, className="bg-primary text-white"),
        dbc.ModalBody([
            # Banner principal
            dbc.Alert([
                html.Div([
                    lucide_icon("line-chart", size="md", className="me-3"),
                    html.Div([
                        html.H5("Sistema de Predicción de Demanda", className="mb-1 alert-heading"),
                        html.P("Utiliza Machine Learning para estimar el consumo futuro de materiales basándose en datos históricos.",
                              className="mb-0 small")
                    ])
                ], className="d-flex align-items-center")
            ], color="primary", className="mb-4"),

            # Tabs para organizar contenido
            dbc.Tabs([
                # Tab 1: Modelos ML
                dbc.Tab([
                    html.Div([
                        html.P("Selecciona el modelo según tus necesidades:", className="text-muted mb-3"),

                        html.H6([lucide_icon("zap", size="xs", className="me-1"), "Modelos Rápidos"], className="text-success mb-2"),
                        modelo_card("Random Forest", "Combina múltiples árboles de decisión. Robusto, estable y excelente para datos con ruido.", recomendado=True, color="success"),
                        modelo_card("Regresión Lineal", "Simple y rápido. Ideal para tendencias lineales claras.", color="info"),

                        html.H6([lucide_icon("brain", size="xs", className="me-1"), "Modelos Avanzados"], className="text-primary mb-2 mt-3"),
                        modelo_card("Gradient Boosting", "Alta precisión construyendo árboles secuencialmente. Bueno para patrones complejos.", color="primary"),
                        modelo_card("XGBoost", "Versión optimizada de Gradient Boosting. Muy rápido y preciso.", color="primary"),

                        html.H6([lucide_icon("calendar", size="xs", className="me-1"), "Series Temporales"], className="text-warning mb-2 mt-3"),
                        modelo_card("Prophet (Meta)", "Diseñado para datos con estacionalidad. Excelente para patrones anuales/semanales.", color="warning"),
                        modelo_card("ARIMA/SARIMAX", "Modelo estadístico clásico. Captura tendencias y estacionalidad.", color="warning"),
                    ], className="p-3")
                ], label="Modelos ML", tab_id="tab-modelos"),

                # Tab 2: Cómo usar
                dbc.Tab([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Div("1", className="badge bg-primary rounded-circle me-2", style={"fontSize": "1rem", "padding": "8px 12px"}),
                                    html.Span("Descargar Plantilla", className="fw-bold")
                                ], className="d-flex align-items-center mb-2"),
                                html.P("Haz clic en el botón de descarga en el sidebar para obtener la plantilla Excel.", className="text-muted small ms-4")
                            ], md=6, className="mb-3"),
                            dbc.Col([
                                html.Div([
                                    html.Div("2", className="badge bg-primary rounded-circle me-2", style={"fontSize": "1rem", "padding": "8px 12px"}),
                                    html.Span("Llenar Datos", className="fw-bold")
                                ], className="d-flex align-items-center mb-2"),
                                html.P("Completa el consumo histórico con fechas, materiales, centros y cantidades.", className="text-muted small ms-4")
                            ], md=6, className="mb-3"),
                            dbc.Col([
                                html.Div([
                                    html.Div("3", className="badge bg-primary rounded-circle me-2", style={"fontSize": "1rem", "padding": "8px 12px"}),
                                    html.Span("Importar Excel", className="fw-bold")
                                ], className="d-flex align-items-center mb-2"),
                                html.P("Sube el archivo usando el botón de importar en el sidebar.", className="text-muted small ms-4")
                            ], md=6, className="mb-3"),
                            dbc.Col([
                                html.Div([
                                    html.Div("4", className="badge bg-primary rounded-circle me-2", style={"fontSize": "1rem", "padding": "8px 12px"}),
                                    html.Span("Configurar y Generar", className="fw-bold")
                                ], className="d-flex align-items-center mb-2"),
                                html.P("Selecciona filtros, modelo y horizonte. Clic en 'Generar' para ver predicciones.", className="text-muted small ms-4")
                            ], md=6, className="mb-3"),
                        ]),

                        html.Hr(),

                        html.H6([lucide_icon("layout-grid", size="xs", className="me-1"), "Modos de Forecast"], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6([lucide_icon("line-chart", size="xs", className="me-1"), "Individual"], className="text-primary"),
                                        html.P("Analiza un material específico con gráficos detallados y métricas completas.", className="small mb-0")
                                    ])
                                ], className="h-100")
                            ], md=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6([lucide_icon("layers", size="xs", className="me-1"), "Masivo"], className="text-primary"),
                                        html.P("Procesa múltiples materiales simultáneamente. Exporta resultados a Excel/PDF.", className="small mb-0")
                                    ])
                                ], className="h-100")
                            ], md=6),
                        ])
                    ], className="p-3")
                ], label="Cómo Usar", tab_id="tab-uso"),

                # Tab 3: Métricas
                dbc.Tab([
                    html.Div([
                        html.P("Interpreta los resultados del modelo:", className="text-muted mb-3"),

                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    dbc.Badge("R²", color="success", className="me-2", style={"fontSize": "0.9rem"}),
                                    html.Strong("Coeficiente de Determinación")
                                ], className="mb-2"),
                                html.P("Mide qué tan bien el modelo explica los datos. Valores de 0 a 1.", className="small mb-2"),
                                dbc.Progress([
                                    dbc.Progress(value=30, color="danger", bar=True, label="Malo <0.5"),
                                    dbc.Progress(value=30, color="warning", bar=True, label="Regular 0.5-0.7"),
                                    dbc.Progress(value=40, color="success", bar=True, label="Bueno >0.7"),
                                ], className="mb-0", style={"height": "20px"})
                            ])
                        ], className="mb-3"),

                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    dbc.Badge("MAE", color="info", className="me-2", style={"fontSize": "0.9rem"}),
                                    html.Strong("Error Absoluto Medio")
                                ], className="mb-2"),
                                html.P("Promedio de la diferencia entre predicción y valor real. Menor es mejor. Se mide en las mismas unidades que los datos.", className="small mb-0")
                            ])
                        ], className="mb-3"),

                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    dbc.Badge("RMSE", color="warning", className="me-2", style={"fontSize": "0.9rem"}),
                                    html.Strong("Raíz del Error Cuadrático Medio")
                                ], className="mb-2"),
                                html.P("Similar al MAE pero penaliza más los errores grandes. Útil para detectar predicciones muy desviadas.", className="small mb-0")
                            ])
                        ]),
                    ], className="p-3")
                ], label="Métricas", tab_id="tab-metricas"),

                # Tab 4: Tips
                dbc.Tab([
                    html.Div([
                        dbc.Alert([
                            html.H6([lucide_icon("lightbulb", size="xs", className="me-1"), "Mejores Prácticas"], className="alert-heading"),
                            html.Hr(),
                            html.Ul([
                                html.Li([html.Strong("Datos históricos: "), "Mínimo 6 meses para predicciones confiables"]),
                                html.Li([html.Strong("Modelo recomendado: "), "Random Forest para la mayoría de casos"]),
                                html.Li([html.Strong("Horizonte: "), "Ajusta según necesidad (7 días a 1 año)"]),
                                html.Li([html.Strong("Confianza 95%: "), "Buen balance entre precisión y rango"]),
                                html.Li([html.Strong("Datos limpios: "), "Elimina valores atípicos o erróneos antes de importar"]),
                            ], className="mb-0")
                        ], color="success", className="mb-3"),

                        dbc.Alert([
                            html.H6([lucide_icon("alert-triangle", size="xs", className="me-1"), "Precauciones"], className="alert-heading"),
                            html.Hr(),
                            html.Ul([
                                html.Li("Prophet y ARIMA requieren más tiempo de procesamiento"),
                                html.Li("Pocos datos (<10 registros) usan promedio móvil automáticamente"),
                                html.Li("Materiales nuevos sin historial no pueden predecirse con precisión"),
                                html.Li("Eventos extraordinarios (pandemia, crisis) pueden afectar predicciones"),
                            ], className="mb-0")
                        ], color="warning"),
                    ], className="p-3")
                ], label="Tips", tab_id="tab-tips"),

            ], id="modal-tabs", active_tab="tab-modelos", className="mb-3"),

        ]),
        dbc.ModalFooter([
            html.Small("Forecast MR v1.0 | Machine Learning para Predicción de Demanda | Manuel Remón | Neuquén, Argentina", className="text-muted me-auto"),
            dbc.Button("Cerrar", id="btn-cerrar-info-modal", color="primary")
        ]),
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
