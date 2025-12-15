"""
Tablero de Forecasting
======================
Forecast y analisis de patrones de consumo con ML
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
import dash_ag_grid as dag
from src.layouts.components import (
    kpi_card, summary_card, empty_state, empty_table, empty_chart,
    empty_filter, empty_search
)
from src.components.icons import lucide_icon
from src.components.icons.icon_button import icon_button
from src.ml.strategies import listar_estrategias
from src.utils.constants import MODELOS_ML, MODELO_DEFAULT


def crear_componentes_descarga():
    """Crea los componentes ocultos para descargas."""
    return html.Div([
        dcc.Download(id="download-forecast-csv"),
        dcc.Download(id="download-forecast-pdf"),
        dcc.Download(id="download-forecast-excel"),
    ], style={"display": "none"})


def crear_modal_gestion_modelos():
    """Crea el modal para gestión de modelos guardados."""
    return dbc.Modal([
        dbc.ModalHeader([
            dbc.ModalTitle([
                lucide_icon("database", size="sm", className="me-2"),
                "Gestión de Modelos"
            ])
        ], close_button=True),
        dbc.ModalBody([
            # Sección guardar modelo actual
            html.Div([
                html.H6([
                    lucide_icon("save", size="xs", className="me-2"),
                    "Guardar Modelo Actual"
                ], className="mb-3"),
                dbc.InputGroup([
                    dbc.Input(
                        id="input-nombre-modelo",
                        placeholder="Nombre del modelo (opcional)",
                        type="text"
                    ),
                    dbc.Button([
                        lucide_icon("save", size="xs", className="me-1"),
                        "Guardar"
                    ], id="btn-guardar-modelo", color="primary")
                ], className="mb-2"),
                html.Div(id="status-guardar-modelo", className="small")
            ], className="mb-4 p-3 bg-light rounded"),

            # Sección modelos guardados
            html.Div([
                html.H6([
                    lucide_icon("list", size="xs", className="me-2"),
                    "Modelos Guardados"
                ], className="mb-3"),
                dcc.Loading(
                    id="loading-lista-modelos",
                    children=[
                        html.Div(id="lista-modelos-guardados", children=[
                            html.P("Seleccione un material para ver modelos guardados",
                                   className="text-muted small")
                        ])
                    ]
                )
            ], className="p-3 bg-light rounded"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cerrar", id="btn-cerrar-modal-modelos", color="secondary")
        ])
    ], id="modal-gestion-modelos", size="lg", is_open=False)


def crear_modal_backtesting():
    """Crea el modal para ejecutar y visualizar backtesting."""
    return dbc.Modal([
        dbc.ModalHeader([
            dbc.ModalTitle([
                lucide_icon("history", size="sm", className="me-2"),
                "Backtesting - Validación Histórica"
            ])
        ], close_button=True),
        dbc.ModalBody([
            # Configuración del backtest
            html.Div([
                html.H6("Configuración", className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Ventana de prueba (días)", className="small"),
                        dbc.Input(id="input-ventana-backtest", type="number",
                                  value=30, min=7, max=90, step=1)
                    ], md=4),
                    dbc.Col([
                        html.Label("Número de pasos", className="small"),
                        dbc.Input(id="input-pasos-backtest", type="number",
                                  value=5, min=2, max=10, step=1)
                    ], md=4),
                    dbc.Col([
                        html.Label(" ", className="small d-block"),
                        dbc.Button([
                            lucide_icon("play", size="xs", className="me-1"),
                            "Ejecutar Backtest"
                        ], id="btn-ejecutar-backtest", color="primary", className="w-100")
                    ], md=4)
                ], className="mb-3")
            ], className="mb-4 p-3 bg-light rounded"),

            # Resultados
            html.Div([
                html.H6("Resultados del Backtest", className="mb-3"),
                dcc.Loading(
                    id="loading-backtest",
                    children=[
                        html.Div(id="resultados-backtest", children=[
                            html.P("Configure y ejecute el backtest para ver resultados",
                                   className="text-muted small text-center py-4")
                        ])
                    ]
                )
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Cerrar", id="btn-cerrar-modal-backtest", color="secondary")
        ])
    ], id="modal-backtesting", size="xl", is_open=False)


def obtener_opciones_modelos():
    """
    Genera opciones de modelos para el dropdown con tooltips.
    Solo incluye modelos que estan instalados y disponibles.
    """
    disponibles = listar_estrategias()
    opciones = []
    for modelo, disponible in disponibles.items():
        if disponible and modelo in MODELOS_ML:
            info = MODELOS_ML[modelo]
            opciones.append({
                "label": info['nombre'],
                "value": modelo,
                "title": info['tooltip']
            })
    return opciones


def crear_controles_forecast() -> html.Div:
    """Crea los controles de configuracion del forecast - UNA SOLA LINEA centrada"""
    # Anchos fijos personalizados
    w_material = {"width": "500px", "minWidth": "500px"}  # Material (dropdown)
    w_codigo = {"width": "130px", "minWidth": "130px"}    # Codigo SAP
    w_small = {"width": "85px", "minWidth": "85px"}       # Centro, Almacen, Confianza
    w_modelo = {"width": "110px", "minWidth": "110px"}    # Modelo ML
    w_horizonte = {"width": "200px", "minWidth": "200px"} # Slider Horizonte
    w_boton = {"width": "95px", "minWidth": "95px"}       # Boton Generar

    return html.Div([
        # Material encontrado (arriba de los filtros)
        html.Div(id="material-encontrado", className="text-center mb-2"),

        # TODOS los filtros en UNA SOLA LINEA centrada
        html.Div([
            # Material (dropdown) - A LA IZQUIERDA
            html.Div([
                html.Label("Material", className="filter-label-unified"),
                dcc.Dropdown(id="select-material-demanda", placeholder="--",
                            className="dash-dropdown filter-dropdown",
                            searchable=True, clearable=True)
            ], style=w_material),
            # Codigo SAP
            html.Div([
                html.Label("Codigo SAP", className="filter-label-unified"),
                html.Div([
                    dbc.Input(id="input-codigo-sap", type="text", placeholder="10301804",
                             className="filter-control", debounce=True, style={"width": "90px"}),
                    dbc.Button(
                        lucide_icon("search", size="sm"),
                        id="btn-buscar-material", size="sm",
                        style={"border": "none", "boxShadow": "none", "background": "none",
                               "color": "#6B7280", "padding": "0", "marginLeft": "4px"}),
                ], className="d-flex align-items-center"),
            ], style=w_codigo),
            # Centro
            html.Div([
                html.Label("Centro", className="filter-label-unified"),
                dcc.Dropdown(id="filtro-centro-demanda", placeholder="--",
                            className="dash-dropdown filter-dropdown")
            ], style=w_small),
            # Almacen
            html.Div([
                html.Label("Almacen", className="filter-label-unified"),
                dcc.Dropdown(id="filtro-almacen-demanda", placeholder="--",
                            className="dash-dropdown filter-dropdown")
            ], style=w_small),
            # Modelo ML
            html.Div([
                html.Label("Modelo ML", className="filter-label-unified"),
                dcc.Dropdown(id="select-modelo-ml", value="random_forest", clearable=False,
                            options=obtener_opciones_modelos(), className="dash-dropdown filter-dropdown")
            ], style=w_modelo),
            # Horizonte slider
            html.Div([
                html.Label("Horizonte", className="filter-label-unified"),
                dcc.Slider(id="slider-horizonte", min=7, max=365, step=None, value=30,
                          marks={7: "7d", 30: "1M", 90: "3M", 180: "6M", 365: "1A"},
                          tooltip={"placement": "bottom", "always_visible": True})
            ], style=w_horizonte),
            # Confianza
            html.Div([
                html.Label("Confianza", className="filter-label-unified"),
                dcc.Dropdown(id="select-confianza", value=0.95, clearable=False,
                            options=[{"label": "80%", "value": 0.80}, {"label": "90%", "value": 0.90},
                                    {"label": "95%", "value": 0.95}, {"label": "99%", "value": 0.99}],
                            className="dash-dropdown filter-dropdown")
            ], style=w_small),
            # Boton Generar
            html.Div([
                html.Label(" ", className="filter-label-unified"),
                dbc.Button([lucide_icon("play", size="sm", className="me-1"), "Generar"],
                          id="btn-generar-forecast", color="primary",
                          className="w-100 filter-action-btn btn-enhanced")
            ], style=w_boton),
        ], className="d-flex flex-nowrap gap-2 align-items-end justify-content-center"),
    ], className="filters-container-enhanced mb-3")


def crear_kpis_demanda() -> html.Div:
    """Crea los KPIs del tablero de demanda usando kpi_card()"""
    return dbc.Row([
        dbc.Col([
            kpi_card(
                titulo="Precision del Modelo",
                valor="--",
                subtitulo="R2 Score",
                icono="target",
                color="success",
                valor_id="forecast-precision-valor",
                tooltip="R2 Score: Que tan bien el modelo explica los datos historicos (0-100%)",
                tooltip_id="tooltip-forecast-precision",
                hoverable=True
            )
        ], md=3),
        dbc.Col([
            kpi_card(
                titulo="Error Promedio (MAE)",
                valor="--",
                subtitulo="Desviacion en unidades",
                icono="alert-circle",
                color="info",
                valor_id="forecast-error-valor",
                tooltip="MAE: Desviacion promedio de la prediccion en unidades",
                tooltip_id="tooltip-forecast-error",
                hoverable=True
            )
        ], md=3),
        dbc.Col([
            kpi_card(
                titulo="Demanda Proyectada",
                valor="--",
                subtitulo="Unidades totales",
                icono="boxes",
                color="primary",
                valor_id="forecast-demanda-valor",
                tooltip="Total de unidades predichas para el horizonte seleccionado",
                tooltip_id="tooltip-forecast-demanda",
                hoverable=True
            )
        ], md=3),
        dbc.Col([
            kpi_card(
                titulo="Tendencia",
                valor="--",
                subtitulo="vs periodo anterior",
                icono="arrow-up",
                color="warning",
                valor_id="forecast-tendencia-valor",
                tooltip="Cambio porcentual vs periodo anterior",
                tooltip_id="tooltip-forecast-tendencia",
                hoverable=True
            )
        ], md=3),
    ], className="g-3 mb-4")


def crear_grafico_forecast() -> html.Div:
    """Crea el grafico principal de forecast"""
    return html.Div([
        html.Div([
            html.Div([
                html.H6("Forecast de Demanda", className="chart-title mb-0"),
                html.Small("Historico + Prediccion con intervalos de confianza",
                          className="text-muted")
            ]),
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button("1M", id="btn-periodo-1m", outline=True, color="light", size="sm"),
                    dbc.Button("3M", id="btn-periodo-3m", outline=True, color="light", size="sm"),
                    dbc.Button("6M", id="btn-periodo-6m", outline=True, color="light", size="sm", active=True),
                    dbc.Button("1A", id="btn-periodo-1a", outline=True, color="light", size="sm"),
                ])
            ])
        ], className="d-flex justify-content-between align-items-center mb-3"),

        dcc.Loading(
            id='loading-grafico-forecast',
            type='default',
            children=[
                dcc.Graph(
                    id="grafico-forecast",
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                        "displaylogo": False
                    },
                    style={"height": "400px"}
                )
            ]
        ),

        # Resumen del forecast
        html.Div(id="forecast-resumen", className="forecast-summary", style={"display": "none"})
    ], className="chart-container chart-container-enhanced")


def crear_graficos_patrones() -> html.Div:
    """Crea los graficos de analisis de patrones con tooltips"""
    return dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H6("Patron Semanal", className="chart-title d-inline"),
                    lucide_icon("info", size="sm", className="ms-2 tooltip-icon", id="tooltip-patron-semanal")
                ]),
                dbc.Tooltip("Consumo promedio historico por dia de la semana",
                           target="tooltip-patron-semanal", placement="top"),
                dcc.Loading(
                    id='loading-grafico-patron-semanal',
                    type='default',
                    children=[
                        dcc.Graph(
                            id="grafico-patron-semanal",
                            config={"displayModeBar": False},
                            style={"height": "250px"}
                        )
                    ]
                )
            ], className="chart-container chart-container-enhanced")
        ], md=4),
        dbc.Col([
            html.Div([
                html.Div([
                    html.H6("Estacionalidad Mensual", className="chart-title d-inline"),
                    lucide_icon("info", size="sm", className="ms-2 tooltip-icon", id="tooltip-estacionalidad")
                ]),
                dbc.Tooltip("Variacion del consumo a lo largo del anio",
                           target="tooltip-estacionalidad", placement="top"),
                dcc.Loading(
                    id='loading-grafico-estacionalidad',
                    type='default',
                    children=[
                        dcc.Graph(
                            id="grafico-estacionalidad",
                            config={"displayModeBar": False},
                            style={"height": "250px"}
                        )
                    ]
                )
            ], className="chart-container chart-container-enhanced")
        ], md=4),
        dbc.Col([
            html.Div([
                html.Div([
                    html.H6("Importancia de Features", className="chart-title d-inline"),
                    lucide_icon("info", size="sm", className="ms-2 tooltip-icon", id="tooltip-features")
                ]),
                dbc.Tooltip("Variables que mas influyen en la prediccion (solo modelo ML completo)",
                           target="tooltip-features", placement="top"),
                dcc.Loading(
                    id='loading-grafico-feature-importance',
                    type='default',
                    children=[
                        dcc.Graph(
                            id="grafico-feature-importance",
                            config={"displayModeBar": False},
                            style={"height": "250px"}
                        )
                    ]
                )
            ], className="chart-container chart-container-enhanced")
        ], md=4),
    ], className="g-3 mb-4")


def crear_metricas_modelo() -> html.Div:
    """Crea el panel de metricas del modelo ML con tooltips"""
    return html.Div([
        html.H6("Metricas del Modelo", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Small("MAE", className="text-muted d-block", id="label-mae"),
                        html.Span("--", id="metrica-mae", className="h5")
                    ], className="text-center p-2"),
                    dbc.Tooltip("Error Absoluto Medio: Diferencia promedio entre prediccion y valor real (unidades)",
                               target="label-mae", placement="top")
                ])
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Small("RMSE", className="text-muted d-block", id="label-rmse"),
                        html.Span("--", id="metrica-rmse", className="h5")
                    ], className="text-center p-2"),
                    dbc.Tooltip("Raiz del Error Cuadratico Medio: Penaliza errores grandes mas que el MAE",
                               target="label-rmse", placement="top")
                ])
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Small("R2", className="text-muted d-block", id="label-r2"),
                        html.Span("--", id="metrica-r2", className="h5")
                    ], className="text-center p-2"),
                    dbc.Tooltip("Coeficiente de Determinacion: Porcentaje de variabilidad explicada por el modelo (0-100%)",
                               target="label-r2", placement="top")
                ])
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Small("MAPE", className="text-muted d-block", id="label-mape"),
                        html.Span("--", id="metrica-mape", className="h5")
                    ], className="text-center p-2"),
                    dbc.Tooltip("Error Porcentual Absoluto Medio: Error como porcentaje del valor real",
                               target="label-mape", placement="top")
                ])
            ], width=3),
        ], className="border rounded py-2",
           style={"backgroundColor": "rgba(59, 130, 246, 0.1)"})
    ])


def crear_tabla_predicciones() -> html.Div:
    """Crea la tabla de predicciones detalladas"""
    column_defs = [
        {
            "field": "fecha",
            "headerName": "Fecha",
            "width": 120,
            "sort": "asc"
        },
        {
            "field": "prediccion",
            "headerName": "Prediccion",
            "width": 120,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format(',.0f')(params.value)"},
            "cellStyle": {"fontWeight": "600", "color": "#3b82f6"}
        },
        {
            "field": "limite_inferior",
            "headerName": "Lim. Inferior",
            "width": 120,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format(',.0f')(params.value)"}
        },
        {
            "field": "limite_superior",
            "headerName": "Lim. Superior",
            "width": 120,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format(',.0f')(params.value)"}
        },
        {
            "field": "dia_semana",
            "headerName": "Dia",
            "width": 100
        },
    ]

    return html.Div([
        html.Div([
            html.H6("Predicciones Detalladas", className="mb-0"),
            html.Div([
                icon_button("Exportar PDF", icon="file-text", id="btn-exportar-pdf", color="danger", outline=True, size="sm", className="me-2"),
                icon_button("Exportar", icon="download", id="btn-exportar-forecast", color="success", outline=True, size="sm")
            ])
        ], className="table-header"),

        dcc.Loading(
            id="loading-tabla-predicciones",
            type="default",
            children=[
                dag.AgGrid(
                    id="tabla-predicciones",
                    columnDefs=column_defs,
                    rowData=[],
                    defaultColDef={"sortable": True, "resizable": True},
                    dashGridOptions={
                        "pagination": True,
                        "paginationPageSize": 20,
                        "paginationPageSizeSelector": [10, 20, 50, 100],
                        "domLayout": "normal",
                    },
                    className="ag-theme-ios-glass",
                    style={"height": "350px", "width": "100%"}
                )
            ]
        )
    ], className="table-container")


def crear_botones_avanzados():
    """Crea los botones para funcionalidades avanzadas."""
    return html.Div([
        dbc.ButtonGroup([
            dbc.Button([
                lucide_icon("database", size="xs", className="me-1"),
                "Modelos"
            ], id="btn-abrir-modal-modelos", color="info", outline=True, size="sm"),
            dbc.Button([
                lucide_icon("history", size="xs", className="me-1"),
                "Backtest"
            ], id="btn-abrir-modal-backtest", color="warning", outline=True, size="sm"),
        ], size="sm")
    ], className="d-flex justify-content-end mb-2")


# Layout principal del tablero de demanda - OPTIMIZADO
layout = html.Div([
    # Componentes de descarga (ocultos)
    crear_componentes_descarga(),

    # Modales
    crear_modal_gestion_modelos(),
    crear_modal_backtesting(),

    # Título de la página
    html.H4("FORECAST INDIVIDUAL", className="mb-3 text-center",
            style={"textShadow": "2px 2px 4px rgba(0,0,0,0.2)", "fontWeight": "700", "letterSpacing": "1px"}),

    # 1. Controles de entrada y configuración
    crear_controles_forecast(),

    # 2. KPIs clave
    crear_kpis_demanda(),

    # 3. Gráfico principal - Lo más importante
    crear_grafico_forecast(),

    # 4. Análisis visual de patrones
    html.Div([
        html.Div([
            html.H5("Análisis de Patrones", className="mb-0"),
            crear_botones_avanzados()
        ], className="d-flex justify-content-between align-items-center mb-3"),
        crear_graficos_patrones(),
    ], className="mb-4"),

    # 5. Detalles técnicos - Métricas y tabla
    html.Div([
        html.H5("Detalles del Modelo", className="mb-3"),
        dbc.Row([
            dbc.Col([
                crear_metricas_modelo(),
                html.Br(),
                html.Div([
                    html.H6("Configuración del Modelo", className="mb-3"),
                    html.Div(id="info-modelo-config", children=[
                        html.P([
                            lucide_icon("info", size="sm", className="me-2"),
                            "Seleccione un material y genere el forecast para ver detalles del modelo."
                        ], className="text-muted small")
                    ])
                ])
            ], md=4),
            dbc.Col([
                crear_tabla_predicciones()
            ], md=8),
        ], className="g-3"),
    ], className="mb-4"),

], className="fade-in")
