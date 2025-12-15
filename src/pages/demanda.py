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
    empty_filter, empty_search, filter_row
)
from src.components.icons import lucide_icon
from src.components.icons.icon_button import icon_button


def crear_controles_forecast() -> html.Div:
    """Crea los controles de configuracion del forecast"""
    return html.Div([
        # Primera fila: Codigo SAP + Centro + Almacen
        dbc.Row([
            dbc.Col([
                html.Label("Codigo SAP", className="filter-label"),
                dbc.InputGroup([
                    dbc.Input(
                        id="input-codigo-sap",
                        type="text",
                        placeholder="Ej: 10301804",
                        className="form-control",
                        debounce=True
                    ),
                    icon_button(
                        icon="search",
                        id="btn-buscar-material",
                        color="primary",
                        outline=True,
                        size="sm",
                        icon_position="only",
                        className="px-3"
                    ),
                ], size="sm"),
                # Dropdown oculto para compatibilidad
                dcc.Dropdown(
                    id="select-material-demanda",
                    style={"display": "none"}
                ),
            ], md=4, lg=4),
            dbc.Col([
                html.Label("Centro", className="filter-label"),
                dcc.Dropdown(
                    id="filtro-centro-demanda",
                    placeholder="Seleccione centro",
                    className="dash-dropdown"
                )
            ], md=4, lg=3),
            dbc.Col([
                html.Label("Almacen", className="filter-label"),
                dcc.Dropdown(
                    id="filtro-almacen-demanda",
                    placeholder="Seleccione almacen",
                    className="dash-dropdown"
                )
            ], md=4, lg=3),
        ], className="mb-3 align-items-end"),

        # Material encontrado (debajo de búsqueda)
        dbc.Row([
            dbc.Col([
                html.Div(id="material-encontrado", className="d-flex align-items-center ps-2")
            ], md=4, lg=4)
        ], className="mb-3"),

        # Segunda fila: Modelo ML, Horizonte, Confianza, Boton
        filter_row([
            {
                "label": "Modelo ML",
                "id": "select-modelo-ml",
                "type": "dropdown",
                "md": 3,
                "value": "random_forest",
                "clearable": False,
                "options": [
                    {"label": "Random Forest", "value": "random_forest"},
                    {"label": "Gradient Boosting", "value": "gradient_boosting"},
                    {"label": "Regresion Lineal", "value": "linear"},
                ]
            },
            {
                "label": "Horizonte",
                "id": "slider-horizonte",
                "type": "slider",
                "md": 4,
                "value": 30,
                "slider_props": {
                    "min": 7,
                    "max": 365,
                    "step": None,
                    "marks": {
                        7: "7d",
                        30: "1M",
                        90: "3M",
                        180: "6M",
                        365: "1A"
                    },
                    "tooltip": {"placement": "bottom", "always_visible": True}
                }
            },
            {
                "label": "Confianza",
                "id": "select-confianza",
                "type": "dropdown",
                "md": 2,
                "value": 0.95,
                "clearable": False,
                "options": [
                    {"label": "80%", "value": 0.80},
                    {"label": "90%", "value": 0.90},
                    {"label": "95%", "value": 0.95},
                    {"label": "99%", "value": 0.99},
                ]
            },
            {
                "label": " ",
                "id": "btn-generar-forecast",
                "type": "button",
                "md": 3,
                "button_props": {
                    "icon": "play",
                    "text": "Generar Forecast",
                    "color": "primary"
                }
            }
        ])
    ])


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
                tooltip_id="tooltip-forecast-precision"
            )
        ], md=3),
        dbc.Col([
            kpi_card(
                titulo="Error Promedio (MAE)",
                valor="--",
                subtitulo="Desviacion en unidades",
                icono="trending-up",
                color="info",
                valor_id="forecast-error-valor",
                tooltip="MAE: Desviacion promedio de la prediccion en unidades",
                tooltip_id="tooltip-forecast-error"
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
                tooltip_id="tooltip-forecast-demanda"
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
                tooltip_id="tooltip-forecast-tendencia"
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
    ], className="chart-container")


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
            ], className="chart-container")
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
            ], className="chart-container")
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
            ], className="chart-container")
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


# Layout principal del tablero de demanda - OPTIMIZADO
layout = html.Div([
    # 1. Controles de entrada y configuración
    html.Div([
        html.H5("Configuración de Forecast", className="mb-3"),
        crear_controles_forecast(),
    ], className="mb-4"),

    # 2. KPIs clave
    crear_kpis_demanda(),

    # 3. Gráfico principal - Lo más importante
    crear_grafico_forecast(),

    # 4. Análisis visual de patrones
    html.Div([
        html.H5("Análisis de Patrones", className="mb-3"),
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
