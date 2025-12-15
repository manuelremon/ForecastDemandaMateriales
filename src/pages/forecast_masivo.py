"""
Tablero de Forecasting Masivo
==============================
Permite generar predicciones para multiples materiales a la vez
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
import dash_ag_grid as dag
from src.layouts.components import (
    kpi_card, empty_state, empty_table, empty_chart, empty_filter, empty_search
)
from src.components.icons import lucide_icon
from src.components.icons.icon_button import icon_button
from src.utils.constants import (
    obtener_opciones_modelos,
    HORIZONTES_PREDICCION,
    NIVELES_CONFIANZA,
    MODELO_DEFAULT,
    HORIZONTE_DEFAULT,
    CONFIANZA_DEFAULT,
    MAX_MATERIALES_MASIVO
)


def crear_area_configuracion() -> html.Div:
    """Area de configuracion del forecast masivo - todos los filtros en una sola linea centrada"""
    # Anchos fijos personalizados
    w_medium = {"width": "112px", "minWidth": "112px"}   # Centro, Almacen, Horizonte, Confianza, Boton
    w_modelo = {"width": "130px", "minWidth": "130px"}   # Modelo ML
    w_limite = {"width": "75px", "minWidth": "75px"}     # Limite

    return html.Div([
        html.Div([
            # Centro (112px)
            html.Div([
                html.Label("Centro", className="filter-label-unified"),
                dcc.Dropdown(id="filtro-centro-masivo", placeholder="--", options=[],
                            className="dash-dropdown filter-dropdown")
            ], style=w_medium),
            # Almacen (112px)
            html.Div([
                html.Label("Almacen", className="filter-label-unified"),
                dcc.Dropdown(id="filtro-almacen-masivo", placeholder="--", options=[],
                            className="dash-dropdown filter-dropdown")
            ], style=w_medium),
            # Modelo ML (130px)
            html.Div([
                html.Label("Modelo ML", className="filter-label-unified"),
                dcc.Dropdown(id="select-modelo-masivo", value=MODELO_DEFAULT, clearable=False,
                            options=obtener_opciones_modelos(incluir_avanzados=True),
                            className="dash-dropdown filter-dropdown")
            ], style=w_modelo),
            # Horizonte (112px)
            html.Div([
                html.Label("Horizonte", className="filter-label-unified"),
                dcc.Dropdown(id="select-horizonte-masivo", value=HORIZONTE_DEFAULT, clearable=False,
                            options=HORIZONTES_PREDICCION, className="dash-dropdown filter-dropdown")
            ], style=w_medium),
            # Nivel de Confianza (112px)
            html.Div([
                html.Label("Confianza", className="filter-label-unified"),
                dcc.Dropdown(id="select-confianza-masivo", value=CONFIANZA_DEFAULT, clearable=False,
                            options=NIVELES_CONFIANZA, className="dash-dropdown filter-dropdown")
            ], style=w_medium),
            # Limite (75px)
            html.Div([
                html.Label("Limite", className="filter-label-unified"),
                dbc.Input(id="input-limite-materiales", type="number", value=50,
                         min=1, max=500, step=10, className="filter-control filter-number")
            ], style=w_limite),
            # Boton Generar (112px)
            html.Div([
                html.Label(" ", className="filter-label-unified"),
                dbc.Button([lucide_icon("play", size="sm", className="me-1"), "Generar"],
                          id="btn-generar-forecast-masivo", color="primary",
                          className="w-100 filter-action-btn btn-enhanced")
            ], style=w_medium),
        ], className="d-flex flex-nowrap gap-3 align-items-end justify-content-center"),
    ], className="filters-container-enhanced mb-3")


def crear_tabla_resultados() -> html.Div:
    """Tabla con resultados del forecast masivo"""
    column_defs = [
        {"field": "codigo", "headerName": "Codigo SAP", "width": 120, "pinned": "left"},
        {"field": "descripcion", "headerName": "Descripcion", "width": 250},
        {"field": "demanda_total", "headerName": "Demanda Total", "width": 130,
         "type": "numericColumn", "valueFormatter": {"function": "d3.format(',.0f')(params.value)"}},
        {"field": "r2", "headerName": "R2", "width": 80,
         "type": "numericColumn", "valueFormatter": {"function": "d3.format('.1%')(params.value)"}},
        {"field": "mae", "headerName": "MAE", "width": 80,
         "type": "numericColumn", "valueFormatter": {"function": "d3.format(',.1f')(params.value)"}},
        {"field": "estado", "headerName": "Estado", "width": 100},
    ]
    return html.Div([
        html.Div([
            html.H6("Resultados del Forecast", className="mb-0"),
            html.Div([
                icon_button("Exportar PDF", icon="file-text", id="btn-exportar-pdf", color="danger", outline=True, size="sm", className="me-2 btn-enhanced"),
                icon_button("Exportar Excel", icon="file-spreadsheet", id="btn-exportar-masivo-excel", color="success", outline=True, size="sm", className="btn-enhanced")
            ])
        ], className="table-header"),
        html.Div(id="status-forecast-masivo", className="mb-3"),
        dcc.Loading(
            type="default",
            children=[
                dag.AgGrid(
                    id="tabla-forecast-masivo",
                    columnDefs=column_defs,
                    rowData=[],
                    defaultColDef={"sortable": True, "resizable": True, "filter": True},
                    dashGridOptions={
                        "pagination": True,
                        "paginationPageSize": 25,
                    },
                    className="ag-theme-ios-glass",
                    style={"height": "500px", "width": "100%"}
                )
            ]
        ),
        dcc.Download(id="download-forecast-masivo-excel"),
    ], className="table-container")


# KPIs
def crear_kpis() -> dbc.Row:
    """Crea los KPIs del forecast masivo usando kpi_card()"""
    return dbc.Row([
        dbc.Col([
            kpi_card(
                titulo="Materiales Procesados",
                valor="--",
                subtitulo="Total analizado",
                icono="package-open",
                color="primary",
                valor_id="kpi-total-materiales-masivo",
                hoverable=True
            )
        ], md=4),
        dbc.Col([
            kpi_card(
                titulo="Demanda Total Proyectada",
                valor="--",
                subtitulo="Unidades totales",
                icono="boxes",
                color="success",
                valor_id="kpi-demanda-total-masivo",
                hoverable=True
            )
        ], md=4),
        dbc.Col([
            kpi_card(
                titulo="Precision Promedio (R2)",
                valor="--",
                subtitulo="Score del modelo",
                icono="target",
                color="info",
                valor_id="kpi-precision-promedio-masivo",
                hoverable=True
            )
        ], md=4),
    ], className="g-3 mb-4")


# Layout principal - OPTIMIZADO
layout = html.Div([
    # Store para estado del forecast
    dcc.Store(id="store-forecast-state", data={"running": False, "progress": 0, "total": 0, "current": 0}),
    # Interval para actualizar progreso (deshabilitado por defecto)
    dcc.Interval(id="interval-forecast-progress", interval=500, disabled=True),

    # Título de la página
    html.H4("FORECAST MASIVO", className="mb-3 text-center",
            style={"textShadow": "2px 2px 4px rgba(0,0,0,0.2)", "fontWeight": "700", "letterSpacing": "1px"}),

    # 1. Configuración del forecast
    crear_area_configuracion(),

    # 2. Barra de progreso mejorada con porcentaje
    html.Div([
        html.Div([
            dbc.Progress(id="progress-forecast-masivo", value=0, striped=True, animated=True,
                        className="progress-enhanced", style={"height": "24px"}),
            html.Span(id="progress-label-masivo", className="progress-label",
                     children="0%")
        ], className="position-relative"),
        html.Small(id="progress-text-masivo", className="text-muted mt-2 d-block",
                  children="Preparando...")
    ], id="progress-container-masivo", style={"display": "none"}, className="mb-4 glass-card-enhanced p-3"),

    # 3. KPIs resumen
    crear_kpis(),

    # 4. Tabla de resultados
    crear_tabla_resultados(),

], className="fade-in")
