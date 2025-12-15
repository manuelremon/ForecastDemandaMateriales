"""
Tablero de Forecasting Masivo
==============================
Permite generar predicciones para multiples materiales a la vez
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
import dash_ag_grid as dag
from src.layouts.components import (
    kpi_card, filter_row, empty_state, empty_table, empty_chart, empty_filter, empty_search
)
from src.components.icons import lucide_icon
from src.components.icons.icon_button import icon_button


def crear_area_entrada() -> html.Div:
    """Area para ingresar codigos SAP"""
    return html.Div([
        # Primera fila: Textarea y Upload
        dbc.Row([
            dbc.Col([
                html.Label("Codigos SAP (uno por linea)", className="filter-label"),
                dcc.Textarea(
                    id="textarea-codigos-masivo",
                    placeholder="Ingrese codigos SAP:\n10301804\n10301805",
                    style={
                        "width": "100%",
                        "height": "200px",
                        "backgroundColor": "var(--bg-primary)",
                        "color": "var(--text-primary)",
                        "border": "1px solid var(--border-color)",
                        "borderRadius": "var(--radius-md)",
                        "padding": "10px"
                    }
                ),
                html.Small("Maximo 50 codigos por lote", className="text-muted mt-1 d-block")
            ], md=6),
            dbc.Col([
                html.Label("O cargar desde archivo", className="filter-label"),
                dcc.Upload(
                    id="upload-codigos-masivo",
                    children=html.Div([
                        lucide_icon("file-up", size="3x", className="mb-2", style={"color": "#BDBEC2"}),
                        html.P("Arrasque un archivo Excel/CSV", className="mb-1"),
                        html.Small("o haga clic para seleccionar", className="text-muted")
                    ], className="text-center py-4"),
                    style={
                        "width": "100%",
                        "height": "200px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderRadius": "var(--radius-lg)",
                        "borderColor": "var(--border-color)",
                        "backgroundColor": "var(--bg-secondary)",
                        "cursor": "pointer"
                    }
                ),
            ], md=6),
        ], className="mb-4"),
        # Segunda fila: Filtros usando filter_row
        filter_row([
            {
                "label": "Centro",
                "id": "filtro-centro-masivo",
                "type": "dropdown",
                "md": 3,
                "placeholder": "",
                "options": []
            },
            {
                "label": "Almacen",
                "id": "filtro-almacen-masivo",
                "type": "dropdown",
                "md": 3,
                "placeholder": "",
                "options": []
            },
            {
                "label": "Modelo ML",
                "id": "select-modelo-masivo",
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
                "label": "Horizonte (dias)",
                "id": "select-horizonte-masivo",
                "type": "dropdown",
                "md": 3,
                "value": 30,
                "clearable": False,
                "options": [
                    {"label": "7 dias", "value": 7},
                    {"label": "30 dias", "value": 30},
                    {"label": "90 dias", "value": 90},
                ]
            },
            {
                "label": "Nivel de Confianza",
                "id": "select-confianza-masivo",
                "type": "dropdown",
                "md": 3,
                "value": 0.95,
                "clearable": False,
                "options": [
                    {"label": "90%", "value": 0.90},
                    {"label": "95%", "value": 0.95},
                ]
            },
            {
                "label": " ",
                "id": "btn-generar-forecast-masivo",
                "type": "button",
                "md": 3,
                "button_props": {
                    "icon": "play",
                    "text": "Generar Forecast Masivo",
                    "color": "primary"
                }
            }
        ])
    ])


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
                icon_button("Exportar PDF", icon="file-text", id="btn-exportar-pdf", color="danger", outline=True, size="sm", className="me-2"),
                icon_button("Exportar Excel", icon="file-spreadsheet", id="btn-exportar-masivo-excel", color="success", outline=True, size="sm")
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
                valor_id="kpi-total-materiales-masivo"
            )
        ], md=4),
        dbc.Col([
            kpi_card(
                titulo="Demanda Total Proyectada",
                valor="--",
                subtitulo="Unidades totales",
                icono="boxes",
                color="success",
                valor_id="kpi-demanda-total-masivo"
            )
        ], md=4),
        dbc.Col([
            kpi_card(
                titulo="Precision Promedio (R2)",
                valor="--",
                subtitulo="Score del modelo",
                icono="target",
                color="info",
                valor_id="kpi-precision-promedio-masivo"
            )
        ], md=4),
    ], className="g-3 mb-4")


# Layout principal - OPTIMIZADO
layout = html.Div([
    # 1. Área de entrada - Compacta
    html.Div([
        html.H5("Ingrese Códigos SAP", className="mb-3"),
        crear_area_entrada(),
    ], className="mb-4"),

    # 2. Barra de progreso
    html.Div([
        dbc.Progress(id="progress-forecast-masivo", value=0, striped=True, animated=True,
                    className="mb-2", style={"height": "20px"}),
    ], id="progress-container-masivo", style={"display": "none"}, className="mb-4"),

    # 3. KPIs resumen
    crear_kpis(),

    # 4. Tabla de resultados
    crear_tabla_resultados(),

], className="fade-in")
