"""
Callbacks de Búsqueda - Demanda
===============================

Búsqueda de materiales por código SAP.
"""
from dash import callback, Output, Input, State, html
import dash_bootstrap_components as dbc
import pandas as pd

from src.data.sap_loader import buscar_material_con_mrp
from src.components.icons import lucide_icon
from src.utils.logger import get_logger

logger = get_logger(__name__)


@callback(
    Output("material-encontrado", "children"),
    Output("select-material-demanda", "value"),
    Input("btn-buscar-material", "n_clicks"),
    State("input-codigo-sap", "value"),
    State("data-store", "data"),
    State("store-excel-data", "data"),
    prevent_initial_call=True
)
def buscar_material_por_codigo(n_clicks, codigo_sap, data, excel_data):
    """
    Busca material en Excel o catálogo completo.

    Prioriza búsqueda en Excel, fallback a BD.

    Args:
        n_clicks: Número de clicks en botón
        codigo_sap: Código a buscar
        data: Datos legacy del data-store
        excel_data: Datos del Excel cargado

    Returns:
        tuple: (mensaje_resultado, codigo_encontrado)
    """
    if not codigo_sap or len(str(codigo_sap).strip()) < 3:
        return html.Span([
            lucide_icon("info", size="sm"),
            " Ingrese al menos 3 caracteres"
        ], className="text-muted"), None

    codigo_buscar = str(codigo_sap).strip()

    # Buscar en Excel primero
    if excel_data and 'consumo' in excel_data:
        df = pd.DataFrame(excel_data['consumo'])
        matches = df[
            df['codigo'].astype(str).str.contains(codigo_buscar, case=False, na=False)
        ]

        if len(matches) > 0:
            codigo_encontrado = matches['codigo'].iloc[0]
            if 'descripcion' in matches.columns and pd.notna(matches['descripcion'].iloc[0]):
                descripcion = str(matches['descripcion'].iloc[0])[:45]
            else:
                descripcion = str(codigo_encontrado)

            n_coincidencias = matches['codigo'].nunique()
            if n_coincidencias == 1:
                return html.Span([
                    lucide_icon("check-circle", size="sm", style={"color": "#4CD964"}),
                    f" {descripcion}"
                ], className="text-success"), str(codigo_encontrado)
            else:
                return html.Span([
                    lucide_icon("info", size="sm", style={"color": "#007AFF"}),
                    f" {descripcion}... ({n_coincidencias} coincidencias)"
                ]), str(codigo_encontrado)

    # Fallback: buscar en BD
    try:
        resultados = buscar_material_con_mrp(codigo_buscar, limite=10)
        if len(resultados) > 0:
            mat = resultados[0]
            badge = dbc.Badge(
                "MRP" if mat.get("tiene_mrp") else "Sin MRP",
                color="success" if mat.get("tiene_mrp") else "secondary",
                className="ms-2"
            )
            return html.Span([
                lucide_icon("check-circle", size="sm"),
                f" {mat['descripcion'][:45]}",
                badge
            ]), mat["codigo"]
    except Exception as e:
        logger.debug(f"BD no disponible para búsqueda: {e}")

    return html.Span([
        lucide_icon("alert-triangle", size="sm", style={"color": "#FF9500"}),
        " No encontrado"
    ], className="text-warning"), None
