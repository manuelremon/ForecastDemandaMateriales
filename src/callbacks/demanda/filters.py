"""
Callbacks de Filtros - Demanda
==============================

Maneja la carga y actualización de filtros:
- Centros
- Almacenes
- Materiales
"""
from dash import callback, Output, Input, State, no_update
import pandas as pd

from src.data.sap_loader import obtener_centros, obtener_almacenes
from src.data.excel_loader import (
    obtener_materiales_desde_excel,
    obtener_centros_desde_excel,
    obtener_almacenes_desde_excel,
    dict_a_df
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@callback(
    Output("filtro-centro-demanda", "options"),
    Output("filtro-centro-demanda", "value"),
    Input("store-excel-data", "data"),
    Input("url", "pathname")
)
def cargar_centros_desde_excel_o_db(excel_data, pathname):
    """
    Carga centros desde Excel si está disponible, sino desde BD.

    Returns:
        tuple: (opciones_dropdown, valor_default)
    """
    if excel_data and 'consumo' in excel_data:
        df = pd.DataFrame(excel_data['consumo'])
        centros = obtener_centros_desde_excel(df)
        opciones = [{"label": "Todos", "value": "Todos"}] + [
            {"label": c, "value": c} for c in centros
        ]
        return opciones, "Todos"

    try:
        centros = obtener_centros()
        return [{"label": c, "value": c} for c in centros], None
    except Exception as e:
        logger.error(f"Error cargando centros: {e}")
        return [], None


@callback(
    Output("filtro-almacen-demanda", "options"),
    Output("filtro-almacen-demanda", "value"),
    Input("filtro-centro-demanda", "value"),
    State("store-excel-data", "data")
)
def cargar_almacenes_desde_excel_o_db(centro_seleccionado, excel_data):
    """
    Carga almacenes filtrados por centro.

    Args:
        centro_seleccionado: Centro seleccionado para filtrar
        excel_data: Datos del Excel cargado

    Returns:
        tuple: (opciones_dropdown, valor_default)
    """
    if excel_data and 'consumo' in excel_data:
        df = pd.DataFrame(excel_data['consumo'])
        almacenes = obtener_almacenes_desde_excel(df, centro_seleccionado)
        opciones = [{"label": "Todos", "value": "Todos"}] + [
            {"label": a, "value": a} for a in almacenes
        ]
        return opciones, "Todos"

    try:
        almacenes = obtener_almacenes(centro_seleccionado)
        return [{"label": a, "value": a} for a in almacenes], None
    except Exception as e:
        logger.error(f"Error cargando almacenes: {e}")
        return [], None


@callback(
    Output("select-material-demanda", "options"),
    Output("select-material-demanda", "style"),
    Input("store-excel-data", "data"),
    Input("data-store", "data")
)
def actualizar_lista_materiales(excel_data, data):
    """
    Actualiza la lista de materiales para selección.

    Prioriza Excel sobre data-store legacy.

    Returns:
        tuple: (opciones_dropdown, estilo_display)
    """
    if excel_data and 'consumo' in excel_data:
        df = pd.DataFrame(excel_data['consumo'])
        opciones = obtener_materiales_desde_excel(df)
        return opciones, {"display": "block", "marginTop": "8px"}

    if data and "records" in data:
        df = dict_a_df(data["records"])
        df_top = df.head(100).copy()
        df_top['descripcion_corta'] = df_top['descripcion'].str[:40] + '...'
        df_top['label'] = df_top['codigo'] + ' - ' + df_top['descripcion_corta']
        return (
            df_top[['label', 'codigo']].rename(columns={'codigo': 'value'}).to_dict('records'),
            {"display": "none"}
        )

    return [], {"display": "none"}


@callback(
    Output("input-codigo-sap", "value"),
    Input("select-material-demanda", "value"),
    prevent_initial_call=True
)
def sincronizar_material_seleccionado(material_seleccionado):
    """
    Sincroniza el material del dropdown al campo de entrada.
    """
    if material_seleccionado:
        return material_seleccionado
    return no_update
