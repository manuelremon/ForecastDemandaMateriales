"""
Callbacks de Exportación - Demanda
==================================

Exportación de forecasts a PDF, CSV y Excel.
"""
from dash import callback, Output, Input, State, dcc, no_update
import pandas as pd
from datetime import datetime
import base64
import io

from src.utils.exporters import ForecastExporter
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Instancia global del exportador
_exporter = None


def get_exporter():
    """Obtiene instancia singleton del exportador."""
    global _exporter
    if _exporter is None:
        _exporter = ForecastExporter()
    return _exporter


@callback(
    Output("download-forecast-csv", "data"),
    Input("btn-exportar-forecast", "n_clicks"),
    State("tabla-predicciones", "rowData"),
    State("input-codigo-sap", "value"),
    prevent_initial_call=True
)
def exportar_forecast_csv(n_clicks, row_data, material):
    """
    Exporta las predicciones a CSV.

    Args:
        n_clicks: Número de clicks en botón
        row_data: Datos de la tabla de predicciones
        material: Código del material

    Returns:
        dict: Datos para descarga del CSV
    """
    if not row_data:
        return no_update

    try:
        df = pd.DataFrame(row_data)
        exporter = get_exporter()

        # Preparar DataFrame con columnas esperadas
        df_export = df.rename(columns={
            'fecha': 'fecha',
            'prediccion': 'prediccion',
            'limite_inferior': 'limite_inferior',
            'limite_superior': 'limite_superior'
        })

        csv_content = exporter.exportar_csv(df_export)

        material_str = str(material).replace('/', '_') if material else 'forecast'
        fecha_str = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"forecast_{material_str}_{fecha_str}.csv"

        logger.info(f"CSV exportado: {filename}")

        return dcc.send_string(csv_content, filename)

    except Exception as e:
        logger.error(f"Error exportando CSV: {e}")
        return no_update


@callback(
    Output("download-forecast-pdf", "data"),
    Input("btn-exportar-pdf", "n_clicks"),
    State("tabla-predicciones", "rowData"),
    State("input-codigo-sap", "value"),
    State("metrica-mae", "children"),
    State("metrica-rmse", "children"),
    State("metrica-r2", "children"),
    State("metrica-mape", "children"),
    State("select-modelo-ml", "value"),
    State("slider-horizonte", "value"),
    State("filtro-centro-demanda", "value"),
    State("filtro-almacen-demanda", "value"),
    prevent_initial_call=True
)
def exportar_forecast_pdf(n_clicks, row_data, material, mae, rmse, r2, mape,
                          modelo, horizonte, centro, almacen):
    """
    Exporta el forecast completo a PDF.

    Args:
        n_clicks: Número de clicks en botón
        row_data: Datos de la tabla de predicciones
        material: Código del material
        mae, rmse, r2, mape: Métricas del modelo
        modelo: Tipo de modelo usado
        horizonte: Días de predicción
        centro, almacen: Filtros aplicados

    Returns:
        dict: Datos para descarga del PDF
    """
    if not row_data:
        return no_update

    try:
        df = pd.DataFrame(row_data)
        exporter = get_exporter()

        # Preparar métricas
        def parse_metric(val):
            if val == "--":
                return 0.0
            try:
                return float(str(val).replace('%', '').replace(',', ''))
            except:
                return 0.0

        metricas = {
            'mae': parse_metric(mae),
            'rmse': parse_metric(rmse),
            'r2': parse_metric(r2) / 100 if '%' not in str(r2) else parse_metric(r2),
            'mape': parse_metric(mape)
        }

        # Info del material
        material_info = {
            'codigo': material or 'N/A',
            'descripcion': f'Material {material}',
            'centro': centro or 'Todos',
            'almacen': almacen or 'Todos'
        }

        # Config
        config = {
            'modelo': modelo,
            'horizonte': horizonte
        }

        # Generar PDF
        pdf_bytes = exporter.exportar_pdf(df, metricas, material_info, config)

        material_str = str(material).replace('/', '_') if material else 'forecast'
        fecha_str = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"forecast_{material_str}_{fecha_str}.pdf"

        logger.info(f"PDF exportado: {filename}")

        # Codificar para descarga
        pdf_base64 = base64.b64encode(pdf_bytes).decode()

        return {
            "content": pdf_base64,
            "filename": filename,
            "base64": True
        }

    except Exception as e:
        logger.error(f"Error exportando PDF: {e}")
        return no_update


@callback(
    Output("download-forecast-excel", "data"),
    Input("btn-exportar-excel", "n_clicks"),
    State("tabla-predicciones", "rowData"),
    State("input-codigo-sap", "value"),
    State("metrica-mae", "children"),
    State("metrica-rmse", "children"),
    State("metrica-r2", "children"),
    State("metrica-mape", "children"),
    prevent_initial_call=True
)
def exportar_forecast_excel(n_clicks, row_data, material, mae, rmse, r2, mape):
    """
    Exporta el forecast a Excel con múltiples hojas.

    Args:
        n_clicks: Número de clicks en botón
        row_data: Datos de la tabla de predicciones
        material: Código del material
        mae, rmse, r2, mape: Métricas del modelo

    Returns:
        dict: Datos para descarga del Excel
    """
    if not row_data:
        return no_update

    try:
        df = pd.DataFrame(row_data)
        exporter = get_exporter()

        # Preparar métricas
        def parse_metric(val):
            if val == "--":
                return 0.0
            try:
                return float(str(val).replace('%', '').replace(',', ''))
            except:
                return 0.0

        metricas = {
            'mae': parse_metric(mae),
            'rmse': parse_metric(rmse),
            'r2': parse_metric(r2) / 100 if '%' not in str(r2) else parse_metric(r2),
            'mape': parse_metric(mape)
        }

        # Generar Excel
        excel_bytes = exporter.exportar_excel(df, metricas, material)

        material_str = str(material).replace('/', '_') if material else 'forecast'
        fecha_str = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"forecast_{material_str}_{fecha_str}.xlsx"

        logger.info(f"Excel exportado: {filename}")

        # Codificar para descarga
        excel_base64 = base64.b64encode(excel_bytes).decode()

        return {
            "content": excel_base64,
            "filename": filename,
            "base64": True
        }

    except Exception as e:
        logger.error(f"Error exportando Excel: {e}")
        return no_update
