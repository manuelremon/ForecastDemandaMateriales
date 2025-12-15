"""
Callbacks del Tablero de Forecasting Masivo
============================================
Procesamiento de multiples materiales con ML
"""
from dash import callback, Output, Input, State, no_update, html
import dash_bootstrap_components as dbc
from dash import dcc
import pandas as pd
import numpy as np
import base64
import io
from datetime import datetime

from src.data.sap_loader import cargar_consumo_historico, buscar_material_con_mrp, obtener_centros, obtener_almacenes
from src.ml.predictor import DemandPredictor
from src.components.icons import lucide_icon
from src.utils.logger import get_logger

logger = get_logger(__name__)


@callback(
    Output("filtro-centro-masivo", "options"),
    Input("url", "pathname")
)
def cargar_centros_masivo(pathname):
    """Carga centros para forecast masivo"""
    try:
        centros = obtener_centros()
        return [{"label": c, "value": c} for c in centros]
    except Exception as e:
        logger.error(f"Cargando centros: {e}")
        return []


@callback(
    Output("filtro-almacen-masivo", "options"),
    Input("filtro-centro-masivo", "value")
)
def cargar_almacenes_masivo(centro_seleccionado):
    """Carga almacenes segun el centro seleccionado"""
    try:
        almacenes = obtener_almacenes(centro_seleccionado)
        return [{"label": a, "value": a} for a in almacenes]
    except Exception as e:
        logger.error(f"Cargando almacenes: {e}")
        return []


def parsear_codigos_texto(texto):
    """Extrae codigos SAP del textarea"""
    if not texto:
        return []
    # Separar por lineas, comas o espacios
    lineas = texto.replace(",", "\n").replace(";", "\n").split("\n")
    codigos = [c.strip() for c in lineas if c.strip()]
    # Limitar a 50 codigos
    return codigos[:50]


def parsear_codigos_archivo(contents, filename):
    """Extrae codigos SAP de archivo Excel/CSV"""
    if not contents:
        return []

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return []

        # Buscar columna de codigos (codigo, material, sap, etc)
        col_codigo = None
        for col in df.columns:
            if any(x in col.lower() for x in ['codigo', 'material', 'sap', 'code']):
                col_codigo = col
                break

        if col_codigo is None and len(df.columns) > 0:
            col_codigo = df.columns[0]

        if col_codigo:
            codigos = df[col_codigo].astype(str).str.strip().tolist()
            return [c for c in codigos if c and c != 'nan'][:50]
    except Exception as e:
        logger.error(f"Parseando archivo: {e}")

    return []


@callback(
    Output("tabla-forecast-masivo", "rowData"),
    Output("kpi-total-materiales-masivo", "children"),
    Output("kpi-demanda-total-masivo", "children"),
    Output("kpi-precision-promedio-masivo", "children"),
    Output("status-forecast-masivo", "children"),
    Output("progress-container-masivo", "style"),
    Output("progress-forecast-masivo", "value"),
    Input("btn-generar-forecast-masivo", "n_clicks"),
    State("textarea-codigos-masivo", "value"),
    State("upload-codigos-masivo", "contents"),
    State("upload-codigos-masivo", "filename"),
    State("filtro-centro-masivo", "value"),
    State("filtro-almacen-masivo", "value"),
    State("select-modelo-masivo", "value"),
    State("select-horizonte-masivo", "value"),
    State("select-confianza-masivo", "value"),
    prevent_initial_call=True
)
def generar_forecast_masivo(n_clicks, texto_codigos, archivo_contents, archivo_filename,
                             centro, almacen, modelo_tipo, horizonte, confianza):
    """Genera forecast para multiples materiales"""

    # Obtener codigos desde texto o archivo
    codigos = parsear_codigos_texto(texto_codigos)

    if archivo_contents and not codigos:
        codigos = parsear_codigos_archivo(archivo_contents, archivo_filename)

    if not codigos:
        return (
            [],
            "--",
            "--",
            "--",
            dbc.Alert("Ingrese codigos SAP o cargue un archivo", color="warning"),
            {"display": "none"},
            0
        )

    # Procesar cada material
    resultados = []
    total_demanda = 0
    total_r2 = 0
    materiales_procesados = 0

    for idx, codigo in enumerate(codigos):
        try:
            # Buscar informacion del material
            info_material = buscar_material_con_mrp(codigo, limite=1)
            descripcion = info_material[0]["descripcion"] if info_material else "Desconocido"

            # Cargar datos historicos con filtros
            df_historico = cargar_consumo_historico(material=codigo, centro=centro, dias=365)

            # Filtrar por almacen si se especifico
            if almacen and len(df_historico) > 0 and "almacen" in df_historico.columns:
                df_historico = df_historico[df_historico["almacen"] == almacen]

            if len(df_historico) == 0:
                resultados.append({
                    "codigo": codigo,
                    "descripcion": descripcion,
                    "demanda_total": 0,
                    "r2": 0,
                    "mae": 0,
                    "estado": "Sin datos"
                })
                continue

            # Agrupar por dia
            df_historico = df_historico.groupby("fecha").agg({
                "cantidad": "sum"
            }).reset_index()

            # Entrenar modelo
            predictor = DemandPredictor(modelo=modelo_tipo)
            metrics = predictor.entrenar(df_historico, "cantidad")

            # Predecir
            df_pred = predictor.predecir(df_historico, periodos=horizonte)
            demanda_total = df_pred["prediccion"].sum()

            resultados.append({
                "codigo": codigo,
                "descripcion": descripcion[:50],
                "demanda_total": round(demanda_total, 0),
                "r2": round(metrics["r2"], 3),
                "mae": round(metrics["mae"], 1),
                "estado": "OK"
            })

            total_demanda += demanda_total
            total_r2 += metrics["r2"]
            materiales_procesados += 1

        except Exception as e:
            logger.error(f"Procesando {codigo}: {e}")
            resultados.append({
                "codigo": codigo,
                "descripcion": "Error",
                "demanda_total": 0,
                "r2": 0,
                "mae": 0,
                "estado": f"Error: {str(e)[:30]}"
            })

    # Calcular KPIs
    precision_promedio = (total_r2 / materiales_procesados * 100) if materiales_procesados > 0 else 0
    sin_datos = len([r for r in resultados if r["estado"] == "Sin datos"])
    con_error = len([r for r in resultados if "Error" in r["estado"]])

    if materiales_procesados == len(codigos):
        status_msg = dbc.Alert([
            lucide_icon("check-circle", size="sm"),
            f"Forecast generado para {materiales_procesados} materiales"
        ], color="success")
    else:
        detalles = []
        if sin_datos > 0:
            detalles.append(f"{sin_datos} sin datos historicos")
        if con_error > 0:
            detalles.append(f"{con_error} con errores")
        status_msg = dbc.Alert([
            lucide_icon("info", size="sm"),
            f"Forecast: {materiales_procesados} exitosos de {len(codigos)} ({', '.join(detalles)})"
        ], color="warning" if materiales_procesados > 0 else "danger")

    return (
        resultados,
        f"{len(codigos)}",
        f"{total_demanda:,.0f}",
        f"{precision_promedio:.1f}%",
        status_msg,
        {"display": "none"},
        100
    )


@callback(
    Output("download-forecast-masivo-excel", "data"),
    Input("btn-exportar-masivo-excel", "n_clicks"),
    State("tabla-forecast-masivo", "rowData"),
    prevent_initial_call=True
)
def exportar_forecast_excel(n_clicks, row_data):
    """Exporta resultados a Excel"""
    if not row_data:
        return no_update

    df = pd.DataFrame(row_data)

    # Renombrar columnas para el Excel
    df.columns = ["Codigo SAP", "Descripcion", "Demanda Total", "R2", "MAE", "Estado"]

    return dcc.send_data_frame(
        df.to_excel,
        f"forecast_masivo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        index=False
    )
