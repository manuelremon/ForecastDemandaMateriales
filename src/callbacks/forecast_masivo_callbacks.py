"""
Callbacks del Tablero de Forecasting Masivo
============================================
Procesamiento de multiples materiales con ML (paralelo)
Con progreso en tiempo real
"""
from dash import callback, Output, Input, State, no_update, html, ctx
import dash_bootstrap_components as dbc
from dash import dcc
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import threading

from src.data.excel_loader import (
    obtener_centros_desde_excel,
    obtener_almacenes_desde_excel,
    filtrar_consumo_por_material
)
from src.ml.predictor import DemandPredictor
from src.components.icons import lucide_icon
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Numero de workers para procesamiento paralelo
NUM_WORKERS = min(8, multiprocessing.cpu_count())

# Estado global del progreso (thread-safe)
_progress_state = {
    "running": False,
    "total": 0,
    "current": 0,
    "message": "",
    "results": None,
    "error": None
}
_progress_lock = threading.Lock()


def _reset_progress():
    """Reinicia el estado del progreso"""
    global _progress_state
    with _progress_lock:
        _progress_state = {
            "running": False,
            "total": 0,
            "current": 0,
            "message": "",
            "results": None,
            "error": None
        }


def _update_progress(current: int, total: int, message: str = ""):
    """Actualiza el progreso de forma thread-safe"""
    global _progress_state
    with _progress_lock:
        _progress_state["current"] = current
        _progress_state["total"] = total
        _progress_state["message"] = message


def _get_progress():
    """Obtiene el estado actual del progreso"""
    with _progress_lock:
        return _progress_state.copy()


def _set_results(results, error=None):
    """Guarda los resultados finales"""
    global _progress_state
    with _progress_lock:
        _progress_state["results"] = results
        _progress_state["error"] = error
        _progress_state["running"] = False


def _set_running(running: bool):
    """Marca si el proceso está corriendo"""
    global _progress_state
    with _progress_lock:
        _progress_state["running"] = running


def procesar_material(args):
    """
    Procesa un material individual (funcion para paralelizar).

    Args:
        args: tupla (codigo, df_material, descripcion, modelo_tipo, horizonte)

    Returns:
        dict con resultado del forecast
    """
    codigo, df_historico, descripcion, modelo_tipo, horizonte = args

    try:
        if len(df_historico) == 0:
            return {
                "codigo": codigo,
                "descripcion": descripcion[:50] if descripcion else codigo,
                "demanda_total": 0,
                "r2": 0,
                "mae": 0,
                "estado": "Sin datos"
            }

        # Entrenar modelo
        predictor = DemandPredictor(modelo=modelo_tipo)
        metrics = predictor.entrenar(df_historico, "cantidad")

        # Predecir
        df_pred = predictor.predecir(df_historico, periodos=horizonte)
        demanda_total = df_pred["prediccion"].sum()

        return {
            "codigo": codigo,
            "descripcion": descripcion[:50] if descripcion else codigo,
            "demanda_total": round(demanda_total, 0),
            "r2": round(metrics["r2"], 3),
            "mae": round(metrics["mae"], 1),
            "estado": "OK"
        }

    except Exception as e:
        logger.error(f"Procesando {codigo}: {e}")
        return {
            "codigo": codigo,
            "descripcion": descripcion[:50] if descripcion else codigo,
            "demanda_total": 0,
            "r2": 0,
            "mae": 0,
            "estado": f"Error: {str(e)[:30]}"
        }


@callback(
    Output("filtro-centro-masivo", "options"),
    Output("filtro-centro-masivo", "value"),
    Input("store-excel-data", "data"),
    Input("url", "pathname")
)
def cargar_centros_masivo(excel_data, pathname):
    """Carga centros desde Excel si esta disponible"""
    if excel_data and 'consumo' in excel_data:
        df = pd.DataFrame(excel_data['consumo'])
        centros = obtener_centros_desde_excel(df)
        opciones = [{"label": "Todos", "value": "Todos"}] + [{"label": c, "value": c} for c in centros]
        return opciones, "Todos"
    else:
        return [], None


@callback(
    Output("filtro-almacen-masivo", "options"),
    Output("filtro-almacen-masivo", "value"),
    Input("filtro-centro-masivo", "value"),
    State("store-excel-data", "data")
)
def cargar_almacenes_masivo(centro_seleccionado, excel_data):
    """Carga almacenes desde Excel si esta disponible"""
    if excel_data and 'consumo' in excel_data:
        df = pd.DataFrame(excel_data['consumo'])
        almacenes = obtener_almacenes_desde_excel(df, centro_seleccionado)
        opciones = [{"label": "Todos", "value": "Todos"}] + [{"label": a, "value": a} for a in almacenes]
        return opciones, "Todos"
    else:
        return [], None


@callback(
    Output("progress-container-masivo", "style", allow_duplicate=True),
    Output("progress-forecast-masivo", "value", allow_duplicate=True),
    Output("progress-label-masivo", "children", allow_duplicate=True),
    Output("progress-text-masivo", "children", allow_duplicate=True),
    Output("interval-forecast-progress", "disabled", allow_duplicate=True),
    Output("store-forecast-state", "data", allow_duplicate=True),
    Input("btn-generar-forecast-masivo", "n_clicks"),
    State("filtro-centro-masivo", "value"),
    State("filtro-almacen-masivo", "value"),
    State("select-modelo-masivo", "value"),
    State("select-horizonte-masivo", "value"),
    State("select-confianza-masivo", "value"),
    State("input-limite-materiales", "value"),
    State("store-excel-data", "data"),
    prevent_initial_call=True
)
def iniciar_forecast_masivo(n_clicks, centro, almacen, modelo_tipo, horizonte, confianza, limite_materiales, excel_data):
    """Inicia el proceso de forecast y muestra la barra de progreso"""

    # Verificar si hay datos de Excel cargados
    if not excel_data or 'consumo' not in excel_data:
        return (
            {"display": "none"},
            0,
            "0%",
            "",
            True,  # Interval disabled
            {"running": False}
        )

    df_excel = pd.DataFrame(excel_data['consumo'])
    df_excel['fecha'] = pd.to_datetime(df_excel['fecha'])

    # Validar y aplicar limite de materiales
    limite = int(limite_materiales) if limite_materiales and limite_materiales > 0 else 50
    limite = min(limite, 500)

    codigos = df_excel['codigo'].unique().tolist()[:limite]

    if not codigos:
        return (
            {"display": "none"},
            0,
            "0%",
            "",
            True,
            {"running": False}
        )

    # Reiniciar progreso y marcar como corriendo
    _reset_progress()
    _set_running(True)

    total = len(codigos)
    _update_progress(0, total, f"Iniciando forecast para {total} materiales...")

    logger.info(f"Iniciando forecast masivo para {total} materiales (limite: {limite}) con {NUM_WORKERS} workers")

    # Iniciar procesamiento en un thread separado
    def procesar_en_background():
        try:
            # Preparar argumentos para procesamiento paralelo
            tareas = []
            for codigo in codigos:
                mat_excel = df_excel[df_excel['codigo'] == codigo]
                if len(mat_excel) > 0 and 'descripcion' in mat_excel.columns:
                    descripcion = mat_excel['descripcion'].iloc[0] if pd.notna(mat_excel['descripcion'].iloc[0]) else codigo
                else:
                    descripcion = codigo

                df_historico = filtrar_consumo_por_material(
                    df_excel,
                    codigo,
                    centro if centro != "Todos" else None,
                    almacen if almacen != "Todos" else None
                )

                tareas.append((codigo, df_historico, descripcion, modelo_tipo, horizonte))

            # Procesar en paralelo con tracking de progreso
            resultados = []
            procesados = 0

            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(procesar_material, tarea): tarea[0] for tarea in tareas}

                for future in as_completed(futures):
                    resultado = future.result()
                    resultados.append(resultado)
                    procesados += 1
                    _update_progress(procesados, total, f"Procesando material {procesados}/{total}")

            # Ordenar resultados
            resultados.sort(key=lambda x: x['codigo'])

            # Guardar resultados
            _set_results(resultados)

            logger.info(f"Forecast masivo completado: {len(resultados)} materiales procesados")

        except Exception as e:
            logger.error(f"Error en forecast masivo: {e}")
            _set_results(None, str(e))

    # Ejecutar en background
    thread = threading.Thread(target=procesar_en_background, daemon=True)
    thread.start()

    # Retornar estado inicial con barra de progreso visible
    return (
        {"display": "block"},  # Mostrar contenedor
        0,
        "0%",
        f"Preparando {total} materiales...",
        False,  # Habilitar interval
        {"running": True, "total": total}
    )


@callback(
    Output("progress-forecast-masivo", "value"),
    Output("progress-label-masivo", "children"),
    Output("progress-text-masivo", "children"),
    Output("interval-forecast-progress", "disabled"),
    Output("tabla-forecast-masivo", "rowData"),
    Output("kpi-total-materiales-masivo", "children"),
    Output("kpi-demanda-total-masivo", "children"),
    Output("kpi-precision-promedio-masivo", "children"),
    Output("status-forecast-masivo", "children"),
    Output("progress-container-masivo", "style"),
    Input("interval-forecast-progress", "n_intervals"),
    State("store-forecast-state", "data"),
    prevent_initial_call=True
)
def actualizar_progreso(n_intervals, forecast_state):
    """Actualiza la barra de progreso y muestra resultados cuando termina"""

    state = _get_progress()

    if state["running"]:
        # Proceso aún corriendo - actualizar progreso
        total = state["total"]
        current = state["current"]
        percent = int((current / total) * 100) if total > 0 else 0

        return (
            percent,
            f"{percent}%",
            state["message"],
            False,  # Mantener interval activo
            no_update,  # No actualizar tabla
            no_update,
            no_update,
            no_update,
            no_update,
            no_update
        )

    # Proceso terminado
    results = state["results"]
    error = state["error"]

    if error:
        # Hubo un error
        return (
            0,
            "Error",
            f"Error: {error}",
            True,  # Desactivar interval
            [],
            "--",
            "--",
            "--",
            dbc.Alert([
                lucide_icon("alert-circle", size="sm", className="me-2"),
                f"Error: {error}"
            ], color="danger"),
            {"display": "none"}
        )

    if results is None:
        # Sin resultados (no debería pasar)
        return (
            no_update, no_update, no_update, no_update,
            no_update, no_update, no_update, no_update,
            no_update, no_update
        )

    # Calcular KPIs
    total_demanda = sum(r['demanda_total'] for r in results)
    materiales_ok = [r for r in results if r['estado'] == 'OK']
    materiales_procesados = len(materiales_ok)
    total_r2 = sum(r['r2'] for r in materiales_ok)

    precision_promedio = (total_r2 / materiales_procesados * 100) if materiales_procesados > 0 else 0
    sin_datos = len([r for r in results if r["estado"] == "Sin datos"])
    con_error = len([r for r in results if "Error" in r["estado"]])

    # Crear mensaje de estado
    if materiales_procesados == len(results):
        status_msg = dbc.Alert([
            lucide_icon("check-circle", size="sm"),
            f" Forecast generado para {materiales_procesados} materiales"
        ], color="success")
    else:
        detalles = []
        if sin_datos > 0:
            detalles.append(f"{sin_datos} sin datos")
        if con_error > 0:
            detalles.append(f"{con_error} errores")
        status_msg = dbc.Alert([
            lucide_icon("info", size="sm"),
            f" {materiales_procesados} exitosos de {len(results)} ({', '.join(detalles)})"
        ], color="warning" if materiales_procesados > 0 else "danger")

    # Reiniciar progreso para próxima ejecución
    _reset_progress()

    return (
        100,
        "100%",
        "Completado",
        True,  # Desactivar interval
        results,
        f"{len(results)}",
        f"{total_demanda:,.0f}",
        f"{precision_promedio:.1f}%",
        status_msg,
        {"display": "none"}  # Ocultar barra de progreso
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
