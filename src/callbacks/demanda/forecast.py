"""
Callbacks de Forecast - Demanda
===============================

Generación de forecast usando ForecastService.
Integra validación de datos, métricas y logging.
"""
from dash import callback, Output, Input, State, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data.sap_loader import cargar_consumo_historico
from src.data.excel_loader import filtrar_consumo_por_material
from src.data.validators import DataValidator
from src.services.forecast_service import ForecastService, ForecastConfig
from src.utils.theme import PLOTLY_TEMPLATE, COLORS, color_con_alpha
from src.utils.logger import get_logger
from src.utils.constants import MODELOS_ML
from src.utils.plotly_helpers import crear_figura_vacia, crear_figura_warning
from src.utils.exceptions import (
    DataValidationError,
    InsufficientDataError,
    ModelTrainingError,
    PredictionError
)
from src.components.icons import lucide_icon
from src.components.activity_panel import (
    log_inicio_carga,
    log_validacion,
    log_entrenamiento,
    log_prediccion,
    log_error,
    log_warning
)
import time

logger = get_logger(__name__)

# Referencia al predictor actual para gestión de modelos
_current_predictor_ref = None


def set_predictor_ref(predictor):
    """Guarda referencia al predictor actual."""
    global _current_predictor_ref
    _current_predictor_ref = predictor


def get_predictor_ref():
    """Obtiene referencia al predictor actual."""
    return _current_predictor_ref

# Instancia global del servicio
_forecast_service = None


def get_forecast_service():
    """Obtiene instancia singleton del ForecastService."""
    global _forecast_service
    if _forecast_service is None:
        _forecast_service = ForecastService()
    return _forecast_service


def _crear_figura_forecast(df_historico, df_pred, confianza):
    """Crea el gráfico principal de forecast."""
    fig = go.Figure()

    # Datos históricos
    fig.add_trace(go.Scatter(
        x=df_historico["fecha"],
        y=df_historico["cantidad"],
        name="Histórico",
        line=dict(color=COLORS["text_secondary"], width=1),
        mode="lines"
    ))

    # Predicción
    fig.add_trace(go.Scatter(
        x=df_pred["fecha"],
        y=df_pred["prediccion"],
        name="Predicción",
        line=dict(color=COLORS["primary"], width=3),
        mode="lines"
    ))

    # Intervalo de confianza
    fig.add_trace(go.Scatter(
        x=pd.concat([df_pred["fecha"], df_pred["fecha"][::-1]]),
        y=pd.concat([df_pred["limite_superior"], df_pred["limite_inferior"][::-1]]),
        fill="toself",
        fillcolor=color_con_alpha('primary', 0.2),
        line=dict(color="rgba(255,255,255,0)"),
        name=f"Intervalo {int(confianza*100)}%",
        hoverinfo='skip'
    ))

    layout_base = {k: v for k, v in PLOTLY_TEMPLATE["layout"].items()
                   if k not in ["legend", "margin"]}
    fig.update_layout(
        **layout_base,
        xaxis_title="Fecha",
        yaxis_title="Demanda",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)", font={"color": COLORS['text_secondary']}
        ),
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified"
    )
    return fig


def _crear_figura_patron_semanal(df_historico):
    """Crea gráfico de patrón semanal."""
    df_historico["dia_semana"] = df_historico["fecha"].dt.day_name()
    patron_semanal = df_historico.groupby("dia_semana")["cantidad"].mean()

    dias_orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dias_es = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
    patron_semanal = patron_semanal.reindex(dias_orden)

    colores_dias = [
        "#3b82f6", "#06b6d4", "#10b981", "#f59e0b",
        "#ef4444", "#8b5cf6", "#ec4899"
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dias_es,
        y=patron_semanal.values,
        marker=dict(
            color=colores_dias,
            line=dict(color="rgba(255,255,255,0.3)", width=2),
            opacity=0.9
        ),
        text=[f"{v:.0f}" for v in patron_semanal.values],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text_primary"]),
        hovertemplate='<b>%{x}</b><br>Demanda: %{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=11, color=COLORS["text_secondary"]),
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(size=10, color=COLORS["text_secondary"]),
            gridcolor=COLORS["grid_color"],
            gridwidth=0.5
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )
    return fig


def _crear_figura_estacionalidad(df_historico):
    """Crea gráfico de estacionalidad mensual."""
    df_historico["mes"] = df_historico["fecha"].dt.month
    patron_mensual = df_historico.groupby("mes")["cantidad"].mean()

    meses_es = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=meses_es[:len(patron_mensual)],
        y=patron_mensual.values,
        mode="lines+markers",
        line=dict(color="#10b981", width=3, shape="spline"),
        marker=dict(size=10, color="#10b981"),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.15)",
        hovertemplate='<b>%{x}</b><br>Demanda: %{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=11, color=COLORS["text_secondary"]),
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(size=10, color=COLORS["text_secondary"]),
            gridcolor=COLORS["grid_color"],
            gridwidth=0.5
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False
    )
    return fig


def _crear_figura_importancia(importance_df):
    """Crea gráfico de importancia de features."""
    if len(importance_df) == 0:
        return crear_figura_vacia("Sin datos de importancia")

    top_features = importance_df.head(8).sort_values("importance")
    max_imp = top_features["importance"].max()

    colores = []
    for imp in top_features["importance"]:
        intensidad = imp / max_imp if max_imp > 0 else 0
        r = int(59 + (0 - 59) * intensidad)
        g = int(130 + (115 - 130) * intensidad)
        b = int(246 + (180 - 246) * intensidad)
        colores.append(f"rgb({r},{g},{b})")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_features["feature"],
        x=top_features["importance"],
        orientation="h",
        marker=dict(color=colores, opacity=0.95),
        text=[f"{v:.1%}" for v in top_features["importance"]],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text_primary"]),
        hovertemplate='<b>%{y}</b><br>Importancia: %{x:.2%}<extra></extra>'
    ))

    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=10, color=COLORS["text_secondary"]),
            gridcolor=COLORS["grid_color"],
            gridwidth=0.5
        ),
        yaxis=dict(
            tickfont=dict(size=11, color=COLORS["text_primary"]),
            autorange="reversed",
            showgrid=False
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=120, r=40, t=20, b=40),
        showlegend=False
    )
    return fig


@callback(
    Output("grafico-forecast", "figure"),
    Output("grafico-patron-semanal", "figure"),
    Output("grafico-estacionalidad", "figure"),
    Output("grafico-feature-importance", "figure"),
    Output("tabla-predicciones", "rowData"),
    Output("metrica-mae", "children"),
    Output("metrica-rmse", "children"),
    Output("metrica-r2", "children"),
    Output("metrica-mape", "children"),
    Output("info-modelo-config", "children"),
    Output("forecast-precision-valor", "children"),
    Output("forecast-error-valor", "children"),
    Output("forecast-demanda-valor", "children"),
    Output("forecast-tendencia-valor", "children"),
    Output("forecast-resumen", "children"),
    Output("forecast-resumen", "style"),
    Input("btn-generar-forecast", "n_clicks"),
    State("input-codigo-sap", "value"),
    State("filtro-centro-demanda", "value"),
    State("filtro-almacen-demanda", "value"),
    State("select-modelo-ml", "value"),
    State("slider-horizonte", "value"),
    State("select-confianza", "value"),
    State("data-store", "data"),
    State("store-excel-data", "data"),
    prevent_initial_call=True
)
def generar_forecast(n_clicks, material, centro, almacen, modelo_tipo,
                     horizonte, confianza, data, excel_data):
    """
    Genera el forecast usando ForecastService.

    Integra:
    - Validación de datos
    - Entrenamiento con métricas
    - Predicción con intervalos
    - Logging estructurado
    """
    fig_vacia = crear_figura_vacia("Seleccione un material")

    valores_default = (
        fig_vacia, fig_vacia, fig_vacia, fig_vacia, [],
        "--", "--", "--", "--",
        html.P("Seleccione un material para generar el forecast",
               className="text-muted small"),
        "--", "--", "--", "--",
        "", {"display": "none"}
    )

    if not material:
        return valores_default

    # Obtener datos históricos
    if excel_data and 'consumo' in excel_data:
        logger.info(f"Usando datos de Excel para material {material}")
        df_consumo = pd.DataFrame(excel_data['consumo'])
        df_consumo['fecha'] = pd.to_datetime(df_consumo['fecha'])
        df_historico = filtrar_consumo_por_material(
            df_consumo, material,
            centro if centro != "Todos" else None,
            almacen if almacen != "Todos" else None
        )
    else:
        df_historico = cargar_consumo_historico(
            material=material, centro=centro, dias=365
        )
        if almacen and len(df_historico) > 0 and "almacen" in df_historico.columns:
            df_historico = df_historico[df_historico["almacen"] == almacen]

    # Validar datos
    if len(df_historico) == 0:
        fig_sin_datos = crear_figura_warning(
            "No hay datos históricos de consumo para este material"
        )
        return (
            fig_sin_datos, fig_sin_datos, fig_sin_datos, fig_sin_datos, [],
            "--", "--", "--", "--",
            html.P("Sin datos históricos disponibles", className="text-warning small"),
            "--", "--", "--", "--",
            "", {"display": "none"}
        )

    # Agrupar por día
    df_historico = df_historico.groupby("fecha").agg({
        "codigo": "first",
        "cantidad": "sum"
    }).reset_index()

    # Log de carga de datos
    log_inicio_carga(len(df_historico), "Excel" if excel_data else "BD")

    # Validar calidad de datos
    validator = DataValidator()
    validacion = validator.validar_completo(df_historico)
    n_issues = len(validacion.get('issues', []))
    log_validacion(validacion.get('score', 0), n_issues)

    if validacion.get('issues'):
        for issue in validacion['issues']:
            if issue.get('severidad') == 'error':
                logger.warning(f"Problema de datos: {issue.get('mensaje')}")
                log_warning(f"Problema de datos: {issue.get('mensaje')[:50]}")

    # Ejecutar forecast usando el servicio
    service = get_forecast_service()
    config = ForecastConfig(
        modelo=modelo_tipo,
        horizonte=horizonte,
        intervalo_confianza=confianza,
        usar_validacion_cruzada=False,  # Más rápido para UI
        auto_tuning=False
    )

    inicio_entrenamiento = time.time()
    try:
        result = service.ejecutar_forecast(df_historico, config)
    except InsufficientDataError as e:
        log_error("Datos", f"Insuficientes: {e.details.get('available', 0)} de {e.details.get('required', 30)}")
        logger.error(f"Datos insuficientes: {e}")
        fig_error = crear_figura_warning(f"Datos insuficientes: {str(e)[:80]}")
        return (
            fig_error, fig_vacia, fig_vacia, fig_vacia, [],
            "--", "--", "--", "--",
            html.P(f"Datos insuficientes: {str(e)[:80]}", className="text-warning small"),
            "--", "--", "--", "--",
            "", {"display": "none"}
        )
    except (ModelTrainingError, PredictionError) as e:
        log_error("Modelo", str(e)[:50])
        logger.error(f"Error en modelo: {e}")
        fig_error = crear_figura_warning(f"Error en modelo: {str(e)[:80]}")
        return (
            fig_error, fig_vacia, fig_vacia, fig_vacia, [],
            "--", "--", "--", "--",
            html.P(f"Error en modelo: {str(e)[:80]}", className="text-danger small"),
            "--", "--", "--", "--",
            "", {"display": "none"}
        )
    except Exception as e:
        log_error("Forecast", str(e)[:50])
        logger.error(f"Error en forecast: {e}")
        fig_error = crear_figura_warning(f"Error: {str(e)[:100]}")
        return (
            fig_error, fig_vacia, fig_vacia, fig_vacia, [],
            "--", "--", "--", "--",
            html.P(f"Error: {str(e)[:100]}", className="text-danger small"),
            "--", "--", "--", "--",
            "", {"display": "none"}
        )

    duracion = time.time() - inicio_entrenamiento

    # Extraer resultados
    df_pred = result.predicciones.copy()
    metrics = result.metricas
    importance = result.importancia_features

    # Log de entrenamiento exitoso
    log_entrenamiento(modelo_tipo, duracion, metrics.get('mae', 0))

    # Guardar referencia al predictor para gestión de modelos
    if hasattr(service, 'predictor') and service.predictor is not None:
        set_predictor_ref(service.predictor)

    # Ajustar intervalos según confianza
    factor_confianza = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = factor_confianza.get(confianza, 1.96)

    std_pred = df_pred["prediccion"].std()
    df_pred["limite_inferior"] = df_pred["prediccion"] - z * std_pred * 0.3
    df_pred["limite_superior"] = df_pred["prediccion"] + z * std_pred * 0.3
    df_pred["limite_inferior"] = df_pred["limite_inferior"].clip(lower=0)

    # Calcular KPIs
    precision_valor = f"{metrics['r2']:.0%}" if metrics["r2"] > 0 else "N/A"
    error_valor = f"{metrics['mae']:.1f}"
    demanda_total = df_pred["prediccion"].sum()
    demanda_valor = f"{demanda_total:,.0f}"

    # Calcular tendencia
    promedio_historico = df_historico["cantidad"].mean()
    promedio_prediccion = df_pred["prediccion"].mean()
    if promedio_historico > 0:
        tendencia = ((promedio_prediccion - promedio_historico) / promedio_historico) * 100
        tendencia_valor = f"{tendencia:+.1f}%"
    else:
        tendencia_valor = "N/A"
        tendencia = 0

    # Crear figuras
    fig_forecast = _crear_figura_forecast(df_historico, df_pred, confianza)
    fig_semanal = _crear_figura_patron_semanal(df_historico.copy())
    fig_estacional = _crear_figura_estacionalidad(df_historico.copy())
    fig_importance = _crear_figura_importancia(importance)

    # Preparar datos para tabla
    df_tabla = df_pred.copy()
    df_tabla["fecha"] = df_tabla["fecha"].dt.strftime("%Y-%m-%d")
    df_tabla["dia_semana"] = pd.to_datetime(df_tabla["fecha"]).dt.day_name()

    dias_map = {
        "Monday": "Lun", "Tuesday": "Mar", "Wednesday": "Mie",
        "Thursday": "Jue", "Friday": "Vie", "Saturday": "Sab", "Sunday": "Dom"
    }
    df_tabla["dia_semana"] = df_tabla["dia_semana"].map(dias_map)

    datos_tabla = df_tabla[["fecha", "prediccion", "limite_inferior",
                            "limite_superior", "dia_semana"]].to_dict("records")

    # Info del modelo con validación
    calidad_datos = validacion.get('score', 0)
    color_calidad = 'success' if calidad_datos >= 80 else ('warning' if calidad_datos >= 60 else 'danger')

    info_modelo = html.Div([
        html.P([
            lucide_icon("bot", size="sm", className="me-2"),
            html.Strong("Modelo: "), MODELOS_ML.get(modelo_tipo, {}).get('nombre', modelo_tipo)
        ], className="mb-2"),
        html.P([
            lucide_icon("calendar", size="sm", className="me-2"),
            html.Strong("Horizonte: "), f"{horizonte} días"
        ], className="mb-2"),
        html.P([
            lucide_icon("target", size="sm", className="me-2"),
            html.Strong("Confianza: "), f"{int(confianza*100)}%"
        ], className="mb-2"),
        html.P([
            lucide_icon("building", size="sm", className="me-2"),
            html.Strong("Centro: "), centro if centro else "Todos"
        ], className="mb-2"),
        html.P([
            lucide_icon("warehouse", size="sm", className="me-2"),
            html.Strong("Almacén: "), almacen if almacen else "Todos"
        ], className="mb-2"),
        html.P([
            lucide_icon("database", size="sm", className="me-2"),
            html.Strong("Datos históricos: "), f"{len(df_historico)} registros"
        ], className="mb-2"),
        html.Hr(className="my-2"),
        html.P([
            lucide_icon("shield-check", size="sm", className="me-2"),
            html.Strong("Calidad datos: "),
            dbc.Badge(f"{calidad_datos:.0f}/100", color=color_calidad)
        ], className="mb-0"),
    ])

    # Generar resumen del forecast
    tendencia_texto = "al alza" if tendencia > 5 else "a la baja" if tendencia < -5 else "estable"
    precision_texto = "alta" if metrics["r2"] > 0.7 else "moderada" if metrics["r2"] > 0.4 else "baja"

    resumen_children = html.Div([
        html.H6([
            lucide_icon("lightbulb", size="sm", className="me-2"),
            "Resumen del Análisis"
        ]),
        html.P([
            f"El modelo {MODELOS_ML.get(modelo_tipo, {}).get('nombre', modelo_tipo)} predice una demanda total de ",
            html.Strong(f"{demanda_total:,.0f} unidades"),
            f" para los próximos {horizonte} días, con un promedio diario de ",
            html.Strong(f"{promedio_prediccion:.1f} unidades"),
            f". La tendencia es {tendencia_texto} ({tendencia_valor}) respecto al periodo histórico. ",
            f"La precisión del modelo es {precision_texto} (R²: {metrics['r2']:.1%}) ",
            f"con un error promedio de {metrics['mae']:.1f} unidades."
        ])
    ])

    # Log de predicción completada
    log_prediccion(horizonte, demanda_total)

    logger.info(
        f"Forecast generado: material={material}, modelo={modelo_tipo}, "
        f"horizonte={horizonte}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.2%}"
    )

    return (
        fig_forecast,
        fig_semanal,
        fig_estacional,
        fig_importance,
        datos_tabla,
        f"{metrics['mae']:.1f}",
        f"{metrics['rmse']:.1f}",
        f"{metrics['r2']:.2%}",
        f"{metrics['mape']:.1f}%",
        info_modelo,
        precision_valor,
        error_valor,
        demanda_valor,
        tendencia_valor,
        resumen_children,
        {"display": "block"}
    )
