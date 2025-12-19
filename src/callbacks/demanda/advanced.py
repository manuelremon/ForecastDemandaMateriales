"""
Callbacks Avanzados - Demanda
=============================

Callbacks para funcionalidades avanzadas:
- Gestión de modelos (guardar/cargar)
- Backtesting y validación histórica
"""
from dash import callback, Output, Input, State, html, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from src.ml.model_registry import ModelRegistry
from src.ml.backtesting import Backtester
from src.ml.predictor import DemandPredictor
from src.data.excel_loader import filtrar_consumo_por_material
from src.utils.theme import COLORS
from src.components.icons import lucide_icon
from src.utils.logger import get_logger
from .forecast import get_predictor_ref, set_predictor_ref

logger = get_logger(__name__)

# Instancias globales
_model_registry = None


def get_model_registry():
    """Obtiene instancia singleton del registry."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


def set_current_predictor(predictor):
    """Guarda referencia al predictor actual (delegado a forecast.py)."""
    set_predictor_ref(predictor)


def get_current_predictor():
    """Obtiene el predictor actual (desde forecast.py)."""
    return get_predictor_ref()


# ============================================================================
# CALLBACKS PARA MODALES
# ============================================================================

@callback(
    Output("modal-gestion-modelos", "is_open"),
    Input("btn-abrir-modal-modelos", "n_clicks"),
    Input("btn-cerrar-modal-modelos", "n_clicks"),
    State("modal-gestion-modelos", "is_open"),
    prevent_initial_call=True
)
def toggle_modal_modelos(n_abrir, n_cerrar, is_open):
    """Abre/cierra el modal de gestión de modelos."""
    return not is_open


@callback(
    Output("modal-backtesting", "is_open"),
    Input("btn-abrir-modal-backtest", "n_clicks"),
    Input("btn-cerrar-modal-backtest", "n_clicks"),
    State("modal-backtesting", "is_open"),
    prevent_initial_call=True
)
def toggle_modal_backtest(n_abrir, n_cerrar, is_open):
    """Abre/cierra el modal de backtesting."""
    return not is_open


# ============================================================================
# CALLBACKS PARA GESTIÓN DE MODELOS
# ============================================================================

@callback(
    Output("status-guardar-modelo", "children"),
    Input("btn-guardar-modelo", "n_clicks"),
    State("input-nombre-modelo", "value"),
    State("input-codigo-sap", "value"),
    State("select-modelo-ml", "value"),
    State("metrica-mae", "children"),
    State("metrica-r2", "children"),
    prevent_initial_call=True
)
def guardar_modelo_actual(n_clicks, nombre, material, modelo_tipo, mae, r2):
    """Guarda el modelo actual en el registry."""
    if not material:
        return dbc.Alert([
            lucide_icon("alert-circle", size="xs", className="me-1"),
            "Primero genere un forecast"
        ], color="warning", className="py-1 px-2 mb-0")

    predictor = get_current_predictor()
    if predictor is None or not predictor.entrenado:
        return dbc.Alert([
            lucide_icon("alert-circle", size="xs", className="me-1"),
            "No hay modelo entrenado para guardar"
        ], color="warning", className="py-1 px-2 mb-0")

    try:
        registry = get_model_registry()

        # Parsear métricas
        def parse_metric(val):
            if val == "--":
                return 0.0
            try:
                return float(str(val).replace('%', '').replace(',', ''))
            except:
                return 0.0

        metricas = {
            'mae': parse_metric(mae),
            'r2': parse_metric(r2) / 100
        }

        # Guardar modelo
        model_id = registry.guardar_modelo(
            modelo=predictor,
            metadata={
                'material': material,
                'modelo_tipo': modelo_tipo,
                'nombre': nombre or f"Modelo {material}",
                'metricas': metricas,
                'fecha_entrenamiento': datetime.now().isoformat()
            }
        )

        logger.info(f"Modelo guardado: {model_id} para material {material}")

        return dbc.Alert([
            lucide_icon("check-circle", size="xs", className="me-1"),
            f"Modelo guardado: {model_id[:8]}..."
        ], color="success", className="py-1 px-2 mb-0")

    except Exception as e:
        logger.error(f"Error guardando modelo: {e}")
        return dbc.Alert([
            lucide_icon("x-circle", size="xs", className="me-1"),
            f"Error: {str(e)[:50]}"
        ], color="danger", className="py-1 px-2 mb-0")


@callback(
    Output("lista-modelos-guardados", "children"),
    Input("modal-gestion-modelos", "is_open"),
    State("input-codigo-sap", "value"),
    prevent_initial_call=True
)
def cargar_lista_modelos(is_open, material):
    """Carga la lista de modelos guardados para el material."""
    if not is_open:
        return no_update

    try:
        registry = get_model_registry()
        modelos = registry.listar_modelos(material=material)

        if not modelos:
            return html.P(
                f"No hay modelos guardados{' para ' + material if material else ''}",
                className="text-muted small"
            )

        # Crear lista de modelos
        items = []
        for modelo in modelos[:10]:  # Limitar a 10
            fecha = modelo.get('fecha_entrenamiento', '')[:10]
            metricas = modelo.get('metricas', {})
            r2 = metricas.get('r2', 0)

            items.append(
                dbc.ListGroupItem([
                    html.Div([
                        html.Div([
                            html.Strong(modelo.get('nombre', 'Sin nombre')[:30]),
                            html.Small(f" ({modelo.get('modelo_tipo', 'N/A')})",
                                       className="text-muted")
                        ]),
                        html.Small([
                            f"Material: {modelo.get('material', 'N/A')} | ",
                            f"R²: {r2*100:.1f}% | ",
                            f"Fecha: {fecha}"
                        ], className="text-muted d-block")
                    ], className="flex-grow-1"),
                    dbc.Button([
                        lucide_icon("download", size="xs")
                    ], id={"type": "btn-cargar-modelo", "index": modelo.get('model_id', '')},
                       color="primary", size="sm", outline=True, className="btn-icon")
                ], className="d-flex justify-content-between align-items-center")
            )

        return dbc.ListGroup(items, flush=True)

    except Exception as e:
        logger.error(f"Error listando modelos: {e}")
        return html.P(f"Error cargando modelos: {str(e)[:50]}", className="text-danger small")


# ============================================================================
# CALLBACKS PARA BACKTESTING
# ============================================================================

@callback(
    Output("resultados-backtest", "children"),
    Input("btn-ejecutar-backtest", "n_clicks"),
    State("input-ventana-backtest", "value"),
    State("input-pasos-backtest", "value"),
    State("input-codigo-sap", "value"),
    State("select-modelo-ml", "value"),
    State("store-excel-data", "data"),
    State("filtro-centro-demanda", "value"),
    State("filtro-almacen-demanda", "value"),
    prevent_initial_call=True
)
def ejecutar_backtesting(n_clicks, ventana, pasos, material, modelo_tipo,
                         excel_data, centro, almacen):
    """Ejecuta backtesting y muestra resultados."""
    if not material:
        return dbc.Alert([
            lucide_icon("alert-circle", size="xs", className="me-1"),
            "Seleccione un material primero"
        ], color="warning")

    if not excel_data or 'consumo' not in excel_data:
        return dbc.Alert([
            lucide_icon("alert-circle", size="xs", className="me-1"),
            "Cargue datos de Excel primero"
        ], color="warning")

    try:
        # Preparar datos
        df_excel = pd.DataFrame(excel_data['consumo'])
        df_excel['fecha'] = pd.to_datetime(df_excel['fecha'])

        df_historico = filtrar_consumo_por_material(
            df_excel, material,
            centro if centro != "Todos" else None,
            almacen if almacen != "Todos" else None
        )

        if len(df_historico) < 60:
            return dbc.Alert([
                lucide_icon("alert-circle", size="xs", className="me-1"),
                f"Datos insuficientes ({len(df_historico)} registros). Se requieren al menos 60."
            ], color="warning")

        # Agrupar por día
        df_historico = df_historico.groupby("fecha").agg({
            "cantidad": "sum"
        }).reset_index()

        # Ejecutar backtest
        backtester = Backtester(modelo_tipo=modelo_tipo)
        reporte = backtester.ejecutar_backtest(
            df=df_historico,
            columna_objetivo='cantidad',
            ventana_test=int(ventana),
            n_pasos=int(pasos)
        )

        logger.info(f"Backtest completado para {material}: MAE promedio = {reporte.mae_promedio:.2f}")

        # Crear visualización
        return _crear_visualizacion_backtest(reporte, df_historico)

    except Exception as e:
        logger.error(f"Error en backtest: {e}")
        return dbc.Alert([
            lucide_icon("x-circle", size="xs", className="me-1"),
            f"Error: {str(e)[:100]}"
        ], color="danger")


def _crear_visualizacion_backtest(reporte, df_historico):
    """Crea la visualización de resultados del backtest."""
    # Métricas resumen
    metricas_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("MAE Promedio", className="card-subtitle text-muted"),
                    html.H4(f"{reporte.mae_promedio:.2f}", className="text-primary")
                ], className="p-2 text-center")
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Desv. Estándar", className="card-subtitle text-muted"),
                    html.H4(f"±{reporte.mae_std:.2f}", className="text-info")
                ], className="p-2 text-center")
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("R² Promedio", className="card-subtitle text-muted"),
                    html.H4(f"{reporte.r2_promedio*100:.1f}%", className="text-success")
                ], className="p-2 text-center")
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Pasos Evaluados", className="card-subtitle text-muted"),
                    html.H4(f"{len(reporte.resultados)}", className="text-secondary")
                ], className="p-2 text-center")
            ])
        ], md=3),
    ], className="mb-3 g-2")

    # Gráfico de resultados
    fig = go.Figure()

    # Línea histórica real
    fig.add_trace(go.Scatter(
        x=df_historico['fecha'],
        y=df_historico['cantidad'],
        name="Histórico Real",
        line=dict(color=COLORS.get('text_secondary', '#64748b'), width=1),
        mode="lines"
    ))

    # Predicciones de cada paso del backtest
    colores = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
               '#06b6d4', '#ec4899', '#14b8a6', '#f97316', '#6366f1']

    for i, resultado in enumerate(reporte.resultados):
        color = colores[i % len(colores)]
        fechas_pred = resultado.get('fechas', [])
        predicciones = resultado.get('predicciones', [])

        if len(fechas_pred) > 0 and len(predicciones) > 0:
            fig.add_trace(go.Scatter(
                x=fechas_pred,
                y=predicciones,
                name=f"Paso {i+1} (MAE: {resultado.get('mae', 0):.1f})",
                line=dict(color=color, width=2, dash='dot'),
                mode="lines",
                opacity=0.7
            ))

    fig.update_layout(
        title="Validación Walk-Forward",
        xaxis_title="Fecha",
        yaxis_title="Demanda",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        height=350
    )

    # Interpretación
    interpretacion = _interpretar_backtest(reporte)

    return html.Div([
        metricas_cards,
        dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig, config={"displayModeBar": False})
            ])
        ], className="mb-3"),
        dbc.Alert([
            lucide_icon("lightbulb", size="xs", className="me-2"),
            html.Strong("Interpretación: "),
            interpretacion
        ], color="info")
    ])


def _interpretar_backtest(reporte):
    """Genera interpretación del backtest."""
    mae = reporte.mae_promedio
    r2 = reporte.r2_promedio
    std = reporte.mae_std

    # Evaluar estabilidad
    cv = (std / mae * 100) if mae > 0 else 0  # Coeficiente de variación

    if r2 >= 0.7 and cv < 30:
        return f"El modelo muestra buena precisión (R²={r2*100:.0f}%) y estabilidad (CV={cv:.0f}%). Es confiable para predicciones."
    elif r2 >= 0.5:
        return f"El modelo tiene precisión moderada (R²={r2*100:.0f}%). Considere ajustar hiperparámetros o probar otros modelos."
    else:
        return f"El modelo tiene precisión baja (R²={r2*100:.0f}%). Se recomienda revisar la calidad de datos o probar modelos alternativos."
