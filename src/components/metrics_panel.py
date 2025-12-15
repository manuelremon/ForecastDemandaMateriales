"""
Panel de Métricas Explicadas para Forecast MR

Proporciona componentes visuales para mostrar métricas
de modelos ML con interpretaciones claras y contextuales.

Author: Manuel Remón
"""

from typing import Dict, Any, Optional, List
from dash import html
import dash_bootstrap_components as dbc


def crear_metrica_card(
    nombre: str,
    valor: float,
    formato: str = ".2f",
    interpretacion: str = "",
    color: str = "primary",
    icono: str = "info",
    mostrar_barra: bool = True,
    porcentaje_barra: Optional[float] = None
) -> dbc.Card:
    """
    Crea una tarjeta de métrica individual.

    Args:
        nombre: Nombre de la métrica
        valor: Valor numérico
        formato: Formato para mostrar (ej: ".2f", ".1%")
        interpretacion: Texto explicativo
        color: Color Bootstrap (primary, success, warning, danger, info)
        icono: Nombre del icono Lucide
        mostrar_barra: Si mostrar barra de progreso
        porcentaje_barra: Porcentaje para la barra (0-100)

    Returns:
        Componente dbc.Card
    """
    from src.components.icons import lucide_icon

    # Formatear valor
    if formato.endswith('%'):
        valor_formateado = f"{valor*100:{formato[:-1]}}%"
    else:
        valor_formateado = f"{valor:{formato}}"

    # Barra de progreso
    barra = None
    if mostrar_barra and porcentaje_barra is not None:
        barra = dbc.Progress(
            value=min(100, max(0, porcentaje_barra)),
            color=color,
            style={"height": "4px"},
            className="mt-2"
        )

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    lucide_icon(icono, size="xs", className=f"text-{color}"),
                    html.Span(nombre, className="ms-2 text-muted small")
                ], className="d-flex align-items-center"),
                html.H4(valor_formateado, className=f"mb-0 text-{color} fw-bold"),
                html.Small(interpretacion, className="text-muted") if interpretacion else None,
                barra
            ])
        ], className="p-3")
    ], className="h-100 shadow-sm")


def crear_panel_metricas(
    metricas: Dict[str, float],
    media_historica: Optional[float] = None,
    mostrar_baseline: bool = False,
    baseline_metricas: Optional[Dict[str, float]] = None
) -> html.Div:
    """
    Crea panel completo con todas las métricas explicadas.

    Args:
        metricas: Diccionario con métricas del modelo
        media_historica: Media histórica para contextualizar MAE
        mostrar_baseline: Si mostrar comparación con baseline
        baseline_metricas: Métricas del modelo baseline

    Returns:
        Componente html.Div con el panel
    """
    r2 = metricas.get('r2', 0)
    mae = metricas.get('mae', 0)
    rmse = metricas.get('rmse', 0)
    mape = metricas.get('mape', 0)

    # Interpretaciones
    interp_r2 = interpretar_r2(r2)
    interp_mae = interpretar_mae(mae, media_historica)
    interp_mape = interpretar_mape(mape)

    # Colores según calidad
    color_r2 = 'success' if r2 >= 0.7 else ('warning' if r2 >= 0.5 else 'danger')
    color_mae = 'success' if media_historica and mae < media_historica * 0.1 else 'info'
    color_mape = 'success' if mape < 15 else ('warning' if mape < 30 else 'danger')

    cards = [
        dbc.Col([
            crear_metrica_card(
                nombre="R² (Precisión)",
                valor=r2 * 100,
                formato=".1f",
                interpretacion=interp_r2,
                color=color_r2,
                icono="target",
                mostrar_barra=True,
                porcentaje_barra=r2 * 100
            )
        ], md=3),
        dbc.Col([
            crear_metrica_card(
                nombre="MAE (Error Medio)",
                valor=mae,
                formato=".1f",
                interpretacion=interp_mae,
                color=color_mae,
                icono="alert-circle",
                mostrar_barra=False
            )
        ], md=3),
        dbc.Col([
            crear_metrica_card(
                nombre="RMSE",
                valor=rmse,
                formato=".1f",
                interpretacion="Penaliza errores grandes",
                color="info",
                icono="activity",
                mostrar_barra=False
            )
        ], md=3),
        dbc.Col([
            crear_metrica_card(
                nombre="MAPE",
                valor=mape,
                formato=".1f",
                interpretacion=interp_mape,
                color=color_mape,
                icono="percent",
                mostrar_barra=True,
                porcentaje_barra=100 - min(mape, 100)
            )
        ], md=3)
    ]

    contenido = [dbc.Row(cards, className="g-3")]

    # Comparación con baseline
    if mostrar_baseline and baseline_metricas:
        contenido.append(html.Hr(className="my-3"))
        contenido.append(crear_comparacion_baseline(metricas, baseline_metricas))

    return html.Div(contenido)


def interpretar_r2(r2: float) -> str:
    """Interpreta el valor de R²"""
    if r2 >= 0.9:
        return "Excelente precisión"
    elif r2 >= 0.8:
        return "Muy buena precisión"
    elif r2 >= 0.7:
        return "Buena precisión"
    elif r2 >= 0.5:
        return "Precisión moderada"
    elif r2 >= 0.3:
        return "Precisión baja"
    else:
        return "Precisión muy baja"


def interpretar_mae(mae: float, media: Optional[float] = None) -> str:
    """Interpreta el valor de MAE"""
    if media is None or media == 0:
        if mae < 10:
            return "Error bajo"
        elif mae < 50:
            return "Error moderado"
        else:
            return "Error alto"

    pct_error = (mae / media) * 100
    if pct_error < 5:
        return f"Excelente ({pct_error:.0f}% de media)"
    elif pct_error < 10:
        return f"Muy bueno ({pct_error:.0f}% de media)"
    elif pct_error < 20:
        return f"Aceptable ({pct_error:.0f}% de media)"
    else:
        return f"Alto ({pct_error:.0f}% de media)"


def interpretar_mape(mape: float) -> str:
    """Interpreta el valor de MAPE"""
    if mape < 10:
        return "Excelente (<10%)"
    elif mape < 20:
        return "Muy bueno (10-20%)"
    elif mape < 30:
        return "Aceptable (20-30%)"
    elif mape < 50:
        return "Mejorable (30-50%)"
    else:
        return "Considere más datos"


def crear_comparacion_baseline(
    metricas_modelo: Dict[str, float],
    metricas_baseline: Dict[str, float]
) -> html.Div:
    """
    Crea sección de comparación con baseline.

    Args:
        metricas_modelo: Métricas del modelo actual
        metricas_baseline: Métricas del modelo baseline

    Returns:
        Componente con la comparación
    """
    mae_modelo = metricas_modelo.get('mae', 0)
    mae_baseline = metricas_baseline.get('mae', 1)

    mejora = ((mae_baseline - mae_modelo) / mae_baseline * 100) if mae_baseline > 0 else 0
    es_mejor = mejora > 0

    color = "success" if es_mejor else "danger"
    icono = "trending-up" if es_mejor else "trending-down"

    from src.components.icons import lucide_icon

    return html.Div([
        html.H6([
            lucide_icon("bar-chart-3", size="xs", className="me-2"),
            "Comparación vs Baseline"
        ], className="text-muted mb-3"),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("Modelo Naive (baseline)", className="small text-muted"),
                    html.H5(f"MAE: {mae_baseline:.1f}", className="mb-0")
                ])
            ], md=4),
            dbc.Col([
                html.Div([
                    html.Span("Modelo Actual", className="small text-muted"),
                    html.H5(f"MAE: {mae_modelo:.1f}", className=f"mb-0 text-{color}")
                ])
            ], md=4),
            dbc.Col([
                html.Div([
                    html.Span("Mejora", className="small text-muted"),
                    html.H5([
                        lucide_icon(icono, size="xs", className=f"text-{color} me-1"),
                        f"{abs(mejora):.1f}%"
                    ], className=f"mb-0 text-{color}")
                ])
            ], md=4)
        ], className="text-center")
    ], className="p-3 bg-light rounded")


def crear_panel_calidad_datos(
    validacion: Dict[str, Any],
    compacto: bool = False
) -> html.Div:
    """
    Crea panel de calidad de datos.

    Args:
        validacion: Resultado de validación
        compacto: Si mostrar versión compacta

    Returns:
        Componente con panel de calidad
    """
    from src.components.icons import lucide_icon

    score = validacion.get('score', 0)
    issues = validacion.get('issues', [])
    outliers = validacion.get('outlier_reports', [])

    # Color según score
    if score >= 80:
        color = "success"
        icono = "check-circle"
        estado = "Excelente"
    elif score >= 60:
        color = "warning"
        icono = "alert-circle"
        estado = "Aceptable"
    else:
        color = "danger"
        icono = "x-circle"
        estado = "Revisar"

    if compacto:
        return html.Div([
            lucide_icon(icono, size="xs", className=f"text-{color} me-1"),
            html.Span(f"Calidad: {score:.0f}/100", className=f"small text-{color}")
        ], className="d-flex align-items-center")

    # Versión completa
    contenido = [
        # Header con score
        html.Div([
            html.Div([
                lucide_icon(icono, size="sm", className=f"text-{color}"),
                html.Div([
                    html.H5(f"{score:.0f}/100", className=f"mb-0 text-{color}"),
                    html.Small(estado, className="text-muted")
                ], className="ms-3")
            ], className="d-flex align-items-center"),
            dbc.Progress(
                value=score,
                color=color,
                style={"height": "8px"},
                className="mt-2"
            )
        ], className="mb-3"),

        # Resumen
        html.P(validacion.get('resumen', ''), className="text-muted small")
    ]

    # Issues (si hay)
    if issues:
        warnings = [i for i in issues if i.get('severidad') in ['warning', 'error']]
        if warnings:
            contenido.append(html.Hr())
            contenido.append(html.H6("Advertencias", className="text-muted"))
            for issue in warnings[:3]:  # Limitar a 3
                severidad = issue.get('severidad', 'info')
                color_issue = 'warning' if severidad == 'warning' else 'danger'
                contenido.append(
                    html.Div([
                        lucide_icon("alert-triangle", size="xs", className=f"text-{color_issue} me-2"),
                        html.Span(issue.get('mensaje', ''), className="small")
                    ], className="mb-1")
                )

    # Outliers (si hay)
    total_outliers = sum(o.get('n_outliers', 0) for o in outliers)
    if total_outliers > 0:
        contenido.append(
            html.Div([
                lucide_icon("filter", size="xs", className="text-info me-2"),
                html.Span(f"{total_outliers} outliers detectados", className="small text-info")
            ], className="mt-2")
        )

    return dbc.Card([
        dbc.CardBody(contenido, className="p-3")
    ], className="shadow-sm")


def crear_panel_actividad(
    logs: List[Dict[str, Any]],
    max_items: int = 10
) -> html.Div:
    """
    Crea panel de actividad/logs.

    Args:
        logs: Lista de entradas de log
        max_items: Máximo de items a mostrar

    Returns:
        Componente con panel de actividad
    """
    from src.components.icons import lucide_icon

    items = []
    for log in logs[:max_items]:
        timestamp = log.get('timestamp', '')
        mensaje = log.get('mensaje', '')
        tipo = log.get('tipo', 'info')

        # Icono y color según tipo
        iconos = {
            'success': ('check', 'success'),
            'warning': ('alert-triangle', 'warning'),
            'error': ('x-circle', 'danger'),
            'info': ('info', 'info')
        }
        icono, color = iconos.get(tipo, ('info', 'info'))

        items.append(
            html.Div([
                html.Span(timestamp, className="text-muted small me-2", style={"width": "60px"}),
                lucide_icon(icono, size="xs", className=f"text-{color} me-2"),
                html.Span(mensaje, className="small")
            ], className="d-flex align-items-center mb-1")
        )

    return html.Div([
        html.H6([
            lucide_icon("activity", size="xs", className="me-2"),
            "Actividad"
        ], className="text-muted mb-3"),
        html.Div(items, style={"maxHeight": "200px", "overflowY": "auto"})
    ], className="p-3 bg-light rounded")


def crear_resumen_forecast(
    metricas: Dict[str, float],
    config: Dict[str, Any],
    predicciones_resumen: Dict[str, Any]
) -> html.Div:
    """
    Crea resumen ejecutivo del forecast.

    Args:
        metricas: Métricas del modelo
        config: Configuración usada
        predicciones_resumen: Resumen de predicciones (total, promedio, etc.)

    Returns:
        Componente con resumen
    """
    from src.components.icons import lucide_icon

    r2 = metricas.get('r2', 0)
    horizonte = config.get('horizonte', 30)
    modelo = config.get('modelo_tipo', 'random_forest')
    demanda_total = predicciones_resumen.get('total', 0)
    demanda_promedio = predicciones_resumen.get('promedio', 0)

    # Generar texto de resumen
    calidad = "alta" if r2 >= 0.7 else ("moderada" if r2 >= 0.5 else "baja")

    texto = f"""
    El modelo {modelo.replace('_', ' ').title()} predice una demanda total de
    {demanda_total:,.0f} unidades para los próximos {horizonte} días
    (promedio diario: {demanda_promedio:,.0f}).
    La precisión del modelo es {calidad} (R²={r2*100:.0f}%).
    """

    return dbc.Alert([
        html.Div([
            lucide_icon("file-text", size="sm", className="me-3"),
            html.Div([
                html.H6("Resumen del Forecast", className="alert-heading mb-1"),
                html.P(texto.strip(), className="mb-0 small")
            ])
        ], className="d-flex")
    ], color="info", className="mb-0")
