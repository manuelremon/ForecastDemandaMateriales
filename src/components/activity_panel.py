"""
Panel de Actividad para Forecast MR

Muestra logs en tiempo real del proceso de forecasting.
Proporciona feedback visual al usuario sobre las operaciones.

Author: Manuel Remón
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dash import html, dcc
import dash_bootstrap_components as dbc
from collections import deque
import threading

from src.components.icons import lucide_icon


# Buffer de actividad global (thread-safe)
_activity_buffer = deque(maxlen=50)
_activity_lock = threading.Lock()


def agregar_actividad(
    mensaje: str,
    tipo: str = "info",
    detalle: Optional[str] = None
) -> None:
    """
    Agrega una entrada al log de actividad.

    Args:
        mensaje: Mensaje principal
        tipo: Tipo de mensaje (success, warning, error, info)
        detalle: Detalle adicional opcional
    """
    with _activity_lock:
        _activity_buffer.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "mensaje": mensaje,
            "tipo": tipo,
            "detalle": detalle
        })


def obtener_actividades(limite: int = 20) -> List[Dict[str, Any]]:
    """
    Obtiene las últimas actividades.

    Args:
        limite: Número máximo de actividades a retornar

    Returns:
        Lista de actividades (más recientes primero)
    """
    with _activity_lock:
        return list(_activity_buffer)[-limite:][::-1]


def limpiar_actividades() -> None:
    """Limpia el buffer de actividades."""
    with _activity_lock:
        _activity_buffer.clear()


def crear_item_actividad(actividad: Dict[str, Any]) -> html.Div:
    """
    Crea un item visual de actividad.

    Args:
        actividad: Diccionario con datos de la actividad

    Returns:
        Componente html.Div con el item
    """
    tipo = actividad.get("tipo", "info")
    iconos = {
        "success": ("check", "success"),
        "warning": ("alert-triangle", "warning"),
        "error": ("x-circle", "danger"),
        "info": ("info", "info"),
        "loading": ("loader", "primary")
    }
    icono, color = iconos.get(tipo, ("info", "info"))

    detalle = actividad.get("detalle")
    detalle_elem = html.Small(
        f"  └─ {detalle}",
        className="text-muted d-block ms-4"
    ) if detalle else None

    return html.Div([
        html.Div([
            html.Span(
                actividad.get("timestamp", ""),
                className="text-muted me-2",
                style={"fontSize": "0.75rem", "fontFamily": "monospace"}
            ),
            lucide_icon(icono, size="xs", className=f"text-{color} me-1"),
            html.Span(
                actividad.get("mensaje", ""),
                className="small"
            )
        ], className="d-flex align-items-center"),
        detalle_elem
    ], className="mb-1 py-1 border-bottom border-light")


def crear_panel_actividad(
    id_componente: str = "panel-actividad",
    altura: str = "200px",
    titulo: str = "Actividad",
    mostrar_limpiar: bool = True
) -> html.Div:
    """
    Crea el panel de actividad completo.

    Args:
        id_componente: ID del componente
        altura: Altura máxima del panel
        titulo: Título del panel
        mostrar_limpiar: Si mostrar botón de limpiar

    Returns:
        Componente html.Div con el panel completo
    """
    header = html.Div([
        html.H6([
            lucide_icon("activity", size="xs", className="me-2"),
            titulo
        ], className="mb-0"),
        dbc.Button(
            lucide_icon("trash-2", size="xs"),
            id=f"{id_componente}-limpiar",
            color="link",
            size="sm",
            className="btn-icon"
        ) if mostrar_limpiar else None
    ], className="d-flex justify-content-between align-items-center mb-2")

    contenido = html.Div(
        id=f"{id_componente}-contenido",
        style={
            "maxHeight": altura,
            "overflowY": "auto",
            "fontSize": "0.85rem"
        },
        children=[
            html.P(
                "Sin actividad reciente",
                className="text-muted small text-center py-3"
            )
        ]
    )

    # Interval para actualización automática
    intervalo = dcc.Interval(
        id=f"{id_componente}-interval",
        interval=2000,  # 2 segundos
        n_intervals=0,
        disabled=True  # Deshabilitado por defecto
    )

    return html.Div([
        header,
        contenido,
        intervalo
    ], id=id_componente, className="p-3 bg-light rounded")


def crear_panel_actividad_compacto(actividades: List[Dict[str, Any]] = None) -> html.Div:
    """
    Crea versión compacta del panel de actividad.

    Args:
        actividades: Lista de actividades a mostrar

    Returns:
        Componente html.Div
    """
    if not actividades:
        actividades = obtener_actividades(5)

    if not actividades:
        return html.Div([
            html.Small("Sin actividad", className="text-muted")
        ])

    items = [crear_item_actividad(act) for act in actividades[:5]]

    return html.Div(items, style={"fontSize": "0.8rem"})


# Funciones de logging convenientes
def log_inicio_carga(n_registros: int, fuente: str = "Excel"):
    """Log de inicio de carga de datos."""
    agregar_actividad(
        f"Datos cargados: {n_registros:,} registros",
        tipo="success",
        detalle=f"Fuente: {fuente}"
    )


def log_validacion(score: float, n_issues: int = 0):
    """Log de validación de datos."""
    tipo = "success" if score >= 70 else ("warning" if score >= 50 else "error")
    agregar_actividad(
        f"Validación: {score:.0f}/100",
        tipo=tipo,
        detalle=f"{n_issues} advertencias" if n_issues > 0 else None
    )


def log_entrenamiento(modelo: str, duracion: float, mae: float):
    """Log de entrenamiento de modelo."""
    agregar_actividad(
        f"Modelo entrenado: {modelo}",
        tipo="success",
        detalle=f"MAE: {mae:.2f} ({duracion:.1f}s)"
    )


def log_prediccion(horizonte: int, total: float):
    """Log de predicción generada."""
    agregar_actividad(
        f"Predicción generada: {horizonte} días",
        tipo="success",
        detalle=f"Demanda total: {total:,.0f}"
    )


def log_error(etapa: str, mensaje: str):
    """Log de error."""
    agregar_actividad(
        f"Error en {etapa}",
        tipo="error",
        detalle=mensaje[:100]
    )


def log_warning(mensaje: str, detalle: str = None):
    """Log de advertencia."""
    agregar_actividad(mensaje, tipo="warning", detalle=detalle)


def log_info(mensaje: str, detalle: str = None):
    """Log informativo."""
    agregar_actividad(mensaje, tipo="info", detalle=detalle)
