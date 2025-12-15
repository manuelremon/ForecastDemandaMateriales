"""
Componentes reutilizables para MRP Analytics
Glassmorphism iOS Design System
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from typing import List, Dict, Any, Optional, Union
from src.utils.formatters import formato_numero, formato_moneda, formato_porcentaje
from src.utils.theme import COLORS, KPI_STYLES, BADGE_COLORS
from src.components.icons import lucide_icon, FA_TO_LUCIDE_MAP


def kpi_card(
    titulo: str,
    valor: str,
    subtitulo: str = "",
    tendencia: dict = None,
    icono: str = "fa-chart-line",
    color: str = "primary",
    tooltip: Optional[str] = None,
    tooltip_id: Optional[str] = None,
    valor_id: Optional[str] = None,
    hoverable: bool = False
) -> Union[dbc.Card, html.Div]:
    """
    Tarjeta KPI con estilo Glassmorphism iOS

    Args:
        titulo: Titulo del KPI
        valor: Valor principal a mostrar
        subtitulo: Texto secundario
        tendencia: Dict con keys: direccion, valor, clase, icono
        icono: Clase FontAwesome del icono
        color: Color del tema (primary, success, warning, danger, info, purple)
        tooltip: Texto del tooltip informativo (opcional)
        tooltip_id: ID unico para el tooltip (requerido si tooltip se proporciona)
        valor_id: ID para el elemento del valor (util para actualizaciones dinamicas)
        hoverable: Si True, agrega efecto hover con elevacion (default False)
    """
    # Colores iOS con fallback
    color_map = {
        "primary": "#007AFF",
        "success": "#4CD964",
        "warning": "#FF9500",
        "danger": "#FF3B30",
        "info": "#5AC8FA",
        "purple": "#5856D6",
    }
    accent_color = color_map.get(color, "#007AFF")
    kpi_class = f"kpi-{color}"
    hover_class = " hoverable" if hoverable else ""

    # Contenido de tendencia
    tendencia_content = []
    if tendencia:
        tendencia_content = [
            lucide_icon(tendencia.get('icono', 'fa-minus'), size="sm", className="me-1"),
            html.Span(tendencia.get('valor', ''), className=tendencia.get('clase', ''))
        ]

    # Titulo con tooltip opcional
    titulo_content = html.H6(
        titulo,
        className="mb-2 text-uppercase",
        style={
            "fontSize": "0.7rem",
            "letterSpacing": "0.5px",
            "fontWeight": "600",
            "color": "#8E8E93"
        }
    )

    if tooltip and tooltip_id:
        titulo_content = html.Div([
            html.H6(titulo, className="mb-2 text-uppercase d-inline",
                   style={
                       "fontSize": "0.7rem",
                       "letterSpacing": "0.5px",
                       "fontWeight": "600",
                       "color": "#8E8E93"
                   }),
            lucide_icon("info", size="xs", className="ms-2", style={"color": "#BDBEC2", "cursor": "pointer"}, id=tooltip_id)
        ])

    # Valor principal
    valor_props = {
        "className": "mb-1",
        "style": {
            "fontWeight": "700",
            "color": accent_color,
            "fontSize": "1.75rem",
            "lineHeight": "1.2"
        }
    }
    if valor_id:
        valor_props["id"] = valor_id

    valor_element = html.H2(valor, **valor_props)

    # Card con glassmorphism
    card = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    titulo_content,
                    valor_element,
                    html.Div([
                        html.Small(subtitulo, className="me-2",
                                  style={"color": "#8E8E93"}) if subtitulo else None,
                        html.Small(tendencia_content, className="d-inline") if tendencia else None,
                    ])
                ], width=9),
                dbc.Col([
                    html.Div([
                        lucide_icon(icono, size="2x", color=accent_color, style={"opacity": "0.25"})
                    ], className="text-end")
                ], width=3, className="d-flex align-items-center justify-content-end")
            ])
        ], style={"padding": "20px"})
    ], className=f"h-100 kpi-card glass-card-enhanced {kpi_class}{hover_class}",
       style={
           "borderRadius": "20px",
       })

    if tooltip and tooltip_id:
        return html.Div([
            card,
            dbc.Tooltip(tooltip, target=tooltip_id, placement="bottom")
        ])

    return card


def status_badge(estado: str, texto: str = None) -> html.Span:
    """
    Badge de estado con colores iOS

    Args:
        estado: Tipo de estado (quiebre, bajo_punto_pedido, normal, etc.)
        texto: Texto a mostrar (si None, usa el estado)
    """
    config = {
        "quiebre": {"bg": "#FF3B30", "texto": "Quiebre"},
        "bajo_punto_pedido": {"bg": "#FF9500", "texto": "Bajo PP"},
        "bajo_seguridad": {"bg": "#FFCC00", "texto": "Bajo SS", "text_color": "#1F1F21"},
        "sobrestock_critico": {"bg": "#5856D6", "texto": "Sobrestock Critico"},
        "sobrestock": {"bg": "#007AFF", "texto": "Sobrestock"},
        "bajo_consumo": {"bg": "#8E8E93", "texto": "Bajo Consumo"},
        "normal": {"bg": "#4CD964", "texto": "Normal"},
    }

    cfg = config.get(estado, {"bg": "#8E8E93", "texto": estado})
    display_text = texto if texto else cfg["texto"]
    text_color = cfg.get("text_color", "#FFFFFF")

    return html.Span(
        display_text,
        style={
            "backgroundColor": cfg["bg"],
            "color": text_color,
            "padding": "6px 12px",
            "borderRadius": "8px",
            "fontSize": "11px",
            "fontWeight": "600",
            "display": "inline-block"
        }
    )


def summary_card(
    titulo: str,
    valor: int,
    color: str = "primary",
    icono: str = "fa-box"
) -> dbc.Card:
    """
    Tarjeta de resumen compacta con glassmorphism - OPTIMIZADA

    Args:
        titulo: Titulo de la tarjeta
        valor: Valor numerico
        color: Color del borde (primary, success, warning, danger, info, purple)
        icono: Clase FontAwesome
    """
    color_map = {
        "primary": "#007AFF",
        "success": "#4CD964",
        "warning": "#FF9500",
        "danger": "#FF3B30",
        "info": "#5AC8FA",
        "purple": "#5856D6",
    }
    accent = color_map.get(color, "#007AFF")

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                lucide_icon(icono, size="sm", color=accent, className="me-2"),
                html.Span(titulo,
                         style={"fontSize": "0.75rem", "color": "#8E8E93", "fontWeight": "500"})
            ], className="mb-1", style={"display": "flex", "alignItems": "center"}),
            html.H3(formato_numero(valor),
                   style={"fontWeight": "700", "color": accent, "margin": "0", "fontSize": "1.4rem"})
        ], className="py-2 px-3", style={"paddingBottom": "8px !important"})
    ], className="summary-card",
       style={
           "background": "rgba(255, 255, 255, 0.85)",
           "backdropFilter": "blur(20px)",
           "WebkitBackdropFilter": "blur(20px)",
           "border": "1px solid rgba(255,255,255,0.25)",
           "borderLeft": f"4px solid {accent}",
           "borderRadius": "12px",
           "boxShadow": "0 4px 16px rgba(31, 31, 33, 0.08)",
           "minHeight": "100px"
       })


def empty_state(
    mensaje: str = "No hay datos disponibles",
    icono: str = "fa-inbox",
    subtitulo: str = "Sube un archivo Excel o CSV para comenzar",
    variant: str = "default",
    action_button: Optional[dict] = None,
    size: str = "medium"
) -> html.Div:
    """
    Estado vacio con estilo glassmorphism iOS

    Args:
        mensaje: Mensaje principal
        icono: Clase FontAwesome del icono
        subtitulo: Texto secundario descriptivo
        variant: Tipo de empty state ('default', 'table', 'chart', 'filter', 'search', 'upload', 'error')
        action_button: Dict opcional con 'text', 'id', 'color' para agregar un boton de accion
        size: Tamano del empty state ('small', 'medium', 'large')
    """
    # Configuracion de variantes predefinidas
    variant_configs = {
        "table": {
            "icono": "table-2",
            "mensaje": "No hay registros para mostrar",
            "subtitulo": "Los datos apareceran aqui cuando apliques filtros o cargues informacion"
        },
        "chart": {
            "icono": "bar-chart-3",
            "mensaje": "No hay datos para graficar",
            "subtitulo": "Selecciona un rango de fechas o ajusta los filtros para visualizar datos"
        },
        "filter": {
            "icono": "filter",
            "mensaje": "No hay resultados con estos filtros",
            "subtitulo": "Intenta ajustar los criterios de busqueda o limpiar los filtros"
        },
        "search": {
            "icono": "search",
            "mensaje": "No se encontraron coincidencias",
            "subtitulo": "Intenta con otros terminos de busqueda"
        },
        "upload": {
            "icono": "cloud-upload",
            "mensaje": "No hay archivo cargado",
            "subtitulo": "Arrastra un archivo o usa el boton para cargar datos"
        },
        "error": {
            "icono": "alert-circle",
            "mensaje": "Error al cargar datos",
            "subtitulo": "Verifica la conexion o intenta nuevamente"
        },
        "forecast": {
            "icono": "sparkles",
            "mensaje": "No hay forecast generado",
            "subtitulo": "Selecciona un material y genera el forecast para ver las predicciones"
        },
        "material": {
            "icono": "package-open",
            "mensaje": "Selecciona un material",
            "subtitulo": "Haz clic en una fila para ver los detalles completos"
        }
    }

    if variant in variant_configs and mensaje == "No hay datos disponibles":
        config = variant_configs[variant]
        mensaje = config["mensaje"]
        icono = config["icono"]
        if subtitulo == "Sube un archivo Excel o CSV para comenzar":
            subtitulo = config["subtitulo"]

    # Configuracion de tamanos
    size_configs = {
        "small": {"icon_size": "fa-2x", "padding": "py-3", "title_class": "h6"},
        "medium": {"icon_size": "fa-4x", "padding": "py-5", "title_class": "h5"},
        "large": {"icon_size": "fa-5x", "padding": "py-5", "title_class": "h4"}
    }

    size_config = size_configs.get(size, size_configs["medium"])

    # Mapear tamanos de icono
    icon_size_map = {"fa-2x": "2x", "fa-4x": "4x", "fa-5x": "5x"}
    icon_pixel_size = icon_size_map.get(size_config['icon_size'], "4x")

    elements = [
        lucide_icon(
            icono,
            size=icon_pixel_size,
            className="mb-3",
            style={"color": "#BDBEC2", "opacity": "0.6"}
        ),
        html.Div(
            mensaje,
            className=f"{size_config['title_class']} mb-2",
            style={"fontWeight": "600", "color": "#8E8E93"}
        ),
        html.P(
            subtitulo,
            style={"maxWidth": "400px", "margin": "0 auto", "color": "#BDBEC2", "fontSize": "0.9rem"}
        )
    ]

    if action_button:
        button_text = action_button.get("text", "Accion")
        button_id = action_button.get("id")
        button_color = action_button.get("color", "primary")
        button_icon = action_button.get("icon")

        button_content = []
        if button_icon:
            button_content.append(lucide_icon(button_icon, size="sm", className="me-2"))
        button_content.append(button_text)

        elements.append(
            dbc.Button(
                button_content,
                id=button_id,
                color=button_color,
                className="mt-3",
                size="sm"
            )
        )

    return html.Div(
        elements,
        className=f"text-center {size_config['padding']}",
        style={
            "borderRadius": "16px",
            "backgroundColor": "rgba(255, 255, 255, 0.5)",
            "backdropFilter": "blur(10px)",
            "WebkitBackdropFilter": "blur(10px)",
            "border": "2px dashed rgba(199, 199, 204, 0.5)"
        }
    )


def empty_table(mensaje: str = None, subtitulo: str = None, show_action: bool = False, action_id: str = None) -> html.Div:
    """Empty state especifico para tablas"""
    action_button = None
    if show_action and action_id:
        action_button = {"text": "Cargar datos", "id": action_id, "color": "primary", "icon": "upload"}
    return empty_state(mensaje=mensaje, subtitulo=subtitulo, variant="table", action_button=action_button, size="medium")


def empty_chart(mensaje: str = None, subtitulo: str = None, chart_type: str = "bar") -> html.Div:
    """Empty state especifico para graficos"""
    chart_icons = {"bar": "bar-chart-3", "line": "line-chart", "pie": "pie-chart", "scatter": "area-chart", "area": "area-chart"}
    icono = chart_icons.get(chart_type, "bar-chart-3")
    return empty_state(mensaje=mensaje, subtitulo=subtitulo, variant="chart", icono=icono, size="medium")


def empty_search(search_term: str = None, subtitulo: str = None) -> html.Div:
    """Empty state especifico para busquedas"""
    mensaje = f"No se encontraron resultados para '{search_term}'" if search_term else "No se encontraron coincidencias"
    return empty_state(mensaje=mensaje, subtitulo=subtitulo, variant="search", size="medium")


def empty_filter(mensaje: str = None, subtitulo: str = None, show_clear_button: bool = True, clear_button_id: str = "btn-limpiar-filtros") -> html.Div:
    """Empty state especifico para filtros sin resultados"""
    action_button = None
    if show_clear_button:
        action_button = {"text": "Limpiar filtros", "id": clear_button_id, "color": "secondary", "icon": "x"}
    return empty_state(mensaje=mensaje, subtitulo=subtitulo, variant="filter", action_button=action_button, size="medium")


def error_state(
    mensaje: str = "Error al cargar los datos",
    icono: str = "alert-circle",
    subtitulo: str = "Ocurrio un problema al procesar la informacion. Intenta nuevamente.",
    show_retry_button: bool = False,
    retry_button_id: str = "btn-reintentar",
    size: str = "medium",
    error_details: Optional[str] = None
) -> html.Div:
    """Estado de error con estilo iOS"""
    size_configs = {
        "small": {"icon_size": "2x", "padding": "py-3", "title_class": "h6"},
        "medium": {"icon_size": "4x", "padding": "py-5", "title_class": "h5"},
        "large": {"icon_size": "5x", "padding": "py-5", "title_class": "h4"}
    }
    size_config = size_configs.get(size, size_configs["medium"])

    elements = [
        lucide_icon(icono, size=size_config['icon_size'], className="mb-3", color="#FF3B30"),
        html.Div(mensaje, className=f"{size_config['title_class']} mb-2", style={"fontWeight": "600", "color": "#FF3B30"}),
        html.P(subtitulo, style={"maxWidth": "500px", "margin": "0 auto", "color": "#8E8E93"})
    ]

    if error_details:
        elements.append(
            html.Details([
                html.Summary("Ver detalles tecnicos", style={"cursor": "pointer", "fontSize": "0.85rem", "marginTop": "1rem", "color": "#8E8E93"}),
                html.Pre(error_details, className="mt-2", style={
                    "fontSize": "0.75rem", "backgroundColor": "rgba(255,59,48,0.1)", "padding": "10px",
                    "borderRadius": "8px", "color": "#1F1F21", "maxWidth": "600px", "margin": "0 auto", "textAlign": "left", "overflow": "auto"
                })
            ], style={"textAlign": "center"})
        )

    if show_retry_button:
        elements.append(dbc.Button([lucide_icon("refresh-cw", size="sm", className="me-2"), "Reintentar"], id=retry_button_id, color="danger", outline=True, className="mt-3", size="sm"))

    return html.Div(
        elements,
        className=f"text-center {size_config['padding']}",
        style={
            "borderRadius": "16px",
            "backgroundColor": "rgba(255, 59, 48, 0.08)",
            "border": "1px solid rgba(255, 59, 48, 0.3)",
            "borderLeftWidth": "4px",
            "borderLeftColor": "#FF3B30"
        }
    )


def warning_state(
    mensaje: str = "Atencion requerida",
    icono: str = "alert-triangle",
    subtitulo: str = "Hay algunos aspectos que requieren tu atencion",
    show_action_button: bool = False,
    action_button_id: str = "btn-accion-warning",
    action_button_text: str = "Revisar",
    size: str = "medium",
    warning_items: Optional[List[str]] = None
) -> html.Div:
    """Estado de advertencia con estilo iOS"""
    size_configs = {
        "small": {"icon_size": "2x", "padding": "py-3", "title_class": "h6"},
        "medium": {"icon_size": "4x", "padding": "py-5", "title_class": "h5"},
        "large": {"icon_size": "5x", "padding": "py-5", "title_class": "h4"}
    }
    size_config = size_configs.get(size, size_configs["medium"])

    elements = [
        lucide_icon(icono, size=size_config['icon_size'], className="mb-3", color="#FF9500"),
        html.Div(mensaje, className=f"{size_config['title_class']} mb-2", style={"fontWeight": "600", "color": "#FF9500"}),
        html.P(subtitulo, style={"maxWidth": "500px", "margin": "0 auto", "color": "#8E8E93"})
    ]

    if warning_items:
        items_list = html.Ul(
            [html.Li([lucide_icon("circle", size="xs", className="me-2", color="#FF9500"), item], className="text-start") for item in warning_items],
            className="mt-3",
            style={"maxWidth": "500px", "margin": "1rem auto 0", "color": "#1F1F21", "fontSize": "0.9rem", "listStyle": "none", "padding": "0"}
        )
        elements.append(items_list)

    if show_action_button:
        elements.append(dbc.Button([lucide_icon("check", size="sm", className="me-2"), action_button_text], id=action_button_id, color="warning", outline=True, className="mt-3", size="sm"))

    return html.Div(
        elements,
        className=f"text-center {size_config['padding']}",
        style={
            "borderRadius": "16px",
            "backgroundColor": "rgba(255, 149, 0, 0.08)",
            "border": "1px solid rgba(255, 149, 0, 0.3)",
            "borderLeftWidth": "4px",
            "borderLeftColor": "#FF9500"
        }
    )


def loading_spinner() -> dbc.Spinner:
    """Spinner de carga iOS style"""
    return dbc.Spinner(color="primary", type="border", size="lg")


def alert_message(mensaje: str, tipo: str = "info") -> dbc.Alert:
    """
    Mensaje de alerta con estilo iOS

    Args:
        mensaje: Texto del mensaje
        tipo: Tipo de alerta (success, warning, danger, info)
    """
    iconos = {
        "success": "check-circle",
        "warning": "alert-triangle",
        "danger": "x-circle",
        "info": "info"
    }

    colors = {
        "success": "#4CD964",
        "warning": "#FF9500",
        "danger": "#FF3B30",
        "info": "#007AFF"
    }

    return dbc.Alert([
        lucide_icon(iconos.get(tipo, 'info'), size="md", className="me-2"),
        mensaje
    ], color=tipo, dismissable=True, duration=6000,
       style={
           "borderRadius": "12px",
           "border": "none",
           "borderLeft": f"4px solid {colors.get(tipo, '#007AFF')}",
           "backgroundColor": "rgba(255,255,255,0.95)",
           "backdropFilter": "blur(10px)",
           "boxShadow": "0 8px 32px rgba(31,31,33,0.15)"
       })


def filter_row(
    filters: List[Dict[str, Any]],
    extra_content: Optional[List[Any]] = None
) -> html.Div:
    """
    Fila de filtros reutilizable con estilo glassmorphism iOS

    Args:
        filters: Lista de diccionarios con configuracion de cada filtro
        extra_content: Lista de componentes adicionales a incluir en la fila
    """
    cols = []

    for filter_config in filters:
        label = filter_config.get("label", "")
        filter_id = filter_config.get("id", "")
        filter_type = filter_config.get("type", "dropdown")
        md = filter_config.get("md", 3)
        placeholder = filter_config.get("placeholder", "")
        options = filter_config.get("options", [])
        value = filter_config.get("value")
        multi = filter_config.get("multi", False)
        clearable = filter_config.get("clearable", True)
        debounce = filter_config.get("debounce", False)
        className = filter_config.get("className", "")
        style = filter_config.get("style", {})

        component = None

        if filter_type == "dropdown":
            component = dcc.Dropdown(
                id=filter_id, options=options, value=value, placeholder=placeholder,
                multi=multi, clearable=clearable, className=f"dash-dropdown {className}".strip(), style=style
            )
        elif filter_type == "input":
            component = dbc.Input(
                id=filter_id, type="text", placeholder=placeholder, value=value,
                debounce=debounce, className=f"form-control {className}".strip(), style=style
            )
        elif filter_type == "number":
            number_props = filter_config.get("number_props", {})
            component = dbc.Input(
                id=filter_id, type="number", placeholder=placeholder, value=value,
                min=number_props.get("min"), max=number_props.get("max"), step=number_props.get("step", 1),
                debounce=debounce, className=f"form-control {className}".strip(), style=style
            )
        elif filter_type == "textarea":
            component = dcc.Textarea(
                id=filter_id, placeholder=placeholder, value=value, className=className,
                style={"width": "100%", "backgroundColor": "rgba(255,255,255,0.85)", "color": "#1F1F21",
                       "border": "1px solid #C7C7CC", "borderRadius": "12px", "padding": "12px", **style}
            )
        elif filter_type == "button":
            button_props = filter_config.get("button_props", {})
            icon = button_props.get("icon")
            text = button_props.get("text", "")
            color = button_props.get("color", "primary")
            button_content = []
            if icon:
                button_content.append(lucide_icon(icon, size="sm", className="me-2"))
            if text:
                button_content.append(text)
            component = dbc.Button(
                button_content, id=filter_id, color=color, className=f"w-100 {className}".strip(), style=style,
                **{k: v for k, v in button_props.items() if k not in ['icon', 'text', 'color']}
            )
        elif filter_type == "slider":
            slider_props = filter_config.get("slider_props", {})
            component = dcc.Slider(
                id=filter_id, min=slider_props.get("min", 0), max=slider_props.get("max", 100),
                step=slider_props.get("step"), value=value or slider_props.get("min", 0),
                marks=slider_props.get("marks", {}),
                tooltip=slider_props.get("tooltip", {"placement": "bottom", "always_visible": True}),
                className=className
            )
        elif filter_type == "upload":
            upload_props = filter_config.get("upload_props", {})
            children = upload_props.get("children", html.Div([
                lucide_icon("file-up", size="3x", className="mb-2", style={"color": "#BDBEC2"}),
                html.P("Arrastre un archivo", className="mb-1", style={"color": "#8E8E93"}),
                html.Small("o haga clic para seleccionar", style={"color": "#BDBEC2"})
            ], className="text-center py-4"))
            component = dcc.Upload(
                id=filter_id, children=children, className=className,
                style={
                    "width": "100%", "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "16px",
                    "borderColor": "#C7C7CC", "backgroundColor": "rgba(255,255,255,0.5)", "cursor": "pointer", **style
                },
                **{k: v for k, v in upload_props.items() if k != 'children'}
            )
        elif filter_type == "custom":
            component = filter_config.get("component")
        else:
            component = dcc.Dropdown(id=filter_id, placeholder=placeholder, className=f"dash-dropdown {className}".strip())

        col_content = []
        if label:
            col_content.append(html.Label(label, style={
                "fontSize": "11px", "fontWeight": "600", "color": "#8E8E93",
                "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "4px", "display": "block"
            }))
        if component is not None:
            col_content.append(component)

        cols.append(dbc.Col(col_content, md=md))

    if extra_content:
        for item in extra_content:
            cols.append(item)

    return html.Div([
        dbc.Row(cols, className="g-2")
    ], className="filters-section mb-3")
