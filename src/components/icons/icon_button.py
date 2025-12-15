"""
Icon Button Component
=====================
Componente de boton con icono integrado
"""
import dash_bootstrap_components as dbc
from dash import html
from typing import Optional, Union, List, Any
from .lucide_icon import lucide_icon
from .icon_registry import IconName


def icon_button(
    text: str = "",
    icon: Union[IconName, str] = None,
    id: Optional[str] = None,
    color: str = "primary",
    size: str = "md",
    outline: bool = False,
    icon_position: str = "left",
    icon_size: str = "sm",
    className: str = "",
    disabled: bool = False,
    loading: bool = False,
    href: Optional[str] = None,
    **kwargs
) -> dbc.Button:
    """
    Boton con icono integrado

    Args:
        text: Texto del boton
        icon: Icono a mostrar (IconName enum o string)
        id: ID del boton
        color: Color Bootstrap (primary, secondary, success, danger, etc.)
        size: Tamano del boton (sm, md, lg)
        outline: Si es True, usa estilo outline
        icon_position: Posicion del icono ('left', 'right', 'only')
        icon_size: Tamano del icono
        className: Clases CSS adicionales
        disabled: Si el boton esta deshabilitado
        loading: Si esta en estado de carga
        href: URL para navegacion

    Returns:
        dbc.Button: Componente boton de Dash Bootstrap
    """
    # Determinar icono y animacion
    display_icon = "loader-2" if loading else icon
    animation = "spin" if loading else None

    # Crear elemento de icono
    if display_icon:
        margin_class = ""
        if icon_position == "left" and text:
            margin_class = "me-2"
        elif icon_position == "right" and text:
            margin_class = "ms-2"

        icon_element = lucide_icon(
            display_icon,
            size=icon_size,
            animation=animation,
            className=margin_class
        )
    else:
        icon_element = None

    # Construir contenido
    content: List[Any] = []
    if icon_position == "only" and icon_element:
        content = [icon_element]
    elif icon_position == "right":
        if text:
            content.append(text)
        if icon_element:
            content.append(icon_element)
    else:  # left (default)
        if icon_element:
            content.append(icon_element)
        if text:
            content.append(text)

    # Crear boton
    return dbc.Button(
        content,
        id=id,
        color=color,
        size=size,
        outline=outline,
        className=className,
        disabled=disabled or loading,
        href=href,
        **kwargs
    )


def action_button(
    action: str,
    id: Optional[str] = None,
    size: str = "sm",
    **kwargs
) -> dbc.Button:
    """
    Boton preconfigurado para acciones comunes

    Args:
        action: Tipo de accion ('export', 'upload', 'refresh', 'save', etc.)
        id: ID del boton
        size: Tamano del boton

    Returns:
        dbc.Button: Boton preconfigurado
    """
    configs = {
        "export": {
            "text": "Exportar",
            "icon": IconName.DOWNLOAD,
            "color": "outline-light"
        },
        "upload": {
            "text": "Subir",
            "icon": IconName.UPLOAD,
            "color": "primary"
        },
        "refresh": {
            "text": "Actualizar",
            "icon": IconName.REFRESH,
            "color": "secondary"
        },
        "save": {
            "text": "Guardar",
            "icon": IconName.CHECK,
            "color": "success"
        },
        "delete": {
            "text": "Eliminar",
            "icon": IconName.CLOSE,
            "color": "danger"
        },
        "approve": {
            "text": "Aprobar",
            "icon": IconName.CHECK,
            "color": "success"
        },
        "calculate": {
            "text": "Calcular",
            "icon": IconName.CALCULATOR,
            "color": "primary"
        },
        "play": {
            "text": "Ejecutar",
            "icon": IconName.PLAY,
            "color": "success"
        },
        "pdf": {
            "text": "PDF",
            "icon": IconName.FILE_PDF,
            "color": "danger"
        },
        "excel": {
            "text": "Excel",
            "icon": IconName.FILE_EXCEL,
            "color": "success"
        },
        "send": {
            "text": "Enviar",
            "icon": IconName.SEND,
            "color": "primary"
        },
        "filter": {
            "text": "Filtrar",
            "icon": IconName.FILTER,
            "color": "secondary"
        },
        "search": {
            "text": "Buscar",
            "icon": IconName.SEARCH,
            "color": "primary"
        },
        "database": {
            "text": "SAP",
            "icon": IconName.DATABASE,
            "color": "info"
        },
    }

    # Obtener configuracion o usar defaults
    config = configs.get(action, {
        "text": action.title(),
        "icon": IconName.BOX,
        "color": "secondary"
    })

    # Merge con kwargs
    config.update(kwargs)

    return icon_button(id=id, size=size, **config)
