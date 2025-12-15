"""
AG Grid Column Definition Factory Functions

This module provides reusable factory functions to create standardized AG Grid
column definitions with consistent formatters, styling, and behavior across
the MRP Analytics application.

Usage:
    from src.utils.grid_helpers import col_numeric, col_currency, col_text

    column_defs = [
        col_text("codigo", "Código", 120, selectable=True),
        col_numeric("stock_actual", "Stock Actual", 110),
        col_currency("costo_total", "Costo Total", 130),
    ]
"""

from typing import Optional, Dict, Any, Literal


# ============================================================================
# Core Column Factory Functions
# ============================================================================

def col_text(
    field: str,
    header: str,
    width: int = 120,
    filter: bool = True,
    selectable: bool = False,
    header_checkbox: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a basic text column definition.

    Args:
        field: The field name from the data
        header: Display name for the column header
        width: Column width in pixels
        filter: Enable column filtering
        selectable: Add checkbox selection to cells
        header_checkbox: Add checkbox selection to header (only if selectable=True)
        **kwargs: Additional AG Grid column properties

    Returns:
        Column definition dictionary

    Example:
        >>> col_text("codigo", "Código", 120)
        >>> col_text("codigo", "Código", 120, selectable=True, header_checkbox=True)
    """
    col_def = {
        "field": field,
        "headerName": header,
        "width": width,
        "filter": filter,
    }

    if selectable:
        col_def["checkboxSelection"] = True
        if header_checkbox:
            col_def["headerCheckboxSelection"] = True

    col_def.update(kwargs)
    return col_def


def col_numeric(
    field: str,
    header: str,
    width: int = 110,
    decimals: int = 0,
    highlighted: bool = False,
    highlight_color: str = "#3b82f6",
    **kwargs
) -> Dict[str, Any]:
    """
    Create a numeric column with standardized formatting.

    Args:
        field: The field name from the data
        header: Display name for the column header
        width: Column width in pixels
        decimals: Number of decimal places (0 for integers)
        highlighted: Apply bold font and color styling
        highlight_color: Color to use if highlighted=True
        **kwargs: Additional AG Grid column properties

    Returns:
        Column definition dictionary

    Example:
        >>> col_numeric("stock_actual", "Stock Actual", 110)
        >>> col_numeric("prediccion", "Predicción", 120, highlighted=True)
        >>> col_numeric("tasa_consumo", "Tasa Consumo", 130, decimals=1)
    """
    format_str = f",.{decimals}f"

    col_def = {
        "field": field,
        "headerName": header,
        "width": width,
        "type": "numericColumn",
        "valueFormatter": {"function": f"d3.format('{format_str}')(params.value)"}
    }

    if highlighted:
        col_def["cellStyle"] = {
            "fontWeight": "600",
            "color": highlight_color
        }

    col_def.update(kwargs)
    return col_def


def col_currency(
    field: str,
    header: str,
    width: int = 130,
    decimals: int = 2,
    symbol: str = "$",
    **kwargs
) -> Dict[str, Any]:
    """
    Create a currency column with standardized formatting.

    Args:
        field: The field name from the data
        header: Display name for the column header
        width: Column width in pixels
        decimals: Number of decimal places
        symbol: Currency symbol (default: $)
        **kwargs: Additional AG Grid column properties

    Returns:
        Column definition dictionary

    Example:
        >>> col_currency("costo_total", "Costo Total", 130)
        >>> col_currency("precio", "Precio", 120, decimals=0)
    """
    format_str = f"{symbol},.{decimals}f"

    col_def = {
        "field": field,
        "headerName": header,
        "width": width,
        "type": "numericColumn",
        "valueFormatter": {"function": f"d3.format('{format_str}')(params.value)"}
    }

    col_def.update(kwargs)
    return col_def


def col_percentage(
    field: str,
    header: str,
    width: int = 100,
    decimals: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a percentage column with standardized formatting.

    Args:
        field: The field name from the data
        header: Display name for the column header
        width: Column width in pixels
        decimals: Number of decimal places
        **kwargs: Additional AG Grid column properties

    Returns:
        Column definition dictionary

    Example:
        >>> col_percentage("tasa_servicio", "Tasa de Servicio", 120)
        >>> col_percentage("variacion", "Variación", 100, decimals=2)
    """
    format_str = f".{decimals}%"

    col_def = {
        "field": field,
        "headerName": header,
        "width": width,
        "type": "numericColumn",
        "valueFormatter": {"function": f"d3.format('{format_str}')(params.value)"}
    }

    col_def.update(kwargs)
    return col_def


def col_date(
    field: str,
    header: str,
    width: int = 120,
    sort: Optional[Literal["asc", "desc"]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a date column definition.

    Args:
        field: The field name from the data
        header: Display name for the column header
        width: Column width in pixels
        sort: Initial sort direction ('asc' or 'desc')
        **kwargs: Additional AG Grid column properties

    Returns:
        Column definition dictionary

    Example:
        >>> col_date("fecha", "Fecha", 120, sort="asc")
        >>> col_date("fecha_entrega", "Fecha Entrega", 140)
    """
    col_def = {
        "field": field,
        "headerName": header,
        "width": width,
    }

    if sort:
        col_def["sort"] = sort

    col_def.update(kwargs)
    return col_def


def col_with_suffix(
    field: str,
    header: str,
    suffix: str,
    width: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a column that appends a suffix to the value.

    Args:
        field: The field name from the data
        header: Display name for the column header
        suffix: Text to append to each value (e.g., ' dias', ' unidades')
        width: Column width in pixels
        **kwargs: Additional AG Grid column properties

    Returns:
        Column definition dictionary

    Example:
        >>> col_with_suffix("lead_time", "Lead Time", " dias", 100)
        >>> col_with_suffix("cantidad", "Cantidad", " unidades", 120)
    """
    col_def = {
        "field": field,
        "headerName": header,
        "width": width,
        "valueFormatter": {"function": f"params.value + '{suffix}'"}
    }

    col_def.update(kwargs)
    return col_def


def col_markdown_badge(
    field: str,
    header: str,
    width: int = 140,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a column that renders markdown content (useful for badges).

    Args:
        field: The field name from the data (should contain markdown)
        header: Display name for the column header
        width: Column width in pixels
        **kwargs: Additional AG Grid column properties

    Returns:
        Column definition dictionary

    Example:
        >>> col_markdown_badge("estado_badge", "Estado", 140)
    """
    col_def = {
        "field": field,
        "headerName": header,
        "width": width,
        "cellRenderer": "markdown"
    }

    col_def.update(kwargs)
    return col_def


# ============================================================================
# Specialized Styled Column Functions
# ============================================================================

def col_priority(
    field: str = "prioridad",
    header: str = "Prioridad",
    width: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a priority column with conditional color styling.

    Applies different background colors and text colors based on priority level:
    - Crítica: Red background
    - Alta: Orange/Yellow background
    - Normal/Other: Cyan background

    Args:
        field: The field name from the data
        header: Display name for the column header
        width: Column width in pixels
        **kwargs: Additional AG Grid column properties

    Returns:
        Column definition dictionary

    Example:
        >>> col_priority()
        >>> col_priority("nivel_prioridad", "Nivel", 120)
    """
    col_def = {
        "field": field,
        "headerName": header,
        "width": width,
        "cellStyle": {"function": """
            params.value === 'Crítica' ? {'backgroundColor': 'rgba(239, 68, 68, 0.3)', 'color': '#fca5a5', 'fontWeight': '600'} :
            params.value === 'Alta' ? {'backgroundColor': 'rgba(245, 158, 11, 0.3)', 'color': '#fcd34d', 'fontWeight': '600'} :
            {'backgroundColor': 'rgba(6, 182, 212, 0.3)', 'color': '#67e8f9', 'fontWeight': '600'}
        """}
    }

    col_def.update(kwargs)
    return col_def


def col_abc_class(
    field: str = "clase_abc",
    header: str = "ABC",
    width: int = 70,
    **kwargs
) -> Dict[str, Any]:
    """
    Create an ABC classification column with conditional background colors.

    Applies different background colors based on ABC classification:
    - A: Red background (high value items)
    - B: Orange background (medium value items)
    - C: Cyan background (low value items)

    Args:
        field: The field name from the data
        header: Display name for the column header
        width: Column width in pixels
        **kwargs: Additional AG Grid column properties

    Returns:
        Column definition dictionary

    Example:
        >>> col_abc_class()
        >>> col_abc_class("clasificacion", "Clase", 80)
    """
    col_def = {
        "field": field,
        "headerName": header,
        "width": width,
        "cellStyle": {"function": """
            params.value === 'A' ? {'backgroundColor': 'rgba(239, 68, 68, 0.2)'} :
            params.value === 'B' ? {'backgroundColor': 'rgba(245, 158, 11, 0.2)'} :
            {'backgroundColor': 'rgba(6, 182, 212, 0.2)'}
        """}
    }

    col_def.update(kwargs)
    return col_def


# ============================================================================
# Grid Options Helper Functions
# ============================================================================

def default_col_def(
    sortable: bool = True,
    resizable: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Create standardized default column definition options.

    Args:
        sortable: Allow sorting on all columns
        resizable: Allow resizing of all columns
        **kwargs: Additional default column properties

    Returns:
        Default column definition dictionary

    Example:
        >>> default_col_def()
        >>> default_col_def(filter=True, floatingFilter=True)
    """
    defaults = {
        "sortable": sortable,
        "resizable": resizable,
    }
    defaults.update(kwargs)
    return defaults


def grid_options(
    pagination: bool = True,
    page_size: int = 20,
    page_size_options: list = None,
    row_selection: Optional[Literal["single", "multiple"]] = None,
    animate_rows: bool = False,
    dom_layout: str = "normal",
    **kwargs
) -> Dict[str, Any]:
    """
    Create standardized grid options configuration.

    Args:
        pagination: Enable pagination
        page_size: Default page size
        page_size_options: List of page size options for selector
        row_selection: Enable row selection ('single' or 'multiple')
        animate_rows: Animate row changes
        dom_layout: Grid layout mode ('normal', 'autoHeight', 'print')
        **kwargs: Additional grid options

    Returns:
        Grid options dictionary

    Example:
        >>> grid_options()
        >>> grid_options(row_selection="multiple", animate_rows=True)
        >>> grid_options(page_size=50, page_size_options=[25, 50, 100])
    """
    if page_size_options is None:
        page_size_options = [10, 20, 50, 100]

    options = {
        "domLayout": dom_layout,
    }

    if pagination:
        options["pagination"] = True
        options["paginationPageSize"] = page_size
        options["paginationPageSizeSelector"] = page_size_options

    if row_selection:
        if row_selection == "single":
            options["rowSelection"] = {"mode": "singleRow"}
        else:  # multiple
            options["rowSelection"] = "multiple"

        if animate_rows:
            options["animateRows"] = True

    options.update(kwargs)
    return options


# ============================================================================
# Convenience Functions for Common Column Sets
# ============================================================================

def cols_stock_metrics(prefix: str = "") -> list:
    """
    Create a standard set of stock-related columns.

    Args:
        prefix: Optional prefix for field names

    Returns:
        List of column definitions for stock metrics

    Example:
        >>> cols_stock_metrics()
        [col_numeric("stock_actual", ...), col_numeric("stock_seguridad", ...)]
    """
    p = f"{prefix}_" if prefix else ""

    return [
        col_numeric(f"{p}stock_actual", "Stock Actual", 110),
        col_numeric(f"{p}stock_seguridad", "Stock Seguridad", 130),
        col_numeric(f"{p}stock_maximo", "Stock Máximo", 120),
        col_numeric(f"{p}punto_reorden", "Punto Reorden", 130),
    ]


def cols_forecast_base() -> list:
    """
    Create standard forecast result columns.

    Returns:
        List of column definitions for forecast results

    Example:
        >>> cols_forecast_base()
        [col_date("fecha", ...), col_numeric("prediccion", ...), ...]
    """
    return [
        col_date("fecha", "Fecha", 120, sort="asc"),
        col_numeric("prediccion", "Predicción", 120, highlighted=True),
        col_numeric("limite_inferior", "Lím. Inferior", 120),
        col_numeric("limite_superior", "Lím. Superior", 120),
    ]


def cols_material_base(selectable: bool = False) -> list:
    """
    Create standard material identification columns.

    Args:
        selectable: Add checkbox selection to the codigo column

    Returns:
        List of column definitions for material identification

    Example:
        >>> cols_material_base()
        >>> cols_material_base(selectable=True)
    """
    return [
        col_text("codigo", "Código", 120, selectable=selectable, header_checkbox=selectable),
        col_text("descripcion", "Descripción", 250, filter=True),
    ]


# ============================================================================
# Export all public functions
# ============================================================================

__all__ = [
    # Core column factories
    "col_text",
    "col_numeric",
    "col_currency",
    "col_percentage",
    "col_date",
    "col_with_suffix",
    "col_markdown_badge",
    # Styled columns
    "col_priority",
    "col_abc_class",
    # Grid configuration
    "default_col_def",
    "grid_options",
    # Convenience sets
    "cols_stock_metrics",
    "cols_forecast_base",
    "cols_material_base",
]
