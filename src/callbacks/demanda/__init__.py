"""
Callbacks Modulares para Demanda/Forecast Individual
=====================================================

Estructura dividida para mejor mantenimiento:
- filters.py: Callbacks de filtros (centros, almacenes, materiales)
- search.py: Búsqueda de materiales
- forecast.py: Generación de forecast con ForecastService
- export.py: Exportación PDF/CSV/Excel
- advanced.py: Gestión de modelos y backtesting
"""

from .filters import (
    cargar_centros_desde_excel_o_db,
    cargar_almacenes_desde_excel_o_db,
    actualizar_lista_materiales,
    sincronizar_material_seleccionado
)
from .search import buscar_material_por_codigo
from .forecast import generar_forecast
from .export import exportar_forecast_csv, exportar_forecast_pdf
from .advanced import (
    toggle_modal_modelos,
    toggle_modal_backtest,
    guardar_modelo_actual,
    cargar_lista_modelos,
    ejecutar_backtesting,
    set_current_predictor
)

__all__ = [
    'cargar_centros_desde_excel_o_db',
    'cargar_almacenes_desde_excel_o_db',
    'actualizar_lista_materiales',
    'sincronizar_material_seleccionado',
    'buscar_material_por_codigo',
    'generar_forecast',
    'exportar_forecast_csv',
    'exportar_forecast_pdf',
    'toggle_modal_modelos',
    'toggle_modal_backtest',
    'guardar_modelo_actual',
    'cargar_lista_modelos',
    'ejecutar_backtesting',
    'set_current_predictor'
]
