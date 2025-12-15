"""
Callbacks del Tablero de Forecasting
=====================================

Módulo principal que importa todos los callbacks modulares.
Ver src/callbacks/demanda/ para implementación detallada.

Estructura:
- filters.py: Filtros de centro, almacén, materiales
- search.py: Búsqueda de materiales
- forecast.py: Generación de forecast con ForecastService
- export.py: Exportación PDF/CSV/Excel
- advanced.py: Gestión de modelos y backtesting
"""

# Importar todos los callbacks desde módulos
from src.callbacks.demanda import (
    # Filtros
    cargar_centros_desde_excel_o_db,
    cargar_almacenes_desde_excel_o_db,
    actualizar_lista_materiales,
    sincronizar_material_seleccionado,
    # Búsqueda
    buscar_material_por_codigo,
    # Forecast
    generar_forecast,
    # Exportación
    exportar_forecast_csv,
    exportar_forecast_pdf,
    # Avanzado (modelos y backtesting)
    toggle_modal_modelos,
    toggle_modal_backtest,
    guardar_modelo_actual,
    cargar_lista_modelos,
    ejecutar_backtesting
)

# Re-exportar para compatibilidad
__all__ = [
    # Filtros
    'cargar_centros_desde_excel_o_db',
    'cargar_almacenes_desde_excel_o_db',
    'actualizar_lista_materiales',
    'sincronizar_material_seleccionado',
    # Búsqueda
    'buscar_material_por_codigo',
    # Forecast
    'generar_forecast',
    # Exportación
    'exportar_forecast_csv',
    'exportar_forecast_pdf',
    # Avanzado
    'toggle_modal_modelos',
    'toggle_modal_backtest',
    'guardar_modelo_actual',
    'cargar_lista_modelos',
    'ejecutar_backtesting'
]
