"""
Capa de Servicios para Forecast MR

Orquesta el flujo completo de forecasting separando
la lógica de negocio de los callbacks de Dash.
"""

from .forecast_service import ForecastService, ForecastConfig, ForecastResult

__all__ = ['ForecastService', 'ForecastConfig', 'ForecastResult']
