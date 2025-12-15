"""
Modulo de Machine Learning para MRP Analytics
==============================================
Modelos predictivos para demanda, stock y optimizacion
"""
from .predictor import DemandPredictor, StockOptimizer

__all__ = [
    'DemandPredictor',
    'StockOptimizer',
]
