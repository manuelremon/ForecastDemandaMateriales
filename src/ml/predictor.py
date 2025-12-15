"""
Predictores de Machine Learning para MRP
=========================================
Modelos de demanda, clasificación y optimización de stock.

Soporta múltiples modelos de forecasting:
- Random Forest (sklearn)
- Gradient Boosting (sklearn)
- XGBoost (si está instalado)
- Prophet (si está instalado)
- ARIMA/SARIMAX (si está instalado)
"""
import pandas as pd
import numpy as np
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import threading

# Scikit-learn imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import joblib  # Para persistencia de modelos

from src.utils.logger import get_logger

# Importar sistema de estrategias
from src.ml.strategies import (
    obtener_estrategia,
    obtener_estrategias_disponibles,
    obtener_nombres_modelos,
    listar_estrategias
)

logger = get_logger(__name__)

# ============================================================================
# Cache de Modelos ML con Persistencia en Disco
# ============================================================================
_model_cache: Dict[str, 'DemandPredictor'] = {}
_cache_lock = threading.Lock()
_cache_max_size = 20  # Reducido para mejor gestión de memoria

# Directorio para persistir modelos en disco
_MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _compute_data_hash(df: pd.DataFrame, codigo: str = None) -> str:
    """Calcula hash único para los datos de entrenamiento"""
    if df is None or len(df) == 0:
        return "empty"
    # Hash basado en: código + longitud + suma de cantidad + fechas
    sample = f"{codigo or 'global'}_{len(df)}_{df['cantidad'].sum() if 'cantidad' in df.columns else 0}"
    if 'fecha' in df.columns:
        sample += f"_{df['fecha'].min()}_{df['fecha'].max()}"
    return hashlib.md5(sample.encode()).hexdigest()[:16]


def _get_model_path(cache_key: str) -> Path:
    """Obtiene la ruta del archivo de modelo en disco"""
    return _MODELS_DIR / f"{cache_key}.joblib"


def _load_from_disk(cache_key: str) -> Optional['DemandPredictor']:
    """Intenta cargar un modelo desde disco"""
    model_path = _get_model_path(cache_key)
    if model_path.exists():
        try:
            predictor = joblib.load(model_path)
            logger.debug(f"Modelo cargado desde disco: {cache_key}")
            return predictor
        except Exception as e:
            logger.warning(f"Error cargando modelo desde disco: {e}")
            # Eliminar archivo corrupto
            model_path.unlink(missing_ok=True)
    return None


def _save_to_disk(cache_key: str, predictor: 'DemandPredictor') -> bool:
    """Guarda un modelo en disco"""
    try:
        model_path = _get_model_path(cache_key)
        joblib.dump(predictor, model_path, compress=3)
        logger.debug(f"Modelo guardado en disco: {cache_key}")
        return True
    except Exception as e:
        logger.warning(f"Error guardando modelo en disco: {e}")
        return False


def get_cached_predictor(
    df: pd.DataFrame,
    codigo: str = None,
    modelo_tipo: str = "random_forest"
) -> Tuple['DemandPredictor', bool]:
    """
    Obtiene predictor desde cache (memoria o disco) o crea uno nuevo.

    RENDIMIENTO: Los modelos se persisten en disco para evitar
    reentrenamiento después de reiniciar el servidor.

    Args:
        df: DataFrame de entrenamiento
        codigo: Código de material
        modelo_tipo: Tipo de modelo

    Returns:
        Tuple (predictor, from_cache) donde from_cache indica si vino del cache
    """
    cache_key = f"{modelo_tipo}_{_compute_data_hash(df, codigo)}"

    # 1. Buscar en cache de memoria
    with _cache_lock:
        if cache_key in _model_cache:
            logger.debug(f"Cache memoria hit: {cache_key}")
            return _model_cache[cache_key], True

    # 2. Buscar en disco
    predictor = _load_from_disk(cache_key)
    if predictor is not None:
        # Guardar en cache de memoria
        with _cache_lock:
            if len(_model_cache) >= _cache_max_size:
                oldest_key = next(iter(_model_cache))
                del _model_cache[oldest_key]
            _model_cache[cache_key] = predictor
        return predictor, True

    # 3. Cache miss - crear y entrenar nuevo predictor
    predictor = DemandPredictor(modelo=modelo_tipo)

    # Si no hay datos, retornar predictor sin entrenar
    if df is None or len(df) == 0:
        return predictor, False

    # Entrenar modelo
    try:
        predictor.entrenar(df)

        # Guardar en disco primero (persistencia)
        _save_to_disk(cache_key, predictor)

        # Guardar en cache de memoria
        with _cache_lock:
            if len(_model_cache) >= _cache_max_size:
                oldest_key = next(iter(_model_cache))
                del _model_cache[oldest_key]
                logger.debug(f"Cache lleno, eliminando: {oldest_key}")

            _model_cache[cache_key] = predictor
            logger.debug(f"Modelo guardado en cache memoria: {cache_key}")

    except Exception as e:
        logger.error(f"Error entrenando predictor: {e}")

    return predictor, False


def clear_model_cache(clear_disk: bool = False):
    """
    Limpia el cache de modelos

    Args:
        clear_disk: Si True, también elimina modelos guardados en disco
    """
    global _model_cache
    with _cache_lock:
        _model_cache.clear()

    if clear_disk:
        # Eliminar archivos de modelos del disco
        try:
            for model_file in _MODELS_DIR.glob("*.joblib"):
                model_file.unlink()
            logger.debug("Cache de modelos en disco limpiado")
        except Exception as e:
            logger.warning(f"Error limpiando cache de disco: {e}")

    logger.debug("Cache de modelos en memoria limpiado")


class DemandPredictor:
    """
    Predictor de demanda basado en series temporales y ML.

    Soporta múltiples modelos de forecasting mediante el patrón Strategy:
    - random_forest: Random Forest (sklearn) - default
    - gradient_boosting: Gradient Boosting (sklearn)
    - linear: Ridge Regression (sklearn)
    - xgboost: XGBoost (requiere instalación)
    - prophet: Facebook Prophet (requiere instalación)
    - arima: ARIMA/SARIMAX (requiere instalación)

    Uso:
        predictor = DemandPredictor(modelo="xgboost")
        metricas = predictor.entrenar(df)
        predicciones = predictor.predecir(df, periodos=30)
    """

    def __init__(self, modelo: str = "random_forest", **kwargs):
        """
        Inicializa el predictor.

        Args:
            modelo: Tipo de modelo. Opciones disponibles:
                - "random_forest" (default)
                - "gradient_boosting"
                - "linear"
                - "xgboost" (si está instalado)
                - "prophet" (si está instalado)
                - "arima" (si está instalado)
            **kwargs: Parámetros adicionales para el modelo
        """
        self.modelo_tipo = modelo
        self._kwargs = kwargs

        # Intentar usar nueva arquitectura de estrategias
        self._estrategia = obtener_estrategia(modelo, **kwargs)

        if self._estrategia is not None:
            # Usar estrategia nueva
            self._usar_legacy = False
            logger.debug(f"Usando estrategia: {self._estrategia.nombre_modelo}")
        else:
            # Fallback a implementación legacy
            self._usar_legacy = True
            logger.debug(f"Usando implementación legacy para: {modelo}")

        # Atributos públicos para compatibilidad
        self.modelo = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.metrics = {}

    @staticmethod
    def modelos_disponibles() -> Dict[str, str]:
        """
        Retorna diccionario de modelos disponibles.

        Returns:
            Dict {identificador: nombre_legible}
        """
        return obtener_nombres_modelos()

    @staticmethod
    def listar_modelos() -> List[str]:
        """
        Lista los identificadores de modelos disponibles.

        Returns:
            Lista de strings con identificadores de modelos
        """
        return obtener_estrategias_disponibles()

    def _preparar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara features temporales para el modelo

        Args:
            df: DataFrame con columnas ['fecha', 'codigo', 'cantidad']

        Returns:
            DataFrame con features extraídas
        """
        df = df.copy()

        # Asegurar que fecha es datetime
        df['fecha'] = pd.to_datetime(df['fecha'])

        # Features temporales
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['dia_mes'] = df['fecha'].dt.day
        df['mes'] = df['fecha'].dt.month
        df['trimestre'] = df['fecha'].dt.quarter
        df['semana_ano'] = df['fecha'].dt.isocalendar().week
        df['es_fin_mes'] = (df['fecha'].dt.is_month_end).astype(int)
        df['es_inicio_mes'] = (df['fecha'].dt.is_month_start).astype(int)

        # Codificación cíclica para mes (captura estacionalidad)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

        # Codificación cíclica para día de la semana
        df['dia_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['dia_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)

        return df

    def _crear_lag_features(
        self,
        df: pd.DataFrame,
        columna: str = 'cantidad',
        lags: List[int] = [1, 7, 14, 30]
    ) -> pd.DataFrame:
        """
        Crea features de lag (valores anteriores)

        Args:
            df: DataFrame ordenado por fecha
            columna: Columna para crear lags
            lags: Lista de períodos de lag

        Returns:
            DataFrame con lag features
        """
        df = df.copy()

        for lag in lags:
            df[f'{columna}_lag_{lag}'] = df[columna].shift(lag)

        # Media móvil
        df[f'{columna}_ma_7'] = df[columna].rolling(window=7, min_periods=1).mean()
        df[f'{columna}_ma_30'] = df[columna].rolling(window=30, min_periods=1).mean()

        # Desviación estándar móvil (volatilidad)
        df[f'{columna}_std_7'] = df[columna].rolling(window=7, min_periods=1).std()

        # Tendencia (diferencia con período anterior)
        df[f'{columna}_diff'] = df[columna].diff()

        return df

    def entrenar(
        self,
        df: pd.DataFrame,
        columna_objetivo: str = 'cantidad',
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Entrena el modelo con datos históricos.

        Args:
            df: DataFrame con ['fecha', 'codigo', 'cantidad']
            columna_objetivo: Columna a predecir
            test_size: Proporción para test

        Returns:
            Dict con métricas de evaluación {'mae', 'rmse', 'r2', 'mape'}
        """
        # Usar estrategia nueva si está disponible
        if not self._usar_legacy and self._estrategia is not None:
            self.metrics = self._estrategia.entrenar(df, columna_objetivo, test_size)
            self.is_trained = self._estrategia.is_trained
            self.feature_names = self._estrategia.feature_names
            self.modelo = self._estrategia.modelo
            self.scaler = self._estrategia.scaler
            # Copiar atributos para compatibilidad
            self._media_historica = self._estrategia._media_historica
            self._std_historica = self._estrategia._std_historica
            self._usar_modelo_simple = self._estrategia._usar_modelo_simple
            return self.metrics

        # Legacy: implementación original
        # Guardar estadísticas básicas para fallback
        self._media_historica = df[columna_objetivo].mean()
        self._std_historica = df[columna_objetivo].std()
        self._ultimo_valor = df[columna_objetivo].iloc[-1] if len(df) > 0 else 0

        # Preparar datos
        df_prep = self._preparar_features(df)

        # Para datasets pequeños, usar solo lags cortos
        n_samples = len(df_prep)
        if n_samples < 35:
            # Dataset pequeño: usar solo lag_1 y ma_7
            lags = [1]
            if n_samples >= 7:
                lags.append(7)
        else:
            lags = [1, 7, 14, 30]

        df_prep = self._crear_lag_features(df_prep, columna_objetivo, lags=lags)

        # Eliminar NaN de lag features
        df_prep = df_prep.dropna()

        # Verificar si hay suficientes muestras para entrenar
        min_samples_required = 5  # Mínimo absoluto para entrenar

        if len(df_prep) < min_samples_required:
            # Fallback: usar modelo simple basado en promedio
            logger.info(f"Solo {len(df_prep)} muestras, usando modelo de promedio móvil")
            self._usar_modelo_simple = True
            self.is_trained = True
            self.feature_names = []

            # Métricas aproximadas basadas en variabilidad histórica
            cv = self._std_historica / self._media_historica if self._media_historica > 0 else 0.5
            self.metrics = {
                'mae': self._std_historica * 0.8 if self._std_historica > 0 else self._media_historica * 0.2,
                'rmse': self._std_historica if self._std_historica > 0 else self._media_historica * 0.25,
                'r2': max(0, 1 - cv),  # Aproximación basada en variabilidad
                'mape': min(cv * 100, 50)  # Cap at 50%
            }
            return self.metrics

        self._usar_modelo_simple = False

        # Definir features base
        base_features = [
            'dia_semana', 'dia_mes', 'mes', 'trimestre',
            'mes_sin', 'mes_cos', 'dia_sin', 'dia_cos',
            'es_fin_mes', 'es_inicio_mes'
        ]

        # Agregar lag features disponibles
        lag_features = []
        for lag in lags:
            lag_features.append(f'{columna_objetivo}_lag_{lag}')

        # Agregar rolling features si existen
        rolling_features = [
            f'{columna_objetivo}_ma_7',
            f'{columna_objetivo}_std_7',
            f'{columna_objetivo}_diff'
        ]
        if n_samples >= 30:
            rolling_features.append(f'{columna_objetivo}_ma_30')

        self.feature_names = base_features + lag_features + rolling_features

        # Filtrar features que existen en el DataFrame
        self.feature_names = [f for f in self.feature_names if f in df_prep.columns]

        X = df_prep[self.feature_names]
        y = df_prep[columna_objetivo]

        # Ajustar test_size para datasets pequeños
        n_available = len(X)
        min_train = 3
        min_test = 1

        if n_available < min_train + min_test:
            # Usar todo para entrenar, sin test split
            X_train, y_train = X, y
            X_test, y_test = X.tail(1), y.tail(1)
        else:
            # Ajustar test_size para asegurar suficientes muestras
            max_test_size = (n_available - min_train) / n_available
            actual_test_size = min(test_size, max_test_size)
            actual_test_size = max(actual_test_size, 1 / n_available)  # Al menos 1 muestra para test

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=actual_test_size, shuffle=False
            )

        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Seleccionar y entrenar modelo (ajustar parámetros para datasets pequeños)
        if self.modelo_tipo == "random_forest":
            self.modelo = RandomForestRegressor(
                n_estimators=min(100, max(10, n_available * 2)),
                max_depth=min(10, max(3, n_available // 3)),
                min_samples_split=max(2, min(5, n_available // 4)),
                random_state=42,
                n_jobs=-1
            )
        elif self.modelo_tipo == "gradient_boosting":
            self.modelo = GradientBoostingRegressor(
                n_estimators=min(100, max(10, n_available * 2)),
                max_depth=min(5, max(2, n_available // 5)),
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.modelo = Ridge(alpha=1.0)

        self.modelo.fit(X_train_scaled, y_train)

        # Evaluar
        y_pred = self.modelo.predict(X_test_scaled)

        # Calcular métricas con protección contra división por cero
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred) if len(y_test) > 1 else 0.0,
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
        }

        self.is_trained = True
        return self.metrics

    def predecir(
        self,
        df_historico: pd.DataFrame,
        periodos: int = 30
    ) -> pd.DataFrame:
        """
        Genera predicciones para períodos futuros.

        Args:
            df_historico: Datos históricos recientes
            periodos: Número de períodos a predecir

        Returns:
            DataFrame con ['fecha', 'prediccion', 'limite_inferior', 'limite_superior']
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Llame a entrenar() primero.")

        # Usar estrategia nueva si está disponible
        if not self._usar_legacy and self._estrategia is not None:
            return self._estrategia.predecir(df_historico, periodos)

        predicciones = []
        df_historico['fecha'] = pd.to_datetime(df_historico['fecha'])
        ultima_fecha = df_historico['fecha'].max()

        # Si estamos usando el modelo simple (datos insuficientes)
        if getattr(self, '_usar_modelo_simple', False):
            logger.debug("Usando modelo simple para predicciones")

            # Calcular tendencia simple si hay suficientes datos
            if len(df_historico) >= 3:
                valores_recientes = df_historico['cantidad'].tail(7).values
                tendencia = (valores_recientes[-1] - valores_recientes[0]) / len(valores_recientes) if len(valores_recientes) > 1 else 0
            else:
                tendencia = 0

            for i in range(1, periodos + 1):
                fecha_pred = ultima_fecha + timedelta(days=i)

                # Predicción basada en media + tendencia ligera
                pred = self._media_historica + (tendencia * i * 0.1)  # Tendencia amortiguada
                pred = max(0, pred)

                # Intervalos basados en desviación estándar
                std = self._std_historica if self._std_historica > 0 else self._media_historica * 0.2

                predicciones.append({
                    'fecha': fecha_pred,
                    'prediccion': pred,
                    'limite_inferior': max(0, pred - 1.5 * std),
                    'limite_superior': pred + 1.5 * std
                })

            return pd.DataFrame(predicciones)

        # Modelo ML completo
        # Preparar datos históricos
        df = self._preparar_features(df_historico)

        # Determinar qué lags usar basado en cantidad de datos
        n_samples = len(df)
        if n_samples < 35:
            lags = [1]
            if n_samples >= 7:
                lags.append(7)
        else:
            lags = [1, 7, 14, 30]

        df = self._crear_lag_features(df, lags=lags)

        # Último registro para features de lag
        ultimo = df.iloc[-1].copy()

        for i in range(1, periodos + 1):
            fecha_pred = ultima_fecha + timedelta(days=i)

            # Crear features para la predicción
            features = {
                'dia_semana': fecha_pred.weekday(),
                'dia_mes': fecha_pred.day,
                'mes': fecha_pred.month,
                'trimestre': (fecha_pred.month - 1) // 3 + 1,
                'mes_sin': np.sin(2 * np.pi * fecha_pred.month / 12),
                'mes_cos': np.cos(2 * np.pi * fecha_pred.month / 12),
                'dia_sin': np.sin(2 * np.pi * fecha_pred.weekday() / 7),
                'dia_cos': np.cos(2 * np.pi * fecha_pred.weekday() / 7),
                'es_fin_mes': int(fecha_pred.day >= 28),
                'es_inicio_mes': int(fecha_pred.day <= 3),
            }

            # Agregar lag features del último registro conocido
            for col in self.feature_names:
                if col not in features and col in ultimo.index:
                    features[col] = ultimo[col]

            # Predecir
            X_pred = pd.DataFrame([features])[self.feature_names]
            X_pred_scaled = self.scaler.transform(X_pred)
            pred = self.modelo.predict(X_pred_scaled)[0]

            predicciones.append({
                'fecha': fecha_pred,
                'prediccion': max(0, pred),  # No permitir negativos
                'limite_inferior': max(0, pred * 0.8),
                'limite_superior': pred * 1.2
            })

        return pd.DataFrame(predicciones)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retorna la importancia de cada feature.

        Returns:
            DataFrame con ['feature', 'importance'] ordenado por importancia
        """
        # Usar estrategia nueva si está disponible
        if not self._usar_legacy and self._estrategia is not None:
            return self._estrategia.get_feature_importance()

        # Legacy
        if not self.is_trained or not hasattr(self.modelo, 'feature_importances_'):
            return pd.DataFrame()

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.modelo.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance

    @property
    def nombre_modelo(self) -> str:
        """Retorna el nombre legible del modelo"""
        if not self._usar_legacy and self._estrategia is not None:
            return self._estrategia.nombre_modelo

        nombres = {
            'random_forest': 'Random Forest',
            'gradient_boosting': 'Gradient Boosting',
            'linear': 'Ridge Regression',
            'xgboost': 'XGBoost',
            'prophet': 'Prophet',
            'arima': 'ARIMA'
        }
        return nombres.get(self.modelo_tipo, self.modelo_tipo)

    def entrenar_con_cv(
        self,
        df: pd.DataFrame,
        columna_objetivo: str = 'cantidad',
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Entrena el modelo con validación cruzada temporal.

        Usa TimeSeriesSplit para respetar la naturaleza temporal de los datos
        y obtener métricas más realistas.

        Args:
            df: DataFrame con ['fecha', 'cantidad']
            columna_objetivo: Columna a predecir
            n_splits: Número de splits para CV

        Returns:
            Dict con métricas de CV y estadísticas
        """
        from sklearn.model_selection import TimeSeriesSplit

        logger.info(f"Entrenando con validación cruzada ({n_splits} splits)")

        # Preparar datos
        df_prep = self._preparar_features(df)
        df_prep = self._crear_lag_features(df_prep, columna_objetivo)
        df_prep = df_prep.dropna()

        if len(df_prep) < n_splits + 5:
            logger.warning(f"Datos insuficientes para {n_splits} splits, usando entrenamiento simple")
            return self.entrenar(df, columna_objetivo)

        # Features
        base_features = [
            'dia_semana', 'dia_mes', 'mes', 'trimestre',
            'mes_sin', 'mes_cos', 'dia_sin', 'dia_cos',
            'es_fin_mes', 'es_inicio_mes'
        ]
        lag_features = [c for c in df_prep.columns if 'lag_' in c or 'ma_' in c or 'std_' in c or 'diff' in c]
        self.feature_names = [f for f in base_features + lag_features if f in df_prep.columns]

        X = df_prep[self.feature_names].values
        y = df_prep[columna_objetivo].values

        # TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = {'mae': [], 'rmse': [], 'r2': [], 'mape': []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Escalar
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Entrenar modelo temporal
            if self.modelo_tipo == "random_forest":
                modelo_temp = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            elif self.modelo_tipo == "gradient_boosting":
                modelo_temp = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            else:
                modelo_temp = Ridge(alpha=1.0)

            modelo_temp.fit(X_train_scaled, y_train)
            y_pred = modelo_temp.predict(X_val_scaled)

            # Métricas del fold
            scores['mae'].append(mean_absolute_error(y_val, y_pred))
            scores['rmse'].append(np.sqrt(mean_squared_error(y_val, y_pred)))
            scores['r2'].append(r2_score(y_val, y_pred) if len(y_val) > 1 else 0)
            mask = y_val != 0
            if mask.any():
                scores['mape'].append(np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask])) * 100)

        # Entrenar modelo final con todos los datos
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if self.modelo_tipo == "random_forest":
            self.modelo = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        elif self.modelo_tipo == "gradient_boosting":
            self.modelo = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        else:
            self.modelo = Ridge(alpha=1.0)

        self.modelo.fit(X_scaled, y)
        self.is_trained = True

        # Guardar estadísticas
        self._media_historica = df[columna_objetivo].mean()
        self._std_historica = df[columna_objetivo].std()
        self._usar_modelo_simple = False

        # Métricas agregadas
        self.metrics = {
            'mae': np.mean(scores['mae']),
            'mae_std': np.std(scores['mae']),
            'rmse': np.mean(scores['rmse']),
            'rmse_std': np.std(scores['rmse']),
            'r2': np.mean(scores['r2']),
            'r2_std': np.std(scores['r2']),
            'mape': np.mean(scores['mape']) if scores['mape'] else 0,
            'cv_scores': scores,
            'n_splits': n_splits
        }

        logger.info(f"CV completada: MAE={self.metrics['mae']:.2f} (±{self.metrics['mae_std']:.2f})")
        return self.metrics

    def _crear_features_temporales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features temporales completos para el modelo.

        Incluye:
        - Features cíclicos (sin/cos para día, mes, semana)
        - Lags y rolling statistics
        - Tendencias
        - Features de calendario

        Args:
            df: DataFrame con ['fecha', 'cantidad']

        Returns:
            DataFrame con todos los features
        """
        df = df.copy()
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.sort_values('fecha').reset_index(drop=True)

        # Features temporales básicos
        df['dia_semana'] = df['fecha'].dt.dayofweek
        df['dia_mes'] = df['fecha'].dt.day
        df['mes'] = df['fecha'].dt.month
        df['trimestre'] = df['fecha'].dt.quarter
        df['semana_ano'] = df['fecha'].dt.isocalendar().week.astype(int)
        df['dia_ano'] = df['fecha'].dt.dayofyear

        # Features de calendario
        df['es_fin_mes'] = df['fecha'].dt.is_month_end.astype(int)
        df['es_inicio_mes'] = df['fecha'].dt.is_month_start.astype(int)
        df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
        df['dias_hasta_fin_mes'] = (df['fecha'] + pd.offsets.MonthEnd(0) - df['fecha']).dt.days

        # Codificación cíclica (captura patrones periódicos)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        df['dia_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['dia_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
        df['semana_sin'] = np.sin(2 * np.pi * df['semana_ano'] / 52)
        df['semana_cos'] = np.cos(2 * np.pi * df['semana_ano'] / 52)

        # Lags
        if 'cantidad' in df.columns:
            for lag in [1, 7, 14, 21, 28, 30]:
                if len(df) > lag:
                    df[f'cantidad_lag_{lag}'] = df['cantidad'].shift(lag)

            # Rolling statistics
            for window in [7, 14, 30]:
                if len(df) >= window:
                    df[f'cantidad_ma_{window}'] = df['cantidad'].rolling(window=window, min_periods=1).mean()
                    df[f'cantidad_std_{window}'] = df['cantidad'].rolling(window=window, min_periods=1).std()

            # EWMA (Exponential Weighted Moving Average)
            for span in [7, 14, 30]:
                if len(df) >= span:
                    df[f'cantidad_ewma_{span}'] = df['cantidad'].ewm(span=span, min_periods=1).mean()

            # Tendencia (diferencia)
            df['cantidad_diff'] = df['cantidad'].diff()
            df['cantidad_diff_7'] = df['cantidad'].diff(7)

            # Aceleración (cambio de tendencia)
            df['cantidad_accel'] = df['cantidad_diff'].diff()

        return df

    def calcular_baseline_metrics(
        self,
        df: pd.DataFrame,
        columna_objetivo: str = 'cantidad'
    ) -> Dict[str, float]:
        """
        Calcula métricas de modelos baseline para comparación.

        Incluye:
        - Naive (último valor)
        - Seasonal naive (valor hace 7 días)
        - Moving average (promedio últimos 30 días)

        Args:
            df: DataFrame con datos
            columna_objetivo: Columna objetivo

        Returns:
            Dict con métricas de cada baseline
        """
        df = df.copy()
        y = df[columna_objetivo].values

        baselines = {}

        # Naive: predicción = último valor
        y_naive = np.roll(y, 1)
        y_naive[0] = y[0]
        baselines['naive_mae'] = mean_absolute_error(y[1:], y_naive[1:])

        # Seasonal naive: predicción = valor hace 7 días
        if len(y) > 7:
            y_seasonal = np.roll(y, 7)
            baselines['seasonal_mae'] = mean_absolute_error(y[7:], y_seasonal[7:])
        else:
            baselines['seasonal_mae'] = baselines['naive_mae']

        # Moving average: predicción = promedio últimos 7 días
        if len(y) > 7:
            y_ma = pd.Series(y).rolling(7, min_periods=1).mean().shift(1).values
            baselines['ma_mae'] = mean_absolute_error(y[1:], y_ma[1:])
        else:
            baselines['ma_mae'] = baselines['naive_mae']

        # Mejor baseline
        baselines['mejor_baseline'] = min(['naive', 'seasonal', 'ma'],
                                          key=lambda x: baselines.get(f'{x}_mae', float('inf')))

        return baselines

    @property
    def _feature_names(self) -> List[str]:
        """Alias para compatibilidad"""
        return self.feature_names

    @property
    def _modelo(self):
        """Alias para compatibilidad"""
        return self.modelo

    @_modelo.setter
    def _modelo(self, value):
        self.modelo = value

    @property
    def _scaler(self):
        """Alias para compatibilidad"""
        return self.scaler

    @_scaler.setter
    def _scaler(self, value):
        self.scaler = value

    @property
    def _entrenado(self) -> bool:
        """Alias para compatibilidad"""
        return self.is_trained

    @_entrenado.setter
    def _entrenado(self, value: bool):
        self.is_trained = value


class StockOptimizer:
    """
    Optimizador de niveles de stock usando ML

    Calcula niveles óptimos de stock de seguridad,
    punto de reorden y EOQ basado en patrones de demanda.
    """

    def __init__(self):
        self.modelo_variabilidad = None
        self.scaler = StandardScaler()

    def calcular_stock_seguridad_ml(
        self,
        df_consumo: pd.DataFrame,
        nivel_servicio: float = 0.95,
        lead_time_dias: int = 14
    ) -> pd.DataFrame:
        """
        Calcula stock de seguridad óptimo por material

        Args:
            df_consumo: DataFrame con ['codigo', 'fecha', 'cantidad']
            nivel_servicio: Nivel de servicio deseado (0-1)
            lead_time_dias: Tiempo de entrega en días

        Returns:
            DataFrame con stock de seguridad por material
        """
        resultados = []

        # Factor Z para nivel de servicio
        z = stats.norm.ppf(nivel_servicio)

        for codigo, grupo in df_consumo.groupby('codigo'):
            grupo = grupo.sort_values('fecha')

            # Calcular estadísticas de demanda
            demanda_diaria = grupo['cantidad'].mean()
            std_demanda = grupo['cantidad'].std()

            # Coeficiente de variación
            cv = std_demanda / demanda_diaria if demanda_diaria > 0 else 0

            # Stock de seguridad = Z * σ_demanda * √(lead_time)
            ss = z * std_demanda * np.sqrt(lead_time_dias)

            # Punto de reorden = demanda_durante_lead_time + stock_seguridad
            rop = (demanda_diaria * lead_time_dias) + ss

            # EOQ simplificado (suponiendo costo orden = 50, costo mantener = 20%)
            demanda_anual = demanda_diaria * 365
            costo_orden = 50
            costo_mantenimiento = 0.2  # 20% del valor

            if demanda_anual > 0:
                eoq = np.sqrt((2 * demanda_anual * costo_orden) / costo_mantenimiento)
            else:
                eoq = 0

            resultados.append({
                'codigo': codigo,
                'demanda_diaria_promedio': round(demanda_diaria, 2),
                'desviacion_demanda': round(std_demanda, 2),
                'coeficiente_variacion': round(cv, 3),
                'stock_seguridad_optimo': round(ss, 0),
                'punto_reorden_optimo': round(rop, 0),
                'eoq_sugerido': round(eoq, 0),
                'nivel_servicio': nivel_servicio,
                'lead_time_dias': lead_time_dias
            })

        return pd.DataFrame(resultados)

    def detectar_anomalias(
        self,
        df_consumo: pd.DataFrame,
        umbral_zscore: float = 3.0
    ) -> pd.DataFrame:
        """
        Detecta consumos anómalos usando Z-score

        Args:
            df_consumo: DataFrame con ['codigo', 'fecha', 'cantidad']
            umbral_zscore: Umbral para considerar anomalía

        Returns:
            DataFrame con anomalías detectadas
        """
        anomalias = []

        for codigo, grupo in df_consumo.groupby('codigo'):
            grupo = grupo.copy()

            # Calcular Z-score
            mean = grupo['cantidad'].mean()
            std = grupo['cantidad'].std()

            if std > 0:
                grupo['zscore'] = (grupo['cantidad'] - mean) / std
                grupo['es_anomalia'] = abs(grupo['zscore']) > umbral_zscore

                # Filtrar anomalías
                anom = grupo[grupo['es_anomalia']].copy()
                if len(anom) > 0:
                    anom['codigo'] = codigo
                    anomalias.append(anom)

        if anomalias:
            return pd.concat(anomalias, ignore_index=True)
        return pd.DataFrame()

    def clasificar_abc_xyz(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clasificación ABC (por valor) y XYZ (por variabilidad)

        Args:
            df: DataFrame con ['codigo', 'valor_total', 'coeficiente_variacion']

        Returns:
            DataFrame con clasificación ABC-XYZ
        """
        df = df.copy()

        # Clasificación ABC por valor (Pareto)
        df = df.sort_values('valor_total', ascending=False)
        df['valor_acumulado'] = df['valor_total'].cumsum()
        df['porcentaje_acumulado'] = df['valor_acumulado'] / df['valor_total'].sum() * 100

        df['clase_abc'] = pd.cut(
            df['porcentaje_acumulado'],
            bins=[0, 80, 95, 100],
            labels=['A', 'B', 'C']
        )

        # Clasificación XYZ por variabilidad (coeficiente de variación)
        df['clase_xyz'] = pd.cut(
            df['coeficiente_variacion'],
            bins=[-np.inf, 0.5, 1.0, np.inf],
            labels=['X', 'Y', 'Z']
        )

        # Clase combinada
        df['clase_abc_xyz'] = df['clase_abc'].astype(str) + df['clase_xyz'].astype(str)

        return df

    def sugerir_politica_stock(self, clase_abc_xyz: str) -> Dict[str, Any]:
        """
        Sugiere política de gestión según clasificación ABC-XYZ

        Args:
            clase_abc_xyz: Clasificación combinada (ej: "AX", "BY", "CZ")

        Returns:
            Dict con recomendaciones de política
        """
        politicas = {
            "AX": {
                "descripcion": "Alto valor, demanda estable",
                "metodo_reposicion": "Punto de reorden fijo",
                "nivel_servicio": 0.99,
                "revision": "Continua",
                "stock_seguridad": "Bajo",
                "prioridad": "Crítica"
            },
            "AY": {
                "descripcion": "Alto valor, demanda variable",
                "metodo_reposicion": "Punto de reorden con revisión periódica",
                "nivel_servicio": 0.97,
                "revision": "Semanal",
                "stock_seguridad": "Medio",
                "prioridad": "Alta"
            },
            "AZ": {
                "descripcion": "Alto valor, demanda errática",
                "metodo_reposicion": "Bajo pedido o consignación",
                "nivel_servicio": 0.95,
                "revision": "Continua con forecast",
                "stock_seguridad": "Alto",
                "prioridad": "Alta"
            },
            "BX": {
                "descripcion": "Valor medio, demanda estable",
                "metodo_reposicion": "Punto de reorden",
                "nivel_servicio": 0.95,
                "revision": "Semanal",
                "stock_seguridad": "Bajo",
                "prioridad": "Media"
            },
            "BY": {
                "descripcion": "Valor medio, demanda variable",
                "metodo_reposicion": "Revisión periódica",
                "nivel_servicio": 0.93,
                "revision": "Quincenal",
                "stock_seguridad": "Medio",
                "prioridad": "Media"
            },
            "BZ": {
                "descripcion": "Valor medio, demanda errática",
                "metodo_reposicion": "Bajo pedido",
                "nivel_servicio": 0.90,
                "revision": "Mensual",
                "stock_seguridad": "Alto",
                "prioridad": "Media-Baja"
            },
            "CX": {
                "descripcion": "Bajo valor, demanda estable",
                "metodo_reposicion": "Lote económico (EOQ)",
                "nivel_servicio": 0.90,
                "revision": "Mensual",
                "stock_seguridad": "Mínimo",
                "prioridad": "Baja"
            },
            "CY": {
                "descripcion": "Bajo valor, demanda variable",
                "metodo_reposicion": "Revisión periódica",
                "nivel_servicio": 0.85,
                "revision": "Mensual",
                "stock_seguridad": "Medio",
                "prioridad": "Baja"
            },
            "CZ": {
                "descripcion": "Bajo valor, demanda errática",
                "metodo_reposicion": "Eliminar o bajo pedido",
                "nivel_servicio": 0.80,
                "revision": "Trimestral",
                "stock_seguridad": "Evaluar eliminación",
                "prioridad": "Muy Baja"
            }
        }

        return politicas.get(clase_abc_xyz, {
            "descripcion": "Sin clasificación",
            "metodo_reposicion": "Por definir",
            "prioridad": "Por evaluar"
        })
