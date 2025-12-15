"""
Predictores de Machine Learning para MRP
=========================================
Modelos de demanda, clasificación y optimización de stock
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
    Predictor de demanda basado en series temporales y ML

    Usa Random Forest y Gradient Boosting para predecir
    demanda futura basada en patrones históricos.
    """

    def __init__(self, modelo: str = "random_forest"):
        """
        Inicializa el predictor

        Args:
            modelo: Tipo de modelo ("random_forest", "gradient_boosting", "linear")
        """
        self.modelo_tipo = modelo
        self.modelo = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.metrics = {}

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
        Entrena el modelo con datos históricos

        Args:
            df: DataFrame con ['fecha', 'codigo', 'cantidad']
            columna_objetivo: Columna a predecir
            test_size: Proporción para test

        Returns:
            Dict con métricas de evaluación
        """
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
        Genera predicciones para períodos futuros

        Args:
            df_historico: Datos históricos recientes
            periodos: Número de períodos a predecir

        Returns:
            DataFrame con predicciones
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado. Llame a entrenar() primero.")

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
        """Retorna la importancia de cada feature"""
        if not self.is_trained or not hasattr(self.modelo, 'feature_importances_'):
            return pd.DataFrame()

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.modelo.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance


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
