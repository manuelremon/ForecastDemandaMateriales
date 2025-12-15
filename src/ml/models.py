"""
Modelos de Machine Learning para MRP
=====================================
Funciones de entrenamiento y evaluación de modelos
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import pickle
import os

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    IsolationForest
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    silhouette_score
)


def train_demand_model(
    df: pd.DataFrame,
    target_col: str = 'cantidad',
    modelo_tipo: str = 'random_forest',
    test_size: float = 0.2
) -> Tuple[Any, Dict[str, float], StandardScaler]:
    """
    Entrena un modelo de predicción de demanda

    Args:
        df: DataFrame con features y target
        target_col: Columna objetivo
        modelo_tipo: Tipo de modelo a entrenar
        test_size: Proporción de datos para test

    Returns:
        Tuple (modelo_entrenado, métricas, scaler)
    """
    # Separar features y target
    feature_cols = [c for c in df.columns if c != target_col and df[c].dtype in ['int64', 'float64']]
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)

    # Split temporal (sin shuffle para series de tiempo)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Seleccionar modelo
    modelos = {
        'random_forest': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1)
    }

    modelo = modelos.get(modelo_tipo, modelos['random_forest'])
    modelo.fit(X_train_scaled, y_train)

    # Evaluar
    y_pred = modelo.predict(X_test_scaled)

    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
    }

    return modelo, metrics, scaler


def train_classification_model(
    df: pd.DataFrame,
    target_col: str = 'clase',
    modelo_tipo: str = 'random_forest'
) -> Tuple[Any, Dict[str, Any], StandardScaler, LabelEncoder]:
    """
    Entrena un modelo de clasificación de materiales

    Args:
        df: DataFrame con features y clase
        target_col: Columna de clase
        modelo_tipo: Tipo de modelo

    Returns:
        Tuple (modelo, métricas, scaler, label_encoder)
    """
    # Separar features y target
    feature_cols = [c for c in df.columns if c != target_col and df[c].dtype in ['int64', 'float64']]
    X = df[feature_cols].fillna(0)
    y = df[target_col]

    # Codificar labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    modelo.fit(X_train_scaled, y_train)

    # Evaluar
    y_pred = modelo.predict(X_test_scaled)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            output_dict=True
        )
    }

    return modelo, metrics, scaler, le


def train_clustering_model(
    df: pd.DataFrame,
    n_clusters: int = 5,
    features: List[str] = None
) -> Tuple[KMeans, Dict[str, float], StandardScaler]:
    """
    Entrena modelo de clustering para segmentación de materiales

    Args:
        df: DataFrame con datos de materiales
        n_clusters: Número de clusters
        features: Lista de features a usar

    Returns:
        Tuple (modelo_kmeans, métricas, scaler)
    """
    if features is None:
        features = ['stock_actual', 'consumo_mensual', 'costo_unitario', 'lead_time']

    # Filtrar features disponibles
    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0)

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar KMeans
    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = modelo.fit_predict(X_scaled)

    # Métricas
    metrics = {
        'silhouette_score': silhouette_score(X_scaled, labels),
        'inertia': modelo.inertia_,
        'n_clusters': n_clusters
    }

    return modelo, metrics, scaler


def train_anomaly_detector(
    df: pd.DataFrame,
    contamination: float = 0.1,
    features: List[str] = None
) -> Tuple[IsolationForest, StandardScaler]:
    """
    Entrena detector de anomalías con Isolation Forest

    Args:
        df: DataFrame con datos
        contamination: Proporción esperada de anomalías
        features: Features a usar

    Returns:
        Tuple (modelo, scaler)
    """
    if features is None:
        features = ['stock_actual', 'consumo_mensual', 'variacion_demanda']

    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_scaled)

    return modelo, scaler


def cross_validate_model(
    modelo: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = 'neg_mean_absolute_error'
) -> Dict[str, float]:
    """
    Realiza validación cruzada de un modelo

    Args:
        modelo: Modelo a evaluar
        X: Features
        y: Target
        cv: Número de folds
        scoring: Métrica de scoring

    Returns:
        Dict con resultados de CV
    """
    # Para series temporales, usar TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=cv)

    scores = cross_val_score(modelo, X, y, cv=tscv, scoring=scoring)

    return {
        'mean_score': -scores.mean() if 'neg' in scoring else scores.mean(),
        'std_score': scores.std(),
        'scores': scores.tolist()
    }


def save_model(
    modelo: Any,
    scaler: StandardScaler,
    path: str,
    metadata: Dict = None
) -> str:
    """
    Guarda modelo y scaler en disco

    Args:
        modelo: Modelo entrenado
        scaler: Scaler utilizado
        path: Ruta donde guardar
        metadata: Metadatos adicionales

    Returns:
        Ruta del archivo guardado
    """
    data = {
        'modelo': modelo,
        'scaler': scaler,
        'metadata': metadata or {}
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)

    return path


def load_model(path: str) -> Tuple[Any, StandardScaler, Dict]:
    """
    Carga modelo y scaler desde disco

    Args:
        path: Ruta del archivo

    Returns:
        Tuple (modelo, scaler, metadata)
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data['modelo'], data['scaler'], data.get('metadata', {})


def get_feature_importance(
    modelo: Any,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Obtiene importancia de features del modelo

    Args:
        modelo: Modelo entrenado (debe tener feature_importances_)
        feature_names: Nombres de features

    Returns:
        DataFrame con importancia ordenada
    """
    if not hasattr(modelo, 'feature_importances_'):
        return pd.DataFrame()

    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': modelo.feature_importances_
    })

    return importance.sort_values('importance', ascending=False).reset_index(drop=True)


def optimize_hyperparameters(
    modelo_tipo: str,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Dict = None
) -> Dict[str, Any]:
    """
    Optimización básica de hiperparámetros

    Args:
        modelo_tipo: Tipo de modelo
        X: Features
        y: Target
        param_grid: Grid de parámetros a probar

    Returns:
        Dict con mejores parámetros y score
    """
    from sklearn.model_selection import GridSearchCV

    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }

    if modelo_tipo == 'random_forest':
        modelo = RandomForestRegressor(random_state=42, n_jobs=-1)
    else:
        modelo = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1, 0.2]
        }

    tscv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(
        modelo,
        param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    grid_search.fit(X_scaled, y)

    return {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
