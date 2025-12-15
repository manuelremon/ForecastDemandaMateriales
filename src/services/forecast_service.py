"""
Servicio Principal de Forecasting para Forecast MR

Orquesta el flujo completo de predicción de demanda:
1. Validación de datos
2. Preprocesamiento
3. Entrenamiento
4. Predicción
5. Evaluación

Separa la lógica de negocio de los callbacks de Dash.

Author: Manuel Remón
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import time

from src.data.validators import DataValidator, DataQualityReport, OutlierMethod
from src.ml.predictor import DemandPredictor
from src.ml.model_registry import ModelRegistry, ModelMetadata, crear_metadata
from src.ml.backtesting import Backtester, BacktestReport
from src.ml.tuning import AutoModelSelector, AutoSelectResult


@dataclass
class ForecastConfig:
    """Configuración para ejecutar forecast"""
    material: str
    horizonte: int = 30
    modelo_tipo: str = 'random_forest'
    nivel_confianza: float = 0.95
    validar_datos: bool = True
    detectar_outliers: bool = True
    metodo_outliers: str = 'iqr'
    remover_outliers: bool = False
    ejecutar_backtest: bool = False
    n_pasos_backtest: int = 3
    guardar_modelo: bool = False
    auto_seleccionar_modelo: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'material': self.material,
            'horizonte': self.horizonte,
            'modelo_tipo': self.modelo_tipo,
            'nivel_confianza': self.nivel_confianza,
            'validar_datos': self.validar_datos,
            'detectar_outliers': self.detectar_outliers,
            'remover_outliers': self.remover_outliers,
            'ejecutar_backtest': self.ejecutar_backtest,
            'auto_seleccionar_modelo': self.auto_seleccionar_modelo
        }


@dataclass
class ForecastResult:
    """Resultado completo de forecasting"""
    exito: bool
    predicciones: Optional[pd.DataFrame]
    metricas: Dict[str, float]
    validacion: Optional[DataQualityReport]
    backtest: Optional[BacktestReport]
    auto_select: Optional[AutoSelectResult]
    config: ForecastConfig
    tiempo_ejecucion: float
    modelo_id: Optional[str] = None
    mensaje: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'exito': self.exito,
            'mensaje': self.mensaje,
            'warnings': self.warnings,
            'predicciones': self.predicciones.to_dict('records') if self.predicciones is not None else None,
            'metricas': {k: round(v, 4) if isinstance(v, float) else v for k, v in self.metricas.items()},
            'validacion': self.validacion.to_dict() if self.validacion else None,
            'backtest': self.backtest.to_dict() if self.backtest else None,
            'auto_select': self.auto_select.to_dict() if self.auto_select else None,
            'config': self.config.to_dict(),
            'tiempo_ejecucion': round(self.tiempo_ejecucion, 2),
            'modelo_id': self.modelo_id
        }


class ForecastService:
    """
    Servicio principal para ejecutar forecasts.

    Orquesta todo el pipeline de predicción de demanda,
    incluyendo validación, entrenamiento y evaluación.

    Ejemplo de uso:
        service = ForecastService()

        config = ForecastConfig(
            material="1000015975",
            horizonte=30,
            modelo_tipo="random_forest",
            validar_datos=True,
            ejecutar_backtest=True
        )

        result = service.ejecutar_forecast(df_historico, config)

        if result.exito:
            print(f"MAE: {result.metricas['mae']}")
            print(result.predicciones)
        else:
            print(f"Error: {result.mensaje}")
    """

    def __init__(
        self,
        registry_path: str = "data/models",
        usar_cache: bool = True
    ):
        """
        Inicializa el servicio.

        Args:
            registry_path: Ruta para guardar modelos
            usar_cache: Si usar caché de modelos
        """
        self.validator = DataValidator()
        self.registry = ModelRegistry(registry_path)
        self.usar_cache = usar_cache
        self._predictors_cache = {}

    def ejecutar_forecast(
        self,
        df: pd.DataFrame,
        config: ForecastConfig,
        columna_fecha: str = 'fecha',
        columna_cantidad: str = 'cantidad'
    ) -> ForecastResult:
        """
        Ejecuta el pipeline completo de forecasting.

        Args:
            df: DataFrame con datos históricos
            config: Configuración del forecast
            columna_fecha: Nombre columna de fechas
            columna_cantidad: Nombre columna de cantidades

        Returns:
            ForecastResult con resultados completos
        """
        inicio = time.time()
        warnings_list = []

        logger.info(f"Iniciando forecast para {config.material} con {config.modelo_tipo}")

        try:
            # 1. Validar datos
            validacion = None
            if config.validar_datos:
                logger.debug("Validando datos...")
                validacion = self.validator.validar_completo(
                    df, columna_fecha, columna_cantidad,
                    OutlierMethod(config.metodo_outliers) if config.detectar_outliers else OutlierMethod.IQR
                )

                if not validacion.is_valid:
                    return ForecastResult(
                        exito=False,
                        predicciones=None,
                        metricas={},
                        validacion=validacion,
                        backtest=None,
                        auto_select=None,
                        config=config,
                        tiempo_ejecucion=time.time() - inicio,
                        mensaje=f"Datos no válidos: {validacion.resumen}"
                    )

                # Advertencias
                for issue in validacion.issues:
                    if issue.severidad.value in ['warning', 'error']:
                        warnings_list.append(issue.mensaje)

                # Limpiar outliers si se solicita
                if config.remover_outliers and validacion.outlier_reports:
                    df = self.validator.limpiar_datos(df, validacion, remover_outliers=True)
                    warnings_list.append(f"Removidos {sum(r.n_outliers for r in validacion.outlier_reports)} outliers")

            # 2. Preparar datos
            df_prep = self._preparar_datos(df, columna_fecha, columna_cantidad)

            if len(df_prep) < 10:
                return ForecastResult(
                    exito=False,
                    predicciones=None,
                    metricas={},
                    validacion=validacion,
                    backtest=None,
                    auto_select=None,
                    config=config,
                    tiempo_ejecucion=time.time() - inicio,
                    mensaje=f"Datos insuficientes: {len(df_prep)} registros (mínimo 10)"
                )

            # 3. Auto-selección de modelo (opcional)
            auto_select = None
            modelo_tipo = config.modelo_tipo

            if config.auto_seleccionar_modelo:
                logger.debug("Auto-seleccionando modelo...")
                try:
                    auto_select = self._auto_seleccionar(df_prep, columna_cantidad)
                    modelo_tipo = auto_select.mejor_modelo
                    warnings_list.append(f"Modelo auto-seleccionado: {modelo_tipo}")
                except Exception as e:
                    logger.warning(f"Error en auto-selección: {e}")
                    warnings_list.append(f"Auto-selección falló, usando {modelo_tipo}")

            # 4. Entrenar modelo
            logger.debug(f"Entrenando modelo {modelo_tipo}...")
            predictor = DemandPredictor(modelo=modelo_tipo)
            metricas = predictor.entrenar(df_prep, columna_cantidad)

            # 5. Generar predicciones
            logger.debug(f"Generando predicciones para {config.horizonte} días...")
            predicciones = predictor.predecir(df_prep, config.horizonte)

            # Ajustar intervalos de confianza
            predicciones = self._ajustar_confianza(
                predicciones, config.nivel_confianza, metricas
            )

            # 6. Backtesting (opcional)
            backtest = None
            if config.ejecutar_backtest:
                logger.debug("Ejecutando backtesting...")
                try:
                    backtest = self._ejecutar_backtest(
                        df_prep, modelo_tipo, columna_fecha, columna_cantidad,
                        config.horizonte, config.n_pasos_backtest
                    )
                except Exception as e:
                    logger.warning(f"Error en backtesting: {e}")
                    warnings_list.append(f"Backtesting falló: {str(e)}")

            # 7. Guardar modelo (opcional)
            modelo_id = None
            if config.guardar_modelo:
                try:
                    modelo_id = self._guardar_modelo(
                        predictor, config, metricas, df_prep, columna_fecha
                    )
                except Exception as e:
                    logger.warning(f"Error guardando modelo: {e}")
                    warnings_list.append(f"No se pudo guardar modelo: {str(e)}")

            tiempo_total = time.time() - inicio
            logger.info(f"Forecast completado en {tiempo_total:.2f}s. MAE: {metricas.get('mae', 0):.2f}")

            return ForecastResult(
                exito=True,
                predicciones=predicciones,
                metricas=metricas,
                validacion=validacion,
                backtest=backtest,
                auto_select=auto_select,
                config=config,
                tiempo_ejecucion=tiempo_total,
                modelo_id=modelo_id,
                mensaje="Forecast generado exitosamente",
                warnings=warnings_list
            )

        except Exception as e:
            logger.error(f"Error en forecast: {e}")
            return ForecastResult(
                exito=False,
                predicciones=None,
                metricas={},
                validacion=validacion if 'validacion' in locals() else None,
                backtest=None,
                auto_select=None,
                config=config,
                tiempo_ejecucion=time.time() - inicio,
                mensaje=f"Error: {str(e)}",
                warnings=warnings_list
            )

    def _preparar_datos(
        self,
        df: pd.DataFrame,
        columna_fecha: str,
        columna_cantidad: str
    ) -> pd.DataFrame:
        """Prepara datos para entrenamiento"""
        df_prep = df.copy()

        # Asegurar tipos
        df_prep[columna_fecha] = pd.to_datetime(df_prep[columna_fecha])
        df_prep[columna_cantidad] = pd.to_numeric(df_prep[columna_cantidad], errors='coerce')

        # Agregar por fecha
        df_prep = df_prep.groupby(columna_fecha)[columna_cantidad].sum().reset_index()

        # Ordenar
        df_prep = df_prep.sort_values(columna_fecha).reset_index(drop=True)

        # Eliminar nulos
        df_prep = df_prep.dropna()

        return df_prep

    def _ajustar_confianza(
        self,
        predicciones: pd.DataFrame,
        nivel_confianza: float,
        metricas: Dict[str, float]
    ) -> pd.DataFrame:
        """Ajusta intervalos de confianza según nivel"""
        # Factores Z por nivel de confianza
        z_scores = {
            0.80: 1.28,
            0.85: 1.44,
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }

        z = z_scores.get(nivel_confianza, 1.96)
        std = metricas.get('std_historica', metricas.get('mae', 0))

        if 'limite_inferior' in predicciones.columns:
            predicciones['limite_inferior'] = predicciones['prediccion'] - z * std * 0.5
            predicciones['limite_superior'] = predicciones['prediccion'] + z * std * 0.5

            # Asegurar no negativos
            predicciones['limite_inferior'] = predicciones['limite_inferior'].clip(lower=0)

        return predicciones

    def _auto_seleccionar(
        self,
        df: pd.DataFrame,
        columna_cantidad: str
    ) -> AutoSelectResult:
        """Auto-selecciona el mejor modelo"""
        # Preparar features (simplificado)
        from src.ml.predictor import DemandPredictor

        predictor = DemandPredictor()
        df_features = predictor._crear_features_temporales(df)

        feature_cols = [c for c in df_features.columns if c not in ['fecha', columna_cantidad]]
        X = df_features[feature_cols].fillna(0).values
        y = df_features[columna_cantidad].values

        selector = AutoModelSelector(
            modelos=['random_forest', 'gradient_boosting', 'linear'],
            optimizar_params=False
        )
        return selector.seleccionar(X, y)

    def _ejecutar_backtest(
        self,
        df: pd.DataFrame,
        modelo_tipo: str,
        columna_fecha: str,
        columna_cantidad: str,
        ventana: int,
        n_pasos: int
    ) -> BacktestReport:
        """Ejecuta backtesting"""
        backtester = Backtester(DemandPredictor, modelo_tipo)
        return backtester.ejecutar(
            df, columna_fecha, columna_cantidad,
            ventana_test=min(ventana, 30),
            n_pasos=n_pasos
        )

    def _guardar_modelo(
        self,
        predictor: DemandPredictor,
        config: ForecastConfig,
        metricas: Dict[str, float],
        df: pd.DataFrame,
        columna_fecha: str
    ) -> str:
        """Guarda modelo en registry"""
        metadata = crear_metadata(
            material=config.material,
            modelo_tipo=config.modelo_tipo,
            metricas=metricas,
            n_muestras=len(df),
            fecha_inicio=str(df[columna_fecha].min().date()),
            fecha_fin=str(df[columna_fecha].max().date()),
            features=predictor._feature_names if hasattr(predictor, '_feature_names') else [],
            descripcion=f"Modelo para {config.material} - Horizonte {config.horizonte} días"
        )

        return self.registry.guardar_modelo(
            modelo=predictor._modelo,
            metadata=metadata,
            scaler=predictor._scaler if hasattr(predictor, '_scaler') else None
        )

    def cargar_y_predecir(
        self,
        model_id: str,
        df: pd.DataFrame,
        horizonte: int,
        columna_fecha: str = 'fecha',
        columna_cantidad: str = 'cantidad'
    ) -> ForecastResult:
        """
        Carga un modelo guardado y genera predicciones.

        Args:
            model_id: ID del modelo guardado
            df: Datos históricos recientes
            horizonte: Días a predecir
            columna_fecha: Columna de fechas
            columna_cantidad: Columna de cantidades

        Returns:
            ForecastResult con predicciones
        """
        inicio = time.time()

        try:
            modelo, metadata, scaler = self.registry.cargar_modelo(model_id)

            # Crear predictor con modelo cargado
            predictor = DemandPredictor(modelo=metadata.modelo_tipo)
            predictor._modelo = modelo
            predictor._scaler = scaler
            predictor._entrenado = True

            # Preparar datos
            df_prep = self._preparar_datos(df, columna_fecha, columna_cantidad)

            # Predecir
            predicciones = predictor.predecir(df_prep, horizonte)

            config = ForecastConfig(
                material=metadata.material,
                horizonte=horizonte,
                modelo_tipo=metadata.modelo_tipo
            )

            return ForecastResult(
                exito=True,
                predicciones=predicciones,
                metricas=metadata.metricas,
                validacion=None,
                backtest=None,
                auto_select=None,
                config=config,
                tiempo_ejecucion=time.time() - inicio,
                modelo_id=model_id,
                mensaje="Predicción con modelo guardado exitosa"
            )

        except Exception as e:
            logger.error(f"Error cargando modelo {model_id}: {e}")
            return ForecastResult(
                exito=False,
                predicciones=None,
                metricas={},
                validacion=None,
                backtest=None,
                auto_select=None,
                config=ForecastConfig(material="", horizonte=horizonte),
                tiempo_ejecucion=time.time() - inicio,
                mensaje=f"Error: {str(e)}"
            )

    def comparar_con_historico(
        self,
        df_historico: pd.DataFrame,
        predicciones: pd.DataFrame,
        columna_fecha: str = 'fecha',
        columna_cantidad: str = 'cantidad'
    ) -> Dict[str, Any]:
        """
        Compara predicciones con datos históricos reales.

        Args:
            df_historico: Datos históricos completos
            predicciones: DataFrame con predicciones
            columna_fecha: Columna de fechas
            columna_cantidad: Columna de cantidades

        Returns:
            Diccionario con comparación y métricas
        """
        # Encontrar fechas comunes
        pred_fechas = set(pd.to_datetime(predicciones['fecha']))
        hist_fechas = set(pd.to_datetime(df_historico[columna_fecha]))

        fechas_comunes = pred_fechas.intersection(hist_fechas)

        if not fechas_comunes:
            return {
                'hay_comparacion': False,
                'mensaje': 'No hay fechas comunes entre predicciones e histórico'
            }

        # Filtrar a fechas comunes
        pred_filtrado = predicciones[predicciones['fecha'].isin(fechas_comunes)].copy()
        hist_filtrado = df_historico[df_historico[columna_fecha].isin(fechas_comunes)].copy()

        # Agrupar histórico por fecha
        hist_agg = hist_filtrado.groupby(columna_fecha)[columna_cantidad].sum().reset_index()

        # Merge
        comparacion = pred_filtrado.merge(
            hist_agg,
            left_on='fecha',
            right_on=columna_fecha,
            how='inner'
        )

        if len(comparacion) == 0:
            return {
                'hay_comparacion': False,
                'mensaje': 'No se pudieron alinear los datos'
            }

        # Calcular métricas
        y_pred = comparacion['prediccion'].values
        y_real = comparacion[columna_cantidad].values

        mae = np.mean(np.abs(y_pred - y_real))
        rmse = np.sqrt(np.mean((y_pred - y_real) ** 2))
        mape = np.mean(np.abs((y_pred - y_real) / np.where(y_real != 0, y_real, 1))) * 100

        ss_res = np.sum((y_real - y_pred) ** 2)
        ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Crear DataFrame de comparación
        comparacion['error'] = np.abs(y_pred - y_real)
        comparacion['error_pct'] = np.abs((y_pred - y_real) / np.where(y_real != 0, y_real, 1)) * 100

        return {
            'hay_comparacion': True,
            'n_dias': len(comparacion),
            'metricas': {
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'mape': round(mape, 2),
                'r2': round(r2, 4)
            },
            'comparacion': comparacion[[
                'fecha', 'prediccion', columna_cantidad, 'error', 'error_pct'
            ]].rename(columns={columna_cantidad: 'real'}).to_dict('records'),
            'resumen': self._generar_resumen_comparacion(mae, r2, len(comparacion))
        }

    def _generar_resumen_comparacion(
        self,
        mae: float,
        r2: float,
        n_dias: int
    ) -> str:
        """Genera resumen de comparación"""
        partes = [f"Comparación basada en {n_dias} días."]

        if r2 > 0.8:
            partes.append(f"Excelente precisión (R²={r2:.2f}).")
        elif r2 > 0.6:
            partes.append(f"Buena precisión (R²={r2:.2f}).")
        elif r2 > 0.4:
            partes.append(f"Precisión moderada (R²={r2:.2f}).")
        else:
            partes.append(f"Precisión baja (R²={r2:.2f}). Considere reentrenar.")

        partes.append(f"Error promedio: {mae:.1f} unidades.")

        return " ".join(partes)


# Función de conveniencia
def ejecutar_forecast_rapido(
    df: pd.DataFrame,
    material: str,
    horizonte: int = 30,
    modelo: str = 'random_forest'
) -> ForecastResult:
    """
    Ejecuta forecast rápido con configuración mínima.

    Args:
        df: DataFrame con datos
        material: Código del material
        horizonte: Días a predecir
        modelo: Tipo de modelo

    Returns:
        ForecastResult
    """
    service = ForecastService()
    config = ForecastConfig(
        material=material,
        horizonte=horizonte,
        modelo_tipo=modelo,
        validar_datos=True,
        ejecutar_backtest=False
    )
    return service.ejecutar_forecast(df, config)
