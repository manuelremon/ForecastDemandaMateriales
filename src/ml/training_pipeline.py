"""
Pipeline de Auto-Training para MRP Analytics
=============================================
Re-entrenamiento automático de modelos ML
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import schedule
import pandas as pd
import numpy as np

from src.ml.predictor import DemandPredictor, StockOptimizer
from src.ml.models import train_demand_model, train_classification_model
from src.data.database import (
    guardar_modelo, cargar_modelo, registrar_entrenamiento,
    get_config, set_config
)
from src.data.sap_loader import (
    cargar_consumo_historico, cargar_stock_ultimo_dia,
    calcular_consumo_promedio, preparar_datos_para_ml
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MRP-Training')


class TrainingPipeline:
    """
    Pipeline de entrenamiento automático

    Características:
    - Re-entrenamiento programado
    - Monitoreo de performance
    - Selección automática del mejor modelo
    - Alertas cuando el modelo degrada
    """

    def __init__(self):
        self.is_running = False
        self.last_training = None
        self.models = {}
        self.metrics_history = []
        self.scheduler_thread = None

    def entrenar_modelo_demanda(
        self,
        material: str = None,
        modelo_tipo: str = 'random_forest'
    ) -> Dict[str, Any]:
        """
        Entrena modelo de predicción de demanda

        Args:
            material: Código de material (None = modelo general)
            modelo_tipo: Tipo de modelo a entrenar

        Returns:
            Dict con resultados del entrenamiento
        """
        logger.info(f"Iniciando entrenamiento de modelo de demanda: {modelo_tipo}")
        start_time = time.time()

        try:
            # Cargar datos históricos
            if material:
                df = preparar_datos_para_ml(material)
            else:
                # Modelo general: usar consumos agregados
                df = cargar_consumo_historico(dias=365)
                if len(df) > 0:
                    df = df.groupby('fecha')['cantidad'].sum().reset_index()
                    df['fecha'] = pd.to_datetime(df['fecha'])

            if len(df) < 30:
                logger.warning(f"Datos insuficientes: {len(df)} registros")
                return {
                    'success': False,
                    'error': 'Datos insuficientes para entrenar',
                    'registros': len(df)
                }

            # Entrenar predictor
            predictor = DemandPredictor(modelo=modelo_tipo)
            metrics = predictor.entrenar(df, 'cantidad')

            # Guardar modelo en base de datos
            modelo_id = guardar_modelo(
                nombre=f"demanda_{material or 'general'}_{modelo_tipo}",
                tipo='demanda',
                modelo=predictor.modelo,
                scaler=predictor.scaler,
                metricas=metrics,
                parametros={'modelo_tipo': modelo_tipo, 'material': material},
                material_id=None  # TODO: obtener material_id
            )

            duracion = time.time() - start_time

            # Registrar entrenamiento
            registrar_entrenamiento(
                modelo_id=modelo_id,
                registros_usados=len(df),
                metricas=metrics,
                duracion_segundos=duracion
            )

            logger.info(f"Entrenamiento completado. MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")

            return {
                'success': True,
                'modelo_id': modelo_id,
                'metricas': metrics,
                'registros_usados': len(df),
                'duracion_segundos': duracion
            }

        except Exception as e:
            logger.error(f"Error en entrenamiento: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def entrenar_todos_modelos(self) -> List[Dict]:
        """
        Entrena todos los modelos del sistema

        Returns:
            Lista de resultados de entrenamiento
        """
        logger.info("Iniciando entrenamiento de todos los modelos")
        resultados = []

        # 1. Modelo general de demanda con diferentes algoritmos
        for modelo_tipo in ['random_forest', 'gradient_boosting']:
            resultado = self.entrenar_modelo_demanda(modelo_tipo=modelo_tipo)
            resultado['tipo'] = 'demanda_general'
            resultado['algoritmo'] = modelo_tipo
            resultados.append(resultado)

        # 2. Modelos específicos para materiales de alto consumo
        df_consumo = calcular_consumo_promedio(dias=90)
        if len(df_consumo) > 0:
            top_materiales = df_consumo.nlargest(10, 'consumo_mensual')['codigo'].tolist()

            for material in top_materiales[:5]:  # Top 5 materiales
                resultado = self.entrenar_modelo_demanda(
                    material=material,
                    modelo_tipo='random_forest'
                )
                resultado['tipo'] = 'demanda_especifico'
                resultado['material'] = material
                resultados.append(resultado)

        self.last_training = datetime.now()
        logger.info(f"Entrenamiento completado. {len(resultados)} modelos entrenados.")

        return resultados

    def evaluar_modelo(
        self,
        tipo: str = 'demanda',
        material_id: int = None
    ) -> Dict[str, Any]:
        """
        Evalúa el rendimiento del modelo actual

        Returns:
            Dict con métricas de evaluación
        """
        modelo, scaler, metricas_guardadas = cargar_modelo(tipo, material_id)

        if modelo is None:
            return {'error': 'Modelo no encontrado'}

        return {
            'metricas_entrenamiento': metricas_guardadas,
            'modelo_activo': True,
            'tipo': tipo
        }

    def verificar_necesidad_reentrenamiento(self) -> bool:
        """
        Verifica si es necesario re-entrenar los modelos

        Criterios:
        - Tiempo desde último entrenamiento
        - Degradación de métricas
        - Nuevos datos disponibles

        Returns:
            True si se debe re-entrenar
        """
        # Obtener intervalo de configuración
        intervalo_dias = int(get_config('retrain_interval_days') or 7)

        if self.last_training is None:
            return True

        dias_desde_ultimo = (datetime.now() - self.last_training).days

        if dias_desde_ultimo >= intervalo_dias:
            logger.info(f"Re-entrenamiento necesario: {dias_desde_ultimo} días desde último entrenamiento")
            return True

        return False

    def programar_reentrenamiento(
        self,
        hora: str = "02:00",
        intervalo_dias: int = 7
    ):
        """
        Programa re-entrenamiento automático

        Args:
            hora: Hora del día para ejecutar (formato HH:MM)
            intervalo_dias: Días entre entrenamientos
        """
        set_config('retrain_interval_days', str(intervalo_dias))

        def job():
            if self.verificar_necesidad_reentrenamiento():
                logger.info("Ejecutando re-entrenamiento programado")
                self.entrenar_todos_modelos()

        # Programar job
        schedule.every(intervalo_dias).days.at(hora).do(job)

        logger.info(f"Re-entrenamiento programado: cada {intervalo_dias} días a las {hora}")

    def iniciar_scheduler(self):
        """Inicia el scheduler en un thread separado"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("Scheduler ya está corriendo")
            return

        self.is_running = True

        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check cada minuto

        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        logger.info("Scheduler de entrenamiento iniciado")

    def detener_scheduler(self):
        """Detiene el scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Scheduler detenido")

    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene estado del pipeline de entrenamiento

        Returns:
            Dict con información de estado
        """
        return {
            'is_running': self.is_running,
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'scheduler_active': self.scheduler_thread.is_alive() if self.scheduler_thread else False,
            'intervalo_dias': int(get_config('retrain_interval_days') or 7),
            'proximo_entrenamiento': self._calcular_proximo_entrenamiento()
        }

    def _calcular_proximo_entrenamiento(self) -> Optional[str]:
        """Calcula fecha del próximo entrenamiento"""
        if self.last_training is None:
            return "Pendiente - nunca entrenado"

        intervalo = int(get_config('retrain_interval_days') or 7)
        proximo = self.last_training + timedelta(days=intervalo)
        return proximo.isoformat()


class ModelMonitor:
    """
    Monitor de rendimiento de modelos

    Detecta degradación y genera alertas
    """

    def __init__(self, umbral_mae: float = 0.2, umbral_r2: float = 0.1):
        self.umbral_mae = umbral_mae  # Degradación máxima aceptable en MAE
        self.umbral_r2 = umbral_r2    # Degradación máxima aceptable en R²
        self.baseline_metrics = {}
        self.alertas = []

    def establecer_baseline(self, tipo: str, metricas: Dict):
        """Establece métricas de referencia"""
        self.baseline_metrics[tipo] = {
            'mae': metricas.get('mae', 0),
            'r2': metricas.get('r2', 0),
            'timestamp': datetime.now()
        }

    def evaluar_degradacion(
        self,
        tipo: str,
        metricas_actuales: Dict
    ) -> Dict[str, Any]:
        """
        Evalúa si el modelo ha degradado

        Returns:
            Dict con análisis de degradación
        """
        if tipo not in self.baseline_metrics:
            return {'degradado': False, 'mensaje': 'Sin baseline establecido'}

        baseline = self.baseline_metrics[tipo]

        mae_actual = metricas_actuales.get('mae', 0)
        mae_baseline = baseline['mae']
        r2_actual = metricas_actuales.get('r2', 0)
        r2_baseline = baseline['r2']

        # Calcular degradación
        if mae_baseline > 0:
            degradacion_mae = (mae_actual - mae_baseline) / mae_baseline
        else:
            degradacion_mae = 0

        degradacion_r2 = r2_baseline - r2_actual  # R² más alto es mejor

        degradado = degradacion_mae > self.umbral_mae or degradacion_r2 > self.umbral_r2

        resultado = {
            'degradado': degradado,
            'degradacion_mae_pct': degradacion_mae * 100,
            'degradacion_r2': degradacion_r2,
            'mae_actual': mae_actual,
            'mae_baseline': mae_baseline,
            'r2_actual': r2_actual,
            'r2_baseline': r2_baseline
        }

        if degradado:
            alerta = {
                'tipo': 'degradacion_modelo',
                'modelo': tipo,
                'timestamp': datetime.now(),
                'detalles': resultado
            }
            self.alertas.append(alerta)
            logger.warning(f"Modelo {tipo} ha degradado: MAE +{degradacion_mae*100:.1f}%")

        return resultado

    def get_alertas(self, ultimas: int = 10) -> List[Dict]:
        """Obtiene últimas alertas"""
        return self.alertas[-ultimas:]


# Singleton del pipeline
_pipeline_instance = None


def get_training_pipeline() -> TrainingPipeline:
    """Obtiene instancia singleton del pipeline"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = TrainingPipeline()
    return _pipeline_instance


def entrenar_modelo_demanda_async(material: str = None, modelo_tipo: str = 'random_forest'):
    """Wrapper para entrenamiento asíncrono"""
    pipeline = get_training_pipeline()
    return pipeline.entrenar_modelo_demanda(material, modelo_tipo)


def iniciar_entrenamiento_programado():
    """Inicia el sistema de entrenamiento programado"""
    pipeline = get_training_pipeline()
    pipeline.programar_reentrenamiento(hora="03:00", intervalo_dias=7)
    pipeline.iniciar_scheduler()
    return pipeline.get_status()


if __name__ == "__main__":
    # Test del pipeline
    pipeline = TrainingPipeline()
    print("Estado del pipeline:", pipeline.get_status())

    # Entrenar un modelo de prueba
    resultado = pipeline.entrenar_modelo_demanda(modelo_tipo='random_forest')
    print("Resultado entrenamiento:", resultado)
