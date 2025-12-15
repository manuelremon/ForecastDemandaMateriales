"""
Módulo de Validación de Datos para Forecast MR

Proporciona validación integral de datos de entrada incluyendo:
- Detección de valores faltantes
- Detección de outliers (IQR, Z-score, Isolation Forest)
- Validación de fechas
- Validación de cantidades
- Reporte de calidad de datos

Author: Manuel Remón
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger


class OutlierMethod(Enum):
    """Métodos disponibles para detección de outliers"""
    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"
    MAD = "mad"  # Median Absolute Deviation


class ValidationSeverity(Enum):
    """Severidad de los problemas encontrados"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Representa un problema encontrado en la validación"""
    campo: str
    mensaje: str
    severidad: ValidationSeverity
    detalles: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'campo': self.campo,
            'mensaje': self.mensaje,
            'severidad': self.severidad.value,
            'detalles': self.detalles
        }


@dataclass
class OutlierReport:
    """Reporte de outliers detectados"""
    metodo: str
    columna: str
    n_outliers: int
    porcentaje: float
    indices: List[int]
    valores: List[float]
    limites: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'metodo': self.metodo,
            'columna': self.columna,
            'n_outliers': self.n_outliers,
            'porcentaje': round(self.porcentaje, 2),
            'indices': self.indices[:20],  # Limitar para no saturar
            'valores': [round(v, 2) for v in self.valores[:20]],
            'limites': {k: round(v, 2) for k, v in self.limites.items()}
        }


@dataclass
class DateValidationResult:
    """Resultado de validación de fechas"""
    is_valid: bool
    fecha_min: Optional[datetime]
    fecha_max: Optional[datetime]
    rango_dias: int
    gaps_detectados: List[Tuple[datetime, datetime]]
    duplicados: int
    fechas_invalidas: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'fecha_min': self.fecha_min.isoformat() if self.fecha_min else None,
            'fecha_max': self.fecha_max.isoformat() if self.fecha_max else None,
            'rango_dias': self.rango_dias,
            'gaps_detectados': len(self.gaps_detectados),
            'duplicados': self.duplicados,
            'fechas_invalidas': self.fechas_invalidas
        }


@dataclass
class DataQualityReport:
    """Reporte completo de calidad de datos"""
    score: float  # 0-100
    total_registros: int
    registros_validos: int
    issues: List[ValidationIssue]
    outlier_reports: List[OutlierReport]
    date_validation: Optional[DateValidationResult]
    completitud_por_columna: Dict[str, float]
    resumen: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_valid(self) -> bool:
        """Datos válidos si no hay errores críticos y score > 50"""
        has_critical = any(i.severidad == ValidationSeverity.CRITICAL for i in self.issues)
        return not has_critical and self.score >= 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': round(self.score, 1),
            'is_valid': self.is_valid,
            'total_registros': self.total_registros,
            'registros_validos': self.registros_validos,
            'issues': [i.to_dict() for i in self.issues],
            'outlier_reports': [o.to_dict() for o in self.outlier_reports],
            'date_validation': self.date_validation.to_dict() if self.date_validation else None,
            'completitud_por_columna': {k: round(v, 1) for k, v in self.completitud_por_columna.items()},
            'resumen': self.resumen,
            'timestamp': self.timestamp.isoformat()
        }


class DataValidator:
    """
    Validador integral de datos de entrada para forecasting.

    Proporciona validación completa incluyendo:
    - Completitud de datos
    - Detección de outliers
    - Validación de fechas
    - Validación de cantidades
    - Score de calidad general

    Ejemplo de uso:
        validator = DataValidator()
        report = validator.validar_completo(df, columna_fecha='fecha', columna_cantidad='cantidad')

        if report.is_valid:
            # Proceder con el análisis
            df_limpio = validator.limpiar_datos(df, report)
        else:
            # Mostrar errores al usuario
            for issue in report.issues:
                print(f"{issue.severidad}: {issue.mensaje}")
    """

    # Configuración por defecto
    MIN_REGISTROS = 10
    MIN_COMPLETITUD = 0.7  # 70% mínimo de datos completos
    IQR_FACTOR = 1.5
    ZSCORE_THRESHOLD = 3.0
    MAD_THRESHOLD = 3.5

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa el validador con configuración opcional.

        Args:
            config: Diccionario con configuración personalizada
        """
        self.config = config or {}
        self._update_config()

    def _update_config(self):
        """Actualiza configuración desde diccionario"""
        self.min_registros = self.config.get('min_registros', self.MIN_REGISTROS)
        self.min_completitud = self.config.get('min_completitud', self.MIN_COMPLETITUD)
        self.iqr_factor = self.config.get('iqr_factor', self.IQR_FACTOR)
        self.zscore_threshold = self.config.get('zscore_threshold', self.ZSCORE_THRESHOLD)

    def validar_completitud(self, df: pd.DataFrame) -> Tuple[Dict[str, float], List[ValidationIssue]]:
        """
        Valida la completitud de datos por columna.

        Args:
            df: DataFrame a validar

        Returns:
            Tuple con diccionario de completitud por columna y lista de issues
        """
        issues = []
        completitud = {}

        for col in df.columns:
            n_validos = df[col].notna().sum()
            pct_completo = (n_validos / len(df)) * 100 if len(df) > 0 else 0
            completitud[col] = pct_completo

            if pct_completo < 50:
                issues.append(ValidationIssue(
                    campo=col,
                    mensaje=f"Columna '{col}' tiene solo {pct_completo:.1f}% de datos completos",
                    severidad=ValidationSeverity.ERROR,
                    detalles={'completitud': pct_completo, 'faltantes': len(df) - n_validos}
                ))
            elif pct_completo < 80:
                issues.append(ValidationIssue(
                    campo=col,
                    mensaje=f"Columna '{col}' tiene {pct_completo:.1f}% de datos completos",
                    severidad=ValidationSeverity.WARNING,
                    detalles={'completitud': pct_completo, 'faltantes': len(df) - n_validos}
                ))

        logger.debug(f"Validación completitud: {len(issues)} issues encontrados")
        return completitud, issues

    def detectar_outliers(
        self,
        df: pd.DataFrame,
        columna: str,
        metodo: OutlierMethod = OutlierMethod.IQR
    ) -> OutlierReport:
        """
        Detecta outliers en una columna numérica.

        Args:
            df: DataFrame con los datos
            columna: Nombre de la columna a analizar
            metodo: Método de detección (IQR, ZSCORE, MAD, ISOLATION_FOREST)

        Returns:
            OutlierReport con los resultados
        """
        if columna not in df.columns:
            raise ValueError(f"Columna '{columna}' no existe en el DataFrame")

        datos = df[columna].dropna()
        if len(datos) == 0:
            return OutlierReport(
                metodo=metodo.value,
                columna=columna,
                n_outliers=0,
                porcentaje=0,
                indices=[],
                valores=[],
                limites={}
            )

        if metodo == OutlierMethod.IQR:
            return self._detectar_outliers_iqr(df, columna, datos)
        elif metodo == OutlierMethod.ZSCORE:
            return self._detectar_outliers_zscore(df, columna, datos)
        elif metodo == OutlierMethod.MAD:
            return self._detectar_outliers_mad(df, columna, datos)
        elif metodo == OutlierMethod.ISOLATION_FOREST:
            return self._detectar_outliers_isolation_forest(df, columna, datos)
        else:
            raise ValueError(f"Método '{metodo}' no soportado")

    def _detectar_outliers_iqr(
        self, df: pd.DataFrame, columna: str, datos: pd.Series
    ) -> OutlierReport:
        """Detección de outliers usando método IQR (Interquartile Range)"""
        Q1 = datos.quantile(0.25)
        Q3 = datos.quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - self.iqr_factor * IQR
        limite_superior = Q3 + self.iqr_factor * IQR

        # Identificar outliers
        mask = (df[columna] < limite_inferior) | (df[columna] > limite_superior)
        outliers_idx = df[mask].index.tolist()
        outliers_vals = df.loc[mask, columna].tolist()

        return OutlierReport(
            metodo="iqr",
            columna=columna,
            n_outliers=len(outliers_idx),
            porcentaje=(len(outliers_idx) / len(df)) * 100,
            indices=outliers_idx,
            valores=outliers_vals,
            limites={
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'limite_inferior': limite_inferior,
                'limite_superior': limite_superior
            }
        )

    def _detectar_outliers_zscore(
        self, df: pd.DataFrame, columna: str, datos: pd.Series
    ) -> OutlierReport:
        """Detección de outliers usando Z-score"""
        media = datos.mean()
        std = datos.std()

        if std == 0:
            return OutlierReport(
                metodo="zscore",
                columna=columna,
                n_outliers=0,
                porcentaje=0,
                indices=[],
                valores=[],
                limites={'media': media, 'std': std, 'threshold': self.zscore_threshold}
            )

        z_scores = np.abs((df[columna] - media) / std)
        mask = z_scores > self.zscore_threshold
        outliers_idx = df[mask].index.tolist()
        outliers_vals = df.loc[mask, columna].tolist()

        return OutlierReport(
            metodo="zscore",
            columna=columna,
            n_outliers=len(outliers_idx),
            porcentaje=(len(outliers_idx) / len(df)) * 100,
            indices=outliers_idx,
            valores=outliers_vals,
            limites={
                'media': media,
                'std': std,
                'threshold': self.zscore_threshold,
                'limite_inferior': media - self.zscore_threshold * std,
                'limite_superior': media + self.zscore_threshold * std
            }
        )

    def _detectar_outliers_mad(
        self, df: pd.DataFrame, columna: str, datos: pd.Series
    ) -> OutlierReport:
        """Detección de outliers usando MAD (Median Absolute Deviation)"""
        mediana = datos.median()
        mad = np.median(np.abs(datos - mediana))

        # Factor de escala para normalizar MAD a std equivalente
        mad_scaled = mad * 1.4826

        if mad_scaled == 0:
            return OutlierReport(
                metodo="mad",
                columna=columna,
                n_outliers=0,
                porcentaje=0,
                indices=[],
                valores=[],
                limites={'mediana': mediana, 'mad': mad}
            )

        modified_z = np.abs((df[columna] - mediana) / mad_scaled)
        mask = modified_z > self.MAD_THRESHOLD
        outliers_idx = df[mask].index.tolist()
        outliers_vals = df.loc[mask, columna].tolist()

        return OutlierReport(
            metodo="mad",
            columna=columna,
            n_outliers=len(outliers_idx),
            porcentaje=(len(outliers_idx) / len(df)) * 100,
            indices=outliers_idx,
            valores=outliers_vals,
            limites={
                'mediana': mediana,
                'mad': mad,
                'mad_scaled': mad_scaled,
                'threshold': self.MAD_THRESHOLD,
                'limite_inferior': mediana - self.MAD_THRESHOLD * mad_scaled,
                'limite_superior': mediana + self.MAD_THRESHOLD * mad_scaled
            }
        )

    def _detectar_outliers_isolation_forest(
        self, df: pd.DataFrame, columna: str, datos: pd.Series
    ) -> OutlierReport:
        """Detección de outliers usando Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            logger.warning("sklearn no disponible, usando IQR como fallback")
            return self._detectar_outliers_iqr(df, columna, datos)

        # Preparar datos
        X = datos.values.reshape(-1, 1)

        # Entrenar modelo
        clf = IsolationForest(
            contamination=0.1,  # Esperamos ~10% de outliers máximo
            random_state=42,
            n_jobs=-1
        )
        predicciones = clf.fit_predict(X)

        # -1 indica outlier en Isolation Forest
        mask_full = pd.Series(False, index=df.index)
        mask_full[datos.index] = predicciones == -1

        outliers_idx = df[mask_full].index.tolist()
        outliers_vals = df.loc[mask_full, columna].tolist()

        return OutlierReport(
            metodo="isolation_forest",
            columna=columna,
            n_outliers=len(outliers_idx),
            porcentaje=(len(outliers_idx) / len(df)) * 100,
            indices=outliers_idx,
            valores=outliers_vals,
            limites={'contamination': 0.1}
        )

    def validar_fechas(
        self,
        df: pd.DataFrame,
        columna_fecha: str,
        detectar_gaps: bool = True,
        gap_threshold_dias: int = 7
    ) -> DateValidationResult:
        """
        Valida la columna de fechas.

        Args:
            df: DataFrame con los datos
            columna_fecha: Nombre de la columna de fechas
            detectar_gaps: Si detectar gaps en la serie temporal
            gap_threshold_dias: Umbral para considerar un gap significativo

        Returns:
            DateValidationResult con los resultados
        """
        if columna_fecha not in df.columns:
            return DateValidationResult(
                is_valid=False,
                fecha_min=None,
                fecha_max=None,
                rango_dias=0,
                gaps_detectados=[],
                duplicados=0,
                fechas_invalidas=len(df)
            )

        # Convertir a datetime si no lo es
        fechas = pd.to_datetime(df[columna_fecha], errors='coerce')

        # Contar fechas inválidas
        fechas_invalidas = fechas.isna().sum()
        fechas_validas = fechas.dropna()

        if len(fechas_validas) == 0:
            return DateValidationResult(
                is_valid=False,
                fecha_min=None,
                fecha_max=None,
                rango_dias=0,
                gaps_detectados=[],
                duplicados=0,
                fechas_invalidas=fechas_invalidas
            )

        # Estadísticas básicas
        fecha_min = fechas_validas.min()
        fecha_max = fechas_validas.max()
        rango_dias = (fecha_max - fecha_min).days

        # Duplicados
        duplicados = fechas_validas.duplicated().sum()

        # Detectar gaps
        gaps = []
        if detectar_gaps and len(fechas_validas) > 1:
            fechas_ordenadas = fechas_validas.sort_values()
            diferencias = fechas_ordenadas.diff()

            for i, diff in enumerate(diferencias):
                if pd.notna(diff) and diff.days > gap_threshold_dias:
                    fecha_inicio = fechas_ordenadas.iloc[i-1]
                    fecha_fin = fechas_ordenadas.iloc[i]
                    gaps.append((fecha_inicio.to_pydatetime(), fecha_fin.to_pydatetime()))

        is_valid = fechas_invalidas == 0 and rango_dias >= 7

        return DateValidationResult(
            is_valid=is_valid,
            fecha_min=fecha_min.to_pydatetime() if pd.notna(fecha_min) else None,
            fecha_max=fecha_max.to_pydatetime() if pd.notna(fecha_max) else None,
            rango_dias=rango_dias,
            gaps_detectados=gaps,
            duplicados=duplicados,
            fechas_invalidas=fechas_invalidas
        )

    def validar_cantidades(
        self,
        df: pd.DataFrame,
        columna_cantidad: str
    ) -> Tuple[Dict[str, Any], List[ValidationIssue]]:
        """
        Valida la columna de cantidades.

        Args:
            df: DataFrame con los datos
            columna_cantidad: Nombre de la columna de cantidades

        Returns:
            Tuple con estadísticas y lista de issues
        """
        issues = []
        stats = {}

        if columna_cantidad not in df.columns:
            issues.append(ValidationIssue(
                campo=columna_cantidad,
                mensaje=f"Columna '{columna_cantidad}' no existe",
                severidad=ValidationSeverity.CRITICAL
            ))
            return stats, issues

        cantidades = pd.to_numeric(df[columna_cantidad], errors='coerce')

        # Estadísticas
        stats = {
            'min': cantidades.min(),
            'max': cantidades.max(),
            'mean': cantidades.mean(),
            'median': cantidades.median(),
            'std': cantidades.std(),
            'sum': cantidades.sum(),
            'zeros': (cantidades == 0).sum(),
            'negativos': (cantidades < 0).sum(),
            'nulos': cantidades.isna().sum()
        }

        # Validaciones
        if stats['negativos'] > 0:
            issues.append(ValidationIssue(
                campo=columna_cantidad,
                mensaje=f"Se encontraron {stats['negativos']} valores negativos",
                severidad=ValidationSeverity.WARNING,
                detalles={'negativos': stats['negativos']}
            ))

        if stats['nulos'] > len(df) * 0.1:
            issues.append(ValidationIssue(
                campo=columna_cantidad,
                mensaje=f"Más del 10% de cantidades son nulas ({stats['nulos']})",
                severidad=ValidationSeverity.ERROR,
                detalles={'nulos': stats['nulos'], 'porcentaje': stats['nulos']/len(df)*100}
            ))

        if stats['zeros'] > len(df) * 0.5:
            issues.append(ValidationIssue(
                campo=columna_cantidad,
                mensaje=f"Más del 50% de cantidades son cero ({stats['zeros']})",
                severidad=ValidationSeverity.WARNING,
                detalles={'zeros': stats['zeros'], 'porcentaje': stats['zeros']/len(df)*100}
            ))

        return stats, issues

    def validar_completo(
        self,
        df: pd.DataFrame,
        columna_fecha: str = 'fecha',
        columna_cantidad: str = 'cantidad',
        metodo_outliers: OutlierMethod = OutlierMethod.IQR,
        columnas_requeridas: Optional[List[str]] = None
    ) -> DataQualityReport:
        """
        Ejecuta validación completa del DataFrame.

        Args:
            df: DataFrame a validar
            columna_fecha: Nombre de la columna de fechas
            columna_cantidad: Nombre de la columna de cantidades
            metodo_outliers: Método para detección de outliers
            columnas_requeridas: Lista de columnas que deben existir

        Returns:
            DataQualityReport con el resultado completo
        """
        logger.info(f"Iniciando validación completa de {len(df)} registros")

        issues: List[ValidationIssue] = []
        outlier_reports: List[OutlierReport] = []
        score = 100.0

        # Validar registros mínimos
        if len(df) < self.min_registros:
            issues.append(ValidationIssue(
                campo='dataset',
                mensaje=f"Dataset tiene solo {len(df)} registros (mínimo: {self.min_registros})",
                severidad=ValidationSeverity.CRITICAL,
                detalles={'registros': len(df), 'minimo': self.min_registros}
            ))
            score -= 30

        # Validar columnas requeridas
        columnas_req = columnas_requeridas or [columna_fecha, columna_cantidad]
        for col in columnas_req:
            if col not in df.columns:
                issues.append(ValidationIssue(
                    campo=col,
                    mensaje=f"Columna requerida '{col}' no existe",
                    severidad=ValidationSeverity.CRITICAL
                ))
                score -= 20

        # Validar completitud
        completitud, completitud_issues = self.validar_completitud(df)
        issues.extend(completitud_issues)

        # Penalizar por baja completitud
        avg_completitud = np.mean(list(completitud.values())) if completitud else 0
        if avg_completitud < 80:
            score -= (80 - avg_completitud) * 0.5

        # Validar fechas
        date_validation = None
        if columna_fecha in df.columns:
            date_validation = self.validar_fechas(df, columna_fecha)
            if not date_validation.is_valid:
                score -= 15
            if date_validation.gaps_detectados:
                issues.append(ValidationIssue(
                    campo=columna_fecha,
                    mensaje=f"Se detectaron {len(date_validation.gaps_detectados)} gaps en la serie temporal",
                    severidad=ValidationSeverity.WARNING,
                    detalles={'gaps': len(date_validation.gaps_detectados)}
                ))
                score -= len(date_validation.gaps_detectados) * 2
            if date_validation.duplicados > 0:
                issues.append(ValidationIssue(
                    campo=columna_fecha,
                    mensaje=f"Se encontraron {date_validation.duplicados} fechas duplicadas",
                    severidad=ValidationSeverity.WARNING,
                    detalles={'duplicados': date_validation.duplicados}
                ))
                score -= 5

        # Validar cantidades
        if columna_cantidad in df.columns:
            cant_stats, cant_issues = self.validar_cantidades(df, columna_cantidad)
            issues.extend(cant_issues)

            # Detectar outliers
            try:
                outlier_report = self.detectar_outliers(df, columna_cantidad, metodo_outliers)
                outlier_reports.append(outlier_report)

                if outlier_report.porcentaje > 10:
                    issues.append(ValidationIssue(
                        campo=columna_cantidad,
                        mensaje=f"Alto porcentaje de outliers: {outlier_report.porcentaje:.1f}%",
                        severidad=ValidationSeverity.WARNING,
                        detalles={'outliers': outlier_report.n_outliers, 'porcentaje': outlier_report.porcentaje}
                    ))
                    score -= outlier_report.porcentaje * 0.5
            except Exception as e:
                logger.warning(f"Error detectando outliers: {e}")

        # Calcular registros válidos
        registros_validos = len(df)
        if columna_cantidad in df.columns:
            registros_validos = df[columna_cantidad].notna().sum()

        # Asegurar score entre 0 y 100
        score = max(0, min(100, score))

        # Generar resumen
        resumen = self._generar_resumen(score, issues, date_validation, outlier_reports)

        logger.info(f"Validación completa. Score: {score:.1f}, Issues: {len(issues)}")

        return DataQualityReport(
            score=score,
            total_registros=len(df),
            registros_validos=registros_validos,
            issues=issues,
            outlier_reports=outlier_reports,
            date_validation=date_validation,
            completitud_por_columna=completitud,
            resumen=resumen
        )

    def _generar_resumen(
        self,
        score: float,
        issues: List[ValidationIssue],
        date_validation: Optional[DateValidationResult],
        outlier_reports: List[OutlierReport]
    ) -> str:
        """Genera un resumen legible del análisis"""
        partes = []

        # Calidad general
        if score >= 90:
            partes.append("Excelente calidad de datos.")
        elif score >= 70:
            partes.append("Buena calidad de datos con algunas observaciones.")
        elif score >= 50:
            partes.append("Calidad de datos aceptable pero mejorable.")
        else:
            partes.append("Calidad de datos insuficiente. Revisar issues críticos.")

        # Críticos
        criticos = [i for i in issues if i.severidad == ValidationSeverity.CRITICAL]
        if criticos:
            partes.append(f"{len(criticos)} problemas críticos requieren atención.")

        # Fechas
        if date_validation:
            if date_validation.rango_dias > 0:
                partes.append(f"Serie temporal de {date_validation.rango_dias} días.")
            if date_validation.gaps_detectados:
                partes.append(f"{len(date_validation.gaps_detectados)} gaps temporales detectados.")

        # Outliers
        total_outliers = sum(r.n_outliers for r in outlier_reports)
        if total_outliers > 0:
            partes.append(f"{total_outliers} outliers detectados.")

        return " ".join(partes)

    def limpiar_datos(
        self,
        df: pd.DataFrame,
        report: DataQualityReport,
        remover_outliers: bool = True,
        imputar_faltantes: bool = False,
        metodo_imputacion: str = 'median'
    ) -> pd.DataFrame:
        """
        Limpia el DataFrame basándose en el reporte de validación.

        Args:
            df: DataFrame original
            report: Reporte de validación
            remover_outliers: Si remover los outliers detectados
            imputar_faltantes: Si imputar valores faltantes
            metodo_imputacion: Método de imputación ('mean', 'median', 'ffill', 'bfill')

        Returns:
            DataFrame limpio
        """
        df_limpio = df.copy()
        registros_inicial = len(df_limpio)

        # Remover outliers
        if remover_outliers and report.outlier_reports:
            indices_outliers = set()
            for outlier_report in report.outlier_reports:
                indices_outliers.update(outlier_report.indices)

            if indices_outliers:
                df_limpio = df_limpio.drop(index=list(indices_outliers), errors='ignore')
                logger.info(f"Removidos {len(indices_outliers)} outliers")

        # Imputar faltantes
        if imputar_faltantes:
            for col in df_limpio.select_dtypes(include=[np.number]).columns:
                if df_limpio[col].isna().any():
                    if metodo_imputacion == 'mean':
                        df_limpio[col].fillna(df_limpio[col].mean(), inplace=True)
                    elif metodo_imputacion == 'median':
                        df_limpio[col].fillna(df_limpio[col].median(), inplace=True)
                    elif metodo_imputacion == 'ffill':
                        df_limpio[col].fillna(method='ffill', inplace=True)
                    elif metodo_imputacion == 'bfill':
                        df_limpio[col].fillna(method='bfill', inplace=True)

        logger.info(f"Limpieza completada: {registros_inicial} -> {len(df_limpio)} registros")
        return df_limpio


# Funciones de conveniencia
def validar_datos_forecast(
    df: pd.DataFrame,
    columna_fecha: str = 'fecha',
    columna_cantidad: str = 'cantidad'
) -> DataQualityReport:
    """
    Función de conveniencia para validar datos de forecast.

    Args:
        df: DataFrame con datos de consumo
        columna_fecha: Nombre de columna de fechas
        columna_cantidad: Nombre de columna de cantidades

    Returns:
        DataQualityReport con el análisis completo
    """
    validator = DataValidator()
    return validator.validar_completo(df, columna_fecha, columna_cantidad)


def detectar_outliers_rapido(
    df: pd.DataFrame,
    columna: str,
    metodo: str = 'iqr'
) -> OutlierReport:
    """
    Función de conveniencia para detectar outliers rápidamente.

    Args:
        df: DataFrame con los datos
        columna: Columna a analizar
        metodo: Método de detección ('iqr', 'zscore', 'mad', 'isolation_forest')

    Returns:
        OutlierReport con los resultados
    """
    validator = DataValidator()
    metodo_enum = OutlierMethod(metodo)
    return validator.detectar_outliers(df, columna, metodo_enum)
