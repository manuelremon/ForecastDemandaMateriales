"""
Funciones de formateo para MRP Analytics
"""
from typing import Union
import locale

# Intentar configurar locale para español
try:
    locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
except (locale.Error, OSError):
    try:
        locale.setlocale(locale.LC_ALL, 'Spanish_Spain.1252')
    except (locale.Error, OSError):
        # Si no se puede configurar locale español, usar el predeterminado
        pass


def formato_moneda(valor: Union[float, int], decimales: int = 2, simbolo: str = "USD") -> str:
    """
    Formatea un valor como moneda

    Args:
        valor: Valor numérico
        decimales: Cantidad de decimales
        simbolo: Símbolo de moneda

    Returns:
        String formateado como moneda
    """
    if valor is None:
        return f"{simbolo} 0.00"

    try:
        valor_formateado = f"{valor:,.{decimales}f}"
        # Cambiar separadores para formato español
        valor_formateado = valor_formateado.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{simbolo} {valor_formateado}"
    except (TypeError, ValueError):
        return f"{simbolo} {valor}"


def formato_numero(valor: Union[float, int], decimales: int = 0) -> str:
    """
    Formatea un número con separadores de miles
    """
    if valor is None:
        return "0"

    try:
        if decimales == 0:
            valor_formateado = f"{int(valor):,}"
        else:
            valor_formateado = f"{valor:,.{decimales}f}"
        # Cambiar separadores para formato español
        valor_formateado = valor_formateado.replace(",", "X").replace(".", ",").replace("X", ".")
        return valor_formateado
    except (TypeError, ValueError):
        return str(valor)


def formato_porcentaje(valor: Union[float, int], decimales: int = 1) -> str:
    """
    Formatea un valor como porcentaje
    """
    if valor is None:
        return "0%"

    try:
        return f"{valor:.{decimales}f}%"
    except (TypeError, ValueError):
        return f"{valor}%"


def formato_tendencia(valor_actual: float, valor_anterior: float) -> dict:
    """
    Calcula y formatea la tendencia entre dos valores

    Returns:
        Dict con dirección, valor, clase CSS e icono
    """
    if valor_anterior == 0:
        return {
            "direccion": "stable",
            "valor": "0%",
            "clase": "text-muted",
            "icono": "fa-minus"
        }

    cambio = ((valor_actual - valor_anterior) / valor_anterior) * 100

    if cambio > 0:
        return {
            "direccion": "up",
            "valor": f"+{cambio:.1f}%",
            "clase": "text-success",
            "icono": "fa-arrow-up"
        }
    elif cambio < 0:
        return {
            "direccion": "down",
            "valor": f"{cambio:.1f}%",
            "clase": "text-danger",
            "icono": "fa-arrow-down"
        }
    else:
        return {
            "direccion": "stable",
            "valor": "0%",
            "clase": "text-muted",
            "icono": "fa-minus"
        }


def formato_dias(dias: Union[float, int]) -> str:
    """
    Formatea días de cobertura
    """
    if dias is None or dias == float('inf'):
        return "∞"

    if dias < 1:
        horas = int(dias * 24)
        return f"{horas}h"
    elif dias < 30:
        return f"{int(dias)}d"
    elif dias < 365:
        meses = dias / 30
        return f"{meses:.1f}m"
    else:
        años = dias / 365
        return f"{años:.1f}a"


def abreviar_numero(valor: Union[float, int]) -> str:
    """
    Abrevia números grandes (K, M, B)
    """
    if valor is None:
        return "0"

    if abs(valor) >= 1_000_000_000:
        return f"{valor/1_000_000_000:.1f}B"
    elif abs(valor) >= 1_000_000:
        return f"{valor/1_000_000:.1f}M"
    elif abs(valor) >= 1_000:
        return f"{valor/1_000:.1f}K"
    else:
        return formato_numero(valor)
