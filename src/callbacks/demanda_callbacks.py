"""
Callbacks del Tablero de Forecasting
=====================================
Integracion con modelos ML para prediccion de demanda
"""
from dash import callback, Output, Input, State, no_update, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.data.sap_loader import cargar_consumo_historico, buscar_material_con_mrp, obtener_centros, obtener_almacenes
from src.data.excel_loader import (
    obtener_materiales_desde_excel,
    filtrar_consumo_por_material,
    obtener_centros_desde_excel,
    obtener_almacenes_desde_excel,
    dict_a_df,
    obtener_filtros_unicos
)
from src.ml.predictor import DemandPredictor
from src.utils.theme import PLOTLY_TEMPLATE, COLORS, color_con_alpha
from src.utils.formatters import formato_numero
from src.utils.logger import get_logger
from src.utils.constants import MODELOS_ML
from src.utils.plotly_helpers import crear_figura_vacia, crear_figura_warning
from src.components.icons import lucide_icon

logger = get_logger(__name__)


@callback(
    Output("filtro-centro-demanda", "options"),
    Output("filtro-centro-demanda", "value"),
    Input("store-excel-data", "data"),
    Input("url", "pathname")
)
def cargar_centros_desde_excel_o_db(excel_data, pathname):
    """Carga centros desde Excel si esta disponible, sino desde BD"""
    if excel_data and 'consumo' in excel_data:
        # Cargar desde Excel
        df = pd.DataFrame(excel_data['consumo'])
        centros = obtener_centros_desde_excel(df)
        opciones = [{"label": "Todos", "value": "Todos"}] + [{"label": c, "value": c} for c in centros]
        return opciones, "Todos"
    else:
        # Cargar desde BD (comportamiento original)
        try:
            centros = obtener_centros()
            return [{"label": c, "value": c} for c in centros], None
        except Exception as e:
            logger.error(f"Cargando centros: {e}")
            return [], None


@callback(
    Output("filtro-almacen-demanda", "options"),
    Output("filtro-almacen-demanda", "value"),
    Input("filtro-centro-demanda", "value"),
    State("store-excel-data", "data")
)
def cargar_almacenes_desde_excel_o_db(centro_seleccionado, excel_data):
    """Carga almacenes desde Excel si esta disponible, sino desde BD"""
    if excel_data and 'consumo' in excel_data:
        # Cargar desde Excel
        df = pd.DataFrame(excel_data['consumo'])
        almacenes = obtener_almacenes_desde_excel(df, centro_seleccionado)
        opciones = [{"label": "Todos", "value": "Todos"}] + [{"label": a, "value": a} for a in almacenes]
        return opciones, "Todos"
    else:
        # Cargar desde BD (comportamiento original)
        try:
            almacenes = obtener_almacenes(centro_seleccionado)
            return [{"label": a, "value": a} for a in almacenes], None
        except Exception as e:
            logger.error(f"Cargando almacenes: {e}")
            return [], None


@callback(
    Output("select-material-demanda", "options"),
    Output("select-material-demanda", "style"),
    Input("store-excel-data", "data"),
    Input("data-store", "data")
)
def actualizar_lista_materiales(excel_data, data):
    """Actualiza la lista de materiales para seleccion desde Excel o data-store"""
    # Si hay datos de Excel, usarlos y mostrar dropdown
    if excel_data and 'consumo' in excel_data:
        df = pd.DataFrame(excel_data['consumo'])
        opciones = obtener_materiales_desde_excel(df)
        return opciones, {"display": "block", "marginTop": "8px"}

    # Si hay data-store (comportamiento legacy)
    if data and "records" in data:
        df = dict_a_df(data["records"])
        df_top = df.head(100).copy()
        df_top['descripcion_corta'] = df_top['descripcion'].str[:40] + '...'
        df_top['label'] = df_top['codigo'] + ' - ' + df_top['descripcion_corta']
        return df_top[['label', 'codigo']].rename(columns={'codigo': 'value'}).to_dict('records'), {"display": "none"}

    return [], {"display": "none"}


@callback(
    Output("input-codigo-sap", "value"),
    Input("select-material-demanda", "value"),
    prevent_initial_call=True
)
def sincronizar_material_seleccionado(material_seleccionado):
    """Sincroniza el material seleccionado del dropdown al campo de entrada"""
    if material_seleccionado:
        return material_seleccionado
    return no_update




@callback(
    Output("material-encontrado", "children"),
    Output("select-material-demanda", "value"),
    Input("btn-buscar-material", "n_clicks"),
    State("input-codigo-sap", "value"),
    State("data-store", "data"),
    State("store-excel-data", "data"),
    prevent_initial_call=True
)
def buscar_material_por_codigo(n_clicks, codigo_sap, data, excel_data):
    """Busca material en Excel o catalogo completo"""
    if not codigo_sap or len(str(codigo_sap).strip()) < 3:
        return html.Span([
            lucide_icon("info", size="sm"),
            " Ingrese al menos 3 caracteres"
        ], className="text-muted"), None

    codigo_buscar = str(codigo_sap).strip()

    # Primero buscar en Excel si esta disponible
    if excel_data and 'consumo' in excel_data:
        df = pd.DataFrame(excel_data['consumo'])
        # Buscar coincidencias (parciales o exactas)
        matches = df[df['codigo'].astype(str).str.contains(codigo_buscar, case=False, na=False)]

        if len(matches) > 0:
            codigo_encontrado = matches['codigo'].iloc[0]
            if 'descripcion' in matches.columns and pd.notna(matches['descripcion'].iloc[0]):
                descripcion = str(matches['descripcion'].iloc[0])[:45]
            else:
                descripcion = str(codigo_encontrado)

            n_coincidencias = matches['codigo'].nunique()
            if n_coincidencias == 1:
                return html.Span([
                    lucide_icon("check-circle", size="sm", style={"color": "#4CD964"}),
                    f" {descripcion}"
                ], className="text-success"), str(codigo_encontrado)
            else:
                return html.Span([
                    lucide_icon("info", size="sm", style={"color": "#007AFF"}),
                    f" {descripcion}... ({n_coincidencias} coincidencias)"
                ]), str(codigo_encontrado)

    # Fallback: buscar en BD si Excel no tiene datos
    try:
        resultados = buscar_material_con_mrp(codigo_buscar, limite=10)
        if len(resultados) > 0:
            mat = resultados[0]
            badge = dbc.Badge("MRP" if mat.get("tiene_mrp") else "Sin MRP",
                            color="success" if mat.get("tiene_mrp") else "secondary",
                            className="ms-2")
            return html.Span([
                lucide_icon("check-circle", size="sm"),
                f" {mat['descripcion'][:45]}",
                badge
            ]), mat["codigo"]
    except Exception as e:
        logger.debug(f"BD no disponible para busqueda: {e}")

    return html.Span([
        lucide_icon("alert-triangle", size="sm", style={"color": "#FF9500"}),
        " No encontrado"
    ], className="text-warning"), None


@callback(
    Output("grafico-forecast", "figure"),
    Output("grafico-patron-semanal", "figure"),
    Output("grafico-estacionalidad", "figure"),
    Output("grafico-feature-importance", "figure"),
    Output("tabla-predicciones", "rowData"),
    Output("metrica-mae", "children"),
    Output("metrica-rmse", "children"),
    Output("metrica-r2", "children"),
    Output("metrica-mape", "children"),
    Output("info-modelo-config", "children"),
    Output("forecast-precision-valor", "children"),
    Output("forecast-error-valor", "children"),
    Output("forecast-demanda-valor", "children"),
    Output("forecast-tendencia-valor", "children"),
    Output("forecast-resumen", "children"),
    Output("forecast-resumen", "style"),
    Input("btn-generar-forecast", "n_clicks"),
    State("input-codigo-sap", "value"),
    State("filtro-centro-demanda", "value"),
    State("filtro-almacen-demanda", "value"),
    State("select-modelo-ml", "value"),
    State("slider-horizonte", "value"),
    State("select-confianza", "value"),
    State("data-store", "data"),
    State("store-excel-data", "data"),
    prevent_initial_call=True
)
def generar_forecast(n_clicks, material, centro, almacen, modelo_tipo, horizonte, confianza, data, excel_data):
    """Genera el forecast usando ML"""
    # Figura vacia por defecto
    fig_vacia = crear_figura_vacia("Seleccione un material")

    valores_default = (
        fig_vacia, fig_vacia, fig_vacia, fig_vacia, [],
        "--", "--", "--", "--",
        html.P("Seleccione un material para generar el forecast", className="text-muted small"),
        "--", "--", "--", "--",
        "", {"display": "none"}
    )

    if not material:
        return valores_default

    # Determinar fuente de datos: Excel o Base de Datos
    if excel_data and 'consumo' in excel_data:
        # Usar datos del Excel cargado
        logger.info(f"Usando datos de Excel para material {material}")
        df_consumo_excel = pd.DataFrame(excel_data['consumo'])
        df_consumo_excel['fecha'] = pd.to_datetime(df_consumo_excel['fecha'])
        df_historico = filtrar_consumo_por_material(
            df_consumo_excel,
            material,
            centro if centro != "Todos" else None,
            almacen if almacen != "Todos" else None
        )
    else:
        # Cargar datos historicos REALES de consumo desde la base de datos
        df_historico = cargar_consumo_historico(material=material, centro=centro, dias=365)

        # Filtrar por almacen si se especifico
        if almacen and len(df_historico) > 0 and "almacen" in df_historico.columns:
            df_historico = df_historico[df_historico["almacen"] == almacen]

    # Validar que hay datos disponibles
    if len(df_historico) == 0:
        fig_sin_datos = crear_figura_warning("No hay datos historicos de consumo para este material")
        return (
            fig_sin_datos, fig_sin_datos, fig_sin_datos, fig_sin_datos, [],
            "--", "--", "--", "--",
            html.P("Sin datos historicos disponibles", className="text-warning small"),
            "--", "--", "--", "--",
            "", {"display": "none"}
        )

    # Agrupar consumo por dia
    df_historico = df_historico.groupby("fecha").agg({
        "codigo": "first",
        "cantidad": "sum"
    }).reset_index()

    # Entrenar modelo
    predictor = DemandPredictor(modelo=modelo_tipo)
    metrics = predictor.entrenar(df_historico, "cantidad")

    # Generar predicciones
    df_pred = predictor.predecir(df_historico, periodos=horizonte)

    # Ajustar intervalos segun confianza
    factor_confianza = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = factor_confianza.get(confianza, 1.96)

    std_pred = df_pred["prediccion"].std()
    df_pred["limite_inferior"] = df_pred["prediccion"] - z * std_pred * 0.3
    df_pred["limite_superior"] = df_pred["prediccion"] + z * std_pred * 0.3
    df_pred["limite_inferior"] = df_pred["limite_inferior"].clip(lower=0)

    # Calcular KPIs
    precision_valor = f"{metrics['r2']:.0%}" if metrics["r2"] > 0 else "N/A"
    error_valor = f"{metrics['mae']:.1f}"
    demanda_total = df_pred["prediccion"].sum()
    demanda_valor = f"{demanda_total:,.0f}"

    # Calcular tendencia
    promedio_historico = df_historico["cantidad"].mean()
    promedio_prediccion = df_pred["prediccion"].mean()
    if promedio_historico > 0:
        tendencia = ((promedio_prediccion - promedio_historico) / promedio_historico) * 100
        tendencia_valor = f"{tendencia:+.1f}%"
    else:
        tendencia_valor = "N/A"
        tendencia = 0

    # 1. Grafico principal de forecast
    fig_forecast = go.Figure()

    # Datos historicos
    fig_forecast.add_trace(go.Scatter(
        x=df_historico["fecha"],
        y=df_historico["cantidad"],
        name="Historico",
        line=dict(color=COLORS["text_secondary"], width=1),
        mode="lines"
    ))

    # Prediccion
    fig_forecast.add_trace(go.Scatter(
        x=df_pred["fecha"],
        y=df_pred["prediccion"],
        name="Prediccion",
        line=dict(color=COLORS["primary"], width=3),
        mode="lines"
    ))

    # Intervalo de confianza
    fig_forecast.add_trace(go.Scatter(
        x=pd.concat([df_pred["fecha"], df_pred["fecha"][::-1]]),
        y=pd.concat([df_pred["limite_superior"], df_pred["limite_inferior"][::-1]]),
        fill="toself",
        fillcolor=color_con_alpha('primary', 0.2),
        line=dict(color="rgba(255,255,255,0)"),
        name=f"Intervalo {int(confianza*100)}%",
        hoverinfo='skip'
    ))

    layout_base = {k: v for k, v in PLOTLY_TEMPLATE["layout"].items() if k not in ["legend", "margin"]}
    fig_forecast.update_layout(
        **layout_base,
        xaxis_title="Fecha",
        yaxis_title="Demanda",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)", font={"color": COLORS['text_secondary']}
        ),
        margin=dict(l=60, r=20, t=50, b=60),
        hovermode="x unified"
    )

    # 2. Patron semanal - MEJORADO VISUALMENTE
    df_historico["dia_semana"] = df_historico["fecha"].dt.day_name()
    patron_semanal = df_historico.groupby("dia_semana")["cantidad"].mean()

    dias_orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dias_es = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
    patron_semanal = patron_semanal.reindex(dias_orden)

    # Colores degradados para días
    colores_dias = [
        "#3b82f6",  # Lun - Azul
        "#06b6d4",  # Mar - Cyan
        "#10b981",  # Mie - Verde
        "#f59e0b",  # Jue - Amarillo
        "#ef4444",  # Vie - Rojo
        "#8b5cf6",  # Sab - Púrpura
        "#ec4899"   # Dom - Rosa
    ]

    fig_semanal = go.Figure()
    fig_semanal.add_trace(go.Bar(
        x=dias_es,
        y=patron_semanal.values,
        marker=dict(
            color=colores_dias,
            line=dict(color="rgba(255,255,255,0.3)", width=2),
            opacity=0.9
        ),
        text=[f"{v:.0f}" for v in patron_semanal.values],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text_primary"], family="-apple-system, BlinkMacSystemFont"),
        hovertemplate='<b style="font-size:13px">%{x}</b><br><b>Demanda:</b> %{y:,.0f}<extra></extra>',
        hoverinfo="x+y"
    ))
    layout_semanal = {k: v for k, v in PLOTLY_TEMPLATE["layout"].items() if k not in ["margin", "xaxis", "yaxis", "plot_bgcolor", "paper_bgcolor"]}
    fig_semanal.update_layout(
        **layout_semanal,
        xaxis=dict(
            title="",
            tickfont=dict(size=11, color=COLORS["text_secondary"], family="-apple-system, BlinkMacSystemFont"),
            gridcolor="rgba(0,0,0,0)",
            showgrid=False,
            linecolor=COLORS["border"],
            linewidth=1
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=10, color=COLORS["text_secondary"]),
            gridcolor=COLORS["grid_color"],
            gridwidth=0.5,
            zeroline=False
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode="x unified",
        showlegend=False
    )

    # 3. Estacionalidad mensual - MEJORADO VISUALMENTE
    df_historico["mes"] = df_historico["fecha"].dt.month
    patron_mensual = df_historico.groupby("mes")["cantidad"].mean()

    meses_es = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

    fig_estacional = go.Figure()
    fig_estacional.add_trace(go.Scatter(
        x=meses_es[:len(patron_mensual)],
        y=patron_mensual.values,
        mode="lines+markers",
        name="Demanda",
        line=dict(
            color="#10b981",
            width=3,
            shape="spline"
        ),
        marker=dict(
            size=10,
            color="#10b981",
            line=dict(color="rgba(255,255,255,0.5)", width=2),
            opacity=0.9,
            symbol="circle"
        ),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.15)",
        hovertemplate='<b style="font-size:13px">%{x}</b><br><b>Demanda:</b> %{y:,.0f}<extra></extra>',
        hoverinfo="x+y"
    ))
    layout_estacional = {k: v for k, v in PLOTLY_TEMPLATE["layout"].items() if k not in ["margin", "xaxis", "yaxis", "plot_bgcolor", "paper_bgcolor"]}
    fig_estacional.update_layout(
        **layout_estacional,
        xaxis=dict(
            title="",
            tickfont=dict(size=11, color=COLORS["text_secondary"], family="-apple-system, BlinkMacSystemFont"),
            gridcolor="rgba(0,0,0,0)",
            showgrid=False,
            linecolor=COLORS["border"],
            linewidth=1
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=10, color=COLORS["text_secondary"]),
            gridcolor=COLORS["grid_color"],
            gridwidth=0.5,
            zeroline=False
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode="x unified",
        showlegend=False
    )

    # 4. Importancia de features - MEJORADO VISUALMENTE
    importance = predictor.get_feature_importance()

    if len(importance) > 0:
        top_features = importance.head(8).sort_values("importance")
        
        # Gradiente de colores para features
        max_importance = top_features["importance"].max()
        colores_features = []
        for imp in top_features["importance"]:
            # Gradiente de azul claro a azul oscuro
            intensidad = imp / max_importance if max_importance > 0 else 0
            r = int(59 + (0 - 59) * intensidad)
            g = int(130 + (115 - 130) * intensidad)
            b = int(246 + (180 - 246) * intensidad)
            colores_features.append(f"rgb({r},{g},{b})")
        
        fig_importance = go.Figure()
        fig_importance.add_trace(go.Bar(
            y=top_features["feature"],
            x=top_features["importance"],
            orientation="h",
            marker=dict(
                color=colores_features,
                line=dict(color="rgba(255,255,255,0.3)", width=1),
                opacity=0.95
            ),
            text=[f"{v:.1%}" for v in top_features["importance"]],
            textposition="outside",
            textfont=dict(size=11, color=COLORS["text_primary"], family="-apple-system, BlinkMacSystemFont"),
            hovertemplate='<b style="font-size:12px">%{y}</b><br><b>Importancia:</b> %{x:.2%}<extra></extra>',
            hoverinfo="y+x"
        ))
        layout_importance = {k: v for k, v in PLOTLY_TEMPLATE["layout"].items() if k not in ["yaxis", "xaxis", "margin", "plot_bgcolor", "paper_bgcolor"]}
        fig_importance.update_layout(
            **layout_importance,
            xaxis=dict(
                title="",
                tickfont=dict(size=10, color=COLORS["text_secondary"]),
                gridcolor=COLORS["grid_color"],
                gridwidth=0.5,
                zeroline=False,
                linecolor=COLORS["border"],
                linewidth=1
            ),
            yaxis=dict(
                title="",
                tickfont=dict(size=11, color=COLORS["text_primary"], family="-apple-system, BlinkMacSystemFont"),
                autorange="reversed",
                gridcolor="rgba(0,0,0,0)",
                showgrid=False,
                linecolor=COLORS["border"],
                linewidth=1
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=120, r=40, t=20, b=40),
            hovermode="closest",
            showlegend=False
        )
    else:
        fig_importance = fig_vacia

    # Preparar datos para tabla
    df_pred["fecha"] = df_pred["fecha"].dt.strftime("%Y-%m-%d")
    df_pred["dia_semana"] = pd.to_datetime(df_pred["fecha"]).dt.day_name()

    dias_map = {
        "Monday": "Lun", "Tuesday": "Mar", "Wednesday": "Mie",
        "Thursday": "Jue", "Friday": "Vie", "Saturday": "Sab", "Sunday": "Dom"
    }
    df_pred["dia_semana"] = df_pred["dia_semana"].map(dias_map)

    datos_tabla = df_pred[["fecha", "prediccion", "limite_inferior",
                           "limite_superior", "dia_semana"]].to_dict("records")

    # Info del modelo
    info_modelo = html.Div([
        html.P([
            lucide_icon("bot", size="sm"),
            html.Strong("Modelo: "), MODELOS_ML.get(modelo_tipo, modelo_tipo)
        ], className="mb-2"),
        html.P([
            lucide_icon("calendar", size="sm"),
            html.Strong("Horizonte: "), f"{horizonte} dias"
        ], className="mb-2"),
        html.P([
            lucide_icon("target", size="sm"),
            html.Strong("Confianza: "), f"{int(confianza*100)}%"
        ], className="mb-2"),
        html.P([
            lucide_icon("building", size="sm"),
            html.Strong("Centro: "), centro if centro else "Todos"
        ], className="mb-2"),
        html.P([
            lucide_icon("warehouse", size="sm"),
            html.Strong("Almacen: "), almacen if almacen else "Todos"
        ], className="mb-2"),
        html.P([
            lucide_icon("database", size="sm"),
            html.Strong("Datos historicos: "), f"{len(df_historico)} registros"
        ], className="mb-0"),
    ])

    # Generar resumen del forecast
    tendencia_texto = "al alza" if tendencia > 5 else "a la baja" if tendencia < -5 else "estable"
    precision_texto = "alta" if metrics["r2"] > 0.7 else "moderada" if metrics["r2"] > 0.4 else "baja"

    resumen_children = html.Div([
        html.H6([
            lucide_icon("lightbulb", size="sm"),
            "Resumen del Analisis"
        ]),
        html.P([
            f"El modelo {MODELOS_ML.get(modelo_tipo, modelo_tipo)} predice una demanda total de ",
            html.Strong(f"{demanda_total:,.0f} unidades"),
            f" para los proximos {horizonte} dias, con un promedio diario de ",
            html.Strong(f"{promedio_prediccion:.1f} unidades"),
            f". La tendencia es {tendencia_texto} ({tendencia_valor}) respecto al periodo historico. ",
            f"La precision del modelo es {precision_texto} (R2: {metrics['r2']:.1%}) ",
            f"con un error promedio de {metrics['mae']:.1f} unidades."
        ])
    ])

    return (
        fig_forecast,
        fig_semanal,
        fig_estacional,
        fig_importance,
        datos_tabla,
        f"{metrics['mae']:.1f}",
        f"{metrics['rmse']:.1f}",
        f"{metrics['r2']:.2%}",
        f"{metrics['mape']:.1f}%",
        info_modelo,
        precision_valor,
        error_valor,
        demanda_valor,
        tendencia_valor,
        resumen_children,
        {"display": "block"}
    )
