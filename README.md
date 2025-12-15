# Forecast - Prediccion de Demanda con ML

Aplicacion Dash para forecasting de demanda de materiales utilizando Machine Learning.

## Caracteristicas

- **Forecast Individual**: Prediccion de demanda para un material especifico con analisis de patrones
- **Forecast Masivo**: Procesamiento batch de multiples materiales simultaneamente
- **Modelos ML**: Random Forest, Gradient Boosting, Regresion Lineal
- **Visualizaciones**: Graficos interactivos con intervalos de confianza
- **Exportacion**: PDF y Excel

## Estructura del Proyecto

```
Forecast/
├── app.py                      # Aplicacion principal
├── requirements.txt            # Dependencias
├── assets/
│   └── styles.css              # Estilos glassmorphism
└── src/
    ├── callbacks/              # Callbacks de Dash
    │   ├── demanda_callbacks.py
    │   └── forecast_masivo_callbacks.py
    ├── components/             # Componentes UI
    │   └── icons/              # Iconos Lucide
    ├── data/                   # Capa de datos
    │   ├── database.py
    │   ├── connection_pool.py
    │   └── sap_loader.py
    ├── layouts/                # Layouts reutilizables
    │   └── components.py
    ├── ml/                     # Modelos ML
    │   ├── predictor.py
    │   ├── models.py
    │   └── training_pipeline.py
    ├── pages/                  # Paginas de la app
    │   ├── demanda.py
    │   └── forecast_masivo.py
    └── utils/                  # Utilidades
        ├── formatters.py
        ├── validators.py
        └── logger.py
```

## Instalacion

```bash
# Clonar repositorio
git clone https://github.com/manuelremon/ForecastDemandaMateriales.git
cd ForecastDemandaMateriales

# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python app.py
```

## Uso

La aplicacion estara disponible en `http://127.0.0.1:8051/`

### Rutas

| Ruta | Descripcion |
|------|-------------|
| `/` | Forecast Individual - Analisis de un material |
| `/masivo` | Forecast Masivo - Procesamiento batch |

### Modelos Disponibles

| Modelo | Descripcion |
|--------|-------------|
| Random Forest | Ensemble de arboles de decision |
| Gradient Boosting | Boosting con arboles de decision |
| Regresion Lineal | Modelo lineal simple |

## Tecnologias

- **Framework**: Dash + Plotly
- **UI**: Dash Bootstrap Components, AG Grid
- **ML**: Scikit-learn
- **Database**: SQLAlchemy
- **Estilos**: Glassmorphism iOS Design

## Licencia

MIT
