# Actividad 3 — Modelo de aprendizaje supervisado

Sistema de transporte masivo: predicción del tiempo de recorrido entre estaciones con un **Random Forest Regressor**.

## Descripción

Este proyecto implementa un modelo de **aprendizaje supervisado** para estimar el tiempo (en minutos) de un recorrido en un sistema de transporte masivo.

Cada registro representa un viaje entre dos estaciones (numeradas del 1 al 8). El modelo aprende a partir de características del recorrido y predice `tiempo_min`.

El dataset es **sintético**: se construye en el propio script porque no se dispone de una fuente real con todas las variables necesarias.

## Archivos

| Archivo | Contenido |
| --- | --- |
| `modelo_supervisado.py` | Dataset, entrenamiento, evaluación y predicción de un nuevo recorrido |
| `README.md` | Este documento |

## Variables del dataset

**Entradas (X)**

| Variable | Significado |
| --- | --- |
| `origen` | Estación de origen (1–8) |
| `destino` | Estación de destino (1–8) |
| `distancia_km` | Distancia aproximada del recorrido |
| `hora` | Hora del viaje |
| `demanda` | Demanda estimada de pasajeros |
| `transbordos` | Número de transbordos |

**Salida (y)**

| Variable | Significado |
| --- | --- |
| `tiempo_min` | Tiempo del recorrido en minutos (variable a predecir) |

El dataset tiene **20 registros**.

## Flujo del programa

1. Crear el dataset en un `DataFrame` de pandas.
2. Mostrar el dataset en consola.
3. Separar características (`X`) y variable objetivo (`y`).
4. Dividir los datos: **80 %** entrenamiento y **20 %** prueba (`random_state=42`).
5. Crear un `RandomForestRegressor` con 100 árboles.
6. Entrenar el modelo con los datos de entrenamiento.
7. Predecir sobre el conjunto de prueba.
8. Evaluar con **MAE** (error medio absoluto) y **R²**.
9. Estimar el tiempo de un recorrido nuevo (origen 1 → destino 6).

Se usa **regresión** porque el objetivo es un valor numérico continuo (minutos), no una clase.

## Requisitos

- Python 3
- pandas
- scikit-learn

Instalación:

```bash
pip install pandas scikit-learn
```

## Cómo ejecutar

Desde esta carpeta:

```bash
python modelo_supervisado.py
```

En Windows, si hay más de un intérprete, conviene usar el mismo Python del editor:

```bash
python modelo_supervisado.py
```

## Salida esperada

El programa imprime tres bloques:

1. **Dataset de transporte masivo** — las 20 filas.
2. **Resultados del modelo** — MAE en minutos y coeficiente R².
3. **Nueva predicción** — tiempo estimado del recorrido 1 → 6 (3.5 km, hora 8, demanda 100, 1 transbordo).

Con la semilla `random_state=42`, una ejecución típica produce aproximadamente:

- Error medio absoluto: **1.61 minutos**
- Coeficiente R²: **0.25**
- Tiempo estimado del nuevo recorrido: **10.51 minutos**

El R² es moderado porque el conjunto es pequeño (20 filas) y el 20 % de prueba deja pocos ejemplos para evaluar. El ejercicio ilustra el pipeline de aprendizaje supervisado, no un modelo de producción.

## Dependencias del código

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
```
