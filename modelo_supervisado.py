# ============================================================
# MODELO DE APRENDIZAJE SUPERVISADO
# SISTEMA DE TRANSPORTE MASIVO
# ============================================================

# Pandas permite crear y trabajar con el dataset
import pandas as pd

# Herramientas de Scikit-learn para dividir los datos,
# crear el modelo y evaluar sus resultados
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# 1. CREACIÓN DEL DATASET
# ============================================================

# Se crea un dataset sintético debido a que no contamos
# con una fuente real que tenga todas las variables necesarias.
#
# Cada fila representa un recorrido entre dos estaciones.
#
# Las estaciones se representan mediante números del 1 al 8.

datos = {
    "origen": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
               6, 6, 7, 7, 1, 2, 3, 4, 5, 6],

    "destino": [2, 7, 3, 1, 4, 2, 5, 3, 6, 4,
                8, 5, 8, 1, 7, 3, 5, 6, 8, 8],

    # Distancia aproximada del recorrido en kilómetros
    "distancia_km": [1.2, 2.1, 1.0, 1.2, 1.3, 1.0,
                     1.8, 1.3, 1.1, 1.8, 2.5, 1.1,
                     3.0, 3.0, 3.2, 2.0, 2.4, 2.1,
                     3.1, 2.5],

    # Hora en la que se realiza el recorrido
    "hora": [7, 7, 8, 8, 8, 8, 9, 9, 9, 9,
             10, 10, 10, 10, 11, 11, 12, 12, 13, 13],

    # Demanda estimada de pasajeros
    "demanda": [80, 65, 90, 70, 110, 85, 70, 95, 60, 75,
                75, 80, 55, 50, 100, 90, 120, 85, 65, 70],

    # Número de transbordos realizados
    "transbordos": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 1, 1, 1, 1],

    # Esta es la variable que queremos predecir
    "tiempo_min": [5, 8, 4, 5, 5, 4, 6, 5, 4, 6,
                   12, 4, 15, 15, 8, 4, 11, 10, 17, 12]
}


# Convertimos el diccionario en un DataFrame de Pandas
df = pd.DataFrame(datos)


# ============================================================
# 2. MOSTRAR EL DATASET
# ============================================================

print("============================================")
print("       DATASET DE TRANSPORTE MASIVO")
print("============================================")

print(df)


# ============================================================
# 3. DEFINIR ENTRADAS Y VARIABLE OBJETIVO
# ============================================================

# X contiene las características que el modelo utilizará
# para realizar sus predicciones.

X = df[
    [
        "origen",
        "destino",
        "distancia_km",
        "hora",
        "demanda",
        "transbordos"
    ]
]


# y contiene el resultado que queremos que el modelo aprenda
# a predecir: el tiempo del recorrido.

y = df["tiempo_min"]


# ============================================================
# 4. DIVIDIR LOS DATOS
# ============================================================

# Separamos el dataset en:
#
# 80% aproximadamente para entrenamiento
# 20% aproximadamente para prueba
#
# random_state=42 permite obtener la misma división cada vez
# que ejecutamos el programa.

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 5. CREAR EL MODELO
# ============================================================

# Random Forest Regressor se utiliza porque queremos
# predecir un valor numérico: el tiempo en minutos.
#
# n_estimators=100 significa que el bosque estará formado
# por 100 árboles de decisión.

modelo = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 6. ENTRENAR EL MODELO
# ============================================================

# Aquí ocurre el aprendizaje supervisado.
#
# El modelo recibe:
#
# X_entrenamiento -> características de los recorridos
# y_entrenamiento -> tiempos reales conocidos
#
# Con estos ejemplos el algoritmo aprende patrones.

modelo.fit(
    X_entrenamiento,
    y_entrenamiento
)


# ============================================================
# 7. REALIZAR PREDICCIONES
# ============================================================

# Utilizamos los datos de prueba para comprobar
# qué tan bien funciona el modelo.

predicciones = modelo.predict(X_prueba)


# ============================================================
# 8. EVALUAR EL MODELO
# ============================================================

# MAE calcula el error absoluto promedio entre el
# tiempo real y el tiempo predicho.

error = mean_absolute_error(
    y_prueba,
    predicciones
)


# R2 mide qué tan bien el modelo explica la variación
# de los valores de tiempo.

precision = r2_score(
    y_prueba,
    predicciones
)


print("\n============================================")
print("          RESULTADOS DEL MODELO")
print("============================================")

print(
    "Error medio absoluto:",
    round(error, 2),
    "minutos"
)

print(
    "Coeficiente R²:",
    round(precision, 2)
)


# ============================================================
# 9. REALIZAR UNA NUEVA PREDICCIÓN
# ============================================================

# Creamos un nuevo recorrido que el modelo no recibe
# como respuesta conocida.
#
# El objetivo es estimar cuánto tiempo podría tardar.

nuevo_recorrido = pd.DataFrame({
    "origen": [1],
    "destino": [6],
    "distancia_km": [3.5],
    "hora": [8],
    "demanda": [100],
    "transbordos": [1]
})


# El modelo utiliza lo aprendido durante el entrenamiento
# para estimar el tiempo del nuevo recorrido.

tiempo_estimado = modelo.predict(
    nuevo_recorrido
)


# ============================================================
# 10. MOSTRAR LA PREDICCIÓN
# ============================================================

print("\n============================================")
print("          NUEVA PREDICCIÓN")
print("============================================")

print(
    "Tiempo estimado del recorrido:",
    round(tiempo_estimado[0], 2),
    "minutos"
)