import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Funciones para calcular MAPE y MAD
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_deviation(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Cargar el archivo CSV en 'datos'
datos = pd.read_csv("bike_rancagua.csv", sep=",")

# Convertir la columna 'Time' a formato datetime
datos['Time'] = pd.to_datetime(datos['Time'])

# Inicializar el modelo de regresión lineal
modelo = LinearRegression()

# CABELLO
# Dividir los datos en características (X) y la variable a predecir (y)
X = datos[['Alameda', 'República de Chile con San Joaquín', 'San Juan con Escrivá de Balaguer']].copy()
y = datos['Cabello'].copy()

# Reemplazar valores nulos en X e y con la media
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Entrenar el modelo con los datos
modelo.fit(X, y)

# Predecir los valores faltantes en 'Cabello' basados en las otras zonas
datos['Cabello_predicho'] = modelo.predict(X)

# Calcular MAPE y MAD para Cabello
mape_cabello = mean_absolute_percentage_error(y, datos['Cabello_predicho'])
mad_cabello = mean_absolute_deviation(y, datos['Cabello_predicho'])
print(f"CABELLO - MAPE: {mape_cabello:.2f}%, MAD: {mad_cabello:.2f}")

# ALAMEDA
# Dividir los datos en características (X) y la variable a predecir (y)
X2 = datos[['Cabello', 'República de Chile con San Joaquín', 'San Juan con Escrivá de Balaguer']].copy()
y2 = datos['Alameda'].copy()

# Reemplazar valores nulos en X e y con la media
X2.fillna(X2.mean(), inplace=True)
y2.fillna(y2.mean(), inplace=True)

# Entrenar el modelo con los datos
modelo.fit(X2, y2)

# Predecir los valores faltantes en 'Alameda' basados en las otras zonas
datos['Alameda_predicho'] = modelo.predict(X2)

# Calcular MAPE y MAD para Alameda
mape_alameda = mean_absolute_percentage_error(y2, datos['Alameda_predicho'])
mad_alameda = mean_absolute_deviation(y2, datos['Alameda_predicho'])
print(f"ALAMEDA - MAPE: {mape_alameda:.2f}%, MAD: {mad_alameda:.2f}")

# REPÚBLICA DE CHILE
# Dividir los datos en características (X) y la variable a predecir (y)
X3 = datos[['Alameda', 'Cabello', 'San Juan con Escrivá de Balaguer']].copy()
y3 = datos['República de Chile con San Joaquín'].copy()

# Reemplazar valores nulos en X e y con la media
X3.fillna(X3.mean(), inplace=True)
y3.fillna(y3.mean(), inplace=True)

# Entrenar el modelo con los datos
modelo.fit(X3, y3)

# Predecir los valores faltantes en 'República de Chile' basados en las otras zonas
datos['República de Chile_predicho'] = modelo.predict(X3)

# Calcular MAPE y MAD para República de Chile
mape_republica = mean_absolute_percentage_error(y3, datos['República de Chile_predicho'])
mad_republica = mean_absolute_deviation(y3, datos['República de Chile_predicho'])
print(f"REPÚBLICA DE CHILE - MAPE: {mape_republica:.2f}%, MAD: {mad_republica:.2f}")

# SAN JUAN
# Dividir los datos en características (X) y la variable a predecir (y)
X4 = datos[['Alameda', 'Cabello', 'República de Chile con San Joaquín']].copy()
y4 = datos['San Juan con Escrivá de Balaguer'].copy()

# Reemplazar valores nulos en X e y con la media
X4.fillna(X4.mean(), inplace=True)
y4.fillna(y4.mean(), inplace=True)

# Entrenar el modelo con los datos
modelo.fit(X4, y4)

# Predecir los valores faltantes en 'San Juan' basados en las otras zonas
datos['San Juan_predicho'] = modelo.predict(X4)

# Calcular MAPE y MAD para San Juan
mape_sanjuan = mean_absolute_percentage_error(y4, datos['San Juan_predicho'])
mad_sanjuan = mean_absolute_deviation(y4, datos['San Juan_predicho'])
print(f"SAN JUAN - MAPE: {mape_sanjuan:.2f}%, MAD: {mad_sanjuan:.2f}")

# Gráficos
fig, axs = plt.subplots(4, figsize=(10, 12), sharex=True)

# Gráfico de Cabello
axs[0].plot(datos['Time'], datos['Cabello'], label='Cabello (original)', linestyle='-', color='blue')
axs[0].plot(datos['Time'], datos['Cabello_predicho'], label='Cabello (predicho)', linestyle='-', color='orange')
axs[0].set_ylabel('Cabello')
axs[0].legend()

# Gráfico de Alameda
axs[1].plot(datos['Time'], datos['Alameda'], label='Alameda (original)', linestyle='-', color='green')
axs[1].plot(datos['Time'], datos['Alameda_predicho'], label='Alameda (predicho)', linestyle='-', color='red')
axs[1].set_ylabel('Alameda')
axs[1].legend()

# Gráfico de República de Chile
axs[2].plot(datos['Time'], datos['República de Chile con San Joaquín'], label='República de Chile (original)',  linestyle='-', color='purple')
axs[2].plot(datos['Time'], datos['República de Chile_predicho'], label='República de Chile (predicho)', linestyle='-', color='brown')
axs[2].set_ylabel('República de Chile')
axs[2].legend()

# Gráfico de San Juan
axs[3].plot(datos['Time'], datos['San Juan con Escrivá de Balaguer'], label='San Juan (original)',  linestyle='-', color='cyan')
axs[3].plot(datos['Time'], datos['San Juan_predicho'], label='San Juan (predicho)', linestyle='-', color='magenta')
axs[3].set_ylabel('San Juan')
axs[3].legend()

axs[3].set_xlabel('Fecha')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
