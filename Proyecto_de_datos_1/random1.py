import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

# Reemplazar valores nulos en X e y con cero
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Entrenar el modelo con los datos
modelo.fit(X, y)

# Predecir los valores faltantes en 'Cabello' basados en las otras zonas
datos['Cabello_predicho'] = modelo.predict(X)

# ALAMEDA
# Dividir los datos en características (X) y la variable a predecir (y)
X2 = datos[['Cabello', 'República de Chile con San Joaquín', 'San Juan con Escrivá de Balaguer']].copy()
y2 = datos['Alameda'].copy()

# Reemplazar valores nulos en X e y con cero
X2.fillna(X2.mean(), inplace=True)
y2.fillna(y2.mean(), inplace=True)

# Entrenar el modelo con los datos
modelo.fit(X2, y2)

# Predecir los valores faltantes en 'Alameda' basados en las otras zonas
datos['Alameda_predicho'] = modelo.predict(X2)

# REPÚBLICA DE CHILE

# Filtrar los datos para que comiencen desde el 30 de abril de 2020
datos_rep_chile = datos[datos['Time'] >= '2020-04-30']

# Dividir los datos en características (X) y la variable a predecir (y)
X3 = datos_rep_chile[['Alameda', 'Cabello', 'San Juan con Escrivá de Balaguer']].copy()
y3 = datos_rep_chile['República de Chile con San Joaquín'].copy()

# Reemplazar valores nulos en X e y con cero
X3.fillna(X3.mean(), inplace=True)
y3.fillna(y3.mean(), inplace=True)

# Entrenar el modelo con los datos
modelo.fit(X3, y3)

# Predecir los valores faltantes en 'República de Chile' basados en las otras zonas
datos_rep_chile['República de Chile_predicho'] = modelo.predict(X3)

# SAN JUAN

# Filtrar los datos para que comiencen desde el 30 de abril de 2020
datos_san_juan = datos[datos['Time'] >= '2020-04-30']

# Dividir los datos en características (X) y la variable a predecir (y)
X4 = datos_san_juan[['Alameda', 'Cabello', 'República de Chile con San Joaquín']].copy()
y4 = datos_san_juan['San Juan con Escrivá de Balaguer'].copy()

# Reemplazar valores nulos en X e y con cero
X4.fillna(X4.mean(), inplace=True)
y4.fillna(y4.mean(), inplace=True)

# Entrenar el modelo con los datos
modelo.fit(X4, y4)

# Predecir los valores faltantes en 'San Juan' basados en las otras zonas
datos_san_juan['San Juan_predicho'] = modelo.predict(X4)

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
axs[2].plot(datos_rep_chile['Time'], datos_rep_chile['República de Chile con San Joaquín'], label='República de Chile (original)',  linestyle='-', color='purple')
axs[2].plot(datos_rep_chile['Time'], datos_rep_chile['República de Chile_predicho'], label='República de Chile (predicho)', linestyle='-', color='brown')
axs[2].set_ylabel('República de Chile')
axs[2].legend()

# Gráfico de San Juan
axs[3].plot(datos_san_juan['Time'], datos_san_juan['San Juan con Escrivá de Balaguer'], label='San Juan (original)',  linestyle='-', color='cyan')
axs[3].plot(datos_san_juan['Time'], datos_san_juan['San Juan_predicho'], label='San Juan (predicho)', linestyle='-', color='magenta')
axs[3].set_ylabel('San Juan')
axs[3].legend()

axs[3].set_xlabel('Fecha')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
