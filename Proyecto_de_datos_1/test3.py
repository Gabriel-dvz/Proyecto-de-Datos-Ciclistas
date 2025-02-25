import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV en 'datos'
datos = pd.read_csv("bike_rancagua.csv", sep=",")

# Convertir la columna 'Time' a formato datetime
datos['Time'] = pd.to_datetime(datos['Time'])
# Dividir los datos en características (X) y la variable a predecir (y)
X = datos[['Alameda', 'Cabello', 'San Juan con Escrivá de Balaguer']].copy()
y = datos['República de Chile con San Joaquín'].copy()

# Reemplazar valores nulos en X e y con la media de la columna correspondiente
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Inicializar el modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos
modelo.fit(X, y)

# Predecir los valores faltantes en 'Cabello' basados en las otras zonas
datos_completos = datos.copy()
datos_completos['República de Chile_predicho'] = modelo.predict(X)

# Mostrar los datos originales y los datos completados
print("Datos originales:")
print(datos_completos[['Time', 'República de Chile con San Joaquín', 'República de Chile_predicho']])

# Graficar los datos originales y los datos completados
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(datos_completos['Time'], datos_completos['República de Chile con San Joaquín'], label='Rep.Chile (original)', marker='o')
ax.plot(datos_completos['Time'], datos_completos['República de Chile_predicho'], label='Rep.Chile (predicho)', linestyle='--')

ax.set_xlabel('Fecha')
ax.set_ylabel('Cantidad de personas')
ax.set_title('Regresión lineal para completar datos faltantes en Rep.Chile')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()