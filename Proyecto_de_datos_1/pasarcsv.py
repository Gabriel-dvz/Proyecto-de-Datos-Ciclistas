import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV en 'datos'
datos = pd.read_csv("bike_rancagua.csv", sep=",")

# Convertir la columna 'Time' a formato datetime
datos['Time'] = pd.to_datetime(datos['Time'])

##############################

# Dividir los datos en características (X) y la variable a predecir (y)
X = datos[['Alameda', 'República de Chile con San Joaquín', 'San Juan con Escrivá de Balaguer']].copy()
y = datos['Cabello'].copy()

X2 = datos[['Cabello', 'República de Chile con San Joaquín', 'San Juan con Escrivá de Balaguer']].copy()
y2 = datos['Alameda'].copy()

X3 = datos[['Cabello', 'Alameda', 'San Juan con Escrivá de Balaguer']].copy()
y3 = datos['República de Chile con San Joaquín'].copy()

X4 = datos[['Cabello', 'Alameda', 'República de Chile con San Joaquín']].copy()
y4 = datos['San Juan con Escrivá de Balaguer'].copy()

###############################

# Reemplazar valores nulos en X e y con la media de la columna correspondiente
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

X2.fillna(X2.mean(), inplace=True)
y2.fillna(y2.mean(), inplace=True)

X3.fillna(X3.mean(), inplace=True)
y3.fillna(y3.mean(), inplace=True)

X4.fillna(X4.mean(), inplace=True)
y4.fillna(y4.mean(), inplace=True)

################################

# Inicializar el modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con los datos
modelo.fit(X, y)

modelo.fit(X2, y2)

modelo.fit(X3, y3)

modelo.fit(X4, y4)

#################

# Predecir los valores faltantes en 'Cabello' basados en las otras zonas
datos_completos = datos.copy()
datos_completos['Cabello_predicho'] = modelo.predict(X)

# Predecir los valores faltantes en 'Cabello' basados en las otras zonas
datos_completos2 = datos.copy()
datos_completos2['Alameda_predicho'] = modelo.predict(X)

# Predecir los valores faltantes en 'Cabello' basados en las otras zonas
datos_completos3 = datos.copy()
datos_completos3['República_predicho'] = modelo.predict(X)

# Predecir los valores faltantes en 'Cabello' basados en las otras zonas
datos_completos4 = datos.copy()
datos_completos4['SanJuan_predicho'] = modelo.predict(X)

################

# Mostrar los datos originales y los datos completados
print("Datos originales:")
print(datos_completos[['Time', 'Cabello', 'Cabello_predicho']])

print("Datos originales2:")
print(datos_completos2[['Time', 'Alameda', 'Alameda_predicho']])

print("Datos originales3:")
print(datos_completos3[['Time', 'República de Chile con San Joaquín', 'República_predicho']])

print("Datos originales4:")
print(datos_completos4[['Time', 'San Juan con Escrivá de Balaguer', 'SanJuan_predicho']])

# Graficar los datos originales y los datos completados
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(datos_completos['Time'], datos_completos['Cabello'], label='Cabello (original)', marker='o')
ax.plot(datos_completos['Time'], datos_completos['Cabello_predicho'], label='Cabello (predicho)', linestyle='--')

ax.set_xlabel('Fecha')
ax.set_ylabel('Cantidad de personas')
ax.set_title('Regresión lineal para completar datos faltantes en Cabello')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


##############

# Graficar los datos originales y los datos completados2
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(datos_completos2['Time'], datos_completos2['Alameda'], label='Alameda (original)', marker='o')
ax.plot(datos_completos2['Time'], datos_completos2['Alameda_predicho'], label='Alameda (predicho)', linestyle='--')

ax.set_xlabel('Fecha')
ax.set_ylabel('Cantidad de personas')
ax.set_title('Regresión lineal para completar datos faltantes en Cabello')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()