import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV en 'datos'
datos = pd.read_csv("bike_rancagua.csv", sep=",")

# Convertir la columna 'Time' a formato datetime
datos['Time'] = pd.to_datetime(datos['Time'])

# Reemplazar los valores NaN con cero
datos.fillna(0, inplace=True)

# Mostrar los datos con valores NaN igualados a cero
print(datos)

# Visualizar los datos de las cuatro calles
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Cabello
ax.plot(datos['Time'], datos['Cabello'], label='Cabello', linestyle='-')

# Plot Alameda
ax.plot(datos['Time'], datos['Alameda'], label='Alameda',  linestyle='-')

# Plot República de Chile con San Joaquín
ax.plot(datos['Time'], datos['República de Chile con San Joaquín'], label='República de Chile con San Joaquín',  linestyle='-')

# Plot San Juan con Escrivá de Balaguer
ax.plot(datos['Time'], datos['San Juan con Escrivá de Balaguer'], label='San Juan con Escrivá de Balaguer',  linestyle='-')

ax.set_xlabel('Fecha')
ax.set_ylabel('Cantidad de personas')
ax.set_title('Cantidad de bicicletas en Rancagua por calle')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
