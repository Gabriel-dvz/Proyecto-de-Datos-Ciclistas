import numpy as np
import pandas as pd
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

# Imputación de valores faltantes usando la media
datos['Cabello_imputado'] = datos['Cabello'].fillna(datos['Cabello'].mean())
datos['Alameda_imputado'] = datos['Alameda'].fillna(datos['Alameda'].mean())
datos['República de Chile con San Joaquín_imputado'] = datos['República de Chile con San Joaquín'].fillna(datos['República de Chile con San Joaquín'].mean())
datos['San Juan con Escrivá de Balaguer_imputado'] = datos['San Juan con Escrivá de Balaguer'].fillna(datos['San Juan con Escrivá de Balaguer'].mean())

# Calcular MAPE y MAD para Cabello
mape_cabello = mean_absolute_percentage_error(datos['Cabello'].dropna(), datos['Cabello_imputado'].dropna())
mad_cabello = mean_absolute_deviation(datos['Cabello'].dropna(), datos['Cabello_imputado'].dropna())
print(f"CABELLO - MAPE: {mape_cabello:.2f}%, MAD: {mad_cabello:.2f}")

# Calcular MAPE y MAD para Alameda
mape_alameda = mean_absolute_percentage_error(datos['Alameda'].dropna(), datos['Alameda_imputado'].dropna())
mad_alameda = mean_absolute_deviation(datos['Alameda'].dropna(), datos['Alameda_imputado'].dropna())
print(f"ALAMEDA - MAPE: {mape_alameda:.2f}%, MAD: {mad_alameda:.2f}")

# Calcular MAPE y MAD para República de Chile
mape_republica = mean_absolute_percentage_error(datos['República de Chile con San Joaquín'].dropna(), datos['República de Chile con San Joaquín_imputado'].dropna())
mad_republica = mean_absolute_deviation(datos['República de Chile con San Joaquín'].dropna(), datos['República de Chile con San Joaquín_imputado'].dropna())
print(f"REPÚBLICA DE CHILE - MAPE: {mape_republica:.2f}%, MAD: {mad_republica:.2f}")

# Calcular MAPE y MAD para San Juan
mape_sanjuan = mean_absolute_percentage_error(datos['San Juan con Escrivá de Balaguer'].dropna(), datos['San Juan con Escrivá de Balaguer_imputado'].dropna())
mad_sanjuan = mean_absolute_deviation(datos['San Juan con Escrivá de Balaguer'].dropna(), datos['San Juan con Escrivá de Balaguer_imputado'].dropna())
print(f"SAN JUAN - MAPE: {mape_sanjuan:.2f}%, MAD: {mad_sanjuan:.2f}")

# Gráficos
fig, axs = plt.subplots(4, figsize=(10, 12), sharex=True)

# Gráfico de Cabello
axs[0].plot(datos['Time'], datos['Cabello'], label='Cabello (original)', linestyle='-', color='blue')
axs[0].plot(datos['Time'], datos['Cabello_imputado'], label='Cabello (imputado)', linestyle='-', color='orange')
axs[0].set_ylabel('Cabello')
axs[0].legend()

# Gráfico de Alameda
axs[1].plot(datos['Time'], datos['Alameda'], label='Alameda (original)', linestyle='-', color='green')
axs[1].plot(datos['Time'], datos['Alameda_imputado'], label='Alameda (imputado)', linestyle='-', color='red')
axs[1].set_ylabel('Alameda')
axs[1].legend()

# Gráfico de República de Chile
axs[2].plot(datos['Time'], datos['República de Chile con San Joaquín'], label='República de Chile (original)',  linestyle='-', color='purple')
axs[2].plot(datos['Time'], datos['República de Chile con San Joaquín_imputado'], label='República de Chile (imputado)', linestyle='-', color='brown')
axs[2].set_ylabel('República de Chile')
axs[2].legend()

# Gráfico de San Juan
axs[3].plot(datos['Time'], datos['San Juan con Escrivá de Balaguer'], label='San Juan (original)',  linestyle='-', color='cyan')
axs[3].plot(datos['Time'], datos['San Juan con Escrivá de Balaguer_imputado'], label='San Juan (imputado)', linestyle='-', color='magenta')
axs[3].set_ylabel('San Juan')
axs[3].legend()

axs[3].set_xlabel('Fecha')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
