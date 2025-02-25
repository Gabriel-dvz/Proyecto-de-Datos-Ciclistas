import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Funciones para calcular MAPE y MAD
def mean_absolute_percentage_error(y_true, y_pred):
    mask = ~np.isnan(y_true)
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def mean_absolute_deviation(y_true, y_pred):
    mask = ~np.isnan(y_true)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))

# Cargar el archivo CSV en 'datos'
datos = pd.read_csv("bike_rancagua.csv", sep=",")

# Convertir la columna 'Time' a formato datetime
datos['Time'] = pd.to_datetime(datos['Time'])

# Guardar los valores originales
cabello_original = datos['Cabello'].copy()
alameda_original = datos['Alameda'].copy()
republica_original = datos['República de Chile con San Joaquín'].copy()
sanjuan_original = datos['San Juan con Escrivá de Balaguer'].copy()

# Calcular la media de toda la serie temporal para cada columna
mean_cabello = datos['Cabello'].mean()
mean_alameda = datos['Alameda'].mean()
mean_republica = datos['República de Chile con San Joaquín'].mean()
mean_sanjuan = datos['San Juan con Escrivá de Balaguer'].mean()

# Imputación de valores faltantes usando la media de toda la serie temporal
datos['Cabello_imputado'] = datos['Cabello'].fillna(mean_cabello)
datos['Alameda_imputado'] = datos['Alameda'].fillna(mean_alameda)
datos['República de Chile con San Joaquín_imputado'] = datos['República de Chile con San Joaquín'].fillna(mean_republica)
datos['San Juan con Escrivá de Balaguer_imputado'] = datos['San Juan con Escrivá de Balaguer'].fillna(mean_sanjuan)

# Calcular MAPE y MAD para los valores donde había datos faltantes
cabello_mask = datos['Cabello'].isna()
mape_cabello = mean_absolute_percentage_error(cabello_original[cabello_mask], datos['Cabello_imputado'][cabello_mask])
mad_cabello = mean_absolute_deviation(cabello_original[cabello_mask], datos['Cabello_imputado'][cabello_mask])
print(f"CABELLO - MAPE: {mape_cabello:.2f}%, MAD: {mad_cabello:.2f}")

alameda_mask = datos['Alameda'].isna()
mape_alameda = mean_absolute_percentage_error(alameda_original[alameda_mask], datos['Alameda_imputado'][alameda_mask])
mad_alameda = mean_absolute_deviation(alameda_original[alameda_mask], datos['Alameda_imputado'][alameda_mask])
print(f"ALAMEDA - MAPE: {mape_alameda:.2f}%, MAD: {mad_alameda:.2f}")

republica_mask = datos['República de Chile con San Joaquín'].isna()
mape_republica = mean_absolute_percentage_error(republica_original[republica_mask], datos['República de Chile con San Joaquín_imputado'][republica_mask])
mad_republica = mean_absolute_deviation(republica_original[republica_mask], datos['República de Chile con San Joaquín_imputado'][republica_mask])
print(f"REPÚBLICA DE CHILE - MAPE: {mape_republica:.2f}%, MAD: {mad_republica:.2f}")

sanjuan_mask = datos['San Juan con Escrivá de Balaguer'].isna()
mape_sanjuan = mean_absolute_percentage_error(sanjuan_original[sanjuan_mask], datos['San Juan con Escrivá de Balaguer_imputado'][sanjuan_mask])
mad_sanjuan = mean_absolute_deviation(sanjuan_original[sanjuan_mask], datos['San Juan con Escrivá de Balaguer_imputado'][sanjuan_mask])
print(f"SAN JUAN - MAPE: {mape_sanjuan:.2f}%, MAD: {mad_sanjuan:.2f}")

# Gráficos
fig, axs = plt.subplots(4, figsize=(10, 12), sharex=True)

# Gráfico de Cabello
axs[0].plot(datos['Time'], cabello_original, label='Cabello (original)', linestyle='-', color='blue')
axs[0].plot(datos['Time'], datos['Cabello_imputado'], label='Cabello (imputado)', linestyle='-', color='orange')
axs[0].set_ylabel('Cabello')
axs[0].legend()

# Gráfico de Alameda
axs[1].plot(datos['Time'], alameda_original, label='Alameda (original)', linestyle='-', color='green')
axs[1].plot(datos['Time'], datos['Alameda_imputado'], label='Alameda (imputado)', linestyle='-', color='red')
axs[1].set_ylabel('Alameda')
axs[1].legend()

# Gráfico de República de Chile
axs[2].plot(datos['Time'], republica_original, label='República de Chile (original)',  linestyle='-', color='purple')
axs[2].plot(datos['Time'], datos['República de Chile con San Joaquín_imputado'], label='República de Chile (imputado)', linestyle='-', color='brown')
axs[2].set_ylabel('República de Chile')
axs[2].legend()

# Gráfico de San Juan
axs[3].plot(datos['Time'], sanjuan_original, label='San Juan (original)',  linestyle='-', color='cyan')
axs[3].plot(datos['Time'], datos['San Juan con Escrivá de Balaguer_imputado'], label='San Juan (imputado)', linestyle='-', color='magenta')
axs[3].set_ylabel('San Juan')
axs[3].legend()

axs[3].set_xlabel('Fecha')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
