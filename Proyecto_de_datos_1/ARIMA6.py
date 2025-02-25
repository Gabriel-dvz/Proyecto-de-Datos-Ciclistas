import numpy as np
import pandas as pd
import openpyxl
from sklearn import metrics
from math import sqrt
import matplotlib.pyplot as plt
import time
import itertools
import warnings
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# Cargar datos
Alameda_pred = pd.read_excel('predicciones_cabello_alameda.xlsx')
df_temp = pd.read_excel('maximos_minimos.xlsx')

# Procesar datos
Alameda_pred.set_index("Time", inplace=True)
Alameda_pred = Alameda_pred.drop('Cabello_predicho', axis=1)
print(Alameda_pred)
print(df_temp)
Alameda_pred.info()

# Separar x e y para el gráfico
x = Alameda_pred.index
y = Alameda_pred["Alameda_predicho"]
y_media = [np.mean(y) for _ in y]

# Visualización de los datos a lo largo de los años
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(x, y, label="Serie Original")
ax1.plot(x, y_media, label="Media de la Serie Original")
ax1.set_ylim(0, np.max(y) * 1.3)
ax1.legend(loc="upper left")
plt.show()

# Análisis de estacionariedad
series = [y]
nombres_series = ["Serie Original"]
for serie, nombre_serie in zip(series, nombres_series):
    print("------------------------------------------------------------------")
    print(f"Estamos trabajando con la serie {nombre_serie}\n")
    resultado_analisis = adfuller(serie)
    valor_estadistico_adf = resultado_analisis[0]
    p_valor = resultado_analisis[1]
    print("Valor estadístico de ADF de las tablas precalculadas: {}".format(-2.89))
    print(f"Valor estadístico de ADF: {valor_estadistico_adf}\n")
    print("Nivel de significación para tomar la serie como estacionaria {}".format(0.05))
    print(f"p-valor: {p_valor}\n")

# Graficar ACF y PACF
LAGS = 24
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 5))
plot_acf(y, ax=ax1, lags=LAGS, title="Autocorrelación")
plot_pacf(y, ax=ax2, lags=LAGS, title="Autocorrelación Parcial")
fig.tight_layout()
plt.show()

# Separar los datos en entrenamiento y prueba
serie_a_predecir = y
y_index = serie_a_predecir.index
date_train = int(len(y_index) * 0.9)
y_train = serie_a_predecir[y_index[:date_train]]
y_test = serie_a_predecir[y_index[date_train:]]

# Grid search para encontrar los mejores parámetros
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(0, 0, 0, 0)]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[0]))
print('SARIMAX: {} x {}'.format(pdq[3], seasonal_pdq[0]))

st = time.time()
best_score = float("inf")
best_params = None
best_seasonal_params = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(y_train, order=param, seasonal_order=param_seasonal,
                          enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            print(f'ARIMA{param}x{param_seasonal}12 - AIC:{results.aic}')
            if results.aic < best_score:
                best_score = results.aic
                best_params = param
                best_seasonal_params = param_seasonal
        except:
            continue

et = time.time()
print(f"La búsqueda de parámetros demora {(et - st) / 60} minutos!")
print(f"El mejor modelo es {best_params}, \nCon un AIC de {best_score}")

# Ajustar el mejor modelo
mod = SARIMAX(y_train, order=best_params, seasonal_order=best_seasonal_params,
              enforce_stationarity=False, enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12), lags=3)
plt.show()

# Hacer predicciones
pred_uc = results.get_forecast(steps=len(y_test))
pred_ci = pred_uc.conf_int()
ax = serie_a_predecir.plot(label='Valores reales', figsize=(15, 10))
pred_uc.predicted_mean.plot(ax=ax, label='Predicción')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Año')
ax.set_ylabel('Alameda')
plt.legend()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo Excel
file_path = 'maximos_minimos.xlsx'
maximos_minimos = pd.read_excel(file_path)

# Mostrar las primeras filas del DataFrame
print(maximos_minimos.head())

# Mostrar información general del DataFrame
print(maximos_minimos.info())

# Describir estadísticamente el DataFrame
print(maximos_minimos.describe())

# Visualización de los datos
maximos_minimos.plot(subplots=True, figsize=(12, 10))
plt.show()





# Rellenar los valores nulos con propagación hacia adelante (pad)
maximos_minimos.fillna(method='pad', inplace=True)

# Verificar si aún hay valores nulos
print(maximos_minimos.info())

# Cargar y procesar la base de datos de predicciones de Alameda
alameda_data_path = 'predicciones_cabello_alameda.xlsx'
alameda_data = pd.read_excel(alameda_data_path)
alameda_data['Time'] = pd.to_datetime(alameda_data['Time'])
alameda_data.set_index('Time', inplace=True)
alameda_data = alameda_data.drop('Cabello_predicho', axis=1)

# Fusionar los DataFrames en función de la fecha
merged_data = pd.merge(alameda_data, maximos_minimos, left_index=True, right_on='Fecha')
merged_data.set_index('Fecha', inplace=True)

print(merged_data.head())

# Visualización de los datos
merged_data.plot(subplots=True, figsize=(12, 10))
plt.show()




import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import time
import matplotlib.pyplot as plt

# Separar los datos en entrenamiento y prueba
date_train = int(len(merged_data) * 0.9)
y_train = merged_data['Alameda_predicho'].iloc[:date_train]
y_test = merged_data['Alameda_predicho'].iloc[date_train:]
exog_train = merged_data[['Temp. Máxima', 'Temp. Mínima']].iloc[:date_train]
exog_test = merged_data[['Temp. Máxima', 'Temp. Mínima']].iloc[date_train:]

# Grid search para encontrar los mejores parámetros
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(0, 0, 0, 0)]

best_score = float("inf")
best_params = None
best_seasonal_params = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(y_train, exog=exog_train, order=param, seasonal_order=param_seasonal,
                          enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            if results.aic < best_score:
                best_score = results.aic
                best_params = param
                best_seasonal_params = param_seasonal
        except:
            continue

# Ajustar el mejor modelo
mod = SARIMAX(y_train, exog=exog_train, order=best_params, seasonal_order=best_seasonal_params,
              enforce_stationarity=False, enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12), lags=3)
plt.show()

# Hacer predicciones
pred_uc = results.get_forecast(steps=len(y_test), exog=exog_test)
pred_ci = pred_uc.conf_int()
ax = merged_data['Alameda_predicho'].plot(label='Valores reales', figsize=(15, 10))
pred_uc.predicted_mean.plot(ax=ax, label='Predicción')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Año')
ax.set_ylabel('Alameda')
plt.legend()
plt.show()