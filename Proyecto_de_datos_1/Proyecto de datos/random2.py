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
from numpy import log
import pmdarima as pm

warnings.filterwarnings("ignore")

# Cargar datos desde el archivo Excel
file_path = 'predicciones_cabello_alameda.xlsx'
data = pd.read_excel(file_path, parse_dates=['Time'], index_col='Time')

# Filtrar la columna Alameda_predicho
data_alameda = data['Alameda_predicho']

print(data_alameda.head())

# Visualizar los datos
data_alameda.plot()
plt.title('Conteo de Bicicletas en Alameda')
plt.ylabel('Número de Bicicletas')
plt.xlabel('Fecha')
plt.show()

# Buscar los mejores parámetros (p, d, q)
modelo = pm.auto_arima(data_alameda, seasonal=False, trace=True)
print(modelo.summary())

from statsmodels.tsa.arima.model import ARIMA

# Ajustar el modelo ARIMA
p, d, q = modelo.order
model = ARIMA(data_alameda, order=(p, d, q))
model_fit = model.fit()

# Resumen del modelo
print(model_fit.summary())

# Hacer predicciones
predicciones = model_fit.predict(start=len(data_alameda), end=len(data_alameda) + 77)
print(predicciones)

# Visualizar las predicciones
plt.figure(figsize=(10, 6))
plt.plot(data_alameda, label='Histórico')
plt.plot(predicciones, label='Predicción', color='red')
plt.title('Predicción de Conteo de Bicicletas en Alameda')
plt.ylabel('Número de Bicicletas')
plt.xlabel('Fecha')
plt.legend()
plt.show()

# Cargar el archivo Excel
file_path_temp = 'maximos_minimos.xlsx'
maximos_minimos = pd.read_excel(file_path_temp, parse_dates=['Fecha'], index_col='Fecha')

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

# Fusionar los DataFrames en función de la fecha
merged_data = pd.merge(data, maximos_minimos, left_index=True, right_index=True, how='inner')

# Filtrar la columna de interés y las variables exógenas
data_alameda = merged_data['Alameda_predicho']
exog = merged_data[['Temp. Máxima', 'Temp. Mínima']] 

print(merged_data.head())

# Buscar los mejores parámetros (p, d, q) con variables exógenas
modelo = pm.auto_arima(data_alameda, exogenous=exog, seasonal=False, trace=True)
print(modelo.summary())

# Ajustar el modelo ARIMA con variables exógenas
p, d, q = modelo.order
model = ARIMA(data_alameda, exog=exog, order=(p, d, q))
model_fit = model.fit()

# Resumen del modelo
print(model_fit.summary())

# Preparar las variables exógenas para el período de predicción (aquí se necesitarían los valores futuros de temp_max y temp_min)
# Asegúrate de proporcionar suficientes datos de exog para el período de predicción
exog_future = exog.iloc[-78:]  # Ajustar según la cantidad de predicciones que se desea hacer

# Hacer predicciones
predicciones = model_fit.predict(start=len(data_alameda), end=len(data_alameda) + 77, exog=exog_future)
print(predicciones)

# Visualizar las predicciones
plt.figure(figsize=(10, 6))
plt.plot(data_alameda, label='Histórico')
plt.plot(predicciones, label='Predicción', color='red')
plt.title('Predicción de Conteo de Bicicletas en Alameda con Variables Exógenas')
plt.ylabel('Número de Bicicletas')
plt.xlabel('Fecha')
plt.legend()
plt.show()
