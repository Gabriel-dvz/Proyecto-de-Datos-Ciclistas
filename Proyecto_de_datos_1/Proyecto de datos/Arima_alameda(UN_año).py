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
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# Cargar datos desde el archivo Excel
file_path = 'predicciones_cabello_alameda.xlsx'
data = pd.read_excel(file_path)

# Convertir la columna de tiempo al formato correcto
data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d')

# Filtrar datos por rango de fechas
#data = data[(data['Time'] >= '2023-01-14') & (data['Time'] <= '2024-01-14')]

# Establecer la columna de tiempo como el índice
data.set_index('Time', inplace=True)

# Filtrar la columna Alameda
data_alameda = data['Alameda_predicho']

# Ver los primeros datos
print(data_alameda.head())

# Visualizar los datos
data_alameda.plot()
plt.title('Conteo de Bicicletas en Alameda')
plt.ylabel('Número de Bicicletas')
plt.xlabel('Fecha')
plt.show()

# Descomponer la serie temporal para verificar estacionalidad
result = seasonal_decompose(data_alameda, model='additive', period=31)
result.plot()
plt.show()

# Buscar los mejores parámetros (p, d, q) incluyendo estacionalidad
modelo = pm.auto_arima(data_alameda, seasonal=True, m=31, trace=True)
print(modelo.summary())

# Ajustar el modelo SARIMA
p, d, q = modelo.order
P, D, Q, s = modelo.seasonal_order
model = SARIMAX(data_alameda, order=(p, d, q), seasonal_order=(P, D, Q, s))
model_fit = model.fit(disp=False)

# Resumen del modelo
print(model_fit.summary())

# Hacer predicciones
pred_start_date = data_alameda.index[-1] + pd.DateOffset(1)
pred_end_date = pred_start_date + pd.DateOffset(77)
predicciones = model_fit.predict(start=pred_start_date, end=pred_end_date)

# Convertir las predicciones a un DataFrame con índice de fecha
predicciones = pd.Series(predicciones, index=pd.date_range(start=pred_start_date, periods=len(predicciones), freq='D'))

# Visualizar las predicciones
plt.figure(figsize=(10, 6))
plt.plot(data_alameda, label='Histórico')
plt.plot(predicciones, label='Predicción', color='red')
plt.title('Predicción de Conteo de Bicicletas en Alameda')
plt.ylabel('Número de Bicicletas')
plt.xlabel('Fecha')
plt.legend()
plt.show()

# Función para calcular el error absoluto medio porcentual
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Función para calcular la desviación absoluta media
def mean_absolute_deviation(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Función para calcular el error absoluto medio porcentual (MAPE)
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Función para calcular la desviación absoluta media (MAD)
def mean_absolute_deviation(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Calcular MAPE y MAD en los datos de entrenamiento
train_predictions = model_fit.predict(start=data_alameda.index[0], end=data_alameda.index[-1])
mape = mean_absolute_percentage_error(data_alameda, train_predictions)
mad = mean_absolute_deviation(data_alameda, train_predictions)

print(f'MAPE: {mape}')
print(f'MAD: {mad}')

