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

# Hacer predicciones dentro del rango de datos disponibles para calcular MAPE y MAD
pred_start = len(data_alameda) - 78
pred_end = len(data_alameda) - 1
predicciones = model_fit.predict(start=pred_start, end=pred_end)
actual = data_alameda.iloc[pred_start:pred_end+1]

# Calcular el MAPE
mape = np.mean(np.abs((actual - predicciones) / actual)) * 100
print('MAPE: {:.2f}%'.format(mape))

# Calcular el MAD
mad = np.mean(np.abs(actual - predicciones))
print('MAD: {:.2f}'.format(mad))

# Visualizar las predicciones
plt.figure(figsize=(10, 6))
plt.plot(data_alameda, label='Histórico')
plt.plot(predicciones, label='Predicción', color='red')
plt.title('Predicción de Conteo de Bicicletas en Alameda')
plt.ylabel('Número de Bicicletas')
plt.xlabel('Fecha')
plt.legend()
plt.show()
