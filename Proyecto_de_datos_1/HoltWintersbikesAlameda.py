import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Leer datos de ventas
data = pd.read_csv('predicciones_cabello_alameda.xlsx')

# Asumimos que los datos tienen una columna llamada 'Time' y otra 'Alameda'
data['Time'] = pd.to_datetime(data['Time'])
data.set_index('Time', inplace=True)

# Aseguramos que la serie temporal tenga una frecuencia diaria explícita
data_series = data['Alameda_predicho'].asfreq('D')

# Aseguramos que todos los valores sean positivos
data_series = data_series[data_series > 0]

# Graficar la serie de tiempo original
plt.figure(figsize=(10, 6))
plt.plot(data_series, label='Original')
plt.title('Flujo de Bicicletas en Alameda')
plt.legend()
plt.show()

# Suavización Exponencial Simple
model_simple = ExponentialSmoothing(data_series, trend=None, seasonal=None, seasonal_periods=12)
fit_simple = model_simple.fit()
data_forecast_simple = fit_simple.fittedvalues

plt.figure(figsize=(10, 6))
plt.plot(data_series, label='Original')
plt.plot(data_forecast_simple, label='Simple Exponential Smoothing', color='red')
plt.title('Suavización Exponencial Simple')
plt.legend()
plt.show()

# Suavización Exponencial con Tendencia
model_trend = ExponentialSmoothing(data_series, trend='add', seasonal=None, seasonal_periods=12)
fit_trend = model_trend.fit()
data_forecast_trend = fit_trend.fittedvalues

plt.figure(figsize=(10, 6))
plt.plot(data_series, label='Original')
plt.plot(data_forecast_trend, label='Exponential Smoothing with Trend', color='red')
plt.title('Suavización Exponencial con Tendencia')
plt.legend()
plt.show()

# Suavización Exponencial con Tendencia y Estacionalidad
model_trend_seasonal = ExponentialSmoothing(data_series, trend='add', seasonal='multiplicative', seasonal_periods=12)
fit_trend_seasonal = model_trend_seasonal.fit()
data_forecast_trend_seasonal = fit_trend_seasonal.fittedvalues

plt.figure(figsize=(10, 6))
plt.plot(data_series, label='Original')
plt.plot(data_forecast_trend_seasonal, label='Exponential Smoothing with Trend and Seasonality', color='red')
plt.title('Suavización Exponencial con Tendencia y Estacionalidad')
plt.legend()
plt.show()

# Predecir utilizando el modelo con tendencia y estacionalidad
forecast_horizon = 50
data_forecast_future = fit_trend_seasonal.forecast(forecast_horizon)

plt.figure(figsize=(10, 6))
plt.plot(data_series, label='Original')
plt.plot(pd.concat([data_forecast_trend_seasonal, data_forecast_future]), label='Forecast', color='red')
plt.title('Predicción con Suavización Exponencial (Tendencia y Estacionalidad)')
plt.legend()
plt.show()



from statsmodels.tsa.seasonal import seasonal_decompose

# Descomponer la serie temporal
result = seasonal_decompose(data_series, model='multiplicative')
result.plot()
plt.show()

