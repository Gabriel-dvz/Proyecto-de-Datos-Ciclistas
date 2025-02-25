from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

# Configuración de gráficos
rcParams['figure.figsize'] = 15, 7

# Cargar datos
try:
    Alameda_pred = pd.read_excel('predicciones_cabello_alameda.xlsx')
    df_temp = pd.read_excel('maximos_minimos.xlsx')
except Exception as e:
    print(f"Error al cargar los datos: {e}")

# Procesar datos
try:
    Alameda_pred.set_index("Time", inplace=True)
    Alameda_pred.index = pd.to_datetime(Alameda_pred.index)
    Alameda_pred = Alameda_pred.drop('Cabello_predicho', axis=1)
    print(Alameda_pred)
    print(df_temp)
    Alameda_pred.info()
except KeyError as e:
    print(f"Error al procesar los datos: {e}")

# Graficar datos
plt.figure(figsize=(15,7))
plt.title("Cantidad de Ciclistas Alameda")
plt.xlabel('Tiempo')
plt.ylabel('Ciclistas')
plt.plot(Alameda_pred)
plt.show()

# Prueba Dickey–Fuller aumentada
print('Results of Dickey Fuller Test:')
dftest = adfuller(Alameda_pred['Alameda_predicho'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value

print(dfoutput)

# Modelo ARIMA estándar
ARIMA_model = pm.auto_arima(Alameda_pred['Alameda_predicho'], 
                            start_p=1, 
                            start_q=1,
                            test='adf',  # utilizar adftest para encontrar el 'd' óptimo
                            max_p=3, max_q=3,  # máximo p y q
                            m=1,  # frecuencia de la serie (diaria en este caso)
                            d=None,  # dejar que el modelo determine 'd'
                            seasonal=False,  # Sin estacionalidad para ARIMA estándar
                            trace=False,  # logs 
                            error_action='warn',  # muestra errores ('ignore' los silencia)
                            suppress_warnings=True,
                            stepwise=True)

# Diagnóstico del modelo
ARIMA_model.plot_diagnostics(figsize=(15,12))
plt.show()

# Función de pronóstico
def forecast(ARIMA_model, periods=100):
    # Pronóstico
    n_periods = periods
    fitted, confint = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)
    
    # Crear un índice de fechas para el pronóstico
    last_date = Alameda_pred.index[-1]
    index_of_fc = pd.date_range(last_date, periods=n_periods, freq='D')

    # Crear series para propósitos de graficación
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Graficar
    plt.figure(figsize=(15,7))
    plt.plot(Alameda_pred['Alameda_predicho'], color='#1f76b4', label='Datos históricos')
    plt.plot(fitted_series, color='darkgreen', label='Predicción')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    plt.title("ARIMA - Predicción de Ciclistas en Alameda")
    plt.xlabel('Tiempo')
    plt.ylabel('Ciclistas')
    plt.legend()
    plt.show()

forecast(ARIMA_model)


# Seasonal - fit stepwise auto-ARIMA
SARIMA_model = pm.auto_arima(Alameda_pred['Alameda_predicho'], start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, 
                         m=365, #12 is the frequncy of the cycle
                         start_P=0, 
                         seasonal=True, #set to seasonal
                         d=None, 
                         D=1, #order of the seasonal differencing
                         trace=False,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

SARIMA_model.plot_diagnostics(figsize=(15,12))
plt.show()