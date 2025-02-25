from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams

from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

# Cargar datos
Alameda_pred = pd.read_excel('predicciones_cabello_alameda.xlsx')
df_temp = pd.read_excel('maximos_minimos.xlsx')

# Procesar datos
Alameda_pred.set_index("Time", inplace=True)
Alameda_pred = Alameda_pred.drop('Cabello_predicho', axis=1)
print(Alameda_pred)
print(df_temp)
Alameda_pred.info()


plt.figure(figsize=(15,7))
plt.title("Cantidad de Ciclistas Alameda")
plt.xlabel('Tiempo')
plt.ylabel('Ciclistas')
plt.plot(Alameda_pred)
plt.show()


#Augmented Dickey–Fuller test:
print('Results of Dickey Fuller Test:')
dftest = adfuller(Alameda_pred['Alameda_predicho'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
    
print(dfoutput)

#Standard ARIMA Model
ARIMA_model = pm.auto_arima(Alameda_pred['Alameda_predicho'], 
                      start_p=1, 
                      start_q=1,
                      test='adf', # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                      d=None,# let model determine 'd'
                      seasonal=False, # No Seasonality for standard ARIMA
                      trace=False, #logs 
                      error_action='warn', #shows errors ('ignore' silences these)
                      suppress_warnings=True,
                      stepwise=True)


ARIMA_model.plot_diagnostics(figsize=(15,12))
plt.show()

def forecast(ARIMA_model, periods=24):
    # Forecast
    n_periods = periods
    fitted, confint = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(Alameda_pred.index[-1] + pd.DateOffset(months=1), periods = n_periods, freq='MS')

    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(figsize=(15,7))
    plt.plot(Alameda_pred['Alameda_predicho'], color='#1f76b4')
    plt.plot(fitted_series, color='darkgreen')
    plt.fill_between(lower_series.index, 
                    lower_series, 
                    upper_series, 
                    color='k', alpha=.15)

    plt.title("ARIMA/SARIMA - Forecast of Airline Passengers")
    plt.show()

forecast(ARIMA_model)

