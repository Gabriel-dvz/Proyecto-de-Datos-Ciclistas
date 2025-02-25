import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# Cargar datos
file_path = 'predicciones_cabello_alameda.xlsx'
alameda_data = pd.read_excel(file_path)

# Procesar datos
alameda_data['Time'] = pd.to_datetime(alameda_data['Time'])
alameda_data.set_index('Time', inplace=True)
alameda_data = alameda_data.drop('Cabello_predicho', axis=1)

# Cargar datos exógenos
exog_path = 'maximos_minimos.xlsx'
exog_data = pd.read_excel(exog_path)

# Procesar datos exógenos
exog_data['Fecha'] = pd.to_datetime(exog_data['Fecha'])
exog_data.set_index('Fecha', inplace=True)

# Fusionar los datos
merged_data = pd.merge(alameda_data, exog_data, left_index=True, right_index=True)

# Verificar y limpiar datos
merged_data = merged_data.interpolate(method='linear')
print(merged_data.info())

# Visualización de los datos
merged_data.plot(subplots=True, figsize=(12, 10))
plt.show()

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

# Verificar y limpiar los datos de pred_ci
pred_ci = pred_ci.apply(pd.to_numeric, errors='coerce')
pred_ci = pred_ci.replace([np.inf, -np.inf], np.nan)
pred_ci = pred_ci.dropna()

# Visualizar las predicciones
ax = merged_data['Alameda_predicho'].plot(label='Valores reales', figsize=(15, 10))
pred_uc.predicted_mean.plot(ax=ax, label='Predicción')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Año')
ax.set_ylabel('Alameda')
plt.legend()
plt.show()
