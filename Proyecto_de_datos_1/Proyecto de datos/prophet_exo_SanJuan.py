import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

# Cargar datos de predicciones
San_Juan_pred = pd.read_excel('predicciones.xlsx')
San_Juan_pred = San_Juan_pred.rename(columns={'Time': 'ds', 'San Juan_predicho': 'y'})

# Cargar el archivo Excel
file_path_temp = 'maximos_minimos.xlsx'
maximos_minimos = pd.read_excel(file_path_temp, parse_dates=['Fecha'])

# Rellenar los valores nulos con propagación hacia adelante (pad)
maximos_minimos.fillna(method='pad', inplace=True)

# Verificar si aún hay valores nulos
print(maximos_minimos.info())

# Renombrar columna de fechas para facilitar la unión
maximos_minimos = maximos_minimos.rename(columns={'Fecha': 'ds'})

# Fusionar los DataFrames en función de la fecha
merged_data = pd.merge(San_Juan_pred, maximos_minimos, on='ds')

# Inicializar el modelo Prophet
m = Prophet()

# Añadir los regresores
m.add_regressor('Temp. Máxima')
m.add_regressor('Temp. Mínima')

# Entrenar el modelo
m.fit(merged_data)

# Crear DataFrame para futuras fechas
future = m.make_future_dataframe(periods=365)

# Añadir los valores futuros de los regresores
# Aquí es necesario proporcionar valores futuros para los regresores
last_known_temps = maximos_minimos.iloc[-1][['Temp. Máxima', 'Temp. Mínima']]
future_temps = pd.concat([maximos_minimos, pd.DataFrame([last_known_temps] * (365), columns=['Temp. Máxima', 'Temp. Mínima'])], ignore_index=True).iloc[:len(future)]

future['Temp. Máxima'] = future_temps['Temp. Máxima']
future['Temp. Mínima'] = future_temps['Temp. Mínima']

# Realizar predicciones
forecast = m.predict(future)

# Calcular MAPE (Error Porcentual Absoluto Medio)
merged_data['abs_diff'] = np.abs(merged_data['y'] - forecast['yhat'][:len(merged_data)])
merged_data['mape'] = np.abs(merged_data['abs_diff'] / merged_data['y'] * 100)
mape_mean = merged_data['mape'].mean()
print(f"MAPE: {mape_mean:.2f}%")

# Calcular MAD (Desviación Absoluta Media)
mad_mean = merged_data['abs_diff'].mean()
print(f"MAD: {mad_mean:.2f}")

# Visualizar las últimas predicciones
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Visualizar los resultados
fig1 = m.plot(forecast)
plt.show()  # Mostrar el gráfico

# Visualizar componentes del pronóstico
fig2 = m.plot_components(forecast)
plt.show()  # Mostrar el gráfico
