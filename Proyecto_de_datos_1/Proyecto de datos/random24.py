import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

# Cargar datos
df = pd.read_excel('predicciones_cabello_alameda.xlsx')

# Renombrar las columnas como 'ds' y 'y'
df = df.rename(columns={'Time': 'ds', 'Alameda_predicho': 'y'})

# Inicializar el modelo Prophet
m = Prophet()

# Entrenar el modelo
m.fit(df)

# Crear DataFrame para futuras fechas
future = m.make_future_dataframe(periods=365)

# Realizar predicciones
forecast = m.predict(future)

# Calcular MAPE (Error Porcentual Absoluto Medio)
df['abs_diff'] = abs(df['y'] - forecast['yhat'])
df['mape'] = np.abs(df['abs_diff'] / df['y'] * 100)
mape_mean = df['mape'].mean()
print(f"MAPE: {mape_mean:.2f}%")

# Calcular MAD (Desviación Absoluta Media)
mad_mean = df['abs_diff'].mean()
print(f"MAD: {mad_mean:.2f}")

# Visualizar las últimas predicciones
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Visualizar los resultados
fig1 = m.plot(forecast)
plt.show()  # Mostrar el gráfico

# Visualizar componentes del pronóstico
fig2 = m.plot_components(forecast)
plt.show()  # Mostrar el gráfico