import pandas as pd
from prophet import Prophet

# Cargar datos
df = pd.read_csv('bike_rancagua.csv')

# Renombrar las columnas como 'ds' y 'y'
df = df.rename(columns={'Time': 'ds', 'Alameda': 'y'})

# Inicializar el modelo Prophet
m = Prophet()

# Entrenar el modelo
m.fit(df)

# Crear DataFrame para futuras fechas
future = m.make_future_dataframe(periods=365)

# Realizar predicciones
forecast = m.predict(future)

# Visualizar las últimas predicciones
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Visualizar los resultados
fig1 = m.plot(forecast)

# Visualizar componentes del pronóstico
fig2 = m.plot_components(forecast)

