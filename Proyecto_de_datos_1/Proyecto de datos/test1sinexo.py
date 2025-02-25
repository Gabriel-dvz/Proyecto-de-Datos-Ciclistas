import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# Cargar los datos desde el archivo proporcionado
file_path = 'export.xlsx'
data = pd.read_excel(file_path)

# Limpiar los datos y seleccionar la columna de interés
data.columns = data.iloc[1]
data = data.drop([0, 1]).reset_index(drop=True)
data = data[['Time', 'Alameda']]
data.columns = ['Time', 'Alameda']

# Convertir la columna 'Time' a formato datetime y establecerla como índice
data['Time'] = pd.to_datetime(data['Time'])
data.set_index('Time', inplace=True)

# Asegurarse de que los datos son numéricos
data['Alameda'] = pd.to_numeric(data['Alameda'])

# Visualizar los datos
data['Alameda'].plot()
plt.title('Conteo de Bicicletas en Alameda')
plt.ylabel('Número de Bicicletas')
plt.xlabel('Fecha')
plt.show()

# Buscar los mejores parámetros (p, d, q)
modelo = pm.auto_arima(data['Alameda'], seasonal=False, trace=True)
print(modelo.summary())

# Ajustar el modelo ARIMA
p, d, q = modelo.order
model = ARIMA(data['Alameda'], order=(p, d, q))
model_fit = model.fit()

# Resumen del modelo
print(model_fit.summary())

# Hacer predicciones
predicciones = model_fit.predict(start=len(data['Alameda']), end=len(data['Alameda']) + 77)
print(predicciones)

# Visualizar las predicciones
plt.figure(figsize=(10, 6))
plt.plot(data['Alameda'], label='Histórico')
plt.plot(predicciones, label='Predicción', color='red')
plt.title('Predicción de Conteo de Bicicletas en Alameda')
plt.ylabel('Número de Bicicletas')
plt.xlabel('Fecha')
plt.legend()
plt.show()
