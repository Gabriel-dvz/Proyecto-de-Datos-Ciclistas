import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos desde el archivo Excel
file_path = 'predicciones.xlsx'
data = pd.read_excel(file_path, parse_dates=['Time'], index_col='Time')

# Filtrar las columnas Alameda_predicho y Cabello_predicho
data_alameda = data['República de Chile_predicho']
data_cabello = data['San Juan_predicho']

# Crear el boxplot con ajustes estéticos
plt.figure(figsize=(12, 8))
plt.boxplot([data_alameda.dropna(), data_cabello.dropna()], labels=['República de Chile', 'San Juan'],
            patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'), whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'), flierprops=dict(markerfacecolor='blue', marker='o', markersize=5))
plt.title('Boxplot de Conteo de Bicicletas en República de Chile y San Juan', fontsize=16)
plt.ylabel('Número de Bicicletas', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
