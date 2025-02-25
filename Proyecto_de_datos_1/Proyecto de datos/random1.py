import pandas as pd

# Ruta al archivo Excel
file_path = 'estacion_metereologica.xlsx'

# Leer el archivo Excel
df = pd.read_excel(file_path)

# Mostrar las primeras filas del DataFrame
print(df.head())

# Si deseas mostrar todo el DataFrame (puede ser muy grande)
# print(df)

# Para mostrar información resumida del DataFrame
print(df.info())

# Para visualizar estadísticas descriptivas
print(df.describe())
