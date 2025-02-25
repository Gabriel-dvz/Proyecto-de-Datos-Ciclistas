import pandas as pd
import matplotlib.pyplot as plt

filas_a_omitir = 4

df_cereza = pd.read_excel('AvanceProductos_202444.xls', skiprows=filas_a_omitir)
df_durazno = pd.read_excel('AvanceProductosDur_202444.xls', skiprows=filas_a_omitir)
df_uva = pd.read_excel('AvanceProductosUva_202444.xls', skiprows=filas_a_omitir)

fila_info1 = df_cereza.iloc[0]
fila_info2 = df_durazno.iloc[0]
fila_info3 = df_uva.iloc[0]

fila_info1 = pd.to_numeric(fila_info1, errors='coerce')
fila_info2 = pd.to_numeric(fila_info2, errors='coerce')
fila_info3 = pd.to_numeric(fila_info3, errors='coerce')


promedio_cer = fila_info1.mean()
promedio_dur = fila_info2.mean()
promedio_uva = fila_info3.mean()

print(df_cereza)
print(df_durazno)
print(df_uva)

print("El promedio de la producción de Cerezas los últimos 24 años es:", promedio_cer)
print("El promedio de la producción de Duraznos los últimos 24 años es:", promedio_dur)
print("El promedio de la producción de Uva los últimos 24 años es:", promedio_uva)

