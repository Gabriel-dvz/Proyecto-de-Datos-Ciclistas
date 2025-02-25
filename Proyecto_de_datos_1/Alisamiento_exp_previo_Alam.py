import pandas as pd
import numpy as np
from random import gauss
from pandas.plotting import autocorrelation_plot
import warnings
import itertools
from random import random

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight') 
# import matplotlib as mpl
import seaborn as sns   

series = pd.read_excel('predicciones_cabello_alameda.xlsx')
series['Time'] = pd.to_datetime(series['Time'])
series.set_index('Time').plot()

plt.show()

#Descomposición 
decomposition = sm.tsa.seasonal_decompose(series['Alameda_predicho'], period=12) 
figure = decomposition.plot()
plt.show()

#Descomposición con additive 
decomposition = sm.tsa.seasonal_decompose(series['Alameda_predicho'], period=12, model='additive') 
figure = decomposition.plot()
plt.show()

#Histograma 

series['Alameda_predicho'].plot.hist(bins=25, alpha=0.5)
plt.show()

#estadisticas:
#La estacionariedad de una serie se puede comprobar examinando la distribución de la serie: dividimos la serie en 2 partes 
#contiguas y calculamos los estadísticos resumidos como la media, la varianza y la autocorrelación. 
#Si las estadísticas son bastante diferentes, entonces no es probable que la serie sea estacionaria.

X = series.Alameda_predicho.values
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
media1, media2 = X1.mean(), X2.mean()
varianza1, varianza2 = X1.var(), X2.var()
print('media:')
print('trozo 1: %.2f vs trozo 2: %.2f' % (media1, media2))
print('varianza:')
print('trozo 1: %.2f vs trozo 2: %.2f' % (varianza1, varianza2))

#Autocorrelación 

plot_acf(X, lags = 12)
plt.show()

#Autocorrelación Parcial 

plot_pacf(X, lags = 12)
plt.show()

