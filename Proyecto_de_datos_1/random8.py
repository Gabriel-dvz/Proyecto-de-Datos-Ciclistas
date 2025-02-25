import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from random import gauss
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from random import random
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from scipy.stats import boxcox

import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight') 

import warnings
warnings.simplefilter(action='ignore', category= FutureWarning) 

alpha = 0.5

df = pd.read_excel('predicciones_cabello_alameda.xlsx', usecols=['Alameda_predicho'])
df['Alameda_predicho'] = df['Alameda_predicho'].clip(lower=0)  # Set any negative values to zero

# Aplicar la transformación Box-Cox manualmente
df['Alameda_predicho'], _ = boxcox(df['Alameda_predicho'] + 1)  # Se suma 1 para evitar valores cero antes de la transformación

df.plot.line()

# Ajustar el modelo de suavizado exponencial
fit1 = ExponentialSmoothing(df, seasonal_periods=12, trend='add', seasonal='add', use_boxcox=False)
fit1 = fit1.fit(smoothing_level=0.5)
fit1.fittedvalues.plot(color='red')
fit1.forecast(12).rename("Holt-Winters smoothing").plot(color='red', legend=True)

plt.ylim(0, 800)
plt.show()
