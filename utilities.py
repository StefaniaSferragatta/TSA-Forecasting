import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def correlation(data : pd.core.frame.DataFrame):
    corr_matrix = data.corr()
    mask = np.zeros_like(corr_matrix) #to plot only a half matrix
    mask[np.triu_indices_from(mask)] = True #to generate a mask for the upper triangle
    sns.heatmap(corr_matrix, mask=mask, square=True,cmap="PuOr_r")
    return

def pair_plot(data : pd.core.frame.DataFrame):
        sns.pairplot(data=data,
                vars=['no_of_rainy_days', 'total_rainfall', 'relative_humidity',
                    'surface_temp', 'co2', 'GMSL(Global Mean Sea Level)',
                    'Anomalies_Land_Ocean_Temperature'],
                kind='scatter',
                markers = '*',
                plot_kws={'color': 'g'},
                diag_kws= {'color': 'm'},
                diag_kind = "kde")

        plt.show()
        plt.clf()
        return

def plot_autocorrelation(df : pd.core.frame.DataFrame):
    for c in df.columns[1:]:
        pd.plotting.autocorrelation_plot(df[c])
        plt.title(f'Autocorrelation of "{c}" ')
        plt.show()
    return 

def acf(df : pd.core.frame.DataFrame):
    for c in df.columns[1:]:
        plot_acf(df[c], lags=36,color='blueviolet')
        plt.title(f'{c}')
        plt.show()
    return

def pacf(df : pd.core.frame.DataFrame):  
    for c in df.columns[1:]:
        plot_pacf(df[c], lags=36,color = 'indianred')
        plt.title(f'Partial autocorrelation of "{c}"')
        plt.show()
    return

def lag_plot(df : pd.core.frame.DataFrame):
    for c in df.columns[1:]:
        pd.plotting.lag_plot(df[c],c='indigo')
        plt.title(f'{c}')
        plt.show()
    return