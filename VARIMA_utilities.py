import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.stattools import grangercausalitytests #for causality check
from statsmodels.tsa.stattools import adfuller #for stationarity check
from statsmodels.stats.stattools import durbin_watson #for autocorrelation check
from statsmodels.tsa.vector_ar.var_model import VAR

def correlation(data : pd.core.frame.DataFrame):
    '''
    Function that takes in input the dataframe and:
    - computes the correlation matrix of the features
    - plot the heatmap
    '''
    corr_matrix = data.corr()
    mask = np.zeros_like(corr_matrix) #to plot only a half matrix
    mask[np.triu_indices_from(mask)] = True #to generate a mask for the upper triangle
    sns.heatmap(corr_matrix, mask=mask, square=True,cmap="PuOr_r")
    return

def pair_plot(data : pd.core.frame.DataFrame):
    '''
    Function that takes in input the dataframe and plot the pairplot of the features.
    '''
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
    '''
    Function that takes in input the dataframe and plot the autocorrelation of the features.
    '''
    for c in df.columns[1:]:
        pd.plotting.autocorrelation_plot(df[c])
        plt.title(f'Autocorrelation of "{c}" ')
        plt.show()
    return 

def acf(df : pd.core.frame.DataFrame):
    '''
    Function that takes in input the dataframe and performs the autocorrelation function of the features
    then plots the result.
    '''
    for c in df.columns[1:]:
        plot_acf(df[c], lags=36,color='blueviolet')
        plt.title(f'{c}')
        plt.show()
    return

def pacf(df : pd.core.frame.DataFrame):  
    '''
    Function that takes in input the dataframe and performs the partial autocorrelation 
    function of the features then plots the result.
    '''
    for c in df.columns[1:]:
        plot_pacf(df[c], lags=36,color = 'indianred')
        plt.title(f'Partial autocorrelation of "{c}"')
        plt.show()
    return

def lag_plot(df : pd.core.frame.DataFrame):
    '''
    Function that takes in input the dataframe and plot the lag scatter plot of the features.
    '''
    for c in df.columns[1:]:
        pd.plotting.lag_plot(df[c],c='indigo')
        plt.title(f'{c}')
        plt.show()
    return

def grangers_causation_matrix(data: pd.core.frame.DataFrame, variables: pd.core.indexes.base.Index, test='ssr_chi2test', verbose=False):    
    '''
    Function that takes in input:
    - the dataframe containing the data to analyze
    - the index of the dataframe.column
    - the type of test

    and performs the grangercausalitytests. It returns a dataframe with the results.
    '''
    maxlag=15
    
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables) #create the df for the results
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False) #perform the test
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def adfuller_test(series: pd.core.series.Series, sig=0.05, name=''):
    '''
    Function that takes in input the series on which perform the adfuller test for the stationarity. 
    It prints the results.
    '''
    res = adfuller(series, autolag='AIC')    
    p_value = round(res[1], 3) 

    if p_value <= sig:
        print(f" {name} : P-Value = {p_value} => Stationary")
    else:
        print(f" {name} : P-Value = {p_value} => Non-stationary")

def invert_transformation(train: pd.core.frame.DataFrame, pred : pd.core.frame.DataFrame):
    '''
    Function that takes in input:
    - the train set
    - the prediction set
    
    and perform the inversion to obtain the forcast, save the result into a dataframe and return it. 
    '''
    forecast = pred.copy()
    columns = train.columns
    for col in columns[1:]:
        forecast[str(col)+'_pred'] = train[col].iloc[-1] + forecast[str(col)+'_pred'].cumsum()
    return forecast