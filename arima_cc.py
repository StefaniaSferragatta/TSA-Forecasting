import pandas as pd 
import numpy as np
from datetime import datetime as dt
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def arima(train : pd.core.series.Series,test : pd.core.series.Series,p : int):
    ''' 
    Function that takes in input:
    - the trainset
    - the test set
    - the value of p (AR)
    
    and defines the ARIMA model, fits it, gets the forecast.
    It returns the predicted values, the lower and the upper series.
    '''
    model = ARIMA(train, order=(p, 1, 1))  
    fitted = model.fit()  

    fc = fitted.get_forecast(len(test))  
    conf = fc.conf_int(alpha=0.05) # 95% confidence

    fc_series = pd.Series(fc.predicted_mean, index=test.index)
    lower_series = pd.Series(conf.iloc[:, 0], index=test.index)
    upper_series = pd.Series(conf.iloc[:, 1], index=test.index)
    return fc_series,lower_series,upper_series

# Plot
def plot(train: pd.core.series.Series,test : np.ndarray, fc_series : pd.core.series.Series,lower_series : pd.core.series.Series,upper_series : pd.core.series.Series, t : str):
    '''
    Function that takes in input:
    - the trainset
    - the test set values
    - the forecasted set
    - the lower bound set
    - the upper bound set
    - the string containing the feature's name

    Plot the forecast VS actuals for each feature with a grey shadow indicates the confidence interval.
    '''
    plt.figure(figsize=(12,5), dpi=200)
    plt.plot(train, label='training', color = 'lightcoral')
    plt.plot(test, label='actual',color = 'peachpuff',linestyle = 'dashed')
    plt.plot(fc_series, label='forecast',color = 'mediumaquamarine')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='darkgrey', alpha=.15)
    plt.title(f'Forecast vs Actuals for {t}')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    return

def forecast_accuracy(forecast : pd.core.series.Series, actual : np.ndarray):
    ''' 
    Function that takes in input:
    - forecast set
    - actual values

    and perform an evaluation of the metod computing the following measures: MAE,RMSE, correlation coefficient.
    '''
    mae = np.mean(np.abs(forecast - actual))    # MAE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr

    return({'mae': mae, 'rmse':rmse,'corr':corr})