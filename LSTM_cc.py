import pandas as pd
from pandas import DataFrame, Series
from pandas import concat
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow import keras

def difference(dataset: np.narray, interval=1):
	'''
	Function that takes in input:
	- the dataset's values 
	- the grade to differenciate (in this case 1)

	and differenciate the data to eliminate the seasonality and make the data stationary.
	It returns a series of the differenciated data.
	'''
	diff = list()
	for i in range(interval, len(dataset)):
		# the first observation in the series is skipped as there is no prior observation with which to calculate a differenced value.
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

def timeseries_to_supervised(data: pd.core.series.Series, lag=1):
    '''
    Function that takes in input:
    - numpy array of the raw time series data 
    - the number of shifted series to create and use as inputs (lag)

    It returns a new dataframe.
	'''

    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# invert differenced value
def inverse_difference(history: np.ndarray, yhat: np.ndarray, interval=1):
	'''
	Functions that takes in input:
	- the raw values
	- the predicted value
	- the number of differenciate apply to the data

	and invert the differenciating process in order to take forecasts made on the differenced 
	series back into their original scale.
	'''
	return yhat + history[-interval]

def scale(train: np.ndarray, test: np.ndarray):
	'''
	Function that takes in input:
	- train dataset
	- test dataset

	and scale them to [-1, 1] to ensure the min/max values of the test data do not influence the model.
	It returns the type of scaler used, the train and the test scaled.
	'''
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
 
def invert_scale(scaler: MinMaxScaler, X: np.ndarray, value: np.ndarray):
	'''
	Function that takes in input:
	- the scaler used to scale the data
	- the predicted value
	- the forecasted value

	and performs an inverse scaling for the given forecasted value.
	'''

	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]
 
# fit an LSTM network to training data
def fit_lstm(train: np.ndarray, batch_size: int, nb_epoch: int, neurons: int):
	'''
	Function that takes in input:
	- the trainset
	- the batch size (1)
	- the number of epochs (10)
	- the number of neurons (4)
	
	and applies the defined model to the data and return it.	
	'''
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model
 
def forecast_lstm(model: Sequential, batch_size: int, X: np.ndarray):
	'''
	Function that takes in input:
	- the model fitted
	- the number of batch's size
	- the values

	and makes a one step forecast. It returns the forecasted values.
	'''
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def robust_lstm(train_scaled: np.ndarray,test_scaled: np.ndarray,scaler: MinMaxScaler,raw_values: np.ndarray):
	'''
	Function that takes in input:
	- the train set
	- the test set
	- the scaler used to scale the data 
	- the dataset's values

	and performs the model fitting and the walk forward validation into a loop, doing it for 10 times.
	At each iteration the RMSE computed is stored and, in the end, returned.
	It's used to plot its distribution.
	'''	
	# repeat experiment
	repeats = 10
	error_scores = list()
	for r in range(repeats):
		# fit the model
		lstm_model = fit_lstm(train_scaled, 1, 10, 4)
		# forecast the entire training dataset to build up state for forecasting
		train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
		lstm_model.predict(train_reshaped, batch_size=1)	
		# walk-forward validation on the test data
		predictions = list()
		for i in range(len(test_scaled)):
			# make one-step forecast
			X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
			yhat = forecast_lstm(lstm_model, 1, X) #yhat = y
			# invert scaling
			yhat = invert_scale(scaler, X, yhat)
			# invert differencing
			yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
			# store forecast
			predictions.append(yhat)
		
		# report performance
		rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
		error_scores.append(rmse)
	return error_scores,predictions