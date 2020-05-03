# OIL-PRICES
# EDA and Time Series Forecasting Brent Oil Prices



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
from matplotlib import pyplot as plt

# Input Data

df = pd.read_csv('C:\\Users\\Siddharth\\Downloads\\BrentOilPrices.csv',engine='python',encoding='latin')

print(df.head())

# Data Preprocessing
# 1) Need to convert Date column to standard format

df['Date'] = pd.to_datetime(df['Date'], format="%b %d, %Y")
print(df.head())

# Data Exploration¶
# Visualizing Full Data as a line plot

g = sns.lineplot(x='Date',y='Price',data = df)
plt.title("Brent Oil Price Trend")
plt.show()

# Function to plot Oil Price Trend between specific period

def plot_price_trend(df, start_date, end_date):
    """
    This function filters the dataframe for the specified date range and
    plots the line plot of the data using seaborn.

    The dataframe may not be indexed on any Datetime column.
    In this case, we use mask to filter out the date.

    PS - There is another function provided later in the notebook
    which used indexed column to filter data
    """
    mask = (df['Date'] > start_date) & (df['Date'] <= end_date)
    sdf = df.loc[mask]
    plt.figure(figsize=(10, 5))
    chart = sns.lineplot(x='Date', y='Price', data=sdf)
    #     chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
    plt.title("Brent Oil Price Trend")
    plt.show()

plot_price_trend(df,'2010-01-01','2019-01-01')

# Forecast Model

# 2) Using ARIMA¶
# Step 1) - First we import the required libraries

from statsmodels.tsa.arima_model import ARIMA    # ARIMA Modeling
from statsmodels.tsa.stattools import adfuller   # Augmented Dickey-Fuller Test for Checking Stationary
from statsmodels.tsa.stattools import acf, pacf  # Finding ARIMA parameters using Autocorrelation
from statsmodels.tsa.seasonal import seasonal_decompose # Decompose the ARIMA Forecast model

# Step 2) - Arima requires the date column to be set as index

arima_df = df
arima_df.columns = ['ds','y']
arima_df.head()

arima_df = df.set_index('ds')
arima_df.head()

print(arima_df.head())

# Step 3) - Next we write a function that plots the Rolling mean and standard deviation and then
# checks the stationarity of the time series using Augmented Dickey - Fuller Test
#
# Credit - https://www.kaggle.com/freespirit08/time-series-for-beginners-with-arima

# Perform Augmented Dickey–Fuller test to check if the given Time series is stationary:
def test_stationarity(ts):
    # Determing rolling statistics
    rolmean = ts.rolling(window=12).mean()
    rolstd = ts.rolling(window=12).std()

    # Plot rolling statistics:
    orig = plt.plot(ts, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts['y'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


#Step 4) - Next, we use this function to check if our given timeseries data is stationary or not

test_stationarity(arima_df)

# Observation - The null hypothesis of ADF test is the Time series is NOT stationary. We see that the
# Test Statistic (-1.95) is higher than 10% Critical Value (-2.56). This means this result is statistically
# significant at 90% confidence interval and so, we fail to reject the null hypothesis.
#
# This means that our time series data is NOT stationary.
#
# Step 5) - Some definitions -
#
# Correlation - Describes how much two variables depend on each other.
#
# Partial Correlation - When multiple variables are involved, two variables may have direct relation as well as
# indirect relation (i.e x1 and x3 are related and x2 and x3 are related. Due to this indirect relation, x1 and
# x2 might be related). This is called partial correlation.
#
# Auto Correlation - In a time series data, variable at a time step is dependent upon its lag values.
# This is called auto-correlation (i.e. variable depending upon its own values)
#
# Partial Autocorrelation - describes correlation of a variable with its lag values after removing the effect of indirect correlation.

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(arima_df)
plot_pacf(arima_df)
plt.show()


# Implementing own function to create ACF plot
def get_acf_plot(ts):
    # calling acf function from stattools
    y = ts['y']
    lag_acf = acf(y, nlags=500)
    plt.figure(figsize=(16, 7))
    plt.plot(lag_acf, marker="o")
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(y)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(y)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')
    plt.xlabel('number of lags')
    plt.ylabel('correlation')
    plt.show()


def get_pacf_plot(ts):
    # calling pacf function from stattools
    y = arima_df['y']
    lag_pacf = pacf(y, nlags=50)
    plt.figure(figsize=(16, 7))
    plt.plot(lag_pacf, marker="o")
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(y)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(y)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.xlabel('number of lags')
    plt.ylabel('correlation')
    plt.show()


get_acf_plot(arima_df)
get_pacf_plot(arima_df)

# Step 6) - Next we see some methods to make the data stationary

# Log Transformation
ts_log = np.log(arima_df)
plt.plot(ts_log)
plt.show()

# Moving Average of last 12 values
moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log,color = 'blue')
plt.plot(moving_avg, color='red')
plt.show()

# Differencing
ts_log_ma_diff = ts_log - moving_avg
ts_log_ma_diff.head(12)
print(ts_log_ma_diff.head(12))  # Having some NaN terms so we remove them in next step

ts_log_ma_diff.dropna(inplace=True)
test_stationarity(ts_log_ma_diff)


# Exponentially weighted moving average
expwighted_avg = ts_log.ewm(halflife=12).mean()

plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')
ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
plt.show()

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

################ DECOMPOSITION ############################################################

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log, period = 30)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)

print(ts_log_decompose)


######################################## FITTING ARIMA MODEL ############################################3
model = ARIMA(ts_log, order=(2, 1, 2))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plt.show()

