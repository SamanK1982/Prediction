import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import ModelCheckpoint , ReduceLROnPlateau
from tensorflow.python.keras.layers.legacy_rnn.rnn_cell_impl import BasicLSTMCell
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , LSTM , Bidirectional , BatchNormalization, GRU, Embedding
from sklearn.metrics import mean_squared_error, median_absolute_error,mean_absolute_percentage_error,r2_score
import math
import plotly.graph_objects as go

def load_data():
    data = yf.download("BTC-USD" , start = "2014-10-01",  interval = '1d')
    #data = yf.download("^GSPC" , start = "2014-10-01",  interval = '1d')
    #data = yf.download("^AMX" , start = "2014-10-01",  interval = '1d')

    #data.head(3)
    # Sort the data points based on indexes just for confirmation
    data.sort_index(inplace = True)

    # Remove any duplicate index
    data = data.loc[~data.index.duplicated(keep='first')]

    # Check for missing values
    data.isnull().sum()

    return data

def data_desc():
    data  = load_data()
    # Get the statistics of the data
    data.describe()

    # Check the trend in Closing Values
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines'))
    fig.update_layout(height=500, width=900,
                      xaxis_title='Date', yaxis_title='Close')
    fig.show()

    cols_plot = ['Open', 'High', 'Low', 'Close']
    axes = data[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(20, 11), subplots=True)
    for ax in axes:
        ax.set_ylabel('price')

def plt_params():
    data = load_data()
    plt.rcParams["figure.figsize"] = (15, 8)
    plt.plot(data['Close'], label="Close price")
    plt.xlabel("Time")
    plt.ylabel("Closing price")

def data_distribution():
    data = load_data()
    fig, ax = plt.subplots(4, 2, figsize=(15, 13))
    sns.boxplot(x=data["Close"], ax=ax[0, 0])
    sns.distplot(data['Close'], ax=ax[0, 1])
    sns.boxplot(x=data["Open"], ax=ax[1, 0])
    sns.distplot(data['Open'], ax=ax[1, 1])
    sns.boxplot(x=data["High"], ax=ax[2, 0])
    sns.distplot(data['High'], ax=ax[2, 1])
    sns.boxplot(x=data["Low"], ax=ax[3, 0])
    sns.distplot(data['Low'], ax=ax[3, 1])
    plt.tight_layout()

def cor_matrix():
    data = load_data()
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), cmap=plt.cm.Blues, annot=True)
    plt.title('correlation between prices',
              fontsize=13)
    plt.show()


if __name__ == '__main__':
    data_desc()
    plt_params()
    data_distribution()
    cor_matrix()




