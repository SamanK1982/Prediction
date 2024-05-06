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
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm.notebook import tnrange
import talib
from data.data import *
# url = 'https://launchpad.net/~mario-mariomedina/+archive/ubuntu/talib/+files'
# ext = '0.4.0-oneiric1_amd64.deb -qO'
# !wget $url/libta-lib0_$ext libta.deb
# !wget $url/ta-lib0-dev_$ext ta.deb
# !dpkg -i libta.deb ta.deb
# !pip install ta-lib


def data_features():
    data = load_data()
    # Filter only required data
    data = data[['Close', 'Volume']]

def technical_indicator():
    data = load_data()
    df = data

    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
    df['RSI_14'] = df['RSI_14'].replace(np.nan, 50)
    df.isna().sum()
    rsi_data = df['RSI_14']
    rsi_data.columns = ['Date', 'RSI']
    rsi_data = rsi_data[rsi_data.index >= '2014-10-01']
    rsi_data = rsi_data.astype(np.float64)

    return data.merge(rsi_data, left_index=True, right_index=True, how='inner')

def test_set():
    # Confirm the Testing Set length
    data = technical_indicator()
    test_length = data[(data.index >= '2022-03-01')].shape[0]



