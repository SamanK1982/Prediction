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
from tensorflow.keras.callbacks import ModelCheckpoint , ReduceLROnPlateau

def gru():
    GRU_model = Sequential()
    GRU_model.add(
        GRU(64, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0.0, unroll=False, use_bias=True,
            reset_after=True))
    GRU_model.add(Dropout(0.3))
    GRU_model.add(Dense(64, activation='elu'))
    GRU_model.add(Dropout(0.3))
    GRU_model.add(Dense(32, activation='elu'))
    GRU_model.add(Dense(1, activation='linear'))

def model():
    save_best = ModelCheckpoint("GRU_best_weights.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, min_lr=0.00001, verbose=1)
    GRU_model = gru()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    return GRU_model, optimizer

if __name__ == '__main__':
    model, optimizer = model()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
