import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.layers.legacy_rnn.rnn_cell_impl import BasicLSTMCell
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization, GRU, Embedding
from sklearn.metrics import mean_squared_error, median_absolute_error, mean_absolute_percentage_error, r2_score
import math
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm.notebook import tnrange
import talib
from data.data import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import MaxPooling1D, Flatten, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.optimizers import Adam

def cnn():
    lr = 0.001
    save_best = ModelCheckpoint("cnn_best_weights.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
    model_cnn = Sequential()
    model_cnn.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Flatten())
    model_cnn.add(Dropout(0.3))
    model_cnn.add(Dense(50, activation='elu'))
    model_cnn.add(Dense(1, activation='linear'))

    # model_cnn.add(Dense(50, activation='relu'))
    # model_cnn.add(Dense(1))
    model_cnn.compile(loss='mse', optimizer=Adam(lr), metrics=['mae'])


def model():
    save_best = ModelCheckpoint("GRU_best_weights.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, min_lr=0.00001, verbose=1)
    CNN_model = cnn()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    return CNN_model, optimizer


if __name__ == '__main__':
    model, optimizer = model()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])