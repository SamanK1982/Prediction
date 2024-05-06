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
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Conv1D, \
    RepeatVector, Reshape, Conv2D
from tensorflow.keras.optimizers import Adam

def cnn_gru():
    model_cnn_gru = Sequential()
    model_cnn_gru.add(Conv2D(128, (3, 3), input_shape=(None, Xtrain.shape[2], 1)))
    model_cnn_gru.add(Dropout(0.2))
    model_cnn_gru.add(Reshape(target_shape=(30, 128)))
    model_cnn_gru.add(GRU(128, return_sequences=False))
    model_cnn_gru.add(Dense(128, activation='relu'))
    model_cnn_gru.add(Dense(1))
    model_cnn_gru.compile(loss='mse', optimizer=optimizer, metrics=['mae'])


def model():
    save_best = ModelCheckpoint("GRU_best_weights.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, min_lr=0.00001, verbose=1)
    CNN_GRU_model = cnn_gru()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    return CNN_GRU_model, optimizer


if __name__ == '__main__':
    # X_train_series_sub = Xtrain.reshape((Xtrain.shape[0], 32, 3, 1))
    # X_valid_series_sub = Xtest.reshape((Xtest.shape[0], 32, 3, 1))
    model, optimizer = model()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])