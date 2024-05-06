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
from cnn, lst, gru, snn_lstm, cnn_gru import *

def model_fit():
    GRU_history = GRU_model.fit(Xtrain, Ytrain,
                                epochs=100,
                                batch_size=64,
                                verbose=1,
                                shuffle=False,
                                validation_data=(Xtest, Ytest),
                                callbacks=[reduce_lr, save_best])
    # Load the best weights
    LSTM_history = model_lstm.fit(Xtrain, Ytrain,
                                  epochs=100,
                                  batch_size=64,
                                  verbose=1,
                                  shuffle=False,
                                  validation_data=(Xtest, Ytest),
                                  callbacks=[reduce_lr, save_best])
    cnn_history = model_cnn.fit(Xtrain, Ytrain,
                                epochs=100,
                                batch_size=64,
                                verbose=1,
                                shuffle=False,
                                validation_data=(Xtest, Ytest),
                                callbacks=[reduce_lr, save_best])
    model_cnn_gru_history = model_cnn_gru.fit(X_train_series_sub, Ytrain,
                                              epochs=100,
                                              batch_size=64,
                                              verbose=1,
                                              shuffle=False,
                                              validation_data=(X_valid_series_sub, Ytest))
    X_train_series_sub = Xtrain.reshape((Xtrain.shape[0], 32, 3, 1))
    X_valid_series_sub = Xtest.reshape((Xtest.shape[0], 32, 3, 1))

    cnn_lstm_history = model_cnn_lstm.fit(X_train_series_sub, Ytrain,
                                          epochs=100,
                                          batch_size=32,
                                          verbose=1,
                                          shuffle=False,
                                          validation_data=(X_valid_series_sub, Ytest))


    return GRU_model, GRU_history

if __name__ == '__main__':
    Xtrain, Xtest, Ytrain, Ytest = X[:-test_length], X[-test_length:], Y[:-test_length], Y[-test_length:]

    Feature_Scaler = MultiDimensionScaler()
    Xtrain = Feature_Scaler.fit_transform(Xtrain)
    Xtest = Feature_Scaler.transform(Xtest)

    Target_Scaler = MinMaxScaler()
    Ytrain = Target_Scaler.fit_transform(Ytrain.reshape(-1, 1))
    Ytest = Target_Scaler.transform(Ytest.reshape(-1, 1))

    X_train_series_sub = Xtrain.reshape((Xtrain.shape[0], 32, 3, 1))
    X_valid_series_sub = Xtest.reshape((Xtest.shape[0], 32, 3, 1))

    model, history = model_fit()
    evaluation(model, 'GRU', history, Xtrain, Xtest)