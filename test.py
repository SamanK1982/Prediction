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


def PredictStockPrice(Model , DataFrame , PreviousDate , feature_length = 32):
    idx_location = DataFrame.index.get_loc(PreviousDate)
    Features = DataFrame.iloc[idx_location - feature_length : idx_location,:].values
    Features = np.expand_dims(Features , axis = 0)
    Features = Feature_Scaler.transform(Features)
    Prediction = Model.predict(Features)
    Prediction = Target_Scaler.inverse_transform(Prediction)
    return Prediction[0][0]

def inference():
    model = load_model()
    PredictStockPrice(loaded_model, data, '2023-01-17')
    floss, fmae = model.evaluate(Xtrain, Ytrain)
    print('Loss = ', '{0:.10f}'.format(floss))
    print('MAE = ', '{0:.10f}'.format(fmae))


if __name__ == '__main__':
    metric=pd.DataFrame()
    Actual = Target_Scaler.inverse_transform(Ytest)
    Actual = np.squeeze(Actual , axis = 1)

    Predictions = Target_Scaler.inverse_transform( GRU_model.predict(Xtest))
    gru_Predictions = np.squeeze(Predictions , axis = 1)
    gru=Metrics(GRU_model,'GRU',Actual, gru_Predictions)
    Predictions = Target_Scaler.inverse_transform( model_lstm.predict(Xtest))
    lstm_Predictions = np.squeeze(Predictions , axis = 1)
    LSTM=Metrics(model_lstm,'LSTM',Actual, lstm_Predictions)
    Predictions = Target_Scaler.inverse_transform( model_cnn.predict(Xtest))
    cnn_Predictions = np.squeeze(Predictions , axis = 1)
    CNN=Metrics(model_cnn,'CNN',Actual, cnn_Predictions)
    X_train_series_sub = Xtrain.reshape((Xtrain.shape[0], 32, 3, 1))
    X_valid_series_sub = Xtest.reshape((Xtest.shape[0], 32, 3, 1))
    Predictions = Target_Scaler.inverse_transform( model_cnn_lstm.predict(X_valid_series_sub))
    cnn_lstm_Predictions = np.squeeze(Predictions , axis = 1)
    CNN_LSTM=Metrics(model_cnn_lstm,'CNN_LSTM',Actual, cnn_lstm_Predictions)
    Predictions = Target_Scaler.inverse_transform( model_cnn_gru.predict(Xtest))
    cnn_gru_Predictions = np.squeeze(Predictions , axis = 1)
    CNN_GRU=Metrics(model_cnn_gru,'CNN_GRU',Actual, cnn_gru_Predictions)

    metric=pd.concat([CNN,gru,LSTM,CNN_GRU,CNN_LSTM],axis=0)

    #visualize and save metrics
    metric.to_csv('drive/My Drive/k2 variable stars_ver1/metrics.csv')

    p = sns.catplot(kind='bar', data=metric, x='model', y='R2', height=5, aspect=1.5)
    p = sns.catplot(kind='bar', data=metric, x='model', y='rmse', height=5, aspect=1.5)
    p = sns.catplot(kind='bar', data=metric, x='model', y='mse', height=5, aspect=1.5)
    p = sns.catplot(kind='bar', data=metric, x='model', y='mae', height=5, aspect=1.5)
    p = sns.catplot(kind='bar', data=metric, x='model', y='mape', height=5, aspect=1.5)

    ax = metric.plot.bar(x='model', y=['mae', 'rmse'], figsize=(12, 6))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[-test_length:], y=Actual, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=data.index[-test_length:], y=gru_Predictions, mode='lines', name='GRU'))
    fig.add_trace(go.Scatter(x=data.index[-test_length:], y=lstm_Predictions, mode='lines', name='LSTM'))
    fig.add_trace(go.Scatter(x=data.index[-test_length:], y=cnn_Predictions, mode='lines', name='CNN'))
    fig.add_trace(go.Scatter(x=data.index[-test_length:], y=cnn_lstm_Predictions, mode='lines', name='CNN_LSTM'))
    fig.add_trace(go.Scatter(x=data.index[-test_length:], y=cnn_gru_Predictions, mode='lines', name='CNN_GRU'))

    fig.update_layout(
        title={
            'text': f'Evaluating different models perfomance in the Test set',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    fig.show()