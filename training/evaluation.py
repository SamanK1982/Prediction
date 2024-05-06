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

def evaluation(model,model_name,history,Xtrain,Xtest):
  print('=================================================================')
  print(f'------------------------------{model_name}--------------------------------')
  print('=================================================================')
  model.summary()
  Predictions = Target_Scaler.inverse_transform( model.predict(Xtest))
  Actual = Target_Scaler.inverse_transform(Ytest)
  Predictions = np.squeeze(Predictions , axis = 1)
  Actual = np.squeeze(Actual , axis = 1)
  print('=================================================================')
  print('--------')
  print('METRICS')
  print('--------')
  Metrics(model,model_name,Actual, Predictions)
  print('=================================================================')

  plt.rcParams["figure.figsize"] = (40,8)
  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.suptitle('Learning Curve')

  ax1.plot(history.history['mae'])
  ax1.plot(history.history['val_mae'])
  ax1.set_title('model MAE')
  ax1.set_ylabel('MAE')
  ax1.set_xlabel('epoch')
  ax1.legend(['Train', 'Validation'], loc='upper left')

  ax2.plot(history.history['loss'])
  ax2.plot(history.history['val_loss'])
  ax2.set_title('model MSE')
  ax2.set_ylabel('MSE')
  ax2.set_xlabel('epoch')
  ax2.legend(['Train', 'Validation'], loc='upper left')

  fig = go.Figure()
  fig.add_trace(go.Scatter(x = data.index[-test_length:] , y = Actual , mode = 'lines' , name='Actual'))
  fig.add_trace(go.Scatter(x = data.index[-test_length:] , y = Predictions , mode = 'lines' , name=f'Predicted_by_{model_name}'))
  fig.update_layout(
      title={
        'text': f'Evaluating {model_name} model perfomance in the Test set',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
  fig.show()
  Total_features = np.concatenate((Xtrain , Xtest) , axis = 0)
  Total_Targets = np.concatenate((Ytrain , Ytest) , axis = 0)
  total_Predictions = model.predict(Total_features)
  total_Predictions = Target_Scaler.inverse_transform(total_Predictions)
  total_Actual = Target_Scaler.inverse_transform(Total_Targets)
  total_Predictions = np.squeeze(total_Predictions , axis = 1)
  total_Actual = np.squeeze(total_Actual , axis = 1)
  fig = go.Figure()
  fig.add_trace(go.Scatter(x = data.index , y = total_Actual , mode = 'lines' , name='Actual'))
  fig.add_trace(go.Scatter(x = data.index , y = total_Predictions , mode = 'lines' , name=f'Predicted_by_{model_name}'))
  fig.update_layout(
    title={
        'text': f'Evaluating {model_name} model perfomance in the entire Dataset',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
  fig.show()

def Metrics(model,model_name,y_test, prediction):
  rmse = math.sqrt(mean_squared_error(y_test, prediction))
  mse=mean_squared_error(y_test, prediction)
  mae=median_absolute_error(y_test, prediction)
  mape=mean_absolute_percentage_error(y_test, prediction)
  r2=r2_score(y_test, prediction)
  result=pd.DataFrame({'model':model_name,'R2':r2,'rmse':rmse,'mse':mse,'mae':mae,'mape':mape},index=[0])
  display(result)
  return result