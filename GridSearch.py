from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.layers.legacy_rnn.rnn_cell_impl import BasicLSTMCell
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , LSTM , Conv2D, Reshape

def create_model(lstm_units, dropout_rate, dense_units):
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', input_shape=(None,Xtrain.shape[2], 1)))
    model.add(Dropout(dropout_rate))
    model.add(Reshape(target_shape=(30, 128)))
    model.add(LSTM(units=lstm_units, dropout=dropout_rate, return_sequences=False))
    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, verbose=0)

param_grid = {
    'lstm_units': [64, 128, 256],
    'dropout_rate': [0.2, 0.3, 0.4],
    'dense_units': [32, 64, 128]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
Y
grid_search.fit(Xtrain, Ytrain)

best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

model = create_model(lstm_units=best_params['lstm_units'], dropout_rate=best_params['dropout_rate'], dense_units=best_params['dense_units'])
