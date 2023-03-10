import optuna

def create_model(trial):
    model = Sequential()
    model.add(LSTM(units=trial.suggest_int('lstm_units', 16, 256, log=True),
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(trial.suggest_float('dropout', 0.0, 0.5)))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)),
                  loss='mean_squared_error', metrics=['mae'])
    return model

def objective(trial):
    model = create_model(trial)

    # Train the model
    batch_size = trial.suggest_int('batch_size', 16, 128, log=True)
    epochs = trial.suggest_int('epochs', 10, 100)
    history = model.fit(X_train, Y_train, 
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, Y_test),
                        callbacks=[
                            ModelCheckpoint(filepath=f'{trial.number}_best_model.h5', monitor='val_loss', save_best_only=True),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
                        ])

    # Evaluate the model
    evaluation(model, f'model_trial_{trial.number}', history, X_train, X_test)

    # Return the validation loss
    return history.history['val_loss'][-1]

# Set the seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Split the data into training and test sets
train_length = int(len(data) * 0.8)
test_length = len(data) - train_length
train_data = data.iloc[:train_length]
test_data = data.iloc[train_length:]
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)
X_train, Y_train = create_dataset(train_scaled, look_back=30)
X_test, Y_test = create_dataset(test_scaled, look_back=30)

# Run the optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and the corresponding validation loss
print('Best trial:')
best_trial = study.best_trial
print(f'  Value: {best_trial.value:.5f}')
print('  Params: ')
for key, value in best_trial.params.items():
    print(f'    {key}: {value}')
