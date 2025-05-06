import pandas as pd
import numpy as np
import optuna
import json
import tensorflow as tf
import gc
import random

def create_model_Vestas(trial):

    lstm_units_1 = trial.suggest_categorical('lstm_units_1', [256, 512])
    lstm_units_2 = trial.suggest_categorical('lstm_units_2', [128, 256])
    lstm_units_3 = trial.suggest_categorical('lstm_units_3', [64, 128])
    dense_units_1 = trial.suggest_categorical('dense_units_1', [256, 512])
    dense_units_2 = trial.suggest_categorical('dense_units_2', [256, 512])
    dropout_rate_1 = trial.suggest_categorical('dropout_rate_1', [0.3, 0.4, 0.5])
    dropout_rate_2 = trial.suggest_categorical('dropout_rate_2', [0.3, 0.4, 0.5])
    learning_rate = trial.suggest_categorical('learning_rate', [0.0005,0.001,0.0015])

    inputs = tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    x = tf.keras.layers.LSTM(lstm_units_1, activation='relu', return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(lstm_units_2, activation="relu", return_sequences=True)(x)
    x = tf.keras.layers.LSTM(lstm_units_3, activation="relu", return_sequences=False)(x)
    x = tf.keras.layers.Dense(dense_units_1, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate_1)(x)
    x = tf.keras.layers.Dense(dense_units_2, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate_2)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model


def create_model_Nordex(trial):

    lstm_units_1 = trial.suggest_categorical('lstm_units_1', [128, 256])
    lstm_units_2 = trial.suggest_categorical('lstm_units_2', [64, 128])
    lstm_units_3 = trial.suggest_categorical('lstm_units_3', [32, 64])
    dense_units_1 = trial.suggest_categorical('dense_units_1', [128, 256])
    dense_units_2 = trial.suggest_categorical('dense_units_2', [128, 256])
    dropout_rate_1 = trial.suggest_categorical('dropout_rate_1', [0.3, 0.4, 0.5])
    dropout_rate_2 = trial.suggest_categorical('dropout_rate_2', [0.3, 0.4, 0.5])
    learning_rate = trial.suggest_categorical('learning_rate', [0.0005,0.001,0.0015])

    
    inputs = tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    x = tf.keras.layers.LSTM(lstm_units_1, activation='relu', return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(lstm_units_2, activation="relu", return_sequences=True)(x)
    x = tf.keras.layers.LSTM(lstm_units_3, activation="relu", return_sequences=False)(x)
    x = tf.keras.layers.Dense(dense_units_1, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate_1)(x)
    x = tf.keras.layers.Dense(dense_units_2, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate_2)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model



def objective_Vestas(trial):
    
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = create_model_Vestas(trial)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10, restore_best_weights=True)

    # Suggest batch size
    batch_size = trial.suggest_categorical('batch_size', [256, 512])

    history = model.fit(x_train, y_train, epochs=50, validation_split=0.2,
                        batch_size=batch_size, verbose=0, shuffle=True, callbacks=[es])

    # Return the validation loss of the last epoch
    val_loss = min(history.history['val_loss'])
    
    tf.keras.backend.clear_session()
    del model, history
    gc.collect

    return val_loss



def objective_Nordex(trial):

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = create_model_Nordex(trial)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10, restore_best_weights=True)

    # Suggest batch size
    batch_size = trial.suggest_categorical('batch_size', [128, 256])

    history = model.fit(x_train, y_train, epochs=50, validation_split=0.2,
                        batch_size=batch_size, verbose=0, shuffle=True, callbacks=[es])

    # Return the validation loss of the last epoch
    val_loss = min(history.history['val_loss'])
    
    tf.keras.backend.clear_session()
    del model, history
    gc.collect

    return val_loss



def best_param(data, filename):
    study = optuna.create_study(direction='minimize')
    # Start the optimization
    if data == 'Vestas':
        study.optimize(objective_Vestas, n_trials=50, gc_after_trial=True, show_progress_bar=True)
    elif data == 'Nordex':
        study.optimize(objective_Nordex, n_trials=50, gc_after_trial=True, show_progress_bar=True)

    # Print the best parameters
    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    with open(filename, 'w') as f:
        json.dump(trial.params, f)




# Limiting GPU memory growth
# Ref: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)]) # Notice here
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)




#VESTAS OPTUNA TRIAL

x_train = np.load('LSTM input/x_train_history-forecast_Vestas.npy')
y_train = np.load('LSTM input/y_train_history-forecast_Vestas.npy')

best_param('Vestas', 'Hyperparameters/LSTM/params_h_V.json')

del x_train, y_train
gc.collect()


x_train = np.load('LSTM input/x_train_forecast-forecast_Vestas.npy')
y_train = np.load('LSTM input/y_train_forecast-forecast_Vestas.npy')

best_param('Vestas', 'Hyperparameters/LSTM/params_f_V.json')

del x_train, y_train
gc.collect()


#NORDEX OPTUNA TRIAL

x_train = np.load('LSTM input/x_train_history-forecast_Nordex.npy')
y_train = np.load('LSTM input/y_train_history-forecast_Nordex.npy')

best_param('Nordex', 'Hyperparameters/LSTM/params_h_N.json')

del x_train, y_train
gc.collect()


x_train = np.load('LSTM input/x_train_forecast-forecast_Nordex.npy')
y_train = np.load('LSTM input/y_train_forecast-forecast_Nordex.npy')

best_param('Nordex', 'Hyperparameters/LSTM/params_f_N.json')

del x_train, y_train
gc.collect()

