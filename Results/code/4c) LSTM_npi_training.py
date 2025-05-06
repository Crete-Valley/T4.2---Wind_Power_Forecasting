import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import gc
import random

import tensorflow as tf
from tabulate import tabulate
from tensorflow.keras.models import load_model


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred))))


def create_LSTM_model(x_train, params):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])))
    model.add(tf.keras.layers.LSTM(params['lstm_units_1'], activation='relu', return_sequences=True))
    model.add(tf.keras.layers.LSTM(params['lstm_units_2'], activation="relu", return_sequences=True))
    model.add(tf.keras.layers.LSTM(params['lstm_units_3'], activation="relu", return_sequences=False))
    model.add(tf.keras.layers.Dense(params['dense_units_1'], activation='relu'))
    model.add(tf.keras.layers.Dropout(params['dropout_rate_1']))
    model.add(tf.keras.layers.Dense(params['dense_units_2'], activation='relu'))
    model.add(tf.keras.layers.Dropout(params['dropout_rate_2']))
    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model





def train_lstm(x_train, x_test, y_train, y_test, i, scaler, params):

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)


    bs = params['batch_size']

    if i == 1 :
        file_name = 'LSTM_h_V'
    elif i == 2 :
        file_name = 'LSTM_f_V'
    elif i == 3 : 
        file_name = 'LSTM_h_N'
    elif i == 4 :
        file_name = 'LSTM_f_N'


    model = create_LSTM_model(x_train, params)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10, restore_best_weights=True)

    history = model.fit(x_train, y_train, epochs=50, validation_split=0.2,
                        batch_size=bs, verbose=1, shuffle=True, callbacks=[es]).history

    model.save('models/LSTM/npi_'+file_name+'.keras')

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('model loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


    model = load_model('models/LSTM/npi_'+file_name+'.keras')

    pred_test = model.predict(x_test)

    np.save('predictions/LSTM/npi_'+file_name+'_predictions.npy', pred_test)


    del history, model
    tf.keras.backend.clear_session()


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


#Load all the scalers

with open('scalers/LSTM_V_h_scaler.pkl', 'rb') as f:
    scaler_h_V = pickle.load(f)
with open('scalers/LSTM_V_f_scaler.pkl', 'rb') as f:
    scaler_f_V = pickle.load(f)
with open('scalers/LSTM_N_h_scaler.pkl', 'rb') as f:
    scaler_h_N = pickle.load(f)
with open('scalers/LSTM_N_f_scaler.pkl', 'rb') as f:
    scaler_f_N = pickle.load(f)


#Load all the hyperparameter files

with open('Hyperparameters/LSTM/params_h_V.json', 'r') as f:
    params_h_V = json.load(f)

with open('Hyperparameters/LSTM/params_f_V.json', 'r') as f:
    params_f_V = json.load(f)

with open('Hyperparameters/LSTM/params_h_N.json', 'r') as f:
    params_h_N = json.load(f)

with open('Hyperparameters/LSTM/params_f_N.json', 'r') as f:
    params_f_N = json.load(f)


#train with the Vestas historic dataset and predict on the forecasted test dataset


x_train_h_V = np.load('LSTM input/npi_x_train_history-forecast_Vestas.npy')
y_train_h_V = np.load('LSTM input/npi_y_train_history-forecast_Vestas.npy')
x_test_f_V = np.load('LSTM input/npi_x_test_history-forecast_Vestas.npy')
y_test_f_V = np.load('LSTM input/npi_y_test_history-forecast_Vestas.npy')

train_lstm(x_train_h_V, x_test_f_V, y_train_h_V, y_test_f_V, 1, scaler_h_V, params_h_V)

del x_train_h_V, y_train_h_V, x_test_f_V, y_test_f_V
gc.collect()


#train with the Vestas forecasted dataset and predict on the forecasted test dataset

x_train_f_V = np.load('LSTM input/npi_x_train_forecast-forecast_Vestas.npy')
y_train_f_V = np.load('LSTM input/npi_y_train_forecast-forecast_Vestas.npy')
x_test_f_V = np.load('LSTM input/npi_x_test_forecast-forecast_Vestas.npy')
y_test_f_V = np.load('LSTM input/npi_y_test_forecast-forecast_Vestas.npy')

train_lstm(x_train_f_V, x_test_f_V, y_train_f_V, y_test_f_V, 2, scaler_f_V, params_f_V)

del x_train_f_V, y_train_f_V, x_test_f_V, y_test_f_V
gc.collect()


#train with the Nordex historic dataset and predict on the forecasted test dataset

x_train_h_N = np.load('LSTM input/npi_x_train_history-forecast_Nordex.npy')
y_train_h_N = np.load('LSTM input/npi_y_train_history-forecast_Nordex.npy')
x_test_f_N = np.load('LSTM input/npi_x_test_history-forecast_Nordex.npy')
y_test_f_N = np.load('LSTM input/npi_y_test_history-forecast_Nordex.npy')

train_lstm(x_train_h_N, x_test_f_N, y_train_h_N, y_test_f_N, 3, scaler_h_N, params_h_N)

del x_train_h_N, y_train_h_N, x_test_f_N, y_test_f_N
gc.collect()


#train with the Nordex forecast dataset and predict on the forecasted test dataset

x_train_f_N = np.load('LSTM input/npi_x_train_forecast-forecast_Nordex.npy')
y_train_f_N = np.load('LSTM input/npi_y_train_forecast-forecast_Nordex.npy')
x_test_f_N = np.load('LSTM input/npi_x_test_forecast-forecast_Nordex.npy')
y_test_f_N = np.load('LSTM input/npi_y_test_forecast-forecast_Nordex.npy')

train_lstm(x_train_f_N, x_test_f_N, y_train_f_N, y_test_f_N, 4, scaler_f_N, params_f_N)

del x_train_f_N, y_train_f_N, x_test_f_N, y_test_f_N
gc.collect()

