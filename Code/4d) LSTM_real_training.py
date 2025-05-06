import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import gc
import random

import tensorflow as tf
from tabulate import tabulate
from tensorflow.keras.models import load_model


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler



def train_lstm(x_train, x_test, y_train, i, params):

    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)


    bs = params['batch_size']

    if i == 1 :
        file_name = 'LSTM_V'

        model = load_model('models/LSTM/npi_LSTM_f_V.keras')

    elif i == 2 :
        file_name = 'LSTM_N'

        model = load_model('models/LSTM/npi_LSTM_f_N.keras')



    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=8, restore_best_weights=True)

    history = model.fit(x_train, y_train, epochs=50, validation_split=0.2,
                        batch_size=bs, verbose=1, shuffle=True, callbacks=[es]).history

    model.save('models/LSTM/real_'+file_name+'.keras')

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('model loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


    model = load_model('models/LSTM/real_'+file_name+'.keras')

    pred_test = model.predict(x_test)

    np.save('predictions/LSTM/real_'+file_name+'_predictions.npy', pred_test)


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



#Load all the hyperparameter files

with open('Hyperparameters/LSTM/params_f_V.json', 'r') as f:
    params_V = json.load(f)

with open('Hyperparameters/LSTM/params_f_N.json', 'r') as f:
    params_N = json.load(f)


#train with the Vestas historic dataset and predict on the forecasted test dataset


x_train_V = np.load('LSTM input/real_x_train_Vestas.npy')
y_train_V = np.load('LSTM input/real_y_train_Vestas.npy')
x_test_V = np.load('LSTM input/real_x_test_Vestas.npy')

train_lstm(x_train_V, x_test_V, y_train_V, 1, params_V)

del x_train_V, y_train_V, x_test_V
gc.collect()



#train with the Nordex historic dataset and predict on the forecasted test dataset

x_train_N = np.load('LSTM input/real_x_train_Nordex.npy')
y_train_N = np.load('LSTM input/real_y_train_Nordex.npy')
x_test_N = np.load('LSTM input/real_x_test_Nordex.npy')

train_lstm(x_train_N, x_test_N, y_train_N, 2, params_N)

del x_train_N, y_train_N, x_test_N
gc.collect()


