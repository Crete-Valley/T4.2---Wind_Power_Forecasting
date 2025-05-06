import pandas as pd
import numpy as np


def convert(path, i):
    
    file = np.load('LSTM input/'+path)

    print(file.shape)
    
    if i == 1:

        modified_file = file[:,:,1:]

    elif i == 2:

        modified_file = file

    print(modified_file.shape)

    np.save('LSTM input/npi_'+path, modified_file)



convert('x_train_forecast-forecast_Nordex.npy', 1)
convert('x_test_forecast-forecast_Nordex.npy', 1)
convert('y_train_forecast-forecast_Nordex.npy', 2)
convert('y_test_forecast-forecast_Nordex.npy', 2)

convert('x_train_forecast-forecast_Vestas.npy', 1)
convert('x_test_forecast-forecast_Vestas.npy', 1)
convert('y_train_forecast-forecast_Vestas.npy', 2)
convert('y_test_forecast-forecast_Vestas.npy', 2)

convert('x_train_history-forecast_Nordex.npy', 1)
convert('x_test_history-forecast_Nordex.npy', 1)
convert('y_train_history-forecast_Nordex.npy', 2)
convert('y_test_history-forecast_Nordex.npy', 2)

convert('x_train_history-forecast_Vestas.npy', 1)
convert('x_test_history-forecast_Vestas.npy', 1)
convert('y_train_history-forecast_Vestas.npy', 2)
convert('y_test_history-forecast_Vestas.npy', 2)



