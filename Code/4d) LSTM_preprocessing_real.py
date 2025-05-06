
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler

def split_sequences(sequences, n_steps, n_outputs):
    X = np.empty((len(sequences) - n_steps, n_steps, sequences.shape[1]), dtype=np.float32)
    y = np.empty((len(sequences) - n_steps, n_outputs), dtype=np.float32)

    j = 0
    count = 0
    print("reached_here")
    for i in range(len(sequences)):
        if i//500 > j:
            print(i)
            j=j+1
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix < len(sequences):
            # gather input and output parts of the pattern

            if (len(sequences.iloc[i:end_ix, :]) == 30 and len(sequences.iloc[end_ix:(end_ix+n_outputs), -1]) == 1):
                seq_x = sequences.iloc[i:end_ix, :].values
                seq_y = sequences.iloc[end_ix:(end_ix+n_outputs), -1].values
                X[count] = seq_x
                y[count] = seq_y
                count += 1
            else :
                #debugging prints
                print(i)
                print(sequences.iloc[i:end_ix, :])
                print(sequences.iloc[end_ix:(end_ix+n_outputs), -1])

    return X, y


def unique_shapes(x, y, lag_, n_features_, num_of_outputs_):
    uniuqe_shapes = []
    for k in range(len(x)):

        if (x[k].shape == (lag_, n_features_)) & (y[k].shape == (num_of_outputs_,)):
                uniuqe_shapes.append(k)

    x = x[uniuqe_shapes]
    y = y[uniuqe_shapes]
    x = np.stack(x)
    y = np.stack(y)
    return x, y


def scale_data(x_train, x_test, y_train, y_test):

    # Reshape features for scaling
    x_train_flat = x_train.reshape(-1, x_train.shape[-1])
    x_test_flat = x_test.reshape(-1, x_test.shape[-1])

    # Initialize scalers
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()

    # Fit scalers on training data
    X_scaler.fit(x_train_flat)
    Y_scaler.fit(y_train)

    # Scale features
    x_train_scaled = X_scaler.transform(x_train_flat).reshape(x_train.shape)
    x_test_scaled = X_scaler.transform(x_test_flat).reshape(x_test.shape)

    # Scale target
    y_train_scaled = Y_scaler.transform(y_train)
    y_test_scaled = Y_scaler.transform(y_test)

    return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, Y_scaler


def split(data):

    data = data[cols]
    train = data.iloc[:int(len(data)*split_ratio),:]
    test = data.iloc[int(len(data)*split_ratio):,]
    return train,test

cols = ['TheoreticalPower',
        'WindSpeed',
        'Temperature',
        'Dewpoint',
        'Rel_Humidity',
        'Pressure',
        'Month',
        'Day',
        'Week',
        'Hour',
        'Minute',
        'Season_Autumn',
        'Season_Spring',
        'Season_Summer',
        'Season_Winter',
        'MeanSpeed',
        'Direction_E',
        'Direction_N',
        'Direction_NEE',
        'Direction_NNE',
        'Direction_NNW',
        'Direction_NWW',
        'Direction_S',
        'Direction_SEE',
        'Direction_SSE',
        'Direction_SSW',
        'Direction_SWW',
        'Direction_W',
        'WindDirectionSin',
        'WindDirectionCos',
        'MeanDirectionSin',
        'MeanDirectionCos']



# Splitting factor for training set and test set
split_ratio = 0.8

lag = 30   #lookbackwindow 6*5 (6*10 minutes = 1 hour) (total 5 hours lookback)

n_features = len(cols)
num_of_outputs = 1


Nordex_data_h = pd.read_csv('datasets/Nordex_N117_preprocessed_history_data.csv', index_col=0)
Vestas_data_h = pd.read_csv('datasets/Vestas_V52_preprocessed_history_data.csv', index_col=0)

Nordex_data_f = pd.read_csv('datasets/Nordex_N117_preprocessed_forecast_data.csv', index_col=0)
Vestas_data_f = pd.read_csv('datasets/Vestas_V52_preprocessed_forecast_data.csv', index_col=0)

j=0
for i in Vestas_data_f['TheoreticalPower']:
    
    if i > 0.85:
        print(i)
    
    j=j+1

print(Nordex_data_f['TheoreticalPower'].min())
print(Nordex_data_f['TheoreticalPower'].max())
print(Vestas_data_f['TheoreticalPower'].min())
print(Vestas_data_f['TheoreticalPower'].max())

#Set values to float32 to reduce memory usage

float64_columns = Vestas_data_h.select_dtypes(include=['float64']).columns
Vestas_data_h[float64_columns] = Vestas_data_h[float64_columns].astype(np.float32)

float64_columns = Vestas_data_f.select_dtypes(include=['float64']).columns
Vestas_data_f[float64_columns] = Vestas_data_f[float64_columns].astype(np.float32)

float64_columns = Nordex_data_h.select_dtypes(include=['float64']).columns
Nordex_data_h[float64_columns] = Nordex_data_h[float64_columns].astype(np.float32)

float64_columns = Nordex_data_f.select_dtypes(include=['float64']).columns
Nordex_data_f[float64_columns] = Nordex_data_f[float64_columns].astype(np.float32)


#GETTING READY THE Vestas_data_h FOR THE LSTM TRAINING


#split into sequences
'''
train_h_V, test_h_V =  split(Vestas_data_h)
train_f_V, test_f_V =  split(Vestas_data_f)

print(train_h_V)
print(test_h_V)

print("split Vestas history")

x_train_h_V_unscaled, y_train_h_V_unscaled = split_sequences(train_h_V, n_steps=lag, n_outputs=num_of_outputs)
x_test_f_V_unscaled, y_test_f_V_unscaled = split_sequences(test_f_V, n_steps=lag, n_outputs=num_of_outputs)

#scaling

x_train_h_V, x_test_f_V, y_train_h_V, y_test_f_V, Y_scaler_h_V = scale_data(x_train_h_V_unscaled, x_test_f_V_unscaled, y_train_h_V_unscaled, y_test_f_V_unscaled)

#Reshape the input

x_train_h_V, y_train_h_V = unique_shapes(x_train_h_V, y_train_h_V, lag, n_features, num_of_outputs)
x_test_f_V, y_test_f_V = unique_shapes(x_test_f_V, y_test_f_V, lag, n_features, num_of_outputs)

#Save the data

print(x_train_h_V.shape)
print(y_train_h_V.shape)
print(x_test_f_V.shape)
print(y_test_f_V.shape)

np.save('LSTM input/x_train_history-forecast_npi_Vestas.npy', x_train_h_V)
np.save('LSTM input/y_train_history-forecast_npi_Vestas.npy', y_train_h_V)
np.save('LSTM input/x_test_history-forecast_npi_Vestas.npy', x_test_f_V)
np.save('LSTM input/y_test_history-forecast_npi_Vestas.npy', y_test_f_V)

del Y_scaler_h_V, train_h_V, train_f_V, test_f_V, test_h_V, x_train_h_V, y_train_h_V, x_test_f_V, y_test_f_V, x_train_h_V_unscaled, y_train_h_V_unscaled, x_test_f_V_unscaled, y_test_f_V_unscaled

#GETTING READY THE Vestas_data_f FOR THE LSTM TRAINING


#split into sequences

column_to_move = 'TheoreticalPower'

train_f_V, test_f_V =  split(Vestas_data_f)

train_f_V = train_f_V[[col for col in train_f_V.columns if col != column_to_move] + [column_to_move]]
test_f_V = test_f_V[[col for col in test_f_V.columns if col != column_to_move] + [column_to_move]]

train_f_V = train_f_V.tail(8700)
train_f_V = train_f_V.reset_index(drop=True)
test_f_V = test_f_V.head(400)
test_f_V = test_f_V.reset_index(drop=True)

print(train_f_V)
print(test_f_V)

print("split Vestas forecast")

x_train_f_V_unscaled, y_train_f_V_unscaled = split_sequences(train_f_V, n_steps=lag, n_outputs=num_of_outputs)
x_test_f_V_unscaled, y_test_f_V_unscaled = split_sequences(test_f_V, n_steps=lag, n_outputs=num_of_outputs)

print(x_train_f_V_unscaled)
print(y_train_f_V_unscaled)

#scaling

x_train_f_V, x_test_f_V, y_train_f_V, y_test_f_V, Y_scaler_f_V = scale_data(x_train_f_V_unscaled, x_test_f_V_unscaled, y_train_f_V_unscaled, y_test_f_V_unscaled)

with open('scalers/real_LSTM_V_scaler.pkl', 'wb') as f:
    pickle.dump(Y_scaler_f_V, f)

#Reshape the input

x_train_f_V, y_train_f_V = unique_shapes(x_train_f_V, y_train_f_V, lag, n_features, num_of_outputs)
x_test_f_V, y_test_f_V = unique_shapes(x_test_f_V, y_test_f_V, lag, n_features, num_of_outputs)

#Save the data
np.save('LSTM input/real_x_train_Vestas.npy', x_train_f_V)
np.save('LSTM input/real_y_train_Vestas.npy', y_train_f_V)
np.save('LSTM input/real_x_test_Vestas.npy', x_test_f_V)


del Y_scaler_f_V, train_f_V, test_f_V, x_train_f_V, y_train_f_V, x_test_f_V, y_test_f_V, x_train_f_V_unscaled, y_train_f_V_unscaled, x_test_f_V_unscaled, y_test_f_V_unscaled


#GETTING READY THE Nordex_data_h FOR THE LSTM TRAINING


#split into sequences

train_h_N, test_h_N =  split(Nordex_data_h)
train_f_N, test_f_N =  split(Nordex_data_f)

print("split Nordex history")

x_train_h_N_unscaled, y_train_h_N_unscaled = split_sequences(train_h_N, n_steps=lag, n_outputs=num_of_outputs)
x_test_f_N_unscaled, y_test_f_N_unscaled = split_sequences(test_h_N, n_steps=lag, n_outputs=num_of_outputs)

#scaling

x_train_h_N, x_test_f_N, y_train_h_N, y_test_f_N, Y_scaler_h_N = scale_data(x_train_h_N_unscaled, x_test_f_N_unscaled, y_train_h_N_unscaled, y_test_f_N_unscaled)

#Reshape the input

x_train_h_N, y_train_h_N = unique_shapes(x_train_h_N, y_train_h_N, lag, n_features, num_of_outputs)
x_test_f_N, y_test_f_N = unique_shapes(x_test_f_N, y_test_f_N, lag, n_features, num_of_outputs)

#Save the data
np.save('LSTM input/x_train_history-forecast_npi_Nordex.npy', x_train_h_N)
np.save('LSTM input/y_train_history-forecast_npi_Nordex.npy', y_train_h_N)
np.save('LSTM input/x_test_history-forecast_npi_Nordex.npy', x_test_f_N)
np.save('LSTM input/y_test_history-forecast_npi_Nordex.npy', y_test_f_N)

del Y_scaler_h_N, train_h_N, train_f_N, test_f_N, test_h_N, x_train_h_N, y_train_h_N, x_test_f_N, y_test_f_N, x_train_h_N_unscaled, y_train_h_N_unscaled, x_test_f_N_unscaled, y_test_f_N_unscaled


#GETTING READY THE Nordex_data_f FOR THE LSTM TRAINING

#split into sequences

train_f_N, test_f_N =  split(Nordex_data_f)

train_f_N = train_f_N[[col for col in train_f_N.columns if col != column_to_move] + [column_to_move]]
test_f_N = test_f_N[[col for col in test_f_N.columns if col != column_to_move] + [column_to_move]]


train_f_N = train_f_N.tail(8700)
train_f_N = train_f_N.reset_index(drop=True)

test_f_N = test_f_N.head(400)
test_f_N = test_f_N.reset_index(drop=True)

print(train_f_N)
print(test_f_N)

print("split Nordex forecast")

x_train_f_N_unscaled, y_train_f_N_unscaled = split_sequences(train_f_N, n_steps=lag, n_outputs=num_of_outputs)
x_test_f_N_unscaled, y_test_f_N_unscaled = split_sequences(test_f_N, n_steps=lag, n_outputs=num_of_outputs)

#scaling

x_train_f_N, x_test_f_N, y_train_f_N, y_test_f_N, Y_scaler_f_N = scale_data(x_train_f_N_unscaled, x_test_f_N_unscaled, y_train_f_N_unscaled, y_test_f_N_unscaled)

with open('scalers/real_LSTM_N_scaler.pkl', 'wb') as f:
    pickle.dump(Y_scaler_f_N, f)

#Reshape the input so all sequences have same shape

x_train_f_N, y_train_f_N = unique_shapes(x_train_f_N, y_train_f_N, lag, n_features, num_of_outputs)
x_test_f_N, y_test_f_N = unique_shapes(x_test_f_N, y_test_f_N, lag, n_features, num_of_outputs)

#Save the data
np.save('LSTM input/real_x_train_Nordex.npy', x_train_f_N)
np.save('LSTM input/real_y_train_Nordex.npy', y_train_f_N)
np.save('LSTM input/real_x_test_Nordex.npy', x_test_f_N)

'''
