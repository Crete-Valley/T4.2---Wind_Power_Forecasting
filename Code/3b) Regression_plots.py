import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def plot(preds_test, actuals, model_name, i):
    
    if i==1:
        actuals_test = (Scaler_f_V.inverse_transform(actuals.reshape(-1, 1))*1000).reshape(-1)
    elif i==2:
        actuals_test = (Scaler_f_N.inverse_transform(actuals.reshape(-1, 1))*1000).reshape(-1)

    error = preds_test - actuals_test

    x = np.arange(0,24)

    plt.figure(figsize=(20, 6))
    plt.plot(x, actuals_test[:24], label='Actual values', color='blue', alpha=0.8)
    plt.plot(x, preds_test[:24], label='Predictions', color='red', linestyle='--')
    plt.fill_between(x, 0, error[:24], label='Error', color='green', alpha=0.3)

    plt.title(model_name)
    plt.xlabel('Time-step')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.show()

def scale_base(x_train, x_test, y_train, y_test):

    y_train_reshaped = y_train.values.reshape(-1,1)
    y_test_reshaped = y_test.values.reshape(-1,1)



    # Initialize scalers
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()

    # Fit scalers on training data
    X_scaler.fit(x_train)
    Y_scaler.fit(y_train_reshaped)

    # Scale features
    x_train_scaled = X_scaler.transform(x_train)
    x_test_scaled = X_scaler.transform(x_test)

    # Scale target
    y_train_scaled = (Y_scaler.transform(y_train_reshaped)).reshape(-1)
    y_test_scaled = (Y_scaler.transform(y_test_reshaped)).reshape(-1)

    return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, Y_scaler

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



Nordex_data_f = pd.read_csv('datasets/Nordex_N117_preprocessed_forecast_data.csv', index_col=0)
Vestas_data_f = pd.read_csv('datasets/Vestas_V52_preprocessed_forecast_data.csv', index_col=0)

X_f_V = Vestas_data_f[cols]

X_f_N = Nordex_data_f[cols]


y_V = Vestas_data_f['ActivePower']

y_N = Nordex_data_f['ActivePower']



#scale datasets


X_train_f_V, X_test_f_V, y_train_f_V, y_test_f_V = train_test_split(X_f_V, y_V, test_size = 0.20, random_state = 42)

X_train_f_V, X_test_f_V, y_train_f_V, y_test_f_V, Scaler_f_V = scale_base(X_train_f_V, X_test_f_V, y_train_f_V, y_test_f_V)

X_train_f_N, X_test_f_N, y_train_f_N, y_test_f_N = train_test_split(X_f_N, y_N, test_size = 0.20, random_state = 42)

X_train_f_N, X_test_f_N, y_train_f_N, y_test_f_N, Scaler_f_N = scale_base(X_train_f_N, X_test_f_N, y_train_f_N, y_test_f_N)

#get the inputs

xgb_f_V = np.load('predictions/XGBRegressor_f_V_predictions.npy')
xgb_f_N = np.load('predictions/XGBRegressor_f_N_predictions.npy')


cat_f_V = np.load('predictions/CatBoostRegressor_f_V_predictions.npy')
cat_f_N = np.load('predictions/CatBoostRegressor_f_N_predictions.npy')


dec_f_V = np.load('predictions/DecisionTreeRegressor_f_V_predictions.npy')
dec_f_N = np.load('predictions/DecisionTreeRegressor_f_N_predictions.npy')



plot(xgb_f_V, y_test_f_V, "XGBoost_Regressor_f_V", 1)
plot(cat_f_V, y_test_f_V, "CatBoost_Regressor_f_V", 1)
plot(dec_f_V, y_test_f_V, "DecisionTree_Regressor_f_V", 1)
plot(xgb_f_N, y_test_f_N, "XGBoost_Regressor_f_N", 2)
plot(xgb_f_N, y_test_f_N, "CatBoost_Regressor_f_N", 2)
plot(xgb_f_N, y_test_f_N, "DecisionTree_Regressor_f_N", 2)