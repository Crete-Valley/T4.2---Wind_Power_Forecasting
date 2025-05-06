import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import MinMaxScaler

def create_time_index(df):

    df.insert(0, 'time_index', range(1, len(df) + 1))



def scale_tft(train_df, test_df):

    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()

    feature_columns1 = [col for col in train_df.columns if col not in ["time_index", "group_id", "ActivePower"]]
    feature_columns2 = [col for col in train_df.columns if col=="ActivePower"]

    scaler1.fit(train_df.loc[:, feature_columns1])
    train_df.loc[:, feature_columns1] = scaler1.transform(train_df.loc[:, feature_columns1]).astype('float64')
    test_df.loc[:, feature_columns1] = scaler1.transform(test_df.loc[:, feature_columns1]).astype('float64')

    scaler2.fit(train_df.loc[:, feature_columns2])
    train_df.loc[:, feature_columns2] = scaler2.transform(train_df.loc[:, feature_columns2]).astype('float64')
    test_df.loc[:, feature_columns2] = scaler2.transform(test_df.loc[:, feature_columns2]).astype('float64')

    return train_df, test_df, scaler2



def split(data):

    train = data.iloc[:int(len(data)*split_ratio),:]
    test = data.iloc[int(len(data)*split_ratio):,]
    return train,test




def set_categories(df,cols):

    df = df.copy()

    for col in cols:

        df[col] = df[col].astype(str).astype("category")

    return df

Nordex_data_h = pd.read_csv('datasets/Nordex_N117_preprocessed_history_data.csv', index_col=0)
Vestas_data_h = pd.read_csv('datasets/Vestas_V52_preprocessed_history_data.csv', index_col=0)

Nordex_data_f = pd.read_csv('datasets/Nordex_N117_preprocessed_forecast_data.csv', index_col=0)
Vestas_data_f = pd.read_csv('datasets/Vestas_V52_preprocessed_forecast_data.csv', index_col=0)

Nordex_data_h = Nordex_data_h.astype('float64')
Vestas_data_h = Vestas_data_h.astype('float64')

Nordex_data_f = Nordex_data_f.astype('float64')
Vestas_data_f = Vestas_data_f.astype('float64')


#need time index column for the model to work

create_time_index(Vestas_data_h)
create_time_index(Vestas_data_f)
create_time_index(Nordex_data_h)
create_time_index(Nordex_data_f)

#need universal group_id since we don't train based on groups

Vestas_data_h['group_id'] = 0
Vestas_data_f['group_id'] = 0
Nordex_data_h['group_id'] = 0
Nordex_data_f['group_id'] = 0


cols = ['Month',
        'Day',
        'Week',
        'Minute',
        'Season_Autumn',
        'Season_Spring',
        'Season_Summer',
        'Season_Winter',
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
        'Direction_W']

#split

split_ratio = 0.8

train_h_V_unscaled, test_h_V_unscaled =  split(Vestas_data_h)
test_h_V_unscaled = test_h_V_unscaled.reset_index(drop=True)

train_f_V_unscaled, test_f_V_unscaled =  split(Vestas_data_f)
test_f_V_unscaled = test_f_V_unscaled.reset_index(drop=True)

train_h_N_unscaled, test_h_N_unscaled =  split(Nordex_data_h)
test_h_N_unscaled = test_h_N_unscaled.reset_index(drop=True)

train_f_N_unscaled, test_f_N_unscaled =  split(Nordex_data_f)
test_f_N_unscaled = test_f_N_unscaled.reset_index(drop=True)

#scaling

train_h_V_scaled, test_h_V_scaled, scaler_V_h  = scale_tft(train_h_V_unscaled, test_f_V_unscaled)
with open('scalers/TFT_V_h_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_V_h, f)

train_f_V_scaled, test_f_V_scaled, scaler_V_f = scale_tft(train_f_V_unscaled, test_f_V_unscaled)
with open('scalers/TFT_V_f_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_V_f, f)

train_h_N_scaled, test_h_N_scaled, scaler_N_h = scale_tft(train_h_N_unscaled, test_f_N_unscaled)
with open('scalers/TFT_N_h_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_N_h, f)

train_f_N_scaled, test_f_N_scaled, scaler_N_f = scale_tft(train_f_N_unscaled, test_f_N_unscaled)
with open('scalers/TFT_N_f_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_N_f, f)


#set categorical features as type categories and strings

train_h_V_scaled = set_categories(train_h_V_scaled, cols)
test_h_V_scaled = set_categories(test_h_V_scaled, cols)
train_f_V_scaled = set_categories(train_f_V_scaled, cols)
test_f_V_scaled = set_categories(test_f_V_scaled, cols)
train_h_N_scaled = set_categories(train_h_N_scaled, cols)
test_h_N_scaled = set_categories(test_h_N_scaled, cols)
train_f_N_scaled = set_categories(train_f_N_scaled, cols)
test_f_N_scaled = set_categories(test_f_N_scaled, cols)

train_h_V_scaled.to_pickle('TFT input/tft_history-forecast_V_train.pkl')
test_h_V_scaled.to_pickle('TFT input/tft_history-forecast_V_test.pkl')
train_f_V_scaled.to_pickle('TFT input/tft_forecast-forecast_V_train.pkl')
test_f_V_scaled.to_pickle('TFT input/tft_forecast-forecast_V_test.pkl')
train_h_N_scaled.to_pickle('TFT input/tft_history-forecast_N_train.pkl')
test_h_N_scaled.to_pickle('TFT input/tft_history-forecast_N_test.pkl')
train_f_N_scaled.to_pickle('TFT input/tft_forecast-forecast_N_train.pkl')
test_f_N_scaled.to_pickle('TFT input/tft_forecast-forecast_N_test.pkl')