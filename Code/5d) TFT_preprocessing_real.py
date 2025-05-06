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

Nordex_data_f = pd.read_csv('datasets/Nordex_N117_preprocessed_forecast_data.csv', index_col=0)
Vestas_data_f = pd.read_csv('datasets/Vestas_V52_preprocessed_forecast_data.csv', index_col=0)

Nordex_data_f = Nordex_data_f.astype('float64')
Vestas_data_f = Vestas_data_f.astype('float64')


Nordex_data_f = Nordex_data_f.drop('ActivePower', axis=1)
Vestas_data_f = Vestas_data_f.drop('ActivePower', axis=1)

column_to_move = 'TheoreticalPower'
new_name = 'ActivePower'

Nordex_data_f = Nordex_data_f[[col for col in Nordex_data_f.columns if col != column_to_move] + [column_to_move]]
Nordex_data_f = Nordex_data_f.rename(columns={column_to_move: new_name})

Vestas_data_f = Vestas_data_f[[col for col in Vestas_data_f.columns if col != column_to_move] + [column_to_move]]
Vestas_data_f = Vestas_data_f.rename(columns={column_to_move: new_name})

#need universal group_id since we don't train based on groups

Vestas_data_f['group_id'] = 0
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

train_f_V_unscaled, test_f_V_unscaled =  split(Vestas_data_f)

train_f_V_unscaled = train_f_V_unscaled.tail(8700)
train_f_V_unscaled = train_f_V_unscaled.reset_index(drop=True)

test_f_V_unscaled = test_f_V_unscaled.head(400)
test_f_V_unscaled = test_f_V_unscaled.reset_index(drop=True)


#need time index column for the model to work

create_time_index(train_f_V_unscaled)
create_time_index(test_f_V_unscaled)


print(train_f_V_unscaled)
print(test_f_V_unscaled)



train_f_N_unscaled, test_f_N_unscaled =  split(Nordex_data_f)

train_f_N_unscaled = train_f_N_unscaled.tail(8700)
train_f_N_unscaled = train_f_N_unscaled.reset_index(drop=True)

test_f_N_unscaled = test_f_N_unscaled.head(400)
test_f_N_unscaled = test_f_N_unscaled.reset_index(drop=True)


create_time_index(train_f_N_unscaled)
create_time_index(test_f_N_unscaled)


print(train_f_N_unscaled)
print(test_f_N_unscaled)

#scaling

train_f_V_scaled, test_f_V_scaled, scaler_V_f = scale_tft(train_f_V_unscaled, test_f_V_unscaled)
with open('scalers/real_TFT_V_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_V_f, f)

print(scaler_V_f.data_min_)
print(scaler_V_f.data_max_)


train_f_N_scaled, test_f_N_scaled, scaler_N_f = scale_tft(train_f_N_unscaled, test_f_N_unscaled)
with open('scalers/real_TFT_N_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_N_f, f)

print(scaler_N_f.data_min_)
print(scaler_N_f.data_max_)

#set categorical features as type categories and strings

train_f_V_scaled = set_categories(train_f_V_scaled, cols)
test_f_V_scaled = set_categories(test_f_V_scaled, cols)
train_f_N_scaled = set_categories(train_f_N_scaled, cols)
test_f_N_scaled = set_categories(test_f_N_scaled, cols)


train_f_V_scaled.to_pickle('TFT input/real_tft_V_train.pkl')
test_f_V_scaled.to_pickle('TFT input/real_tft_V_test.pkl')
train_f_N_scaled.to_pickle('TFT input/real_tft_N_train.pkl')
test_f_N_scaled.to_pickle('TFT input/real_tft_N_test.pkl')