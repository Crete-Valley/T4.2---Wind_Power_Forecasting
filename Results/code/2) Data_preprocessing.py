

import metpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import gc

def sin_wind(x):
    return np.sin(x*np.pi/180)

def cos_wind(x):
    return np.cos(x*np.pi/180)

def scale(x):
    return x*0.001

def data_preprocessing(data):

    # Get the dummies for column 'WindOrientation','Season'
    dummies_dir = pd.get_dummies(data['WindOrientation'], prefix='Direction', dtype=int)
    dummies_sea = pd.get_dummies(data['Season'], prefix='Season', dtype=int)

    # Find the original column position
    col_position_dir = data.columns.get_loc('WindOrientation')
    col_position_sea = data.columns.get_loc('Season')

    # Drop the original column 'WindOrientation','Season'
    data.drop('WindOrientation', axis=1, inplace=True)
    data.drop('Season', axis=1, inplace=True)

    # Insert the dummy columns at the original position
    for i, col in enumerate(dummies_dir.columns):
        data.insert(col_position_dir + i-1, col, dummies_dir[col])

    for i, col in enumerate(dummies_sea.columns):
        data.insert(col_position_sea + i-1, col, dummies_sea[col])

    data["WindDirectionSin"] = data["WindDirection"].apply(sin_wind)
    data["WindDirectionCos"] = data["WindDirection"].apply(cos_wind)

    data["MeanDirectionSin"] = data["MeanDirection"].apply(sin_wind)
    data["MeanDirectionCos"] = data["MeanDirection"].apply(cos_wind)

    data.drop('WindDirection', axis=1, inplace=True)
    data.drop('MeanDirection', axis=1, inplace=True)

    data["ActivePower"] = data["ActivePower"].apply(scale)    #Convert kW to MW
    data["TheoreticalPower"] = data["TheoreticalPower"].apply(scale)      #Convert kW to MW
    data["Pressure"] = data["Pressure"].apply(scale)      #Convert hPa to bar

directiondict = {0:"N", 30:"NNE", 60:"NEE", 90:"E", 120:"SEE", 150:"SSE", 180:"S", 210:"SSW", 240:"SWW", 270:"W", 300:"NWW", 330:"NNW"}

def wind_direction(x):
    return directiondict[x]


def mean_speed(x):
    x = round(x,2)
    a = x//1
    a,b = a+0.25,a+0.75
    if x < a:
        x = a - 0.25
    else:
        x = b -0.25
    return x



def mean_direction(x):
    list=[]
    i=15
    while i<=375:
        list.append(i)
        i+=30

    for i in list:
        if x < i:
            x=i-15
            if x==360:
                return 0
            else:
                return x

Nordex_data_h = pd.read_csv('datasets/Nordex_N117_final_history_data.csv', index_col=0)
Vestas_data_h = pd.read_csv('datasets/Vestas_V52_final_history_data.csv', index_col=0)

Nordex_data_f = pd.read_csv('datasets/Nordex_N117_final_forecast_data.csv', index_col=0)
Vestas_data_f = pd.read_csv('datasets/Vestas_V52_final_forecast_data.csv', index_col=0)


print(Vestas_data_h)
print(Nordex_data_h)

print(Vestas_data_f)
print(Nordex_data_f)

#fix wrong timestamp values in Vestas dataset

for i in range(len(Vestas_data_h)):
    if len(Vestas_data_h.loc[i,('Timestamp')]) == 8 or len(Vestas_data_h.loc[i,('Timestamp')]) == 9 or len(Vestas_data_h.loc[i,('Timestamp')]) == 10:
        Vestas_data_h.loc[i,('Timestamp')] = Vestas_data_h.loc[i,('Timestamp')] + ' 0:00'

    if len(Vestas_data_f.loc[i,('Timestamp')]) == 8 or len(Vestas_data_f.loc[i,('Timestamp')]) == 9 or len(Vestas_data_f.loc[i,('Timestamp')]) == 10:
        Vestas_data_f.loc[i,('Timestamp')] = Vestas_data_f.loc[i,('Timestamp')] + ' 0:00'




#Populate datasets with new feature columns generated from the timestamp column


Nordex_data_h.loc[Nordex_data_h['ActivePower'] < 0, 'ActivePower'] = 0
Nordex_data_f.loc[Nordex_data_f['ActivePower'] < 0, 'ActivePower'] = 0

seasons_dict = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}

Vestas_data_h['Timestamp'] = pd.to_datetime(Vestas_data_h['Timestamp'],format='%m/%d/%Y %H:%M')
Vestas_data_h['Month'] = Vestas_data_h['Timestamp'].dt.month
Vestas_data_h['Week'] = Vestas_data_h['Timestamp'].dt.day // 7 + 1
Vestas_data_h['Day'] = Vestas_data_h['Timestamp'].dt.day
Vestas_data_h['Hour'] = Vestas_data_h['Timestamp'].dt.hour
Vestas_data_h['Minute'] = Vestas_data_h['Timestamp'].dt.minute
Vestas_data_h['Season'] = Vestas_data_h['Month'].map(seasons_dict)

Vestas_data_f['Timestamp'] = pd.to_datetime(Vestas_data_f['Timestamp'],format='%m/%d/%Y %H:%M')
Vestas_data_f['Month'] = Vestas_data_f['Timestamp'].dt.month
Vestas_data_f['Week'] = Vestas_data_f['Timestamp'].dt.day // 7 + 1
Vestas_data_f['Day'] = Vestas_data_f['Timestamp'].dt.day
Vestas_data_f['Hour'] = Vestas_data_f['Timestamp'].dt.hour
Vestas_data_f['Minute'] = Vestas_data_f['Timestamp'].dt.minute
Vestas_data_f['Season'] = Vestas_data_f['Month'].map(seasons_dict)

Vestas_data_h.drop(columns=['Timestamp'],inplace=True)
Vestas_data_f.drop(columns=['Timestamp'],inplace=True)


Nordex_data_h['Timestamp'] = pd.to_datetime(Nordex_data_h['Timestamp'],format='%d %m %Y %H:%M')
Nordex_data_h['Month'] = Nordex_data_h['Timestamp'].dt.month
Nordex_data_h['Day'] = Nordex_data_h['Timestamp'].dt.day
Nordex_data_h['Week'] = Nordex_data_h['Timestamp'].dt.day // 7 + 1
Nordex_data_h['Hour'] = Nordex_data_h['Timestamp'].dt.hour
Nordex_data_h['Minute'] = Nordex_data_h['Timestamp'].dt.minute
Nordex_data_h['Season'] = Nordex_data_h['Month'].map(seasons_dict)

Nordex_data_f['Timestamp'] = pd.to_datetime(Nordex_data_f['Timestamp'],format='%d %m %Y %H:%M')
Nordex_data_f['Month'] = Nordex_data_f['Timestamp'].dt.month
Nordex_data_f['Day'] = Nordex_data_f['Timestamp'].dt.day
Nordex_data_f['Week'] = Nordex_data_f['Timestamp'].dt.day // 7 + 1
Nordex_data_f['Hour'] = Nordex_data_f['Timestamp'].dt.hour
Nordex_data_f['Minute'] = Nordex_data_f['Timestamp'].dt.minute
Nordex_data_f['Season'] = Nordex_data_f['Month'].map(seasons_dict)

Nordex_data_h.drop(columns=['Timestamp'],inplace=True)
Nordex_data_f.drop(columns=['Timestamp'],inplace=True)



#Fix certain values in WindDirection which exceeded the 0-360 degrees space


for i in range(len(Vestas_data_h)):
    if Vestas_data_h.loc[i,('WindDirection')] < 0:
        Vestas_data_h.loc[i,('WindDirection')] = 360 + (-1*(abs(Vestas_data_h.loc[i,('WindDirection')])%360))
    elif Vestas_data_h.loc[i,('WindDirection')] > 360:
        Vestas_data_h.loc[i,('WindDirection')] = Vestas_data_h.loc[i,('WindDirection')]%360

    if Vestas_data_f.loc[i,('WindDirection')] < 0:
        Vestas_data_f.loc[i,('WindDirection')] = 360 + (-1*(abs(Vestas_data_f.loc[i,('WindDirection')])%360))
    elif Vestas_data_f.loc[i,('WindDirection')] > 360:
        Vestas_data_f.loc[i,('WindDirection')] = Vestas_data_f.loc[i,('WindDirection')]%360




#Populate datasets with new feature columns generated from the WindSpeed and WindDirection columns


Vestas_data_h['MeanSpeed'] = Vestas_data_h['WindSpeed'].apply(mean_speed)
Vestas_data_h["MeanDirection"] = Vestas_data_h["WindDirection"].apply(mean_direction)
Vestas_data_h["WindOrientation"] = Vestas_data_h["MeanDirection"].apply(wind_direction)

Vestas_data_f['MeanSpeed'] = Vestas_data_f['WindSpeed'].apply(mean_speed)
Vestas_data_f["MeanDirection"] = Vestas_data_f["WindDirection"].apply(mean_direction)
Vestas_data_f["WindOrientation"] = Vestas_data_f["MeanDirection"].apply(wind_direction)


Nordex_data_h['MeanSpeed'] = Nordex_data_h['WindSpeed'].apply(mean_speed)
Nordex_data_h["MeanDirection"] = Nordex_data_h["WindDirection"].apply(mean_direction)
Nordex_data_h['WindOrientation'] = Nordex_data_h['MeanDirection'].apply(wind_direction)

Nordex_data_f['MeanSpeed'] = Nordex_data_f['WindSpeed'].apply(mean_speed)
Nordex_data_f["MeanDirection"] = Nordex_data_f["WindDirection"].apply(mean_direction)
Nordex_data_f['WindOrientation'] = Nordex_data_f['MeanDirection'].apply(wind_direction)




#Brings every feature values closer to each other before scaling so that there are no dominant features


data_preprocessing(Vestas_data_h)
data_preprocessing(Nordex_data_h)

data_preprocessing(Vestas_data_f)
data_preprocessing(Nordex_data_f)



#Rearrange ActivePower column to be the last


first_col = Vestas_data_h.pop("ActivePower")
Vestas_data_h["ActivePower"] = first_col

first_col = Vestas_data_f.pop("ActivePower")
Vestas_data_f["ActivePower"] = first_col

first_col = Nordex_data_h.pop("ActivePower")
Nordex_data_h["ActivePower"] = first_col

first_col = Nordex_data_f.pop("ActivePower")
Nordex_data_f["ActivePower"] = first_col

pd.set_option('display.max_columns', None)
print(Vestas_data_h)
print(Nordex_data_h)

print(Vestas_data_f)
print(Nordex_data_f)

Vestas_data_h.to_csv('datasets/Vestas_V52_preprocessed_history_data.csv')
Nordex_data_h.to_csv('datasets/Nordex_N117_preprocessed_history_data.csv')

Vestas_data_f.to_csv('datasets/Vestas_V52_preprocessed_forecast_data.csv')
Nordex_data_f.to_csv('datasets/Nordex_N117_preprocessed_forecast_data.csv')
