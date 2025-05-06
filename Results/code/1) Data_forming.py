import metpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import gc



from scipy.interpolate import interp1d
from metpy.units import units
from metpy.calc import density
from metpy.calc import add_height_to_pressure
from metpy.calc import mixing_ratio_from_relative_humidity
from metpy.calc import relative_humidity_from_dewpoint



def calculate_theoritical_power(pressure, temperature, rel_humidity, wind_speed, turbine_swept_area, turbine_efficiency):

    heighted_pressure = pressure * units.hPa                          # 1st APPROXIMATION

    heighted_temperature = temperature * units.degC     # 2nd APPROXIMATION

    heighted_rel_humidity = rel_humidity

    mixing_ratio = mixing_ratio_from_relative_humidity(heighted_pressure, heighted_temperature, heighted_rel_humidity).to('g/kg')

    air_density = density(heighted_pressure, heighted_temperature, mixing_ratio, 0.62195691)

    theoritical_turbine_power = 0.5 * (air_density.magnitude * turbine_swept_area * (wind_speed**3) * turbine_efficiency)

    return theoritical_turbine_power

def add_random_variation(value):
    #random multiplier between -3% and +3%
    variation = np.random.uniform(-0.03, 0.03)
    return value * (1 + variation)

Nordex_data_h = pd.read_csv('datasets/Nordex_N117_raw_data.csv')
Nordex_weather_h = pd.read_csv('datasets/Nordex_N117_meteo_data.csv')
Vestas_data_h = pd.read_csv('datasets/Vestas_V52_raw_data.csv')
Vestas_data_h = Vestas_data_h[:653007]
Vestas_weather_h = pd.read_csv('datasets/Vestas_V52_meteo_data.csv')


turbine_height_nordex = 106
surface_elevation_nordex = 787
turbine_swept_area_nordex = 10715



turbine_height_vestas = 60
surface_elevation_vestas = 9
turbine_swept_area_vestas = 2124

#Create function that takes the theoretical power of each wind turbine based on the wind speed

# Wind speed data
wind_speed = np.array([2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5])

# Cp data from Vestas V52
cp_values_vestas = np.array([0, 0, 0, 0, 0, 0.05, 0.270, 0.370, 0.450, 0.48, 0.49, 0.47, 0.48, 0.47, 0.45, 0.44, 0.47, 0.46, 0.45, 0.45, 0.42, 0.39, 0.38, 0.35, 0.32, 0.29, 0.26, 0.24, 0.21, 0.19, 0.18, 0.16, 0.15])

# Cp data from Nordex N117
cp_values_nordex = np.array([0, 0, 0, 0, 0, 0.096, 0.192, 0.312, 0.381, 0.419, 0.439, 0.45, 0.457, 0.461, 0.464, 0.465, 0.463, 0.457, 0.448, 0.434, 0.409, 0.379, 0.346, 0.312, 0.28, 0.25, 0.223, 0.2, 0.18, 0.163, 0.147, 0.134, 0.122])


# Create interpolation functions
cp_from_windspeed_vestas = interp1d(wind_speed, cp_values_vestas, kind='cubic', fill_value="extrapolate")
cp_from_windspeed_nordex = interp1d(wind_speed, cp_values_nordex, kind='cubic', fill_value="extrapolate")

# Plotting the original data and the interpolation
wind_speed_dense = np.linspace(2.5, 16.5, 141)  # More points for a smoother plot
cp_dense_vestas = cp_from_windspeed_vestas(wind_speed_dense)
cp_dense_nordex = cp_from_windspeed_nordex(wind_speed_dense)

# Example of using the function to predict a value
wind_speed_value = 4.7
cp_query_vestas = cp_from_windspeed_vestas(wind_speed_value)
print(f"Vestas V52 Cp value at wind speed {wind_speed_value} m/s is approximately {cp_query_vestas:.2f}")

cp_query_nordex = cp_from_windspeed_nordex(wind_speed_value)
print(f"Nordex N117 Cp value at wind speed {wind_speed_value} m/s is approximately {cp_query_nordex:.2f}")

# Plotting
plt.figure(figsize=(20, 10))
plt.plot(Nordex_data_h['ActivePower'], label='Active', linestyle = '--')

plt.title('Plot of Active and Theoretical')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(20, 10))
plt.plot(Vestas_data_h['ActivePower'], label='Active', linestyle = '--')

plt.title('Plot of Active and Theoretical')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()

#Create simulated forecasting data


Nordex_data_f = Nordex_data_h.copy()
Vestas_data_f = Vestas_data_h.copy()

Nordex_weather_f = Nordex_weather_h.copy()
Vestas_weather_f = Vestas_weather_h.copy()


Nordex_data_f['WindDirection'] = Nordex_data_f['WindDirection'].apply(add_random_variation)
Nordex_data_f['WindSpeed'] = Nordex_data_f['WindSpeed'].apply(add_random_variation)

Nordex_weather_f['temperature_2m (°C)'] = Nordex_weather_f['temperature_2m (°C)'].apply(add_random_variation)
Nordex_weather_f['dew_point_2m (°C)'] = Nordex_weather_f['dew_point_2m (°C)'].apply(add_random_variation)
Nordex_weather_f['pressure_msl (hPa)'] = Nordex_weather_f['pressure_msl (hPa)'].apply(add_random_variation)


Vestas_data_f['WindDirection'] = Vestas_data_f['WindDirection'].apply(add_random_variation)
Vestas_data_f['WindSpeed'] = Vestas_data_f['WindSpeed'].apply(add_random_variation)

Vestas_weather_f['temperature_2m (°C)'] = Vestas_weather_f['temperature_2m (°C)'].apply(add_random_variation)
Vestas_weather_f['dew_point_2m (°C)'] = Vestas_weather_f['dew_point_2m (°C)'].apply(add_random_variation)
Vestas_weather_f['pressure_msl (hPa)'] = Vestas_weather_f['pressure_msl (hPa)'].apply(add_random_variation)


Nordex_data_f.drop(columns=['TheoreticalPower'],inplace=True)

#Insert weather data to the datasets


temperature_h = []
dewpoint_h = []
rel_humidity_h = []
pressure_h = []

temperature_f = []
dewpoint_f = []
rel_humidity_f = []
pressure_f = []

for i in range(len(Vestas_data_h)):
    if i == 0 or i == 1 or i == 2:
        temp_h = (Vestas_weather_h.iloc[0]['temperature_2m (°C)'] - (0.0065 * (turbine_height_vestas - 2)))
        dp_h = (Vestas_weather_h.iloc[0]['dew_point_2m (°C)'] - (0.0018 * (turbine_height_vestas - 2)))

        temperature_h.append(temp_h)
        dewpoint_h.append(dp_h)

        temp_f = (Vestas_weather_f.iloc[0]['temperature_2m (°C)'] - (0.0065 * (turbine_height_vestas - 2)))
        dp_f = (Vestas_weather_f.iloc[0]['dew_point_2m (°C)'] - (0.0018 * (turbine_height_vestas - 2)))

        temperature_f.append(temp_f)
        dewpoint_f.append(dp_f)

        rel_humidity_h.append((relative_humidity_from_dewpoint(temp_h * units.degC, dp_h * units.degC)).magnitude)
        pressure_h.append((add_height_to_pressure(Vestas_weather_h.iloc[0]['pressure_msl (hPa)'] * units.hPa, (surface_elevation_vestas + turbine_height_vestas) * units.meter)).magnitude)

        rel_humidity_f.append((relative_humidity_from_dewpoint(temp_f * units.degC, dp_f * units.degC)).magnitude)
        pressure_f.append((add_height_to_pressure(Vestas_weather_f.iloc[0]['pressure_msl (hPa)'] * units.hPa, (surface_elevation_vestas + turbine_height_vestas) * units.meter)).magnitude)

    elif i > 2:
        df_index = (i + 3)//6

        temp_h = (Vestas_weather_h.iloc[df_index]['temperature_2m (°C)'] - (0.0065 * (turbine_height_vestas - 2)))
        dp_h = (Vestas_weather_h.iloc[df_index]['dew_point_2m (°C)'] - (0.0018 * (turbine_height_vestas - 2)))

        temperature_h.append(temp_h)
        dewpoint_h.append(dp_h)

        temp_f = (Vestas_weather_f.iloc[df_index]['temperature_2m (°C)'] - (0.0065 * (turbine_height_vestas - 2)))
        dp_f = (Vestas_weather_f.iloc[df_index]['dew_point_2m (°C)'] - (0.0018 * (turbine_height_vestas - 2)))

        temperature_f.append(temp_f)
        dewpoint_f.append(dp_f)

        rel_humidity_h.append((relative_humidity_from_dewpoint(temp_h * units.degC, dp_h * units.degC)).magnitude)
        pressure_h.append((add_height_to_pressure(Vestas_weather_h.iloc[df_index]['pressure_msl (hPa)'] * units.hPa, (surface_elevation_vestas + turbine_height_vestas) * units.meter)).magnitude)

        rel_humidity_f.append((relative_humidity_from_dewpoint(temp_f * units.degC, dp_f * units.degC)).magnitude)
        pressure_f.append((add_height_to_pressure(Vestas_weather_f.iloc[df_index]['pressure_msl (hPa)'] * units.hPa, (surface_elevation_vestas + turbine_height_vestas) * units.meter)).magnitude)


Vestas_data_h.insert(4, 'Temperature', temperature_h)
Vestas_data_h.insert(5, 'Dewpoint', dewpoint_h)
Vestas_data_h.insert(6, 'Rel_Humidity', rel_humidity_h)
Vestas_data_h.insert(7, 'Pressure', pressure_h)

Vestas_data_f.insert(4, 'Temperature', temperature_f)
Vestas_data_f.insert(5, 'Dewpoint', dewpoint_f)
Vestas_data_f.insert(6, 'Rel_Humidity', rel_humidity_f)
Vestas_data_f.insert(7, 'Pressure', pressure_f)



temperature_h = []
dewpoint_h = []
rel_humidity_h = []
pressure_h = []

temperature_f = []
dewpoint_f = []
rel_humidity_f = []
pressure_f = []

for i in range(len(Nordex_data_h)):
    if i == 0 or i == 1 or i == 2:
        temp_h = (Nordex_weather_h.iloc[0]['temperature_2m (°C)'] - (0.0065 * (turbine_height_nordex - 2)))
        dp_h = (Nordex_weather_h.iloc[0]['dew_point_2m (°C)'] - (0.0018 * (turbine_height_nordex - 2)))

        temperature_h.append(temp_h)
        dewpoint_h.append(dp_h)

        temp_f = (Nordex_weather_f.iloc[0]['temperature_2m (°C)'] - (0.0065 * (turbine_height_nordex - 2)))
        dp_f = (Nordex_weather_f.iloc[0]['dew_point_2m (°C)'] - (0.0018 * (turbine_height_nordex - 2)))

        temperature_f.append(temp_f)
        dewpoint_f.append(dp_f)

        rel_humidity_h.append((relative_humidity_from_dewpoint(temp_h * units.degC, dp_h * units.degC)).magnitude)
        pressure_h.append((add_height_to_pressure(Nordex_weather_h.iloc[0]['pressure_msl (hPa)'] * units.hPa, (surface_elevation_nordex + turbine_height_nordex) * units.meter)).magnitude)

        rel_humidity_f.append((relative_humidity_from_dewpoint(temp_f * units.degC, dp_f * units.degC)).magnitude)
        pressure_f.append((add_height_to_pressure(Nordex_weather_f.iloc[0]['pressure_msl (hPa)'] * units.hPa, (surface_elevation_nordex + turbine_height_nordex) * units.meter)).magnitude)

    elif i > 2:
        df_index = (i + 3)//6

        temp_h = (Nordex_weather_h.iloc[df_index]['temperature_2m (°C)'] - (0.0065 * (turbine_height_nordex - 2)))
        dp_h = (Nordex_weather_h.iloc[df_index]['dew_point_2m (°C)'] - (0.0018 * (turbine_height_nordex - 2)))

        temperature_h.append(temp_h)
        dewpoint_h.append(dp_h)

        temp_f = (Nordex_weather_f.iloc[df_index]['temperature_2m (°C)'] - (0.0065 * (turbine_height_nordex - 2)))
        dp_f = (Nordex_weather_f.iloc[df_index]['dew_point_2m (°C)'] - (0.0018 * (turbine_height_nordex - 2)))

        temperature_f.append(temp_f)
        dewpoint_f.append(dp_f)

        rel_humidity_h.append((relative_humidity_from_dewpoint(temp_h * units.degC, dp_h * units.degC)).magnitude)
        pressure_h.append((add_height_to_pressure(Nordex_weather_h.iloc[df_index]['pressure_msl (hPa)'] * units.hPa, (surface_elevation_nordex + turbine_height_nordex) * units.meter)).magnitude)

        rel_humidity_f.append((relative_humidity_from_dewpoint(temp_f * units.degC, dp_f * units.degC)).magnitude)
        pressure_f.append((add_height_to_pressure(Nordex_weather_f.iloc[df_index]['pressure_msl (hPa)'] * units.hPa, (surface_elevation_nordex + turbine_height_nordex) * units.meter)).magnitude)


Nordex_data_h.insert(4, 'Temperature', temperature_h)
Nordex_data_h.insert(5, 'Dewpoint', dewpoint_h)
Nordex_data_h.insert(6, 'Rel_Humidity', rel_humidity_h)
Nordex_data_h.insert(7, 'Pressure', pressure_h)

Nordex_data_f.insert(4, 'Temperature', temperature_f)
Nordex_data_f.insert(5, 'Dewpoint', dewpoint_f)
Nordex_data_f.insert(6, 'Rel_Humidity', rel_humidity_f)
Nordex_data_f.insert(7, 'Pressure', pressure_f)

#Add the theoretical power column to the datasets


th_power_h = []
th_power_f = []

for i in range(len(Vestas_data_h)):
    print(i)
    wind_speed_h = Vestas_data_h.iloc[i]['WindSpeed']
    wind_speed_f = Vestas_data_f.iloc[i]['WindSpeed']

    if wind_speed_h < 3:
        th_power_h.append(0)
    elif wind_speed_h > 16 and wind_speed_h <= 25:
        th_power_h.append(851.9)
    elif wind_speed_h > 25:
        th_power_h.append(0)
    else :
        power = calculate_theoritical_power(Vestas_data_h.iloc[i]['Pressure'],
                                            Vestas_data_h.iloc[i]['Temperature'],
                                            Vestas_data_h.iloc[i]['Rel_Humidity'],
                                            wind_speed_h,
                                            turbine_swept_area_vestas,
                                            cp_from_windspeed_vestas(wind_speed_h))

        th_power_h.append(round(power * 0.001, 1))

    if wind_speed_f < 3:
        th_power_f.append(0)
    elif wind_speed_f > 16 and wind_speed_f <= 25:
        th_power_f.append(851.9)
    elif wind_speed_f > 25:
        th_power_f.append(0)
    else :
        power = calculate_theoritical_power(Vestas_data_f.iloc[i]['Pressure'],
                                            Vestas_data_f.iloc[i]['Temperature'],
                                            Vestas_data_f.iloc[i]['Rel_Humidity'],
                                            wind_speed_f,
                                            turbine_swept_area_vestas,
                                            cp_from_windspeed_vestas(wind_speed_f))

        th_power_f.append(round(power * 0.001, 1))

Vestas_data_h.insert(2, 'TheoreticalPower', th_power_h)
Vestas_data_f.insert(2, 'TheoreticalPower', th_power_f)

th_power = []


for i in range(len(Nordex_data_f)):
    print(i)
    wind_speed = Nordex_data_f.iloc[i]['WindSpeed']
    if wind_speed < 3:
        th_power.append(0)
    elif wind_speed > 13 and wind_speed <= 25:
        th_power.append(3600)
    elif wind_speed > 25:
        th_power.append(0)
    else :
        power = calculate_theoritical_power(Nordex_data_f.iloc[i]['Pressure'],
                                            Nordex_data_f.iloc[i]['Temperature'],
                                            Nordex_data_f.iloc[i]['Rel_Humidity'],
                                            wind_speed,
                                            turbine_swept_area_nordex,
                                            cp_from_windspeed_nordex(wind_speed))

        th_power.append(round(power * 0.001, 1))

Nordex_data_f.insert(2, 'TheoreticalPower', th_power)

#Fix certain outliers in the Vestas dataset


percentages_h = 0
sum_h = 0

percentages_f = 0
sum_f = 0

for i in range(len(Vestas_data_h[65000:580000])):
    if Vestas_data_h.loc[i,('ActivePower')] <= 851  or Vestas_data_h.loc[i,('ActivePower')] > 0 :
        if Vestas_data_h.loc[i,('TheoreticalPower')] > 0 :
            sum_h += 1
            percentages_h = percentages_h + (Vestas_data_h.loc[i,('ActivePower')] / Vestas_data_h.loc[i,('TheoreticalPower')])

        if Vestas_data_f.loc[i,('TheoreticalPower')] > 0 :
            sum_f += 1
            percentages_f = percentages_f + (Vestas_data_f.loc[i,('ActivePower')] / Vestas_data_f.loc[i,('TheoreticalPower')])

mean_percentage_h = percentages_h/sum_h
mean_percentage_f = percentages_f/sum_f

max_power = Vestas_data_h.loc[Vestas_data_h['ActivePower'] < 860, 'ActivePower'].max()

min_optimal_wind_speed_h = round(Vestas_data_h.loc[Vestas_data_h['ActivePower'] == max_power, 'WindSpeed'].min(), 1)
min_optimal_wind_speed_f = round(Vestas_data_f.loc[Vestas_data_f['ActivePower'] == max_power, 'WindSpeed'].min(), 1)





for i in range(len(Vestas_data_h)):
    if Vestas_data_h.loc[i,('ActivePower')] > 851 or Vestas_data_h.loc[i,('ActivePower')] < 0:
        if Vestas_data_h.loc[i,('WindSpeed')] > min_optimal_wind_speed_h and Vestas_data_h.loc[i,('WindSpeed')] <= 25:
            Vestas_data_h.loc[i,('ActivePower')] = max_power
        elif Vestas_data_h.loc[i,('WindSpeed')] < 3  or Vestas_data_h.loc[i,('WindSpeed')] > 25:
            Vestas_data_h.loc[i,('ActivePower')] = 0
        elif Vestas_data_h.loc[i,('WindSpeed')] >= 3 and Vestas_data_h.loc[i,('WindSpeed')] <= min_optimal_wind_speed_h :
            Vestas_data_h.loc[i,('ActivePower')] = (round(Vestas_data_h.loc[i,('TheoreticalPower')]*mean_percentage_h, 1))

        if Vestas_data_f.loc[i,('WindSpeed')] > min_optimal_wind_speed_f and Vestas_data_f.loc[i,('WindSpeed')] <= 25:
            Vestas_data_f.loc[i,('ActivePower')] = max_power
        elif Vestas_data_f.loc[i,('WindSpeed')] < 3  or Vestas_data_f.loc[i,('WindSpeed')] > 25:
            Vestas_data_f.loc[i,('ActivePower')] = 0
        elif Vestas_data_f.loc[i,('WindSpeed')] >= 3 and Vestas_data_f.loc[i,('WindSpeed')] <= min_optimal_wind_speed_f :
            Vestas_data_f.loc[i,('ActivePower')] = (round(Vestas_data_f.loc[i,('TheoreticalPower')]*mean_percentage_f, 1))




max = 0
for i in range(len(Vestas_data_h[0:63000])):
    if Vestas_data_h.loc[i,('ActivePower')] > max and Vestas_data_h.loc[i,('ActivePower')] < 800:
        max = Vestas_data_h.loc[i,('ActivePower')]


for i in range(len(Vestas_data_h.loc[0:63000])):
    if Vestas_data_h.loc[i,('ActivePower')] < max and Vestas_data_h.loc[i,('ActivePower')] > 700:
        wind_h = Vestas_data_h.loc[i,('WindSpeed')]
        wind_f = Vestas_data_f.loc[i,('WindSpeed')]

        if wind_h > min_optimal_wind_speed_h and wind_h <= 25:
            Vestas_data_h.loc[i,('ActivePower')] = max_power
        else:
            Vestas_data_h.loc[i,('ActivePower')] = (round(Vestas_data_h.loc[i,('TheoreticalPower')]*mean_percentage_h, 1))

        if wind_f > min_optimal_wind_speed_f and wind_f <= 25:
            Vestas_data_f.loc[i,('ActivePower')] = max_power
        else:
            Vestas_data_f.loc[i,('ActivePower')] = (round(Vestas_data_f.loc[i,('TheoreticalPower')]*mean_percentage_f, 1))

print(Vestas_data_h)
print(Vestas_data_f)

print(Nordex_data_h)
print(Nordex_data_f)

Vestas_data_h.to_csv('datasets/Vestas_V52_final_history_data.csv')
Nordex_data_h.to_csv('datasets/Nordex_N117_final_history_data.csv')

Vestas_data_f.to_csv('datasets/Vestas_V52_final_forecast_data.csv')
Nordex_data_f.to_csv('datasets/Nordex_N117_final_forecast_data.csv')