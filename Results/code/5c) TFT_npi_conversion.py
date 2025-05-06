import pandas as pd
import pickle


def create_npi(filepath):

    df = pd.read_pickle('TFT input/'+filepath)

    df2 = df.drop('TheoreticalPower', axis=1)

    df2.to_pickle('TFT input/npi_'+filepath)


create_npi('tft_history-forecast_V_train.pkl')
create_npi('tft_history-forecast_V_test.pkl')
create_npi('tft_forecast-forecast_V_train.pkl')
create_npi('tft_forecast-forecast_V_test.pkl')
create_npi('tft_history-forecast_N_train.pkl')
create_npi('tft_history-forecast_N_test.pkl')
create_npi('tft_forecast-forecast_N_train.pkl')
create_npi('tft_forecast-forecast_N_test.pkl')










