import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joypy
from joypy import joyplot
import seaborn as sns
import pickle
import pandas as pd
from tabulate import tabulate

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def plot_comparative_results(dataset, model, period):

    if dataset == 'V':

        data = 'Vestas'
        
        offset = 50

        actuals = lstm_scaler_f_V.inverse_transform(npi_y_test_lstm_f_V)*1000
        actuals_plot = actuals[offset+1:offset+period+1]

        if model == 'LSTM':

            preds1_scaled = np.load('predictions/LSTM/LSTM_f_V_predictions.npy')
            preds2_scaled = np.load('predictions/LSTM/npi_LSTM_f_V_predictions.npy')

            preds1 = lstm_scaler_f_V.inverse_transform(preds1_scaled)*1000
            preds1 = preds1.ravel()
            preds1_plot = preds1[offset+2:offset+period+2]

            preds2 = lstm_scaler_f_V.inverse_transform(preds2_scaled)*1000
            preds2 = preds2.ravel()
            preds2_plot = preds2[offset+2:offset+period+2]

            error1 = preds1[offset+2:offset+period+2] - actuals[offset+1:offset+period+1].squeeze()
            error1 = error1.reshape(-1)

            error2 = preds2[offset+2:offset+period+2] - actuals[offset+1:offset+period+1].squeeze()
            error2 = error2.reshape(-1)

            name_1 = 'LSTM'
            name_2 = 'npi_LSTM'

        
        elif model == 'TFT':

            preds1_scaled = np.load('predictions/TFT/TFT_f_V_predictions.npy')
            preds2_scaled = np.load('predictions/TFT/npi_TFT_f_V_predictions.npy')


            preds1 = ((tft_scaler_f_V.inverse_transform(preds1_scaled.reshape(-1, 1)))).reshape(-1)*1000
            preds1 = preds1.ravel()
            preds1_plot = preds1[offset:offset+period]
            preds2 = ((tft_scaler_f_V.inverse_transform(preds2_scaled.reshape(-1, 1)))).reshape(-1)*1000
            preds2 = preds2.ravel()
            preds2_plot = preds2[offset:offset+period]
            

            error1 = preds1[offset:offset+period] - actuals[offset+1:offset+period+1].squeeze()
            error1 = error1.reshape(-1)

            error2 = preds2[offset:offset+period] - actuals[offset+1:offset+period+1].squeeze()
            error2 = error2.reshape(-1)

            name_1 = 'TFT'
            name_2 = 'npi_TFT'

    elif dataset == 'N':

        data = 'Nordex'
        
        offset = 360

        actuals = lstm_scaler_f_N.inverse_transform(npi_y_test_lstm_f_N)*1000
        actuals_plot = actuals[offset+1:offset+period+1]

        if model == 'LSTM':

            preds1_scaled = np.load('predictions/LSTM/LSTM_f_N_predictions.npy')
            preds2_scaled = np.load('predictions/LSTM/npi_LSTM_f_N_predictions.npy')

            preds1 = lstm_scaler_f_N.inverse_transform(preds1_scaled)*1000
            preds1 = preds1.ravel()
            preds1_plot = preds1[offset+2:offset+period+2]

            preds2 = lstm_scaler_f_N.inverse_transform(preds2_scaled)*1000
            preds2 = preds2.ravel()
            preds2_plot = preds2[offset+2:offset+period+2]

            error1 = preds1[offset+2:offset+period+2] - actuals[offset+1:offset+period+1].squeeze()
            error1 = error1.reshape(-1)

            error2 = preds2[offset+2:offset+period+2] - actuals[offset+1:offset+period+1].squeeze()
            error2 = error2.reshape(-1)

            name_1 = 'LSTM'
            name_2 = 'npi_LSTM'

        
        elif model == 'TFT':

            preds1_scaled = np.load('predictions/TFT/TFT_f_N_predictions.npy')
            preds2_scaled = np.load('predictions/TFT/npi_TFT_f_N_predictions.npy')


            preds1 = ((tft_scaler_f_N.inverse_transform(preds1_scaled.reshape(-1, 1)))).reshape(-1)*1000
            preds1 = preds1.ravel()
            preds1_plot = preds1[offset:offset+period]
            preds2 = ((tft_scaler_f_N.inverse_transform(preds2_scaled.reshape(-1, 1)))).reshape(-1)*1000
            preds2 = preds2.ravel()
            preds2_plot = preds2[offset:offset+period]
            

            error1 = preds1[offset:offset+period] - actuals[offset+1:offset+period+1].squeeze()
            error1 = error1.reshape(-1)

            error2 = preds2[offset:offset+period] - actuals[offset+1:offset+period+1].squeeze()
            error2 = error2.reshape(-1)

            name_1 = 'TFT'
            name_2 = 'npi_TFT'


    x = np.arange(0,period)
    zeros = np.zeros(period)

    plt.figure(figsize=(20, 10))
    plt.plot(x, actuals_plot, color='#fe9929', alpha=1, linestyle='--',  linewidth = 1.7, label='Ground Truth')
    plt.plot(x, preds1_plot, color='#8856a7', alpha=0.8, linewidth = 1.5, label= name_1)
    plt.plot(x, preds2_plot, color='#74c476', alpha=0.8, linewidth = 1.5, label= name_2)
    plt.fill_between(x, zeros, error1, color='#810f7c', alpha=0.4, label='{} error'.format(name_1))
    plt.fill_between(x, zeros, error2, color='#31a354', alpha=0.4, label='{} Error'.format(name_2))

    plt.xlabel('Time-step')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Comparative/compared_{}_{}_plot.png'.format(model,data), dpi=300)
    plt.show()
    



def error_plots(dataset, model, period):

    if dataset == 'V':

        offset = 50

        data = 'Vestas'

        if model == 'LSTM':

            true_measurements_unscaled = lstm_scaler_f_V.inverse_transform(npi_y_test_lstm_f_V)*1000
            true_measurements_unscaled = true_measurements_unscaled.ravel()         

            preds_LSTM_scaled = np.load('predictions/LSTM/LSTM_f_V_predictions.npy')
            preds_LSTM_npi_scaled = np.load('predictions/LSTM/npi_LSTM_f_V_predictions.npy')

            predictions_1 = lstm_scaler_f_V.inverse_transform(preds_LSTM_scaled)*1000
            predictions_1 = predictions_1.ravel()
            predictions_2 = lstm_scaler_f_V.inverse_transform(preds_LSTM_npi_scaled)*1000
            predictions_2 = predictions_2.ravel()


            errors_1 = np.abs(predictions_1[offset+2:offset+period+2] - true_measurements_unscaled[offset+1:offset+period+1])
            errors_1 = errors_1.reshape(-1)

            errors_2 = np.abs(predictions_2[offset+2:offset+period+2] - true_measurements_unscaled[offset+1:offset+period+1])
            errors_2 = errors_2.reshape(-1)

            error_data = {
                'LSTM': errors_1,
                'npi_LSTM': errors_2
            }

            name_1 = 'LSTM'
            name_2 = 'npi_LSTM'
        
        elif model == 'TFT':      

            y_test = npi_test_tft_f_V['ActivePower'][30:-24]
            true_measurements_unscaled = ((tft_scaler_f_V.inverse_transform(y_test.values.reshape(-1, 1)))).reshape(-1)*1000
            true_measurements_unscaled= true_measurements_unscaled.ravel()

            preds_TFT_scaled = np.load('predictions/TFT/TFT_f_V_predictions.npy')
            preds_TFT_npi_scaled = np.load('predictions/TFT/npi_TFT_f_V_predictions.npy')

            predictions_1 = ((tft_scaler_f_V.inverse_transform(preds_TFT_scaled.reshape(-1, 1)))).reshape(-1)*1000
            predictions_1 = predictions_1.ravel()
            predictions_2 = ((tft_scaler_f_V.inverse_transform(preds_TFT_npi_scaled.reshape(-1, 1)))).reshape(-1)*1000
            predictions_2 = predictions_2.ravel()

            errors_1 = np.abs(predictions_1[offset:offset+period] - true_measurements_unscaled[offset+1:offset+period+1])
            errors_1 = errors_1.reshape(-1)

            errors_2 = np.abs(predictions_2[offset:offset+period] - true_measurements_unscaled[offset+1:offset+period+1])
            errors_2 = errors_2.reshape(-1)

            error_data = {
                'TFT': errors_1,
                'npi_TFT': errors_2
            }

            name_1 = 'TFT'
            name_2 = 'npi_TFT'

    elif dataset == 'N':

        offset = 360

        data = 'Nordex'

        if model == 'LSTM':

            true_measurements_unscaled = lstm_scaler_f_N.inverse_transform(npi_y_test_lstm_f_N)*1000
            true_measurements_unscaled = true_measurements_unscaled.ravel()

            preds_LSTM_scaled = np.load('predictions/LSTM/LSTM_f_N_predictions.npy')
            preds_LSTM_npi_scaled = np.load('predictions/LSTM/npi_LSTM_f_N_predictions.npy')

            predictions_1 = lstm_scaler_f_N.inverse_transform(preds_LSTM_scaled)*1000
            predictions_1 = predictions_1.ravel()
            predictions_2 = lstm_scaler_f_N.inverse_transform(preds_LSTM_npi_scaled)*1000
            predictions_2 = predictions_2.ravel()

            errors_1 = np.abs(predictions_1[offset+2:offset+period+2] - true_measurements_unscaled[offset+1:offset+period+1])
            errors_1 = errors_1.reshape(-1)

            errors_2 = np.abs(predictions_2[offset+2:offset+period+2] - true_measurements_unscaled[offset+1:offset+period+1])
            errors_2 = errors_2.reshape(-1)      

            error_data = {
                'LSTM': errors_1,
                'npi_LSTM': errors_2
            }

            name_1 = 'LSTM'
            name_2 = 'npi_LSTM'


        elif model == 'TFT':

            y_test = npi_test_tft_f_N['ActivePower'][30:-24]
            true_measurements_unscaled = ((tft_scaler_f_N.inverse_transform(y_test.values.reshape(-1, 1)))).reshape(-1)*1000
            true_measurements_unscaled = true_measurements_unscaled.ravel()

            preds_TFT_scaled = np.load('predictions/TFT/TFT_f_N_predictions.npy')
            preds_TFT_npi_scaled = np.load('predictions/TFT/npi_TFT_f_N_predictions.npy')

            predictions_1 = ((tft_scaler_f_N.inverse_transform(preds_TFT_scaled.reshape(-1, 1)))).reshape(-1)*1000
            predictions_1 = predictions_1.ravel()
            predictions_2 = ((tft_scaler_f_N.inverse_transform(preds_TFT_npi_scaled.reshape(-1, 1)))).reshape(-1)*1000
            predictions_2 = predictions_2.ravel()

            errors_1 = np.abs(predictions_1[offset:offset+period] - true_measurements_unscaled[offset+1:offset+period+1])
            errors_1 = errors_1.reshape(-1)

            errors_2 = np.abs(predictions_2[offset:offset+period] - true_measurements_unscaled[offset+1:offset+period+1])
            errors_2 = errors_2.reshape(-1)

            error_data = {
                'TFT': errors_1,
                'npi_TFT': errors_2
            }

            name_1 = 'TFT'
            name_2 = 'npi_TFT'

    # Violin plot
    sns.violinplot(data=[errors_2, errors_1])
    plt.xticks([0, 1], [name_2, name_1])
    plt.ylabel("Error Distributions")

    plt.savefig('plots/Error plots/compared_Error_violin_plot_{}_{}.png'.format(model, dataset), dpi=500)
    plt.show()


# Load scalers 

with open('scalers/LSTM_V_h_scaler.pkl', 'rb') as f:
    lstm_scaler_h_V = pickle.load(f)
with open('scalers/LSTM_V_f_scaler.pkl', 'rb') as f:
    lstm_scaler_f_V = pickle.load(f)
with open('scalers/LSTM_N_h_scaler.pkl', 'rb') as f:
    lstm_scaler_h_N = pickle.load(f)
with open('scalers/LSTM_N_f_scaler.pkl', 'rb') as f:
    lstm_scaler_f_N = pickle.load(f)


with open('scalers/TFT_V_h_scaler.pkl', 'rb') as f:
    tft_scaler_h_V = pickle.load(f)
with open('scalers/TFT_V_f_scaler.pkl', 'rb') as f:
    tft_scaler_f_V = pickle.load(f)
with open('scalers/TFT_N_h_scaler.pkl', 'rb') as f:
    tft_scaler_h_N = pickle.load(f)
with open('scalers/TFT_N_f_scaler.pkl', 'rb') as f:
    tft_scaler_f_N = pickle.load(f)


# Load normal data

y_test_lstm_h_V = np.load('LSTM input/y_test_history-forecast_Vestas.npy')
y_test_lstm_f_V = np.load('LSTM input/y_test_forecast-forecast_Vestas.npy')
y_test_lstm_h_N = np.load('LSTM input/y_test_history-forecast_Nordex.npy')
y_test_lstm_f_N = np.load('LSTM input/y_test_forecast-forecast_Nordex.npy')

test_tft_h_V = pd.read_pickle('TFT input/tft_history-forecast_V_test.pkl')
test_tft_f_V = pd.read_pickle('TFT input/tft_forecast-forecast_V_test.pkl')
test_tft_h_N = pd.read_pickle('TFT input/tft_history-forecast_N_test.pkl')
test_tft_f_N = pd.read_pickle('TFT input/tft_forecast-forecast_N_test.pkl')



# Load test data

npi_y_test_lstm_h_V = np.load('LSTM input/npi_y_test_history-forecast_Vestas.npy')
npi_y_test_lstm_f_V = np.load('LSTM input/npi_y_test_forecast-forecast_Vestas.npy')
npi_y_test_lstm_h_N = np.load('LSTM input/npi_y_test_history-forecast_Nordex.npy')
npi_y_test_lstm_f_N = np.load('LSTM input/npi_y_test_forecast-forecast_Nordex.npy')


npi_test_tft_h_V = pd.read_pickle('TFT input/npi_tft_history-forecast_V_test.pkl')
npi_test_tft_f_V = pd.read_pickle('TFT input/npi_tft_forecast-forecast_V_test.pkl')
npi_test_tft_h_N = pd.read_pickle('TFT input/npi_tft_history-forecast_N_test.pkl')
npi_test_tft_f_N = pd.read_pickle('TFT input/npi_tft_forecast-forecast_N_test.pkl')


plot_comparative_results("V", "LSTM", 144)
plot_comparative_results("V", "TFT", 144)
plot_comparative_results("N", "LSTM", 144)
plot_comparative_results("N", "TFT", 144)

error_plots("V", "LSTM", 144)
error_plots("V", "TFT", 144)
error_plots("N", "LSTM", 144)
error_plots("N", "TFT", 144)

