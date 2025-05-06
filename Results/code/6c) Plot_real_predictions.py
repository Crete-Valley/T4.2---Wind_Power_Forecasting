import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
from tabulate import tabulate

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred))))


def metrics_day_results(dataset):


    if dataset == 'LSTM':

        predictions_V = np.load("predictions/LSTM/real_LSTM_V_predictions.npy")
        predictions_N = np.load("predictions/LSTM/real_LSTM_N_predictions.npy")
        
        preds_list = [predictions_V, predictions_N]
        actuals_list = [npi_y_test_lstm_f_V[:len(predictions_V)], npi_y_test_lstm_f_N[:len(predictions_N)]]
        scalers_list = [lstm_real_scaler_V, lstm_real_scaler_N]
        actuals_scalers_list = [lstm_scaler_f_V, lstm_scaler_f_N]
        names_list = ["V","N"]
        offset_list = [50,200]
        
        RMSE_values = []
        MAE_values = []
        MAPE_values = []
        R2_values = []

        for preds,actuals,scaler,actuals_scaler,offset,name in zip(preds_list,actuals_list,scalers_list,actuals_scalers_list,offset_list,names_list):
            
            if name == 'V':
                actuals_scaler = scaler

            preds_unscaled = scaler.inverse_transform(preds)*1000
            actuals_unscaled = actuals_scaler.inverse_transform(actuals)*1000
            
            rmse_test = np.sqrt(mean_squared_error(actuals_unscaled[offset+1:offset-1+144], preds_unscaled[offset+2:offset+144]))
            mae_test = mean_absolute_error(actuals_unscaled[offset+1:offset-1+144], preds_unscaled[offset+2:offset+144])
            mape_test = mean_absolute_percentage_error(actuals_unscaled[offset+1:offset-1+144], preds_unscaled[offset+2:offset+144])
            r2_score_test = r2_score(actuals_unscaled[offset+1:offset-1+144], preds_unscaled[offset+2:offset+144])*100
                        
            RMSE_values.append(rmse_test)
            MAE_values.append(mae_test)
            MAPE_values.append(mape_test)
            R2_values.append(r2_score_test)


        RMSE_values = [float(value) for value in RMSE_values]
        MAE_values = [float(value) for value in MAE_values]
        MAPE_values = [float(value) for value in MAPE_values]
        R2_values = [float(value) for value in R2_values]
            

        dataset_names = ["Vestas", "Nordex"]
        train_names = ["Forecasted data", "Forecasted data"]
        test_names = ["Forecasted data", "Forecasted data"]

        metrics_df = pd.DataFrame({
            'Dataset': dataset_names,
            'Training data': train_names,
            'Testing Data': test_names,
            'RMSE': [RMSE_values[0], RMSE_values[1]],
            'MAE': [MAE_values[0], MAE_values[1]],
            'MAPE': [MAPE_values[0], MAPE_values[1]],
            'R2': [R2_values [0], R2_values [1]]
            })
        

        metrics_df = metrics_df.round(3)

        fig,ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')

        # Adjust vertical space by setting row heights
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)  # Scale width by 1, height by 2 (change 2 to a larger value for more height)

        for key, cell in table.get_celld().items():
            cell.set_fontsize(12)  # Adjust font size if needed
            cell.set_linewidth(0.5)  # Adjust line width if needed

        plt.tight_layout()
        plt.suptitle("Realistic LSTM models next day results", y=0.7)
        # Save plot
        plt.savefig('results/real_lstm_next_day_results.png', bbox_inches='tight', dpi=300)
        plt.show()

    if dataset == 'TFT':

        predictions_tft_V = np.load("predictions/TFT/real_TFT_V_predictions.npy")
        predictions_tft_N = np.load("predictions/TFT/real_TFT_N_predictions.npy")

        preds_list = [predictions_tft_V, predictions_tft_N]
        actuals_list = [npi_test_tft_f_V[:(len(predictions_tft_V)+54)], npi_test_tft_f_N[:(len(predictions_tft_N)+54)]]
        scalers_list = [tft_real_scaler_V, tft_real_scaler_N]
        actuals_scalers_list = [tft_scaler_f_V, tft_scaler_f_N]
        names_list = ["V","N"]
        offset_list = [50,200]


        RMSE_values = []
        MAE_values = []
        MAPE_values = []
        R2_values = []

        for preds,actuals,scaler,actuals_scaler,offset,name in zip(preds_list,actuals_list,scalers_list,actuals_scalers_list,offset_list,names_list):
            
            y_test = actuals['ActivePower'][30:-24]

            if name == 'V':
                actuals_scaler = scaler

            preds_unscaled = ((scaler.inverse_transform(preds.reshape(-1, 1)))).reshape(-1)*1000
            actuals_unscaled = ((actuals_scaler.inverse_transform(y_test.values.reshape(-1, 1)))).reshape(-1)*1000

            rmse_test = np.sqrt(mean_squared_error(actuals_unscaled[offset+1:offset+144], preds_unscaled[offset:offset-1+144]))
            mae_test = mean_absolute_error(actuals_unscaled[offset+1:offset+144], preds_unscaled[offset:offset-1+144])
            mape_test = mean_absolute_percentage_error(actuals_unscaled[offset+1:offset+144], preds_unscaled[offset:offset-1+144])
            r2_score_test = r2_score(actuals_unscaled[offset+1:offset+144], preds_unscaled[offset:offset-1+144])*100

            RMSE_values.append(rmse_test)
            MAE_values.append(mae_test)
            MAPE_values.append(mape_test)
            R2_values.append(r2_score_test)



        dataset_names = ["Vestas","Nordex"]
        train_names = ["Forecasted data", "Forecasted data"]
        test_names = ["Forecasted data", "Forecasted data"]
        
        metrics_df = pd.DataFrame({
            'Dataset' : dataset_names,
            'Training Data' : train_names,
            'Testing Data' : test_names,
            'RMSE': [RMSE_values[0], RMSE_values[1]],
            'MAE': [MAE_values[0], MAE_values[1]],
            'MAPE': [MAPE_values[0], MAPE_values[1]],
            'R2': [R2_values [0], R2_values [1]]
            })

        
        metrics_df = metrics_df.round(3)

        fig,ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')

        # Adjust vertical space by setting row heights
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)  # Scale width by 1, height by 2 (change 2 to a larger value for more height)

        for key, cell in table.get_celld().items():
            cell.set_fontsize(12)  # Adjust font size if needed
            cell.set_linewidth(0.5)  # Adjust line width if needed

        plt.tight_layout()

        plt.suptitle("Realistic TFT models next day results", y=0.7)
        # Save plot
        plt.savefig('results/real_tft_next_day_results.png', bbox_inches='tight', dpi=300)
        plt.show()


def metrics_total_results(dataset):


    if dataset == 'LSTM':

        predictions_V = np.load("predictions/LSTM/real_LSTM_V_predictions.npy")
        predictions_N = np.load("predictions/LSTM/real_LSTM_N_predictions.npy")
        
        preds_list = [predictions_V, predictions_N]
        actuals_list = [npi_y_test_lstm_f_V[:len(predictions_V)], npi_y_test_lstm_f_N[:len(predictions_N)]]
        scalers_list = [lstm_real_scaler_V, lstm_real_scaler_N]
        actuals_scalers_list = [lstm_scaler_f_V, lstm_scaler_f_N]
        names_list = ["V","N"]
               
        
        RMSE_values = []
        MAE_values = []
        MAPE_values = []
        R2_values = []


        for preds,actuals,scaler,actuals_scaler,name in zip(preds_list,actuals_list,scalers_list,actuals_scalers_list,names_list):
            
            if name == 'V':
                actuals_scaler = scaler

            preds_unscaled = scaler.inverse_transform(preds)*1000
            actuals_unscaled = actuals_scaler.inverse_transform(actuals)*1000
            
            rmse_test = np.sqrt(mean_squared_error(actuals_unscaled[+1:-1], preds_unscaled[2:]))
            mae_test = mean_absolute_error(actuals_unscaled[+1:-1], preds_unscaled[2:])
            mape_test = mean_absolute_percentage_error(actuals_unscaled[+1:-1], preds_unscaled[2:])
            r2_score_test = r2_score(actuals_unscaled[+1:-1], preds_unscaled[2:])*100
                        
            RMSE_values.append(rmse_test)
            MAE_values.append(mae_test)
            MAPE_values.append(mape_test)
            R2_values.append(r2_score_test)


        RMSE_values = [float(value) for value in RMSE_values]
        MAE_values = [float(value) for value in MAE_values]
        MAPE_values = [float(value) for value in MAPE_values]
        R2_values = [float(value) for value in R2_values]
                

        dataset_names = ["Vestas","Nordex"]
        train_names = ["Forecasted data", "Forecasted data"]
        test_names = ["Forecasted data", "Forecasted data"]

        metrics_df = pd.DataFrame({
            'Dataset': dataset_names,
            'Training data': train_names,
            'Testing Data': test_names,
            'RMSE': [RMSE_values[0], RMSE_values[1]],
            'MAE': [MAE_values[0], MAE_values[1]],
            'MAPE': [MAPE_values[0], MAPE_values[1]],
            'R2': [R2_values [0], R2_values [1]]
            })

        metrics_df = metrics_df.round(3)

        fig,ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')

        # Adjust vertical space by setting row heights
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)  # Scale width by 1, height by 2 (change 2 to a larger value for more height)

        for key, cell in table.get_celld().items():
            cell.set_fontsize(12)  # Adjust font size if needed
            cell.set_linewidth(0.5)  # Adjust line width if needed

        plt.tight_layout()
        plt.suptitle("Realistic LSTM models total results", y=0.7)
        # Save plot
        plt.savefig('results/real_lstm_total_results.png', bbox_inches='tight', dpi=300)
        plt.show()

    if dataset == 'TFT':

        predictions_tft_V = np.load("predictions/TFT/real_TFT_V_predictions.npy")
        predictions_tft_N = np.load("predictions/TFT/real_TFT_N_predictions.npy")

        preds_list = [predictions_tft_V, predictions_tft_N]
        actuals_list = [npi_test_tft_f_V[:(len(predictions_tft_V)+54)], npi_test_tft_f_N[:(len(predictions_tft_N)+54)]]
        scalers_list = [tft_real_scaler_V, tft_real_scaler_N]
        actuals_scalers_list = [tft_scaler_f_V, tft_scaler_f_N]
        names_list = ["V","N"]


        RMSE_values = []
        MAE_values = []
        MAPE_values = []
        R2_values = []


        for preds,actuals,scaler,actuals_scaler,name in zip(preds_list,actuals_list,scalers_list,actuals_scalers_list,names_list):
            
            y_test = actuals['ActivePower'][30:-24]

            if name == 'V':
                actuals_scaler = scaler

            preds_unscaled = ((scaler.inverse_transform(preds.reshape(-1, 1)))).reshape(-1)*1000
            actuals_unscaled = ((actuals_scaler.inverse_transform(y_test.values.reshape(-1, 1)))).reshape(-1)*1000

            rmse_test = np.sqrt(mean_squared_error(actuals_unscaled[1:], preds_unscaled[:-1]))
            mae_test = mean_absolute_error(actuals_unscaled[1:], preds_unscaled[:-1])
            mape_test = mean_absolute_percentage_error(actuals_unscaled[1:], preds_unscaled[:-1])
            r2_score_test = r2_score(actuals_unscaled[1:], preds_unscaled[:-1])*100

            RMSE_values.append(rmse_test)
            MAE_values.append(mae_test)
            MAPE_values.append(mape_test)
            R2_values.append(r2_score_test)


        dataset_names = ["Vestas", "Nordex"]
        train_names = ["Forecasted data", "Forecasted data"]
        test_names = ["Forecasted data", "Forecasted data"]
        
        metrics_df = pd.DataFrame({
            'Dataset' : dataset_names,
            'Training Data' : train_names,
            'Testing Data' : test_names,
            'RMSE': [RMSE_values[0], RMSE_values[1]],
            'MAE': [MAE_values[0], MAE_values[1]],
            'MAPE': [MAPE_values[0], MAPE_values[1]],
            'R2': [R2_values [0], R2_values [1]]
            })

        metrics_df = metrics_df.round(3)

        fig,ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')

        # Adjust vertical space by setting row heights
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)  # Scale width by 1, height by 2 (change 2 to a larger value for more height)

        for key, cell in table.get_celld().items():
            cell.set_fontsize(12)  # Adjust font size if needed
            cell.set_linewidth(0.5)  # Adjust line width if needed

        plt.tight_layout()
        plt.suptitle("Realistic TFT models total results", y=0.7)
        plt.savefig('results/real_tft_total_results.png', bbox_inches='tight', dpi=300)
        plt.show()




def plot_predictions(model, dataset, scaler, actuals_scaler, period, y_test):
    
    if dataset == 'V':
        offset = 50
        dataset_name = 'Vestas'
    elif dataset == 'N':
        offset = 200
        dataset_name = 'Nordex'

    if model == 'LSTM':
        
        if dataset == "V":
            actuals_scaler = scaler

        model_name = 'real_lstm_forecast_'+dataset_name
        pred_scaled = np.load('predictions/LSTM/real_LSTM_'+dataset+'_predictions.npy')
        preds = scaler.inverse_transform(pred_scaled)*1000
        actuals = actuals_scaler.inverse_transform(y_test[:len(preds)])*1000
        preds = preds.ravel()
        actuals = actuals.ravel()
        
        x = np.arange(0,period)
        zeros = np.zeros(period)

        error = preds[offset+2:offset+period+2] - actuals[offset+1:offset+period+1]
        
        error = error.reshape(-1)

        plt.figure(figsize=(20, 10))
        plt.plot(x, actuals[offset+1:offset+period+1], label='Actual values', color='blue', alpha=0.7)
        plt.plot(x, preds[offset+2:offset+period+2], label='Predictions', color='red', linestyle='--')
        plt.fill_between(x, zeros, error, color='green', alpha=0.3, label='Error')

        plt.title(model_name)
        plt.xlabel('Time-step')
        plt.ylabel('Power Consumption')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/real_lstm_'+dataset+'_plot.png', dpi=300)
        plt.show()


    elif model == 'TFT':
        
        if dataset == "V":
            actuals_scaler = scaler

        model_name = 'real_tft_forecast_'+dataset_name
        pred_scaled = np.load('predictions/TFT/real_TFT_'+dataset+'_predictions.npy')
        
        y_test = y_test[:(len(pred_scaled)+54)]
        y_test = y_test['ActivePower'][30:-24]

        preds = ((scaler.inverse_transform(pred_scaled.reshape(-1, 1)))).reshape(-1)*1000
        actuals = ((actuals_scaler.inverse_transform(y_test.values.reshape(-1, 1)))).reshape(-1)*1000

        x = np.arange(0,period)
        zeros = np.zeros(period)
        error = preds[offset:offset+period] - actuals[offset+1:offset+period+1]

        plt.figure(figsize=(20, 10))
        plt.plot(x, actuals[offset+1:offset+period+1], label='Actual values', color='blue', alpha=0.7)
        plt.plot(x, preds[offset:offset+period], label='Predictions', color='red', linestyle='--')
        plt.fill_between(x, zeros, error, color='green', alpha=0.3, label='Error')

        plt.title(model_name)
        plt.xlabel('Time-step')
        plt.ylabel('Power Consumption')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/real_tft_'+dataset+'_plot.png', dpi=300)
        plt.show()




#actuals scalers

with open('scalers/LSTM_V_f_scaler.pkl', 'rb') as f:
    lstm_scaler_f_V = pickle.load(f)
with open('scalers/LSTM_N_f_scaler.pkl', 'rb') as f:
    lstm_scaler_f_N = pickle.load(f)


with open('scalers/TFT_V_f_scaler.pkl', 'rb') as f:
    tft_scaler_f_V = pickle.load(f)
with open('scalers/TFT_N_f_scaler.pkl', 'rb') as f:
    tft_scaler_f_N = pickle.load(f)


#real scalers

with open('scalers/real_LSTM_V_scaler.pkl', 'rb') as f:
    lstm_real_scaler_V = pickle.load(f)
with open('scalers/real_LSTM_N_scaler.pkl', 'rb') as f:
    lstm_real_scaler_N = pickle.load(f)


with open('scalers/real_TFT_V_scaler.pkl', 'rb') as f:
    tft_real_scaler_V = pickle.load(f)
with open('scalers/real_TFT_N_scaler.pkl', 'rb') as f:
    tft_real_scaler_N = pickle.load(f)


# Load the test data

npi_y_test_lstm_f_V = np.load('LSTM input/npi_y_test_forecast-forecast_Vestas.npy')
npi_y_test_lstm_f_N = np.load('LSTM input/npi_y_test_forecast-forecast_Nordex.npy')


npi_test_tft_f_V = pd.read_pickle('TFT input/npi_tft_forecast-forecast_V_test.pkl')
npi_test_tft_f_N = pd.read_pickle('TFT input/npi_tft_forecast-forecast_N_test.pkl')


#Check results for LSTM

metrics_total_results('LSTM')
metrics_day_results('LSTM')


plot_predictions('LSTM', 'V', lstm_scaler_f_V, lstm_real_scaler_V, 144, npi_y_test_lstm_f_V)
plot_predictions('LSTM', 'N', lstm_scaler_f_N, lstm_real_scaler_N, 144, npi_y_test_lstm_f_N)

#Check results for TFT

metrics_total_results('TFT')
metrics_day_results('TFT')


plot_predictions('TFT', 'V', tft_scaler_f_V, tft_real_scaler_V, 144, npi_test_tft_f_V)
plot_predictions('TFT', 'N', tft_scaler_f_N, tft_real_scaler_N, 144, npi_test_tft_f_N)



