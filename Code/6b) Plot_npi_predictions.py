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

        predictions_V_h = np.load("predictions/LSTM/npi_LSTM_h_V_predictions.npy")
        predictions_V_f = np.load("predictions/LSTM/npi_LSTM_f_V_predictions.npy")
        predictions_N_h = np.load("predictions/LSTM/npi_LSTM_h_N_predictions.npy")
        predictions_N_f = np.load("predictions/LSTM/npi_LSTM_f_N_predictions.npy")
        
        preds_list = [predictions_V_h, predictions_V_f, predictions_N_h, predictions_N_f]
        actuals_list = [npi_y_test_lstm_h_V, npi_y_test_lstm_f_V, npi_y_test_lstm_h_N, npi_y_test_lstm_f_N]
        scalers_list = [lstm_scaler_h_V, lstm_scaler_f_V, lstm_scaler_h_N, lstm_scaler_f_N]
        period_list = [144]
        offset_list = [50,50,360,360]
    

        for period in period_list:

            RMSE_values = []
            MAE_values = []
            MAPE_values = []
            R2_values = []


            for preds,actuals,scaler,offset in zip(preds_list,actuals_list,scalers_list,offset_list):
                
                preds_unscaled = scaler.inverse_transform(preds)*1000
                actuals_unscaled = scaler.inverse_transform(actuals)*1000
                
                rmse_test = np.sqrt(mean_squared_error(actuals_unscaled[offset+1:offset+1+period], preds_unscaled[offset+2:offset+2+period]))
                mae_test = mean_absolute_error(actuals_unscaled[offset+1:offset+1+period], preds_unscaled[offset+2:offset+2+period])
                mape_test = mean_absolute_percentage_error(actuals_unscaled[offset+1:offset+1+period], preds_unscaled[offset+2:offset+2+period])
                r2_score_test = r2_score(actuals_unscaled[offset+1:offset+1+period], preds_unscaled[offset+2:offset+2+period])*100
                            
                RMSE_values.append(rmse_test)
                MAE_values.append(mae_test)
                MAPE_values.append(mape_test)
                R2_values.append(r2_score_test)

            RMSE_values = [float(value) for value in RMSE_values]
            MAE_values = [float(value) for value in MAE_values]
            MAPE_values = [float(value) for value in MAPE_values]
            R2_values = [float(value) for value in R2_values]
                

            dataset_names = ["Vestas", "Vestas", "Nordex", "Nordex"]
            train_names = ["Historical data", "Forecasted data", "Historical data", "Forecasted data"]
            test_names = ["Forecasted data", "Forecasted data", "Forecasted data", "Forecasted data"]

            metrics_df = pd.DataFrame({
                'Dataset': dataset_names,
                'Training data': train_names,
                'Testing Data': test_names,
                'RMSE': [RMSE_values[0], RMSE_values[1], RMSE_values[2], RMSE_values[3]],
                'MAE': [MAE_values[0], MAE_values[1], MAE_values[2], MAE_values[3]],
                'MAPE': [MAPE_values[0], MAPE_values[1], MAPE_values[2], MAPE_values[3]],
                'R2': [R2_values [0], R2_values [1], R2_values [2], R2_values [3]]
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
            plt.suptitle("Non physics informed LSTM models results {}".format(period), y=0.8)
            # Save plot
            plt.savefig('results/LSTM/npi_lstm_results_{}.png'.format(period), bbox_inches='tight', dpi=300)
            plt.show()

    if dataset == 'TFT':

        predictions_V_h = np.load("predictions/TFT/npi_TFT_h_V_predictions.npy")
        predictions_V_f = np.load("predictions/TFT/npi_TFT_f_V_predictions.npy")
        predictions_N_h = np.load("predictions/TFT/npi_TFT_h_N_predictions.npy")
        predictions_N_f = np.load("predictions/TFT/npi_TFT_f_N_predictions.npy")

        preds_list = [predictions_V_h, predictions_V_f, predictions_N_h, predictions_N_f]
        actuals_list = [npi_test_tft_h_V, npi_test_tft_f_V, npi_test_tft_h_N, npi_test_tft_f_N]
        scalers_list = [tft_scaler_h_V, tft_scaler_f_V, tft_scaler_h_N, tft_scaler_f_N]
        period_list = [144]
        offset_list = [50,50,360,360]


        for period in period_list:

            RMSE_values = []
            MAE_values = []
            MAPE_values = []
            R2_values = []

            for preds,actuals,scaler,offset in zip(preds_list,actuals_list,scalers_list,offset_list):
                
                y_test = actuals['ActivePower'][30:-24]

                preds_unscaled = ((scaler.inverse_transform(preds.reshape(-1, 1)))).reshape(-1)*1000
                actuals_unscaled = ((scaler.inverse_transform(y_test.values.reshape(-1, 1)))).reshape(-1)*1000

                rmse_test = np.sqrt(mean_squared_error(actuals_unscaled[offset+1:offset+1+period], preds_unscaled[offset:offset+period]))
                mae_test = mean_absolute_error(actuals_unscaled[offset+1:offset+1+period], preds_unscaled[offset:offset+period])
                mape_test = mean_absolute_percentage_error(actuals_unscaled[offset+1:offset+1+period], preds_unscaled[offset:offset+period])
                r2_score_test = r2_score(actuals_unscaled[offset+1:offset+1+period], preds_unscaled[offset:offset+period])*100

                RMSE_values.append(rmse_test)
                MAE_values.append(mae_test)
                MAPE_values.append(mape_test)
                R2_values.append(r2_score_test)

            dataset_names = ["Vestas", "Vestas", "Nordex", "Nordex"]
            train_names = ["Historical data", "Forecasted data", "Historical data", "Forecasted data"]
            test_names = ["Forecasted data", "Forecasted data", "Forecasted data", "Forecasted data"]
            
            metrics_df = pd.DataFrame({
                'Dataset' : dataset_names,
                'Training Data' : train_names,
                'Testing Data' : test_names,
                'RMSE': [RMSE_values[0], RMSE_values[1], RMSE_values[2], RMSE_values[3]],
                'MAE': [MAE_values[0], MAE_values[1], MAE_values[2], MAE_values[3]],
                'MAPE': [MAPE_values[0], MAPE_values[1], MAPE_values[2], MAPE_values[3]],
                'R2': [R2_values [0], R2_values [1], R2_values [2], R2_values [3]]
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

            plt.suptitle("Non physics informed TFT models results {}".format(period), y=0.8)
            # Save plot
            plt.savefig('results/TFT/npi_tft_results_{}.png'.format(period), bbox_inches='tight', dpi=300)
            plt.show()


def metrics_total_results(dataset):


    if dataset == 'LSTM':

        predictions_V_h = np.load("predictions/LSTM/npi_LSTM_h_V_predictions.npy")
        predictions_V_f = np.load("predictions/LSTM/npi_LSTM_f_V_predictions.npy")
        predictions_N_h = np.load("predictions/LSTM/npi_LSTM_h_N_predictions.npy")
        predictions_N_f = np.load("predictions/LSTM/npi_LSTM_f_N_predictions.npy")
        
        preds_list = [predictions_V_h, predictions_V_f, predictions_N_h, predictions_N_f]
        actuals_list = [npi_y_test_lstm_h_V, npi_y_test_lstm_f_V, npi_y_test_lstm_h_N, npi_y_test_lstm_f_N]
        scalers_list = [lstm_scaler_h_V, lstm_scaler_f_V, lstm_scaler_h_N, lstm_scaler_f_N]
               
        
        RMSE_values = []
        MAE_values = []
        MAPE_values = []
        R2_values = []

        for preds,actuals,scaler in zip(preds_list,actuals_list,scalers_list):
            
            preds_unscaled = scaler.inverse_transform(preds)*1000
            actuals_unscaled = scaler.inverse_transform(actuals)*1000
            
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
                

        dataset_names = ["Vestas", "Vestas", "Nordex", "Nordex"]
        train_names = ["Historical data", "Forecasted data", "Historical data", "Forecasted data"]
        test_names = ["Forecasted data", "Forecasted data", "Forecasted data", "Forecasted data"]

        metrics_df = pd.DataFrame({
            'Dataset': dataset_names,
            'Training data': train_names,
            'Testing Data': test_names,
            'RMSE': [RMSE_values[0], RMSE_values[1], RMSE_values[2], RMSE_values[3]],
            'MAE': [MAE_values[0], MAE_values[1], MAE_values[2], MAE_values[3]],
            'MAPE': [MAPE_values[0], MAPE_values[1], MAPE_values[2], MAPE_values[3]],
            'R2': [R2_values [0], R2_values [1], R2_values [2], R2_values [3]]
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
        plt.suptitle("Non physics informed LSTM models total results", y=0.8)
        # Save plot
        plt.savefig('results/LSTM/npi_lstm_total_results.png', bbox_inches='tight', dpi=300)
        plt.show()


    if dataset == 'TFT':

        predictions_V_h = np.load("predictions/TFT/npi_TFT_h_V_predictions.npy")
        predictions_V_f = np.load("predictions/TFT/npi_TFT_f_V_predictions.npy")
        predictions_N_h = np.load("predictions/TFT/npi_TFT_h_N_predictions.npy")
        predictions_N_f = np.load("predictions/TFT/npi_TFT_f_N_predictions.npy")

        preds_list = [predictions_V_h, predictions_V_f, predictions_N_h, predictions_N_f]
        actuals_list = [npi_test_tft_h_V, npi_test_tft_f_V, npi_test_tft_h_N, npi_test_tft_f_N]
        scalers_list = [tft_scaler_h_V, tft_scaler_f_V, tft_scaler_h_N, tft_scaler_f_N]


        RMSE_values = []
        MAE_values = []
        MAPE_values = []
        R2_values = []

        for preds,actuals,scaler in zip(preds_list,actuals_list,scalers_list):
            
            y_test = actuals['ActivePower'][30:-24]

            preds_unscaled = ((scaler.inverse_transform(preds.reshape(-1, 1)))).reshape(-1)*1000
            actuals_unscaled = ((scaler.inverse_transform(y_test.values.reshape(-1, 1)))).reshape(-1)*1000

            rmse_test = np.sqrt(mean_squared_error(actuals_unscaled[1:], preds_unscaled[:-1]))
            mae_test = mean_absolute_error(actuals_unscaled[1:], preds_unscaled[:-1])
            mape_test = mean_absolute_percentage_error(actuals_unscaled[1:], preds_unscaled[:-1])
            r2_score_test = r2_score(actuals_unscaled[1:], preds_unscaled[:-1])*100

            RMSE_values.append(rmse_test)
            MAE_values.append(mae_test)
            MAPE_values.append(mape_test)
            R2_values.append(r2_score_test)

        dataset_names = ["Vestas", "Vestas", "Nordex", "Nordex"]
        train_names = ["Historical data", "Forecasted data", "Historical data", "Forecasted data"]
        test_names = ["Forecasted data", "Forecasted data", "Forecasted data", "Forecasted data"]
        
        metrics_df = pd.DataFrame({
            'Dataset' : dataset_names,
            'Training Data' : train_names,
            'Testing Data' : test_names,
            'RMSE': [RMSE_values[0], RMSE_values[1], RMSE_values[2], RMSE_values[3]],
            'MAE': [MAE_values[0], MAE_values[1], MAE_values[2], MAE_values[3]],
            'MAPE': [MAPE_values[0], MAPE_values[1], MAPE_values[2], MAPE_values[3]],
            'R2': [R2_values [0], R2_values [1], R2_values [2], R2_values [3]]
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
        plt.suptitle("Non physics informed TFT models total results", y=0.8)
        plt.savefig('results/TFT/npi_tft_total_results.png', bbox_inches='tight', dpi=300)
        plt.show()


def plot_comparative_results(dataset, period):

    if dataset == 'V':
        
        offset = 50

        actuals = lstm_scaler_f_V.inverse_transform(npi_y_test_lstm_f_V)*1000

        preds1_scaled = np.load('predictions/LSTM/npi_LSTM_f_V_predictions.npy')
        preds2_scaled = np.load('predictions/TFT/npi_TFT_f_V_predictions.npy')

        preds1 = lstm_scaler_f_V.inverse_transform(preds1_scaled)*1000
        preds1 = preds1.ravel()
        preds2 = ((tft_scaler_f_V.inverse_transform(preds2_scaled.reshape(-1, 1)))).reshape(-1)*1000
        preds2 = preds2.ravel()

        x = np.arange(0,period)
        zeros = np.zeros(period)


        error1 = preds1[offset+2:offset+period+2] - actuals[offset+1:offset+period+1].squeeze()
        error1 = error1.reshape(-1)

        error2 = preds2[offset:offset+period] - actuals[offset+1:offset+period+1].squeeze()
        error2 = error2.reshape(-1)


        plt.figure(figsize=(20, 10))
        plt.plot(x, actuals[offset+1:offset+period+1], color='#fe9929', alpha=1, linestyle='--',  linewidth = 1.7, label='Ground Truth')
        plt.plot(x, preds1[offset+2:offset+period+2], color='#8856a7', alpha=0.8, linewidth = 1.5, label='LSTM Predictions')
        plt.plot(x, preds2[offset:offset+period], color='#74c476', alpha=0.8, linewidth = 1.5, label='TFT Predictions')
        plt.fill_between(x, zeros, error1, color='#810f7c', alpha=0.4, label='LSTM Error')
        plt.fill_between(x, zeros, error2, color='#31a354', alpha=0.4, label='TFT Error')

        plt.xlabel('Time-step')
        plt.ylabel('Power Consumption')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/Comparative/npi_Vestas_plot.png', dpi=300)
        plt.show()
    
    elif dataset == 'N':
        
        offset = 360

        actuals = lstm_scaler_f_N.inverse_transform(npi_y_test_lstm_f_N)*1000

        preds1_scaled = np.load('predictions/LSTM/npi_LSTM_f_N_predictions.npy')
        preds2_scaled = np.load('predictions/TFT/npi_TFT_f_N_predictions.npy')

        preds1 = lstm_scaler_f_N.inverse_transform(preds1_scaled)*1000
        preds1 = preds1.ravel()
        preds2 = ((tft_scaler_f_N.inverse_transform(preds2_scaled.reshape(-1, 1)))).reshape(-1)*1000
        preds2 = preds2.ravel()

        x = np.arange(0,period)
        zeros = np.full((period), 0)

        error1 = preds1[offset+2:offset+period+2] - actuals[offset+1:offset+period+1].squeeze()
        error1 = error1.reshape(-1)

        error2 = preds2[offset:offset+period] - actuals[offset+1:offset+period+1].squeeze()
        error2 = error2.reshape(-1)



        plt.figure(figsize=(20, 10))
        plt.plot(x, actuals[offset+1:offset+period+1], color='#fe9929', alpha=1, linestyle='--',  linewidth = 1.7, label='Ground Truth')
        plt.plot(x, preds1[offset+2:offset+period+2], color='#8856a7', alpha=0.8,  linewidth = 1.5, label='LSTM Predictions')
        plt.plot(x, preds2[offset:offset+period], color='#74c476', alpha=0.8,  linewidth = 1.5, label='TFT Predictions')
        plt.fill_between(x, zeros, error1, color='#810f7c', alpha=0.4, label='LSTM Error')
        plt.fill_between(x, zeros, error2, color='#31a354', alpha=0.4, label='TFT Error')

        plt.xlabel('Time-step')
        plt.ylabel('Power Consumption')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/Comparative/npi_Nordex_plot.png', dpi=300)
        plt.show()



def error_plots(dataset, period):

    if dataset == 'V':

        offset = 50
        true_measurements_unscaled_LSTM = lstm_scaler_f_V.inverse_transform(npi_y_test_lstm_f_V)*1000
        true_measurements_unscaled_LSTM = true_measurements_unscaled_LSTM.ravel()

        y_test = npi_test_tft_f_V['ActivePower'][30:-24]
        true_measurements_unscaled_TFT = ((tft_scaler_f_V.inverse_transform(y_test.values.reshape(-1, 1)))).reshape(-1)*1000
        true_measurements_unscaled_TFT = true_measurements_unscaled_TFT.ravel()

        preds_LSTM_scaled = np.load('predictions/LSTM/npi_LSTM_f_V_predictions.npy')
        preds_TFT_scaled = np.load('predictions/TFT/npi_TFT_f_V_predictions.npy')

        predictions_LSTM = lstm_scaler_f_V.inverse_transform(preds_LSTM_scaled)*1000
        predictions_LSTM = predictions_LSTM.ravel()
        predictions_TFT = ((tft_scaler_f_V.inverse_transform(preds_TFT_scaled.reshape(-1, 1)))).reshape(-1)*1000
        predictions_TFT = predictions_TFT.ravel()

        data = 'Vestas'


    elif dataset == 'N':

        offset = 360
        true_measurements_unscaled_LSTM = lstm_scaler_f_N.inverse_transform(npi_y_test_lstm_f_N)*1000
        true_measurements_unscaled_LSTM = true_measurements_unscaled_LSTM.ravel()

        y_test = npi_test_tft_f_N['ActivePower'][30:-24]
        true_measurements_unscaled_TFT = ((tft_scaler_f_N.inverse_transform(y_test.values.reshape(-1, 1)))).reshape(-1)*1000
        true_measurements_unscaled_TFT = true_measurements_unscaled_TFT.ravel()

        preds_LSTM_scaled = np.load('predictions/LSTM/npi_LSTM_f_N_predictions.npy')
        preds_TFT_scaled = np.load('predictions/TFT/npi_TFT_f_N_predictions.npy')

        predictions_LSTM = lstm_scaler_f_N.inverse_transform(preds_LSTM_scaled)*1000
        predictions_LSTM = predictions_LSTM.ravel()
        predictions_TFT = ((tft_scaler_f_N.inverse_transform(preds_TFT_scaled.reshape(-1, 1)))).reshape(-1)*1000
        predictions_TFT = predictions_TFT.ravel()

        data = 'Nordex'


    LSTM_errors = np.abs(predictions_LSTM[offset+2:offset+period+2] - true_measurements_unscaled_LSTM[offset+1:offset+period+1])
    LSTM_errors = LSTM_errors.reshape(-1)

    TFT_errors = np.abs(predictions_TFT[offset:offset+period] - true_measurements_unscaled_TFT[offset+1:offset+period+1])
    TFT_errors = TFT_errors.reshape(-1)

    error_data = {
        'LSTM': LSTM_errors,
        'TFT': TFT_errors
    }

    # Violin plot
    sns.violinplot(data=[LSTM_errors, TFT_errors])
    plt.xticks([0, 1], ["LSTM Error", "TFT Error"])
    plt.ylabel("Error Distributions")

    plt.savefig('plots/Error plots/npi_Error_violin_plot_{}.png'.format(dataset), dpi=500)
    plt.show()


def plot_predictions(model, type, dataset, scaler, period, y_test):
    
    if dataset == 'V':
        offset = 50
        dataset_name = 'Vestas'
    elif dataset == 'N':
        offset = 360
        dataset_name = 'Nordex'

    if type == 'h':
        type_name = 'history'
    elif type == 'f':
        type_name = 'forecast'

    if model == 'LSTM':
        
        dir = 'LSTM/npi_lstm'
        model_name = 'lstm_'+type_name+'_'+dataset_name
        pred_scaled = np.load('predictions/LSTM/npi_LSTM_'+type+'_'+dataset+'_predictions.npy')
        preds = scaler.inverse_transform(pred_scaled)*1000
        actuals = scaler.inverse_transform(y_test)*1000
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
        plt.savefig('plots/'+dir+'_'+type+'_'+dataset+'_plot.png', dpi=300)
        plt.show()


    elif model == 'TFT':
        
        dir = 'TFT/npi_tft'
        model_name = 'tft_'+type_name+'_'+dataset_name
        y_test = y_test['ActivePower'][30:-24]
        pred_scaled = np.load('predictions/TFT/npi_TFT_'+type+'_'+dataset+'_predictions.npy')
        preds = ((scaler.inverse_transform(pred_scaled.reshape(-1, 1)))).reshape(-1)*1000
        actuals = ((scaler.inverse_transform(y_test.values.reshape(-1, 1)))).reshape(-1)*1000

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
        plt.savefig('plots/'+dir+'_'+type+'_'+dataset+'_plot.png', dpi=300)
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



# Load test data

npi_y_test_lstm_h_V = np.load('LSTM input/npi_y_test_history-forecast_Vestas.npy')
npi_y_test_lstm_f_V = np.load('LSTM input/npi_y_test_forecast-forecast_Vestas.npy')
npi_y_test_lstm_h_N = np.load('LSTM input/npi_y_test_history-forecast_Nordex.npy')
npi_y_test_lstm_f_N = np.load('LSTM input/npi_y_test_forecast-forecast_Nordex.npy')


npi_test_tft_h_V = pd.read_pickle('TFT input/npi_tft_history-forecast_V_test.pkl')
npi_test_tft_f_V = pd.read_pickle('TFT input/npi_tft_forecast-forecast_V_test.pkl')
npi_test_tft_h_N = pd.read_pickle('TFT input/npi_tft_history-forecast_N_test.pkl')
npi_test_tft_f_N = pd.read_pickle('TFT input/npi_tft_forecast-forecast_N_test.pkl')



#Check results for LSTM

#metrics_total_results('LSTM')
metrics_day_results('LSTM')

plot_predictions('LSTM', 'h', 'V', lstm_scaler_h_V, 144, npi_y_test_lstm_h_V)
plot_predictions('LSTM', 'f', 'V', lstm_scaler_f_V, 144, npi_y_test_lstm_f_V)
plot_predictions('LSTM', 'h', 'N', lstm_scaler_h_N, 144, npi_y_test_lstm_h_N)
plot_predictions('LSTM', 'f', 'N', lstm_scaler_f_N, 144, npi_y_test_lstm_f_N)


#Comparative results
plot_comparative_results("V", 144)

plot_comparative_results("N", 144)


#Error plots

error_plots("V", 144)

error_plots("N", 144)



#Check results for TFT

#metrics_total_results('TFT')
metrics_day_results('TFT')

plot_predictions('TFT', 'h', 'V', tft_scaler_h_V, 144, npi_test_tft_h_V)
plot_predictions('TFT', 'f', 'V', tft_scaler_f_V, 144, npi_test_tft_f_V)
plot_predictions('TFT', 'h', 'N', tft_scaler_h_N, 144, npi_test_tft_h_N)
plot_predictions('TFT', 'f', 'N', tft_scaler_f_N, 144, npi_test_tft_f_N)


