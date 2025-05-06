import metpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import gc


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor,XGBRFRegressor
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


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


def model_selection(i):
    if i == 0:
        model = RandomForestRegressor(max_depth=10, random_state=42)
        model_name = "RandomForestRegressor"
    elif i == 1:
        model = CatBoostRegressor(random_state=42,verbose=False)
        model_name = "CatBoostRegressor"
    elif i == 2:
        model = GradientBoostingRegressor(random_state=42)
        model_name = "GradientBoostingRegressor"
    elif i == 3:
        model = LinearRegression()
        model_name = "LinearRegressor"
    elif i == 4:
        model = ExtraTreesRegressor(max_depth=10, random_state=42)
        model_name = "ExtraTreesRegressor"
    elif i == 5:
        model = AdaBoostRegressor(random_state=42)
        model_name = "AdaBoostRegressor"
    elif i == 6:
        model = DecisionTreeRegressor(max_depth=10, random_state=42)
        model_name = "DecisionTreeRegressor"
    elif i == 7:
        model = XGBRegressor(random_state=42)
        model_name = "XGBRegressor"
    elif i == 8:
        model = XGBRFRegressor(random_state=42)
        model_name = "XGBRFRegressor"

    return model, model_name

def results_df(i):
    if i == 0:
        df = random_forest_regressor_results
    elif i == 1:
        df = catBoost_Regressor_results
    elif i == 2:
        df = gradientBoosting_Regressor_results
    elif i == 3:
        df = linear_Regression_results
    elif i == 4:
        df = extraTrees_Regressor_results
    elif i == 5:
        df = adaBoost_Regressor_results
    elif i == 6:
        df = decisionTree_Regressor_results
    elif i == 7:
        df = xGB_Regressor_results
    elif i == 8:
        df = xGBRF_Regressor_results
    
    return df



random_forest_regressor_results = pd.DataFrame(columns=['Dataset', 'Predictions on :', 'MAE', 'RMSE', 'MAPE', 'R2'])
gradientBoosting_Regressor_results = pd.DataFrame(columns=['Dataset', 'Predictions on :', 'MAE', 'RMSE', 'MAPE', 'R2'])
linear_Regression_results = pd.DataFrame(columns=['Dataset', 'Predictions on :', 'MAE', 'RMSE', 'MAPE', 'R2'])
extraTrees_Regressor_results = pd.DataFrame(columns=['Dataset', 'Predictions on :', 'MAE', 'RMSE', 'MAPE', 'R2'])
adaBoost_Regressor_results = pd.DataFrame(columns=['Dataset', 'Predictions on :', 'MAE', 'RMSE', 'MAPE', 'R2'])
decisionTree_Regressor_results = pd.DataFrame(columns=['Dataset', 'Predictions on :', 'MAE', 'RMSE', 'MAPE', 'R2'])
xGB_Regressor_results = pd.DataFrame(columns=['Dataset', 'Predictions on :', 'MAE', 'RMSE', 'MAPE', 'R2'])
xGBRF_Regressor_results = pd.DataFrame(columns=['Dataset', 'Predictions on :', 'MAE', 'RMSE', 'MAPE', 'R2'])
catBoost_Regressor_results = pd.DataFrame(columns=['Dataset', 'Predictions on :', 'MAE', 'RMSE', 'MAPE', 'R2'])

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

Nordex_data_h = pd.read_csv('datasets/Nordex_N117_preprocessed_history_data.csv', index_col=0)
Vestas_data_h = pd.read_csv('datasets/Vestas_V52_preprocessed_history_data.csv', index_col=0)

Nordex_data_f = pd.read_csv('datasets/Nordex_N117_preprocessed_forecast_data.csv', index_col=0)
Vestas_data_f = pd.read_csv('datasets/Vestas_V52_preprocessed_forecast_data.csv', index_col=0)

X_h_V = Vestas_data_h[cols]

X_f_V = Vestas_data_f[cols]

X_h_N = Nordex_data_h[cols]

X_f_N = Nordex_data_f[cols]


y_V = Vestas_data_f['ActivePower']

y_N = Nordex_data_f['ActivePower']



#scale datasets


X_train_h_V, X_test_h_V, y_train_h_V, y_test_h_V = train_test_split(X_h_V, y_V, test_size = 0.20, random_state = 42)

X_train_h_V, X_test_h_V, y_train_h_V, y_test_h_V, Scaler_h_V = scale_base(X_train_h_V, X_test_h_V, y_train_h_V, y_test_h_V)


X_train_f_V, X_test_f_V, y_train_f_V, y_test_f_V = train_test_split(X_f_V, y_V, test_size = 0.20, random_state = 42)

X_train_f_V, X_test_f_V, y_train_f_V, y_test_f_V, Scaler_f_V = scale_base(X_train_f_V, X_test_f_V, y_train_f_V, y_test_f_V)


X_train_h_N, X_test_h_N, y_train_h_N, y_test_h_N = train_test_split(X_h_N, y_N, test_size = 0.20, random_state = 42)

X_train_h_N, X_test_h_N, y_train_h_N, y_test_h_N, Scaler_h_N = scale_base(X_train_h_N, X_test_h_N, y_train_h_N, y_test_h_N)


X_train_f_N, X_test_f_N, y_train_f_N, y_test_f_N = train_test_split(X_f_N, y_N, test_size = 0.20, random_state = 42)

X_train_f_N, X_test_f_N, y_train_f_N, y_test_f_N, Scaler_f_N = scale_base(X_train_f_N, X_test_f_N, y_train_f_N, y_test_f_N)




#loop through each model and train it 4 times in each dataset and predict to the training historic values and the testing forecasting values


for i in range(0,9):
    if i>0:
        break
    print(i)

    results_dataframe = results_df(i)

    x = np.arange(0,72)

    model, model_name = model_selection(i)

    model.fit(X_train_h_V, y_train_h_V)
    pred_test = model.predict(X_test_f_V)     #prediction for the forecasted values

    preds_test = (Scaler_f_V.inverse_transform(pred_test.reshape(-1, 1))*1000).reshape(-1)
    actuals_test = (Scaler_f_V.inverse_transform(y_test_f_V.reshape(-1, 1))*1000).reshape(-1)
    error = preds_test - actuals_test


    np.save('predictions/Regressor Models/'+model_name+'_h_V_predictions.npy', preds_test)

    rmse_test = np.sqrt(mean_squared_error(actuals_test, preds_test))
    mae_test = mean_absolute_error(actuals_test, preds_test)
    mape_test = mean_absolute_percentage_error(actuals_test, preds_test)
    r2_score_test = r2_score(actuals_test, preds_test)*100


    plt.figure(figsize=(20, 10))
    plt.plot(x, actuals_test[:72], label='Actual values', color='blue', alpha=0.7)
    plt.plot(x, preds_test[:72], label='Predictions', color='red', linestyle='--')
    plt.fill_between(x, 0, error[:72], color='green', alpha=0.3, label='Error')

    plt.title(model_name+'_historical_Vestas')
    plt.xlabel('Time-step')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Regressor Models/'+model_name+'_h_V_plot.png', dpi=100)

    results_dataframe.loc[0] = ['Vestas_data_history', 'Test_forecast', mae_test, rmse_test, mape_test, r2_score_test]



    model, model_name = model_selection(i)

    model.fit(X_train_f_V, y_train_f_V)

    pred_test = model.predict(X_test_f_V)

    preds_test = (Scaler_f_V.inverse_transform(pred_test.reshape(-1, 1))*1000).reshape(-1)
    actuals_test = (Scaler_f_V.inverse_transform(y_test_f_V.reshape(-1, 1))*1000).reshape(-1)
    error = preds_test - actuals_test

    np.save('predictions/Regressor Models/'+model_name+'_f_V_predictions.npy', preds_test)

    rmse_test = np.sqrt(mean_squared_error(actuals_test, preds_test))
    mae_test = mean_absolute_error(actuals_test, preds_test)
    mape_test = mean_absolute_percentage_error(actuals_test, preds_test)
    r2_score_test = r2_score(actuals_test, preds_test)*100


    plt.figure(figsize=(20, 10))
    plt.plot(x, actuals_test[:72], label='Actual values', color='blue', alpha=0.7)
    plt.plot(x, preds_test[:72], label='Predictions', color='red', linestyle='--')
    plt.fill_between(x, 0, error[:72], color='green', alpha=0.3, label='Error')

    plt.title(model_name+'_forecasted_Vestas')
    plt.xlabel('Time-step')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Regressor Models/'+model_name+'_f_V_plot.png', dpi=100)

    results_dataframe.loc[1] = ['Vestas_data_forecast', 'Test_forecast', mae_test, rmse_test, mape_test, r2_score_test]



    model, model_name = model_selection(i)

    model.fit(X_train_h_N, y_train_h_N)

    pred_test = model.predict(X_test_f_N)    #prediction for the forecasted values

    preds_test = (Scaler_f_N.inverse_transform(pred_test.reshape(-1, 1))*1000).reshape(-1)
    actuals_test = (Scaler_f_N.inverse_transform(y_test_f_N.reshape(-1, 1))*1000).reshape(-1)
    error = preds_test - actuals_test

    np.save('predictions/Regressor Models/'+model_name+'_h_N_predictions.npy', preds_test)

    rmse_test = np.sqrt(mean_squared_error(actuals_test, preds_test))
    mae_test = mean_absolute_error(actuals_test, preds_test)
    mape_test = mean_absolute_percentage_error(actuals_test, preds_test)
    r2_score_test = r2_score(actuals_test, preds_test)*100


    plt.figure(figsize=(20, 10))
    plt.plot(x, actuals_test[:72], label='Actual values', color='blue', alpha=0.7)
    plt.plot(x, preds_test[:72], label='Predictions', color='red', linestyle='--')
    plt.fill_between(x, 0, error[:72], color='green', alpha=0.3, label='Error')

    plt.title(model_name+'_historical_Nordex')
    plt.xlabel('Time-step')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Regressor Models/'+model_name+'_h_N_plot.png', dpi=100)

    results_dataframe.loc[2] = ['Nordex_data_history', 'Train on_forecast', mae_test, rmse_test, mape_test, r2_score_test]



    model, model_name = model_selection(i)

    model.fit(X_train_f_N, y_train_f_N)

    pred_test = model.predict(X_test_f_N)

    preds_test = (Scaler_f_N.inverse_transform(pred_test.reshape(-1, 1))*1000).reshape(-1)
    actuals_test = (Scaler_f_N.inverse_transform(y_test_f_N.reshape(-1, 1))*1000).reshape(-1)
    error = preds_test - actuals_test

    np.save('predictions/Regressor Models/'+model_name+'_f_N_predictions.npy', preds_test)

    rmse_test = np.sqrt(mean_squared_error(actuals_test, preds_test))
    mae_test = mean_absolute_error(actuals_test, preds_test)
    mape_test = mean_absolute_percentage_error(actuals_test, preds_test)
    r2_score_test = r2_score(actuals_test, preds_test)*100


    plt.figure(figsize=(20, 10))
    plt.plot(x, actuals_test[:72], label='Actual values', color='blue', alpha=0.7)
    plt.plot(x, preds_test[:72], label='Predictions', color='red', linestyle='--')
    plt.fill_between(x, 0, error[:72], color='green', alpha=0.3, label='Error')

    plt.title(model_name+'_forecasted_Nordex')
    plt.xlabel('Time-step')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Regressor Models/'+model_name+'_f_N_plot.png', dpi=100)

    results_dataframe.loc[3] = ['Nordex_data_forecast', 'Test_forecast', mae_test, rmse_test, mape_test, r2_score_test]

for i in range(9):
    if i == 0:
        print("random_forest_regressor_results")
    elif i == 1:
        print("catBoost_Regressor_results")
    elif i == 2:
        print("gradientBoosting_Regressor_results")
    elif i == 3:
        print("linear_Regression_results")
    elif i == 4:
        print("extraTrees_Regressor_results")
    elif i == 5:
        print("adaBoost_Regressor_results")
    elif i == 6:
        print("decisionTree_Regressor_results")
    elif i == 7:
        print("xGB_Regressor_results")
    elif i == 8:
        print("xGBRF_Regressor_results")
    print()

    results_dataframe = results_df(i)

    print(results_dataframe)

    print()

#save each regressor model results

random_forest_regressor_results.to_csv('results/random_forest_regressor_results.csv')


catBoost_Regressor_results.to_csv('results/catBoost_Regressor_results.csv')
gradientBoosting_Regressor_results.to_csv('results/gradientBoosting_Regressor_results.csv')
linear_Regression_results.to_csv('results/linear_Regression_results.csv')
extraTrees_Regressor_results.to_csv('results/extraTrees_Regressor_results.csv')
adaBoost_Regressor_results.to_csv('results/adaBoost_Regressor_results.csv')
decisionTree_Regressor_results.to_csv('results/decisionTree_Regressor_results.csv')
xGB_Regressor_results.to_csv('results/xGB_Regressor_results.csv')
xGBRF_Regressor_results.to_csv('results/xGBRF_Regressor_results.csv')