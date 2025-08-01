import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import gc
import warnings
import random
import tensorflow as tf
import lightning.pytorch as pl
import pytorch_forecasting as pf
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")
# Ref: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)]) # Notice here
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


class MinValidationLossCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float("inf")  # Initialize with a very high value

    def on_validation_end(self, trainer, pl_module):
        # Access the latest validation loss
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Worker {worker_id} initialized with seed {seed}")


def create_Time_Dataset(train, test):

    max_encoder_length = 30
    max_prediction_length = 24

    training_cutoff = len(train) - max_prediction_length


    # Define the source training dataset
    training = TimeSeriesDataSet(
        train[lambda x: x.time_index <= training_cutoff],
        time_idx="time_index",
        target="ActivePower",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=["Month",
                                         "Day",
                                         "Week",
                                         "Minute",
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
                                         'Direction_W'],
        time_varying_known_reals=['WindSpeed',
                                  'Temperature',
                                  'Dewpoint',
                                  'Rel_Humidity',
                                  'Pressure',
                                  'MeanSpeed',
                                  'WindDirectionSin',
                                  'WindDirectionCos',
                                  'MeanDirectionSin',
                                  'MeanDirectionCos'],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["ActivePower"],
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
        allow_missing_timesteps=False,
        categorical_encoders={"Month": NaNLabelEncoder(add_nan=True)},
        scalers = None
    )

    # Define the source validation and testing dataset
    validation = TimeSeriesDataSet.from_dataset(training, train, predict=True, stop_randomization=True)
    testing = TimeSeriesDataSet.from_dataset(training, test)


    return training, validation, testing

def TFT_model(dataset, params):

    model = TemporalFusionTransformer.from_dataset(
        dataset,
        hidden_size = params['hidden_size'],
        hidden_continuous_size = params['hidden_continuous_size'],
        attention_head_size = params['attention_head_size'],
        dropout = params['dropout'],
        learning_rate = params['learning_rate'],
        lstm_layers = params['lstm_layers'],
        loss = QuantileLoss(),
        optimizer = "Adam",
        log_interval = 10,
        log_val_interval = 10,
        reduce_on_plateau_patience = 4
    )

    return model



def fit_predict_TFT(train, test, i, params):

    training, validation, testing = create_Time_Dataset(train, test)

    batch_size = params['batch_size']

    if i==1:
        file_name = "TFT_h_V"
    elif i==2:
        file_name = "TFT_f_V"
    elif i==3:
        file_name = "TFT_h_N"
    elif i==4:
        file_name = "TFT_f_N"


    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=3, pin_memory=True, prefetch_factor=5, worker_init_fn=worker_init_fn, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=3, pin_memory=True, prefetch_factor=5, worker_init_fn=worker_init_fn, persistent_workers=True)
    test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=3, pin_memory=True, prefetch_factor=5, worker_init_fn=worker_init_fn, persistent_workers=True)



    pl.seed_everything(42)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=8, verbose=False, mode="min")
    logger = TensorBoardLogger("lightning_logs", log_graph=False)
    min_val_loss_callback = MinValidationLossCallback()

    model = TFT_model(training, params)

    trainer = pl.Trainer(
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs = 50,
        enable_model_summary=True,
        gradient_clip_val = 0.3,
        callbacks = [early_stop_callback, min_val_loss_callback],
        logger = logger,
        enable_checkpointing = False,
        log_every_n_steps=10,  # Log every step
        num_sanity_val_steps=0  # To avoid running validation before training starts
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    trainer.save_checkpoint("models/TFT/npi_"+file_name+".ckpt")

    model = TemporalFusionTransformer.load_from_checkpoint("models/TFT/npi_"+file_name+".ckpt")

    predictions_scaled_gpu_test = model.predict(test_dataloader)
    predictions_scaled_test = (predictions_scaled_gpu_test.cpu()).numpy()
    predictions_scaled_test = predictions_scaled_test[:,:1].ravel()
    predictions_scaled_test = predictions_scaled_test[1:]

    np.save('predictions/TFT/npi'+file_name+'_predictions.npy', predictions_scaled_test)

    del model, trainer, training, validation, testing, train_dataloader, val_dataloader, test_dataloader, predictions_scaled_gpu_test, predictions_scaled_test
    torch.cuda.empty_cache()
    gc.collect()
    

np.random.seed(42)
torch.manual_seed(42)


with open('Hyperparameters/TFT/params_h_V.json', 'r') as f:
    params_h_V = json.load(f)

with open('Hyperparameters/TFT/params_f_V.json', 'r') as f:
    params_f_V = json.load(f)

with open('Hyperparameters/TFT/params_h_N.json', 'r') as f:
    params_h_N = json.load(f)

with open('Hyperparameters/TFT/params_f_N.json', 'r') as f:
    params_f_N = json.load(f)



train_h_V = pd.read_pickle('TFT input/npi_tft_history-forecast_V_train.pkl')
test_h_V = pd.read_pickle('TFT input/npi_tft_history-forecast_V_test.pkl')

fit_predict_TFT(train_h_V, test_h_V, 1, params_h_V)


train_f_V = pd.read_pickle('TFT input/npi_tft_forecast-forecast_V_train.pkl')
test_f_V = pd.read_pickle('TFT input/npi_tft_forecast-forecast_V_test.pkl')

fit_predict_TFT(train_f_V, test_f_V, 2, params_f_V)

train_h_N = pd.read_pickle('TFT input/npi_tft_history-forecast_N_train.pkl')
test_h_N = pd.read_pickle('TFT input/npi_tft_history-forecast_N_test.pkl')


fit_predict_TFT(train_h_N, test_h_N, 3, params_h_N)

train_f_N = pd.read_pickle('TFT input/npi_tft_forecast-forecast_N_train.pkl')
test_f_N = pd.read_pickle('TFT input/npi_tft_forecast-forecast_N_test.pkl')

fit_predict_TFT(train_f_N, test_f_N, 4, params_h_N)


