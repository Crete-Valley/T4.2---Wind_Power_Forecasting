import pandas as pd
import numpy as np
import random
import json
import optuna
import gc
import warnings
import os


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


torch.set_float32_matmul_precision("medium")

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



def create_Time_Dataset(train):

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
        time_varying_known_reals=['TheoreticalPower',
                                  'WindSpeed',
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
        categorical_encoders={
            "Month": NaNLabelEncoder(add_nan=True),
            "Season_Autumn": NaNLabelEncoder(add_nan=True),
            "Season_Spring": NaNLabelEncoder(add_nan=True),
            "Season_Summer": NaNLabelEncoder(add_nan=True),
            "Season_Winter": NaNLabelEncoder(add_nan=True)
        },
        scalers = None
    )

    # Define the source validation and testing dataset
    validation = TimeSeriesDataSet.from_dataset(training, train, predict=True, stop_randomization=True)

    return training, validation


def objective_Vestas(trial):

    pl.seed_everything(42)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=8, verbose=False, mode="min")
    logger = TensorBoardLogger("lightning_logs", name=f"trial_{trial.number}", log_graph=False)
    min_val_loss_callback = MinValidationLossCallback()

    # Define other hyperparameters
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    hidden_continuous_size = trial.suggest_categorical('hidden_continuous_size', [16, 32, 64])
    attention_head_size = trial.suggest_int('attention_head_size', 4, 8)
    learning_rate = trial.suggest_float('learning_rate', 0.00001, 0.001)
    dropout = trial.suggest_float('dropout', 0.3, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [256, 512])
    lstm_layers = trial.suggest_categorical('lstm_layers', [2, 4])


    training, validation = create_Time_Dataset(train)


    trainer = pl.Trainer(
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs = 40,
        enable_model_summary=True,
        gradient_clip_val = 0.3,
        callbacks = [early_stop_callback, min_val_loss_callback],
        logger = logger,
        enable_checkpointing = False,
        log_every_n_steps=10,  # Log every step
        num_sanity_val_steps=0  # To avoid running validation before training starts
    )


    # Define the model
    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=hidden_size,
        hidden_continuous_size=hidden_continuous_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        learning_rate=learning_rate,
        lstm_layers=lstm_layers,
        loss = QuantileLoss(),
        optimizer="Adam",
        log_interval=10,
        log_val_interval = 10,
        reduce_on_plateau_patience=4
    )

    # Define the data loaders
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=3, pin_memory=True, prefetch_factor=5, persistent_workers=True, worker_init_fn=worker_init_fn)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=3, pin_memory=True, prefetch_factor=5, persistent_workers=True, worker_init_fn=worker_init_fn)
 
    print(f"Starting trial {trial.number} with params: {trial.params}")

    # Optimize the model using the trainer
    trainer.fit(model, train_dataloader, val_dataloader)

    # Compute minimum validation loss
    val_loss = min_val_loss_callback.best_val_loss

    del model, train_dataloader, val_dataloader
    torch.cuda.empty_cache()
    gc.collect()

    return val_loss



def objective_Nordex(trial):

    pl.seed_everything(42)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=8, verbose=False, mode="min")
    logger = TensorBoardLogger("lightning_logs", name=f"trial_{trial.number}", log_graph=False)
    min_val_loss_callback = MinValidationLossCallback()


    # Define other hyperparameters
    hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64])
    hidden_continuous_size = trial.suggest_categorical('hidden_continuous_size', [8, 16, 32])
    attention_head_size = trial.suggest_int('attention_head_size', 2, 6)
    learning_rate = trial.suggest_float('learning_rate', 0.00001, 0.001)
    dropout = trial.suggest_float('dropout', 0.3, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [128, 256])
    lstm_layers = trial.suggest_categorical('lstm_layers', [2, 4])

    
    training, validation = create_Time_Dataset(train)


    trainer = pl.Trainer(
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs = 40,
        enable_model_summary=True,
        gradient_clip_val = 0.3,
        callbacks = [early_stop_callback, min_val_loss_callback],
        logger = logger,
        enable_checkpointing = False,
        log_every_n_steps=10,  # Log every step
        num_sanity_val_steps=0  # To avoid running validation before training starts
    )


    # Define the model
    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=hidden_size,
        hidden_continuous_size=hidden_continuous_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        learning_rate=learning_rate,
        lstm_layers=lstm_layers,
        loss = QuantileLoss(),
        optimizer="Ranger",
        log_interval=10,
        log_val_interval = 10,
        reduce_on_plateau_patience=4
    )

    # Define the data loaders
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=3, prefetch_factor=5, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=3, prefetch_factor=5, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)

    print(f"Starting trial {trial.number} with params: {trial.params}")

    # Optimize the model using the trainer
    trainer.fit(model, train_dataloader, val_dataloader)

    # Compute minimum validation loss
    val_loss = min_val_loss_callback.best_val_loss

    del model, train_dataloader, val_dataloader
    torch.cuda.empty_cache()
    gc.collect()

    return val_loss



def best_param(data, filename):
    study = optuna.create_study(direction='minimize')
    # Start the optimization
    if data == 'Vestas':
        study.optimize(objective_Vestas, n_trials=100, gc_after_trial=True)
    elif data == 'Nordex':
        study.optimize(objective_Nordex, n_trials=100, gc_after_trial=True)

    # Print the best parameters
    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    with open(filename, 'w') as f:
        json.dump(trial.params, f)


#save best hyperparameters for each base TFT model (source, target, target_2)

train = pd.read_pickle('TFT input/tft_history-forecast_V_train.pkl')

best_param('Vestas', 'Hyperparameters/TFT/params_h_V.json')



train = pd.read_pickle('TFT input/tft_forecast-forecast_V_train.pkl')

best_param('Vestas', 'Hyperparameters/TFT/params_f_V.json')



train = pd.read_pickle('TFT input/tft_history-forecast_N_train.pkl')

best_param('Nordex', 'Hyperparameters/TFT/params_h_N.json')



train = pd.read_pickle('TFT input/tft_forecast-forecast_N_train.pkl')

best_param('Nordex', 'Hyperparameters/TFT/params_f_N.json')
