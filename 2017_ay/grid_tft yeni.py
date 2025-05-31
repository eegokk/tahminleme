import os
import warnings
import cx_Oracle
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, SMAPE
from pytorch_forecasting.data import GroupNormalizer
import random
import gc

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Veritabanı bağlantısı
username = 'ECINAR'
password = '123'
dsn = '127.0.0.1:1521/orcl'

try:
    connection = cx_Oracle.connect(username, password, dsn)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM ECINAR.YK_GGD_AYLIK")
    columns = [col[0] for col in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)
    cursor.close()
    connection.close()
    print("✅ Veri tabanı bağlantısı başarılı.")
except cx_Oracle.DatabaseError as e:
    print("❌ Bağlantı hatası:", e)

# Veri ön işleme
df.columns = [col.upper() for col in df.columns]
df["TARIH"] = pd.to_datetime(df["TARIH"])
df = df.sort_values("TARIH")
df["time_idx"] = (df["TARIH"] - df["TARIH"].min()).dt.days // 30
df["AY"] = 0


# Grid Search Parametreleri
hidden_sizes = [16]
dropouts = [0.1,0.2]
epoch_list = [2]
batch_size = [16,32]

results = []

for hidden_size in hidden_sizes:
    for dropout in dropouts:
        for ep in epoch_list:
            for bs in batch_size:
                print(dropout)
                print(f"\n➡️ Grid Search: hidden_size={hidden_size}, dropout={dropout}, ep={ep}, bs={batch_size}")

                max_encoder_length = 12
                max_prediction_length = 12
                training_cutoff = df["time_idx"].max() - max_prediction_length
                print('time_idx', df.time_idx)
                print('DF', len(df))
                print('training_cutoff', training_cutoff)
                
                training = TimeSeriesDataSet(
                    df[df.time_idx <= training_cutoff],
                    time_idx="time_idx",
                    target="SAYI",
                    group_ids=["AY"],
                    max_encoder_length=max_encoder_length,
                    max_prediction_length=max_prediction_length,
                    time_varying_known_reals=["time_idx"],
                    time_varying_unknown_reals=["SAYI"],
                    target_normalizer=GroupNormalizer(groups=["AY"]),
                    allow_missing_timesteps=True)

                validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
                train_loader = training.to_dataloader(train=True, batch_size=bs, num_workers=0)
                val_loader = validation.to_dataloader(train=False, batch_size=bs, num_workers=0)
                
                print('val_loader', len(val_loader))
                
                tft = TemporalFusionTransformer.from_dataset(
                    training,
                    learning_rate=0.03,
                    hidden_size=hidden_size,
                    attention_head_size=1,
                    dropout=dropout,
                    loss=SMAPE(),
                    log_interval=1,
                    reduce_on_plateau_patience=4
                )

                early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
                lr_logger = LearningRateMonitor()
                trainer = pl.Trainer(
                    max_epochs=ep,
                    gpus=1 if torch.cuda.is_available() else 0,
                    gradient_clip_val=0.1,
                    callbacks=[early_stop, lr_logger],
                    logger=TensorBoardLogger("lightning_logs")
                )

                trainer.fit(
                    model=tft,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader
                )
               
                val_predictions, x_val = tft.predict(val_loader, mode="prediction", return_x=True)
                val_predicted = val_predictions[0].detach().cpu().numpy().squeeze()
                val_true = x_val["decoder_target"][0].detach().cpu().numpy().squeeze()

                mae = mean_absolute_error(val_true, val_predicted)
                rmse = mean_squared_error(val_true, val_predicted, squared=False)
                mape = mean_absolute_percentage_error(val_true, val_predicted)
                mase = mae / np.mean(np.abs(np.diff(val_true)))

                results.append({
                    "hidden_size": hidden_size,
                    "dropout": dropout,
                    "ep": ep,
                    "batch_size": bs,
                    #"MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape * 100,
                    "MASE": mase
                })
                del tft
                gc.collect()
                torch.cuda.empty_cache()
            
# Sonuçları yazdır
results_df = pd.DataFrame(results)
print("\n📊 Grid Search Sonuçları:")
print(results_df.sort_values("MASE"))
