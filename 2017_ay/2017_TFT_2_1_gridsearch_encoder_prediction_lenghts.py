import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cx_Oracle
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, SMAPE, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Oracle bağlantısı
username = 'ECINAR'
password = '123'
dsn = '127.0.0.1:1521/orcl'

connection = cx_Oracle.connect(username, password, dsn)
cursor = connection.cursor()
query = "SELECT * FROM ECINAR.YK_GGD_AYLIK"
cursor.execute(query)
columns = [col[0] for col in cursor.description]
data = cursor.fetchall()
df = pd.DataFrame(data, columns=columns)
cursor.close()
connection.close()

# Veri ön işleme
df.columns = [col.upper() for col in df.columns]
df["TARIH"] = pd.to_datetime(df["TARIH"])
df = df.sort_values("TARIH")
df["time_idx"] = (df["TARIH"] - df["TARIH"].min()).dt.days // 30
df["AY"] = 0

# Grid search parametreleri
encoder_lengths = [12, 24]
prediction_lengths = [6, 12]
loss_functions = [SMAPE(), QuantileLoss()]  # QuantileLoss() varsayılan q=0.5 ile medyan tahmini yapar

results = []

for encoder_len in encoder_lengths:
    for pred_len in prediction_lengths:
        for loss_fn in loss_functions:
            print(f"\n🔧 Testing config: encoder={encoder_len}, prediction={pred_len}, loss={loss_fn.__class__.__name__}")

            training_cutoff = df["time_idx"].max() - pred_len

            training = TimeSeriesDataSet(
                df[df.time_idx <= training_cutoff],
                time_idx="time_idx",
                target="SAYI",
                group_ids=["AY"],
                max_encoder_length=encoder_len,
                max_prediction_length=pred_len,
                time_varying_known_reals=["time_idx"],
                time_varying_unknown_reals=["SAYI"],
                target_normalizer=GroupNormalizer(groups=["AY"]),
                allow_missing_timesteps=True)

            validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
            train_loader = training.to_dataloader(train=True, batch_size=16, num_workers=0)
            val_loader = validation.to_dataloader(train=False, batch_size=16, num_workers=0)

            tft = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=0.03,
                hidden_size=16,
                attention_head_size=1,
                dropout=0.1,
                loss=loss_fn,
                log_interval=1,
                reduce_on_plateau_patience=4,)

            early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
            lr_logger = LearningRateMonitor()
            trainer = pl.Trainer(
                max_epochs=15,
                gpus=1 if torch.cuda.is_available() else 0,
                gradient_clip_val=0.1,
                callbacks=[early_stop, lr_logger],
                logger=TensorBoardLogger("lightning_logs", name=f"run_e{encoder_len}_p{pred_len}_{loss_fn.__class__.__name__}", default_hp_metric=False))

            trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

            val_predictions, x_val = tft.predict(val_loader, mode="prediction", return_x=True)
            val_predicted = val_predictions[0].detach().cpu().numpy().squeeze()
            val_true = x_val["decoder_target"][0].detach().cpu().numpy().squeeze()

            mae = mean_absolute_error(val_true, val_predicted)
            rmse = mean_squared_error(val_true, val_predicted, squared=False)
            mape = mean_absolute_percentage_error(val_true, val_predicted)
            mase = mae / np.mean(np.abs(np.diff(val_true)))

            results.append({
                "encoder_len": encoder_len,
                "prediction_len": pred_len,
                "loss": loss_fn.__class__.__name__,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape * 100,
                "MASE": mase
            })

# Sonuçları DataFrame olarak yazdır
results_df = pd.DataFrame(results)
print("\n📊 Grid Search Sonuçları:")
print(results_df.sort_values("RMSE"))
