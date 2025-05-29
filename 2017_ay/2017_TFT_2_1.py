import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cx_Oracle
import pandas as pd
import numpy as np
np.float = float
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, SMAPE
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random

SEED = 42  
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    #torch.cuda.manual_seed_all(SEED) #GPU  yüklenmediğinden kaldırıldı.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Bağlantı bilgileri
username = 'ECINAR'
password = '123'
dsn = '127.0.0.1:1521/orcl'

try:
    connection = cx_Oracle.connect(username, password, dsn)
    print("✅ Veritabanı bağlantısı başarılı.")
    cursor = connection.cursor()
    query = "SELECT * FROM ECINAR.YK_GGD_AYLIK"
    cursor.execute(query)
    columns = [col[0] for col in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)
    cursor.close()
    connection.close()
except cx_Oracle.DatabaseError as e:
    print("❌ Veritabanı bağlantı hatası:", e)

# VERİ HAZIRLAMA
df.columns = [col.upper() for col in df.columns]
df["TARIH"] = pd.to_datetime(df["TARIH"])
df = df.sort_values("TARIH")
df["time_idx"] = (df["TARIH"] - df["TARIH"].min()).dt.days // 30  # Aylık indeks
df["AY"] = 0  # Tek grup için sabit ID

# TFT MODEL VERİ SETİ
max_encoder_length = 12
max_prediction_length = 12
training_cutoff = df["time_idx"].max() - max_prediction_length

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
train_loader = training.to_dataloader(train=True, batch_size=16, num_workers=0)
val_loader = validation.to_dataloader(train=False, batch_size=16, num_workers=0)

# TFT MODELİNİ OLUŞTUR
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate= 0.04,    #0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=SMAPE(),
    log_interval=1,
    reduce_on_plateau_patience=4)

# EĞİTİM
early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=40,
    gpus=1 if torch.cuda.is_available() else 0,
    gradient_clip_val=0.1,
    callbacks=[early_stop, lr_logger],
    logger=TensorBoardLogger("lightning_logs"))

trainer.fit(
    model=tft,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader)

# GEÇMİŞ VERİ TAHMİNİ
val_predictions, x_val = tft.predict(val_loader, mode="prediction", return_x=True)
val_predicted = val_predictions[0].detach().cpu().numpy().squeeze()
val_true = x_val["decoder_target"][0].detach().cpu().numpy().squeeze()
val_time_idx = x_val["decoder_time_idx"][0].detach().cpu().numpy()
time_map = df.drop_duplicates("time_idx").set_index("time_idx")["TARIH"]
val_dates = [time_map.get(i) for i in val_time_idx]

# GELECEK TAHMİNİ
future_idx = np.arange(df["time_idx"].max() + 1, df["time_idx"].max() + 1 + max_prediction_length)
future_dates = pd.date_range(df["TARIH"].max() + pd.DateOffset(months=1), periods=max_prediction_length, freq="MS")
future_df = pd.DataFrame({
    "TARIH": future_dates,
    "time_idx": future_idx,
    "AY": 0,
    "SAYI": np.nan
})

df_extended = pd.concat([df, future_df], ignore_index=True)
df_extended["SAYI"] = df_extended["SAYI"].fillna(method="ffill")
full_dataset = TimeSeriesDataSet.from_dataset(training, df_extended, predict=True, stop_randomization=True)
full_loader = full_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
full_predictions, x_full = tft.predict(full_loader, mode="prediction", return_x=True)

full_predicted = full_predictions[0].detach().cpu().numpy().squeeze()
full_decoder_idx = x_full["decoder_time_idx"][0].detach().cpu().numpy()
time_map_full = df_extended.drop_duplicates("time_idx").set_index("time_idx")["TARIH"]
full_dates = [time_map_full.get(i) for i in full_decoder_idx]

val_threshold = df["TARIH"].max()
future_dates_final = [d for d in full_dates if d > val_threshold]
future_preds = full_predicted[-len(future_dates_final):]

# GRAFİK
plt.figure(figsize=(12, 6))
plt.plot(df["TARIH"], df["SAYI"], label='Gerçek Veri', color='blue', linewidth=2)
plt.plot(val_dates, val_predicted, label='Geçmiş Tahmin', linestyle='--', color='orange', linewidth=2)
plt.plot(future_dates_final, future_preds, label='TFT Gelecek Tahminleri', color='red', linewidth=2)
plt.title('TFT ile Aylık Gönüllü Geri Dönüş Tahmini')
plt.xlabel("Tarih")
plt.ylabel("Geri Dönüş Sayısı")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# HATA METRİKLERİ
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

print("\nGelecek 12 Aylık Tahmin:\n")
for tarih, tahmin in zip(future_dates_final, future_preds.astype(int)):
    print(f"{tarih.strftime('%Y-%m-%d')} -> Tahmin: {tahmin:>6}")

print("\n Gerçek Değerler vs Tahmin:\n")
for tarih, gercek, tahmin in zip(val_dates, val_true.astype(int), val_predicted.astype(int)):
    print(f"{tarih.strftime('%Y-%m-%d')} -> Gerçek:{gercek:>6},  Tahmin:{tahmin:>6}")


mae = mean_absolute_error(val_true, val_predicted)
rmse = mean_squared_error(val_true, val_predicted, squared=False)
mape = mean_absolute_percentage_error(val_true, val_predicted)
mase = mae / np.mean(np.abs(np.diff(val_true)))

print("\nHATA METRİKLERİ")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape * 100:.2f}%")
print(f"MASE : {mase:.2f}")

