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
max_encoder_length = 24
max_prediction_length = 6
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
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=SMAPE(),
    log_interval=1,
    reduce_on_plateau_patience=4,)


# EĞİTİM
early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=30,
    gpus=1 if torch.cuda.is_available() else 0,
    gradient_clip_val=0.1,
    callbacks=[early_stop, lr_logger],
    logger=TensorBoardLogger("lightning_logs"))

# Eğitimi başlat
trainer.fit(
    model=tft,  # tft bir TemporalFusionTransformer örneği
    train_dataloaders=train_loader,
    val_dataloaders=val_loader)



# TAHMİN VE GRAFİK
predictions, x = tft.predict(val_loader, mode="prediction", return_x=True)

# Gerçek ve tahmin edilen değerleri çıkar (ilk örnek için idx=0)
predicted = predictions[0].detach().cpu().numpy().squeeze()
true = x["decoder_target"][0].detach().cpu().numpy().squeeze()
decoder_time_idx = x["decoder_time_idx"][0].detach().cpu().numpy()

# decoder time_idx'e karşılık gelen gerçek tarihleri df üzerinden çek
time_map = df.drop_duplicates("time_idx").set_index("time_idx")["TARIH"]
dates = [time_map.get(i) for i in decoder_time_idx]

# Kontrol
print("Tahmin shape:", predicted.shape)
print("Tarih sayısı :", len(dates))

# Grafik
plt.figure(figsize=(12, 6))
plt.plot(dates, true, label='Gerçek Veri', linewidth=2)
plt.plot(dates, predicted, label='TFT Tahminleri', linestyle='--', marker='o')
plt.title('TFT ile Aylık Gönüllü Geri Dönüş Tahmini')
plt.xlabel("Tarih")
plt.ylabel("Geri Dönüş Sayısı")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Tahmin edilen değerler:", predicted.astype(int))
print("Gerçek değerler:", true)