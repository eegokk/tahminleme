# TFT (Temporal Fusion Transformer) 

import cx_Oracle
import pandas as pd
import torch
import pytorch_lightning as pl
#from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from pytorch_forecasting.models import TemporalFusionTransformer


# Bağlantı bilgileri
username = 'ECINAR'  # Veritabanı kullanıcı adınız
password = '123'  # Veritabanı şifreniz
dsn = '127.0.0.1:1521/orcl'  # Veritabanı bağlantı adresi (localhost, port ve service name)

try:
    # Oracle veritabanına bağlantı
    connection = cx_Oracle.connect(username, password, dsn)
    print("Bağlantı başarılı ✅")

    # Bağlantıyı kontrol etmek için bir sorgu çalıştıralım
    cursor = connection.cursor()
    query = "SELECT * FROM ECINAR.YK_GGD_SAYI "
    cursor.execute(query)

    # Sütun adlarını al
    columns = [col[0] for col in cursor.description]

    # Verileri al
    data = cursor.fetchall()

    # DataFrame'e dönüştür
    df = pd.DataFrame(data, columns=columns)

    # Sonuçları yazdır
    #print("Veriler DataFrame olarak alındı:")
    #print(df)
    #print(df.head())  # İlk 5 satır
    print(df.columns)         # Sütun isimlerini göster
    print(df.iloc[:, :2])  

    # Bağlantıyı kapat
    cursor.close()
    connection.close()

except cx_Oracle.DatabaseError as e:
    print("Veritabanı bağlantı hatası:", e)
    
#dataframe oluştur    
df = pd.DataFrame(data, columns=columns)    
df['tarih'] = pd.to_datetime(df['TARIH'])
df.set_index('TARIH', inplace=True)
df.rename(columns={'SAYI': 'geri_donus_sayisi'}, inplace=True)
df = df.sort_values('tarih').reset_index(drop=True)


# TFT modeli, her gözlem için time_idx, group_id, target, ve tarihsel bilgiler gerektirir:
# Gerekli sütunları oluştur
df['time_idx'] = (df['tarih'] - df['tarih'].min()).dt.days
df['group_id'] = "ggd"  # çünkü tek bir zaman serimiz var

# train/test ayır
max_time_idx = df['time_idx'].max()
training_cutoff = max_time_idx - 30  # son 30 günü test seti olarak bırak

# TimeSeriesDataset nesnelerini oluştur
training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="geri_donus_sayisi",
    group_ids=["group_id"],
    max_encoder_length=60,
    max_prediction_length=30,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["geri_donus_sayisi"],
    target_normalizer=GroupNormalizer(groups=["group_id"]),
)


validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=32, num_workers=0)



# Eğitim geri çağrımları
early_stop = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, mode="min")
checkpoint = ModelCheckpoint(
    dirpath="tft_checkpoints",
    filename="best-checkpoint",
    monitor="val_loss",
    save_top_k=1,
    mode="min"
)

# Eğitim ayarları
pl.seed_everything(42)
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="auto",
    devices="auto",
    gradient_clip_val=0.1,
    callbacks=[early_stop, checkpoint],
    log_every_n_steps=10,
    enable_checkpointing=True,
)

# Modeli oluştur
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=SMAPE(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# Modeli eğit
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# En iyi modeli yükle
best_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)

# Tahmin ve görselleştirme
raw_predictions, x = best_model.predict(val_dataloader, mode="raw", return_x=True)
#best_model.plot_prediction(x, raw_predictions, idx=0)
#plt.show()

# Tahmin edilen ve gerçek değerleri alalım (1. örnek için)
decoder_actuals = x["decoder_target"][0].detach().cpu().numpy()
decoder_preds = raw_predictions["prediction"][0].detach().cpu().numpy()

plt.figure(figsize=(12, 5))
plt.plot(decoder_actuals, label="Gerçek (decoder_target)", marker='o')
plt.plot(decoder_preds, label="Tahmin (prediction)", marker='x')
plt.title("Temporal Fusion Transformer - Tahmin vs. Gerçek (Decoder Dönemi)")
plt.xlabel("Zaman Adımı (Decoder)")
plt.ylabel("Geri Dönüş Sayısı")
plt.grid(True)
plt.legend()
plt.show()


# Geçmiş (encoder_input) ve gelecek (decoder tahmini) birleştirme
encoder_length = x["encoder_lengths"][0]
encoder_input = x["encoder_target"][0][:encoder_length].detach().cpu().numpy()
decoder_actual = x["decoder_target"][0].detach().cpu().numpy()
decoder_pred = raw_predictions["prediction"][0].detach().cpu().numpy()

full_actual = np.concatenate([encoder_input, decoder_actual])
full_pred = np.concatenate([encoder_input, decoder_pred])  # Geçmişteki tahmin = gerçek

plt.figure(figsize=(14, 5))
plt.plot(full_actual, label="Gerçek (Encoder + Decoder)", marker='o')
plt.plot(full_pred, label="Tahmin (Decoder Sonrası)", marker='x')
plt.axvline(x=encoder_length - 1, color='gray', linestyle='--', label='Encoder Bitişi')
plt.title("Gerçek ve Tahmin Değerleri - Tüm Zaman Aralığı")
plt.xlabel("Zaman Adımı")
plt.ylabel("Geri Dönüş Sayısı")
plt.legend()
plt.grid(True)
plt.show()

import pytorch_lightning as pl




