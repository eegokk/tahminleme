import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import random
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

username = 'ECINAR' 
password = '123'    
dsn = '127.0.0.1:1521/orcl'  

# Veriyi al
connection = cx_Oracle.connect(username, password, dsn)
cursor = connection.cursor()
cursor.execute("SELECT * FROM ECINAR.YK_GGD_AYLIK")
columns = [col[0] for col in cursor.description]
data = cursor.fetchall()
df = pd.DataFrame(data, columns=columns)
cursor.close()
connection.close()

# Tarih ve sıralama
df['TARIH'] = pd.to_datetime(df['TARIH'], errors='coerce')
df.dropna(subset=['TARIH'], inplace=True)
df = df.sort_values('TARIH')
df.set_index('TARIH', inplace=True)
df.rename(columns={'SAYI': 'geri_donus_sayisi'}, inplace=True)

# Özellik mühendisliği
future_event_date = pd.to_datetime("2025-12-01")
df['esad_devrildi'] = (df.index >= future_event_date).astype(int)
df['ay'] = df.index.month
df['mevsim'] = ((df.index.month - 1) // 3 + 1)

# Özellik vektörü
features = df[['geri_donus_sayisi', 'esad_devrildi', 'ay', 'mevsim']].values
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_target = scaled_features[:, 0].reshape(-1, 1)
scaled_series = scaled_features
series = df['geri_donus_sayisi']

# Sequence oluşturma
def create_multivariate_sequences(data, look_back=35, target_column_index=0):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back, target_column_index])
    return np.array(X), np.array(y)

look_back = 35
X, y = create_multivariate_sequences(scaled_series, look_back)

# Train-validation bölme
split_index = int(len(X) * 0.99)
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Model
model_lstm = Sequential()
model_lstm.add(LSTM(64, return_sequences=True, input_shape=(look_back, scaled_series.shape[1])))
model_lstm.add(Dropout(0.1))
model_lstm.add(LSTM(32))
model_lstm.add(Dropout(0.1))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model_lstm.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=150, verbose=1, callbacks=[early_stop])

# Geçmiş tahmin
train_predictions = model_lstm.predict(X, verbose=0)
y_true = scaled_target[look_back:]
train_predictions = scaler.inverse_transform(np.hstack([train_predictions, np.zeros((len(train_predictions), scaled_series.shape[1]-1))]))[:, 0]
y_true_inv = scaler.inverse_transform(np.hstack([y_true, np.zeros((len(y_true), scaled_series.shape[1]-1))]))[:, 0]

# Gelecek tahmin
last_sequence = scaled_series[-look_back:]
future_predictions = []
forecast_dates = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=10, freq='MS')

for i in range(10):
    input_seq = last_sequence.reshape((1, look_back, scaled_series.shape[1]))
    pred = model_lstm.predict(input_seq, verbose=0)
    future_predictions.append(pred[0][0])

    esad_flag = 1 if forecast_dates[i] >= future_event_date else 0
    next_month = forecast_dates[i].month
    next_season = (next_month - 1) // 3 + 1

    new_step = np.array([[pred[0][0], esad_flag, next_month, next_season]])
    new_step_scaled = scaler.transform(new_step)
    last_sequence = np.vstack([last_sequence[1:], new_step_scaled])

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions_inv = scaler.inverse_transform(np.hstack([future_predictions, np.zeros((10, scaled_series.shape[1]-1))]))[:, 0]

# DataFrame
train_pred_dates = series.index[look_back : look_back + len(train_predictions)]
in_sample_df = pd.DataFrame({'Tarih': train_pred_dates, 'Gerçek': y_true_inv, 'Tahmin': train_predictions})

# Hata metrikleri
mae = mean_absolute_error(y_true_inv, train_predictions)
rmse = np.sqrt(mean_squared_error(y_true_inv, train_predictions))
mape = np.mean(np.abs((y_true_inv - train_predictions) / y_true_inv)) * 100
mase = mae / np.mean(np.abs(np.diff(y_true_inv)))

print("\nGelecek Tahminleri:")
for date, pred in zip(forecast_dates, future_predictions_inv):
    print(f"{date.strftime('%Y-%m')} → {int(round(pred))}")

# Yazılı çıktı
print("\nHATA METRİKLERİ:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"MASE : {mase:.2f}")



# Grafik
plt.figure(figsize=(12, 6))
plt.plot(series, label='Gerçek Veri', linewidth=2)
plt.plot(in_sample_df['Tarih'], in_sample_df['Tahmin'], label='Geçmiş Tahmin (LSTM)', linestyle='--')
plt.plot(forecast_dates, future_predictions_inv, label='Gelecek Tahmin (LSTM)', color='red', linewidth=2)
plt.title('LSTM ile Geçmiş ve Gelecek Tahmin')
plt.xlabel("Tarih")
plt.ylabel("geri_donus_sayisi")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()