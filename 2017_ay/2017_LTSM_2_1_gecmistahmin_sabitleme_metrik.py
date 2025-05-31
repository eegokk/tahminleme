import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import random
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Bağlantı bilgileri
username = 'ECINAR' 
password = '123'    
dsn = '127.0.0.1:1521/orcl'  

try:
    # Oracle veritabanına bağlantı
    connection = cx_Oracle.connect(username, password, dsn)
    print("Bağlantı başarılı ✅")

    cursor = connection.cursor()
    query = "SELECT * FROM ECINAR.YK_GGD_AYLIK"
    cursor.execute(query)

    columns = [col[0] for col in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)

    print(df.columns)         
    print(df.iloc[:, :2])  

    cursor.close()
    connection.close()

except cx_Oracle.DatabaseError as e:
    print("Veritabanı bağlantı hatası:", e)

# Zaman serisi verinizi hazırlayın
df['TARIH'] = pd.to_datetime(df['TARIH'])  
df.set_index('TARIH', inplace=True)     
series = df['SAYI'].sort_index()
df.rename(columns={'SAYI': 'geri_donus_sayisi'}, inplace=True)   

# Veriyi normalize et
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(series.values.reshape(-1, 1))


# LSTM için veri oluştur
def create_sequences(data, look_back=5):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)


look_back = 5
X, y = create_sequences(scaled_series, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# LSTM Modeli
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X, y, epochs=20, verbose=1)

# GEçmiş tahmini
train_predictions = model_lstm.predict(X, verbose=0)
train_predictions = scaler.inverse_transform(train_predictions)
y_true = scaler.inverse_transform(y.reshape(-1, 1))


# Gelecek 10 ay tahmini
last_sequence = scaled_series[-look_back:]
input_seq = last_sequence.reshape((1, look_back, 1))
future_predictions = []

for _ in range(10):
    pred = model_lstm.predict(input_seq, verbose=0)
    future_predictions.append(pred[0][0])
    last_sequence = np.append(last_sequence[1:], pred[0][0])
    input_seq = last_sequence.reshape((1, look_back, 1))
forecast_dates = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=10, freq='MS')

# Ölçeği geri al
arr = np.array(future_predictions).reshape(-1, 1)
arr = np.nan_to_num(arr)
future_predictions = scaler.inverse_transform(arr).flatten()

# Tahmin tarihlerini oluştur
train_pred_dates = series.index[look_back:]

# DataFrame olarak göster
in_sample_df = pd.DataFrame({
    'Tarih': train_pred_dates,
    'Gerçek': y_true.flatten(),
    'Tahmin': train_predictions.flatten()
})
in_sample_df.set_index('Tarih', inplace=True)

print("\nGeçmiş Tahminleri:")
print(in_sample_df.tail(10))  # Son 10 tahmin


# Yazdır
print("Gelecek Tahminleri (LTSM):")
for date, pred in zip(forecast_dates, future_predictions):
    print(f"{date.strftime('%Y-%m')} → {int(round(pred))}")



# Grafik
plt.figure(figsize=(12, 6))
plt.plot(series, label='Gerçek Veri', linewidth=2)
plt.plot(in_sample_df.index, in_sample_df['Tahmin'], label='Geçmiş Tahmin (LSTM)', linestyle='--')
plt.plot(forecast_dates, future_predictions, label='Gelecek Tahmin (LSTM)', color='red', linewidth=2)
plt.title('LSTM ile Geçmiş ve Gelecek Tahmin')
plt.xlabel("Tarih")
plt.ylabel("geri_donus_sayisi")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Hata Metrikleri
mae = mean_absolute_error(y_true, train_predictions)
rmse = np.sqrt(mean_squared_error(y_true, train_predictions))
mape = np.mean(np.abs((y_true - train_predictions) / y_true)) * 100
mase = mae / np.mean(np.abs(np.diff(y_true.flatten())))  # naive model: y[t] = y[t-1]

print("\nHATA METRİKLERİ :")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"MASE : {mase:.2f}")
