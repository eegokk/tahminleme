import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import random
import tensorflow as tf
import tensorflow.keras.backend as K

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Oracle bağlantısı
username = 'ECINAR' 
password = '123'    
dsn = '127.0.0.1:1521/orcl'  

try:
    connection = cx_Oracle.connect(username, password, dsn)
    print("Bağlantı başarılı ✅")
    cursor = connection.cursor()
    query = "SELECT * FROM ECINAR.YK_GGD_AYLIK"
    cursor.execute(query)
    columns = [col[0] for col in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)
    cursor.close()
    connection.close()
except cx_Oracle.DatabaseError as e:
    print("Veritabanı bağlantı hatası:", e)

# Zaman serisi hazırlanması
df['TARIH'] = pd.to_datetime(df['TARIH'])  
df.set_index('TARIH', inplace=True)
series = df['SAYI'].sort_index()

# Normalizasyon
#scaler = MinMaxScaler()
scaler = StandardScaler()
series_scaled = scaler.fit_transform(series.values.reshape(-1, 1))

# Sequence oluşturma
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 18   #12  
X, y = create_sequences(series_scaled, look_back)
X = X.reshape((X.shape[0], look_back, 1))

# RNN modeli kurulumu
model_rnn = Sequential()
#model_rnn.add(SimpleRNN(50, activation='tanh', input_shape=(look_back, 1)))
model_rnn.add(SimpleRNN(64, activation='tanh', return_sequences=True, input_shape=(look_back, 1)))
#model_rnn.add(Dropout(0.1))
model_rnn.add(SimpleRNN(32, activation='tanh'))
#model_rnn.add(Dropout(0.2))
model_rnn.add(Dense(1))
#loss_fn = lambda y_true, y_pred: K.mean(tf.math.log(tf.math.cosh(y_pred - y_true)))
model_rnn.compile(optimizer='adam', loss='mse')
#model_rnn.compile(optimizer='adam', loss=loss_fn)
model_rnn.fit(X, y, epochs=200, verbose=1)
#model_rnn.fit(X, y, epochs=200, verbose=1, validation_split=0.05)

# Geçmiş tahmin (train) çıkarımı
rnn_train_pred_scaled = model_rnn.predict(X, verbose=0)
rnn_train_pred = scaler.inverse_transform(rnn_train_pred_scaled).flatten()
actual_past = series[look_back:]

# Gelecek tahminleri
n_forecast = 10
last_seq = series_scaled[-look_back:]
future_preds_scaled = []

for _ in range(n_forecast):
    input_seq = last_seq.reshape(1, look_back, 1)
    pred = model_rnn.predict(input_seq, verbose=0)[0][0]
    future_preds_scaled.append(pred)
    last_seq = np.append(last_seq[1:], [[pred]], axis=0)

rnn_forecast = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()
forecast_dates = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=n_forecast, freq='MS')

# Yazdırma
print("\n Gelecek Tahminleri (RNN):")
for date, value in zip(forecast_dates, rnn_forecast):
    print(f"{date.strftime('%Y-%m')} → {int(value)}")

# Hata metrikleri
mae = mean_absolute_error(actual_past, rnn_train_pred)
rmse = np.sqrt(mean_squared_error(actual_past, rnn_train_pred))
mape = np.mean(np.abs((actual_past - rnn_train_pred) / actual_past)) * 100
naive = actual_past.shift(1).dropna()
mase = mae / np.mean(np.abs(actual_past[1:] - naive))

print(f"\nHATA METRİKLERİ")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"MASE : {mase:.2f}")

# Grafikle gösterim
plt.figure(figsize=(12, 6))
plt.plot(series, label='Gerçek Veri', linewidth=2.5)
plt.plot(actual_past.index, rnn_train_pred, label='Geçmiş Tahmin (RNN)',  color='orange', linestyle='--')
plt.plot(forecast_dates, rnn_forecast, label='Gelecek Tahmin (RNN)', color='red', linewidth=2)
plt.title('RNN Tahmini (Geçmiş + Gelecek)')
plt.xlabel("Tarih")
plt.ylabel("Geri Dönüş Sayısı")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




print("\nGEÇMİŞ VERİ & TAHMİN KARŞILAŞTIRMASI:")
comparison_df = pd.DataFrame({
    'Gerçek': actual_past.values.astype(int),
    'Tahmin': rnn_train_pred.astype(int)
}, index=actual_past.index)

comparison_df['Hata'] = comparison_df['Gerçek'] - comparison_df['Tahmin']
comparison_df['Mutlak Hata'] = np.abs(comparison_df['Hata'])

# Sadece son 10 ayı göstermek istersen:
print(comparison_df.tail(15)) 
# print(comparison_df)   #tüm veri