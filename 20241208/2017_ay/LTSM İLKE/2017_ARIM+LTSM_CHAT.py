import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
tf.random.set_seed(42)

# Bağlantı bilgileri
username = 'ECINAR'  # Veritabanı kullanıcı adınız
password = '123'     # Veritabanı şifreniz
dsn = '127.0.0.1:1521/orcl'  # Veritabanı bağlantı adresi (localhost, port ve service name)

try:
    # Oracle veritabanına bağlantı
    connection = cx_Oracle.connect(username, password, dsn)
    print("Bağlantı başarılı ✅")

    # Bağlantıyı kontrol etmek için bir sorgu çalıştıralım
    cursor = connection.cursor()
    query = "SELECT * FROM ECINAR.YK_GGD_AYLIK"
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
    

    
# 1. Tarih formatı ve sıralama
df.columns = [col.lower() for col in df.columns]
df['tarih'] = pd.to_datetime(df['tarih'])
df.set_index('tarih', inplace=True)
df.sort_index(inplace=True)
series = df['sayi'].astype(float)

# 1. look_back değeri başta tanımlansın
look_back = 3

# 2. ARIMA modeli ve tahmini
arima_model = ARIMA(series, order=(1, 1, 1)).fit()
arima_pred = arima_model.predict(start=look_back + 1, end=len(series) - 1, typ='levels')

# 3. ARIMA residual'ları (hatayı) hesapla
residuals = series[look_back + 1:] - arima_pred

# 4. LSTM için residual'ları ölçekle
scaler = MinMaxScaler()
res_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

# 5. Sequence oluştur
X, y = [], []
for i in range(look_back, len(res_scaled)):
    X.append(res_scaled[i - look_back:i])
    y.append(res_scaled[i])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 6. LSTM modeli


model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, batch_size=8, verbose=0)

# 7. LSTM tahmini
lstm_pred = model.predict(X)
lstm_pred_inv = scaler.inverse_transform(lstm_pred)

# 8. Nihai tahmin
# ARIMA pred, LSTM residual pred ile toplanıyor
final_pred = arima_pred[-len(lstm_pred_inv):].values + lstm_pred_inv.flatten()

# 9. Gerçek değerleri aynı uzunlukta kes
actual = series[look_back + 1:].values[-len(final_pred):]

# 10. Hata metrikleri

mae = mean_absolute_error(actual, final_pred)
rmse = np.sqrt(mean_squared_error(actual, final_pred))
mape = mean_absolute_percentage_error(actual, final_pred) * 100
naive_forecast = actual[:-1]  # Bir önceki değeri tahmin olarak al
naive_actual = actual[1:]
mae_naive = mean_absolute_error(naive_actual, naive_forecast)
mase = mae / mae_naive

# 11. Grafik
plt.figure(figsize=(12, 6))
plt.plot(actual, label='Gerçek')
plt.plot(final_pred, label='ARIMA + LSTM Tahmin')
plt.legend()
plt.grid(True)
plt.title("ARIMA + LSTM Tahmini")
plt.tight_layout()
plt.show()

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"MASE : {mase:.2f}")

print("ARIMA pred:", len(arima_pred))
print("LSTM pred:", len(lstm_pred_inv))
print("Actual    :", len(actual))
