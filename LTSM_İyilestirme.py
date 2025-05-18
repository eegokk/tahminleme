import cx_Oracle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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
    print(df.head())  # İlk 5 satır
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
df = df.sort_index()
df["weekday"] = df.index.weekday
df["month"] = df.index.month  # Değerleri idealize etmek için eklendi
df["dayofweek"] = df.index.dayofweek  # Değerleri idealize etmek için eklendi
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)  # Değerleri idealize etmek için eklendi


# 3. VERİYİ ÖLÇEKLE
scaler = MinMaxScaler()
#scaled_data = scaler.fit_transform(df[['geri_donus_sayisi']])  # Değerleri idealize etmek için değiştirildi
scaled_data = scaler.fit_transform(df[['geri_donus_sayisi', 'month', 'dayofweek', 'is_weekend']])  # Değerleri idealize etmek için eklendi


# 4. SEQUENCE OLUŞTUR
X, y = [], []
window_size = 5  # ikinci kodla aynı pencere

for i in range(len(scaled_data) - window_size):
    X.append(scaled_data[i:i+window_size])
    y.append(scaled_data[i+window_size])

X, y = np.array(X), np.array(y)

print("X shape:", X.shape)

# 5. MODELİ KUR
model = Sequential()
#model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1))) # Değerleri idealize etmek için değiştirildi
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2]))) # Değerleri idealize etmek için eklendi
model.add(LSTM(32)) # Değerleri idealize etmek için eklendi
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# 6. MODELİ EĞİT
model.fit(X, y, epochs=50, batch_size=1, verbose=1)

# 7. TAHMİN YAP
predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)

# Gelecek tahminleri
future_steps = 30  # Kaç gün sonrasını tahmin etmek istiyorsan burada belirle
last_sequence = scaled_data[-window_size:]  # Son pencere verisi
future_predictions = []

current_sequence = last_sequence.copy()

for _ in range(future_steps):
    prediction = model.predict(current_sequence.reshape(1, window_size, 1))
    future_predictions.append(prediction[0, 0])
    
    # Yeni tahmini diziye ekle, eskisini çıkar (kayan pencere)
    current_sequence = np.append(current_sequence[1:], prediction, axis=0)

# Ölçekten çıkar (inverse transform)
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Gelecek tarihleri oluştur
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)

# 8. Gelecek tahmini eklenmiş Grafik Gösterimi
plt.figure(figsize=(10,6))
plt.plot(df.index[window_size:], df['geri_donus_sayisi'][window_size:], label='Gerçek')
plt.plot(df.index[window_size:], predicted, label='Tahmin')
plt.plot(future_dates, future_predictions, label='İleri Tahmin', linestyle='dashed')
plt.legend()
plt.title("Gerçek vs LSTM Tahmin ve Gelecek Tahminleri")
plt.show()

# Gerçek Değeri Tahminleyen Grafik
#plt.figure(figsize=(10,6))
#plt.plot(df.index[window_size:], df['geri_donus_sayisi'][window_size:], label='Gerçek')
#plt.plot(df.index[window_size:], predicted, label='Tahmin')
#plt.legend()
#plt.title("Gerçek vs LSTM Tahmin")
#plt.show()


#Değerleri hesaplamak için eklendi.
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Gerçek ve tahmin değerlerini hizala (window_size sonrası)
y_true = df['geri_donus_sayisi'][window_size:].values
y_pred = predicted.flatten()

# MAE
mae = mean_absolute_error(y_true, y_pred)

# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# MAPE
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# SMAPE (kendi formülümüzle)
smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

# Yazdır
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"SMAPE: {smape:.2f}%")
