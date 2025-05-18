import cx_Oracle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import random
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Rastgelelikleri sabitle
os.environ['TF_DETERMINISTIC_OPS'] = '1'
#os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


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
#df = pd.DataFrame(data, columns=columns)  #Değerleri idealize etmek için çıkarıldı.   
df['tarih'] = pd.to_datetime(df['TARIH'])
df.set_index('TARIH', inplace=True)
df.rename(columns={'SAYI': 'geri_donus_sayisi'}, inplace=True)
df = df.sort_index()
#df["weekday"] = df.index.weekday  #Değerleri idealize etmek için çıkarıldı. 


 #Değerleri idealize etmek için eklendi.
df['month'] = df.index.month
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# 4. Özellik ve hedef ayrımı
feature_cols = ['month', 'dayofweek', 'is_weekend']
target_col = ['geri_donus_sayisi']

X_raw = df[feature_cols]
y_raw = df[target_col]

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(X_raw)
y_scaled = target_scaler.fit_transform(y_raw)

#Değerleri idealize etmek için çıkarıldı.
# 3. VERİYİ ÖLÇEKLE
#scaler = MinMaxScaler()
#scaled_data = scaler.fit_transform(df[['geri_donus_sayisi']])

# 4. SEQUENCE OLUŞTUR

window_size = 17
X, y = [], []
for i in range(len(y_scaled) - window_size):
#for i in range(len(scaled_data) - window_size): #Değerleri idealize etmek için değiştirildi.
   # X.append(scaled_data[i:i+window_size]) #Değerleri idealize etmek için değiştirildi.
   # y.append(scaled_data[i+window_size]) #Değerleri idealize etmek için değiştirildi.
   seq_x = np.hstack([ 
       y_scaled[i:i+window_size],
       X_scaled[i:i+window_size, :]
   ])
   X.append(seq_x)
   y.append(y_scaled[i + window_size])

X = np.array(X)
y = np.array(y)


# 5. MODELİ KUR
model = Sequential()
#model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))  #Değerleri idealize etmek için değiştirildi.
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(32)) #Değerleri idealize etmek için eklendi.
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()



# 6. MODELİ EĞİT
#model.fit(X, y, epochs=50, batch_size=1, verbose=1)
history = model.fit(
    X, y,
    epochs=50,
    batch_size=1,
    validation_split=0.1,
    verbose=1
)


# 7. TAHMİN YAP
#predicted = model.predict(X)  #Değerleri idealize etmek için değiştirildi.
#predicted = scaler.inverse_transform(predicted)  #Değerleri idealize etmek için değiştirildi.
y_pred = model.predict(X)
y_pred_inverse = target_scaler.inverse_transform(y_pred)
y_true_inverse = target_scaler.inverse_transform(y)

# Gelecek tahminleri
future_steps = 30  # Kaç gün sonrasını tahmin etmek istiyorsan burada belirle
last_sequence = np.hstack([
    y_scaled[-window_size:], 
    X_scaled[-window_size:, :]
])  
future_predictions = []
current_sequence = last_sequence.copy()

for _ in range(future_steps):
    prediction = model.predict(current_sequence.reshape(1, window_size, 4)) 
    future_predictions.append(prediction[0, 0])    
    dummy_features = np.zeros((1, 3))  # ay, gün, hafta sonu sahte veri
    new_step = np.concatenate((prediction, dummy_features), axis=1)
    current_sequence = np.append(current_sequence[1:], new_step, axis=0)
 

# Ölçekten çıkar (inverse transform)
future_predictions = target_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))  # ✅ DOĞRU


# Gelecek tarihleri oluştur
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)

# 8. Gelecek tahmini eklenmiş Grafik Gösterimi
plt.figure(figsize=(10,6))
plt.plot(df.index[window_size:], df['geri_donus_sayisi'][window_size:], label='Gerçek')
plt.plot(df.index[window_size:], y_pred_inverse, label='Tahmin') 
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


# Gerçek ve tahmin değerlerini hizala (window_size sonrası)
y_true = df['geri_donus_sayisi'][window_size:].values
y_pred = y_pred_inverse.flatten() 

# MAE
mae = mean_absolute_error(y_true, y_pred) # Değerleri idealize etmek için değiştirildi.
mae = mean_absolute_error(y_true_inverse, y_pred_inverse)

# RMSE
#rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # Değerleri idealize etmek için değiştirildi.
rmse = np.sqrt(mean_squared_error(y_true_inverse, y_pred_inverse))

# MAPE
#mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 # Değerleri idealize etmek için değiştirildi.
mape = np.mean(np.abs((y_true_inverse - y_pred_inverse) / y_true_inverse)) * 100

# SMAPE (kendi formülümüzle)
#smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) # Değerleri idealize etmek için değiştirildi.
smape = 100 * np.mean(2 * np.abs(y_pred_inverse - y_true_inverse) / (np.abs(y_pred_inverse) + np.abs(y_true_inverse))) 

# MASE hesapla
naive_forecast = y_true[1:]  # Gerçek değerler (bir gün sonrası)
naive_prediction = y_true[:-1]  # Naive model: her değeri bir önceki günün değeri olarak tahmin eder

naive_mae = mean_absolute_error(naive_forecast, naive_prediction)
mase = mae / naive_mae


print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"SMAPE: {smape:.2f}%")  
print(f"MASE: {mase:.2f}")                                                                 



# Hata hesapla
hatalar_df = pd.DataFrame({
    'gercek': y_true,
    'tahmin': y_pred,
}, index=df.index[window_size:])

hatalar_df['hata'] = hatalar_df['gercek'] - hatalar_df['tahmin']
hatalar_df['mutlak_hata'] = np.abs(hatalar_df['hata'])
                                                         
plt.figure(figsize=(12,5))
plt.plot(hatalar_df.index, hatalar_df['hata'], label='Hata (Gerçek - Tahmin)')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Günlük Tahmin Hataları')
plt.ylabel('Hata')
plt.xlabel('Tarih')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


aylik_hatalar = hatalar_df.resample('M').mean()

plt.figure(figsize=(10,4))
plt.bar(aylik_hatalar.index, aylik_hatalar['mutlak_hata'], width=20)
plt.title('Aylık Ortalama Mutlak Hata')
plt.ylabel('Ortalama Hata')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


en_kotu = hatalar_df.sort_values(by='mutlak_hata', ascending=False).head(10)
print("En yüksek sapmalı 10 gün:")
print(en_kotu)


hatalar_df['weekday'] = hatalar_df.index.dayofweek
print(hatalar_df.groupby('weekday')['mutlak_hata'].mean())
