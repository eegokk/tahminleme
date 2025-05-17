# -*- coding: utf-8 -*-
"""
Created on Sat May 10 21:46:29 2025

@author: cinar
"""


# 1. GEREKLİ KÜTÜPHANELER
import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


# 2. ORACLE'DAN VERİYİ ÇEK
username = 'ECINAR'
password = '123'
dsn = '127.0.0.1:1521/orcl'

try:
    connection = cx_Oracle.connect(username, password, dsn)
    cursor = connection.cursor()
    
    query = "SELECT TARIH, SAYI FROM ECINAR.YK_GGD_SAYI"  
    df = pd.read_sql(query, con=connection)
    
    cursor.close()
    connection.close()
    
    # 3. CSV'YE KAYDET
    df.to_csv("veri.csv", index=False)
    print("Veri başarıyla çekildi ve kaydedildi ✅")

except cx_Oracle.DatabaseError as e:
    print("Veritabanı hatası:", e)

# 4. VERİYİ CSV'DEN OKU
df = pd.read_csv("veri.csv")
df['TARIH'] = pd.to_datetime(df['TARIH'])
df = df.sort_values('TARIH')
df = df.set_index('TARIH')

# 5. LSTM İÇİN VERİ HAZIRLIĞI
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['SAYI']])

X, y = [], []
window_size = 5

for i in range(len(scaled_data) - window_size):
    X.append(scaled_data[i:i+window_size])
    y.append(scaled_data[i+window_size])

X, y = np.array(X), np.array(y)

# 6. MODELİ OLUŞTUR
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 7. MODELİ EĞİT
model.fit(X, y, epochs=50, batch_size=1, verbose=1)

# 8. TAHMİN YAP
predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)

# 9. GÖRSELLEŞTİR
%matplotlib inline
plt.figure(figsize=(10,6))
plt.plot(df.index[window_size:], df['SAYI'][window_size:], label='Gerçek')
plt.plot(df.index[window_size:], predicted, label='Tahmin')
plt.legend()
plt.title("Gerçek vs LSTM Tahmin")
plt.show()
