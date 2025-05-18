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

# 2. TARİH DÜZENLEME
#df['TARIH'] = pd.to_datetime(df['TARIH'])
#df = df.sort_values('TARIH')
#df.set_index('TARIH', inplace=True)

# 3. VERİYİ ÖLÇEKLE
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['geri_donus_sayisi']])

# 4. SEQUENCE OLUŞTUR
X, y = [], []
window_size = 5  # ikinci kodla aynı pencere

for i in range(len(scaled_data) - window_size):
    X.append(scaled_data[i:i+window_size])
    y.append(scaled_data[i+window_size])

X, y = np.array(X), np.array(y)

# 5. MODELİ KUR
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 6. MODELİ EĞİT
model.fit(X, y, epochs=50, batch_size=1, verbose=1)

# 7. TAHMİN YAP
predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)

# 8. GRAFİKLE GÖSTER
plt.figure(figsize=(10,6))
plt.plot(df.index[window_size:], df['geri_donus_sayisi'][window_size:], label='Gerçek')
plt.plot(df.index[window_size:], predicted, label='Tahmin')
plt.legend()
plt.title("Gerçek vs LSTM Tahmin")
plt.show()
