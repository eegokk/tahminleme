import cx_Oracle 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Bağlantı bilgileri
username = 'ECINAR' 
password = '123' 
dsn = '127.0.0.1:1521/orcl'  
try:
    # Oracle veritabanına bağlantı
    connection = cx_Oracle.connect(username, password, dsn)
    print("Bağlantı başarılı ✅")

    # Bağlantıyı kontrol etmek için çalıştırılan sorggu
    cursor = connection.cursor()
    query = "SELECT * FROM ECINAR.YK_GGD_AYLIK_TUMVERI "
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

# Veriyi ölçeklendir
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['geri_donus_sayisi']])

# LSTM Girdilerini Oluştur
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

look_back = 10
X, y = create_sequences(scaled_data, look_back)

# LSTM girişi şekli: [örnek sayısı, zaman adımı, özellik sayısı]
X = X.reshape((X.shape[0], X.shape[1], 1))

# LTSM modelini kur ve eğit
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

history = model.fit(X, y, epochs=50, batch_size=16, verbose=1)

# Gelecek tahmini 
future_steps = 10  # Kaç hafta ileriye tahmin yapılacak
last_sequence = scaled_data[-look_back:]
predictions = []

for _ in range(future_steps):
    input_seq = last_sequence.reshape((1, look_back, 1))
    next_pred = model.predict(input_seq)[0][0]
    predictions.append(next_pred)
    last_sequence = np.append(last_sequence[1:], [[next_pred]], axis=0)

# Ölçekten çıkar
predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Gelecekteki tarihleri oluştur
last_date = df.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=future_steps, freq='W-MON')

# Tahminleri DataFrame’e al
forecast_df = pd.DataFrame(predicted_values, index=future_dates, columns=['tahmin'])

# Görselleştirme 
plt.figure(figsize=(12,6))
plt.plot(df['geri_donus_sayisi'], label='Gerçek')
plt.plot(forecast_df, label='Tahmin', linestyle='--')
plt.title("LSTM ile Haftalık Tahmin")
plt.legend()
plt.grid(True)
plt.show()