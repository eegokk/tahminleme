import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

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
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.values.reshape(-1, 1))

# Sequence oluşturma
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 12  
X, y = create_sequences(series_scaled, look_back)
X = X.reshape((X.shape[0], look_back, 1))

# RNN modeli kurulumu
model_rnn = Sequential()
model_rnn.add(SimpleRNN(50, activation='tanh', input_shape=(look_back, 1)))
model_rnn.add(Dense(1))
model_rnn.compile(optimizer='adam', loss='mse')
model_rnn.fit(X, y, epochs=100, verbose=1)

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



# Grafikle gösterim
plt.figure(figsize=(12, 6))
plt.plot(series, label='Gerçek Veri', linewidth=2)
plt.plot(forecast_dates, rnn_forecast, label='RNN Tahminleri', linestyle='--', marker='o')
plt.title('RNN ile Aylık Gönüllü Geri Dönüş Tahmini')
plt.xlabel("Tarih")
plt.ylabel("Geri Dönüş Sayısı")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






