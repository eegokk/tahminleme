import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pmdarima import auto_arima



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
    #print(df.head())  # İlk 5 satır
    print(df.columns)         # Sütun isimlerini göster
    print(df.iloc[:, :2])  

    # Bağlantıyı kapat
    cursor.close()
    connection.close()

except cx_Oracle.DatabaseError as e:
    print("Veritabanı bağlantı hatası:", e)
    
    
    
# Zaman serisi verinizi hazırlayın
df['TARIH'] = pd.to_datetime(df['TARIH'])  # Zaman sütunu varsa
df.set_index('TARIH', inplace=True)        # TARIH sütununu index olarak ayarla


# Tahmin etmek istediğiniz sütunu belirleyin sıralı
series = df['SAYI'].sort_index()

    
# Otomatik model seçimi auto_arima
stepwise_model = auto_arima(series,
                             start_p=0, start_q=0,
                             max_p=5, max_q=5,
                             d=1,             # fark derecesi
                             seasonal=True,  # haftasonları düşen veri için 
                             m=7,  # haftalık döngü
                             trace=True,      # süreç bilgisi yazdır
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)
    
# Seçilen modeli yazdır
print("Otomatik seçilen ARIMA parametreleri:", stepwise_model.order)


    # ARIMA modelini kur ve tahmin et
#model_arima = ARIMA(series, order=(1, 1, 1))
#model_arima_fit = model_arima.fit()
model_arima = ARIMA(series, order=stepwise_model.order)
model_arima_fit = model_arima.fit()


# ARIMA artıklarını al
fitted_values = model_arima_fit.fittedvalues
residuals = series[1:] - fitted_values  # İlk fark alınmış seride ilk eleman düşer


# Veriyi normalize et (Eğitim verisini look_back ile hazırlayıp LSTM'e ver)
#scaler = MinMaxScaler()
#res_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))
scaler = StandardScaler()
res_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

# LSTM için veri oluştur
def create_sequences(data, look_back=5):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

#look_back = 5
look_back = 30
X, y = create_sequences(res_scaled, look_back)

# LSTM girdi şekli: [örnek, zaman adımı, özellik]
X = X.reshape((X.shape[0], X.shape[1], 1))

# LSTM Modeli
model_lstm = Sequential()
#model_lstm.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
model_lstm.add(LSTM(50, activation='tanh', input_shape=(look_back, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X, y, epochs=20, verbose=1)



# Gelecek 10 gün için ARIMA tahmini
future_arima = model_arima_fit.forecast(steps=10)

# LSTM ile artık tahmini
last_res = res_scaled[-look_back:]
input_seq = last_res.reshape((1, look_back, 1))

future_residuals = []

for _ in range(10):
    pred = model_lstm.predict(input_seq, verbose=0)
    future_residuals.append(pred[0][0])

       # input_seq'u güncelle (kaydır ve yeni değeri ekle)
    last_res = np.append(last_res[1:], pred[0][0])
    input_seq = last_res.reshape((1, look_back, 1))


# Artıkları orijinal ölçeğe döndür
#future_residuals = scaler.inverse_transform(np.array(future_residuals).reshape(-1, 1)).flatten()
arr = np.array(future_residuals).reshape(-1, 1)
arr = np.nan_to_num(arr)  # NaN → 0, inf → max
future_residuals = scaler.inverse_transform(arr).flatten()





# ARIMA + LSTM Hibrit tahmin
hybrid_forecast = future_arima + future_residuals

# Tahmin tarihlerini oluştur
forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=10)

#tahminleri yazdırma
print("Hybrid Forecast:", hybrid_forecast)
print("Forecast Dates:", forecast_dates)
print("ARIMA forecast:\n", future_arima)
print("LSTM residuals:\n", future_residuals)

print("Future ARIMA:", future_arima)
print("Future residuals (inverse scaled):", future_residuals)
print("Hybrid Forecast:", hybrid_forecast)

#ltsm tarafında nan çıktığından bu kontrol eklendi. bu kontrol sonucunda da artıkları
#orjinal ölçeğe dönüştür aşamasında güncelleme yapıldı
print("ARIMA NaN var mı?", np.any(pd.isna(future_arima)))
print("LSTM NaN var mı?", np.any(pd.isna(future_residuals)))
print("Hybrid NaN var mı?", np.any(pd.isna(hybrid_forecast)))



# Grafik
plt.figure(figsize=(12, 6))
plt.plot(series, label='Gerçek Veri', linewidth=2)
plt.plot(forecast_dates, hybrid_forecast, label='ARIMA+LSTM Tahmin', color='red', linewidth=2)
plt.title('ARIMA + LSTM Hibrit Tahmin')
plt.xlabel("Tarih")
plt.ylabel("Sayı")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


