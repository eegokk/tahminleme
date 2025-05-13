"""
Spyder Editor

This is a temporary script file.
"""
import cx_Oracle
import pandas as pd

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
    
    
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Zaman serisi verinizi hazırlayın
df['TARIH'] = pd.to_datetime(df['TARIH'])  # Zaman sütunu varsa
df.set_index('TARIH', inplace=True)        # TARIH sütununu index olarak ayarla

# Tahmin etmek istediğiniz sütunu belirleyin
series = df['SAYI']  # Örnek veri sütunu

# Grafikle kontrol et
series.plot(title='Zaman Serisi Verisi')
plt.show()

# ARIMA modelini kur (örnek: ARIMA(1,1,1))
model = ARIMA(series, order=(1, 1, 1))
model_fit = model.fit()

# Model özetini yazdır
print(model_fit.summary())

# Gelecek 10 günü tahmin et
forecast = model_fit.forecast(steps=10)
print("Tahmin sonuçları:")
print(forecast)

# Tahminleri grafikte göster
plt.plot(series, label='Gerçek Veri')
forecast_index = pd.date_range(start=series.index[-1], periods=11, freq='D')[1:]
plt.plot(forecast_index, forecast, label='Tahmin', color='red')
plt.legend()
plt.title('ARIMA Tahmini')
plt.show()
