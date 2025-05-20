import cx_Oracle
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score



# Bağlantı bilgileri
username = 'ECINAR' 
password = '123' 
dsn = '127.0.0.1:1521/orcl'  

try:
    # Oracle veritabanına bağlantı
    connection = cx_Oracle.connect(username, password, dsn)
    print("Bağlantı başarılı ✅")

    # Bağlantıyı kontrol etmek için çalıştırılan sorgu
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
    
#dataframe oluştur    
df = pd.DataFrame(data, columns=columns)    
df['tarih'] = pd.to_datetime(df['TARIH'])
df.set_index('TARIH', inplace=True)
df.rename(columns={'SAYI': 'geri_donus_sayisi'}, inplace=True)
df = df.sort_index()
#df = df.asfreq('D') eksik verim olmadığından bu alanlar çıkarıldı
#df.ffill(inplace=True)


# Veri analizi grafiği 
df['geri_donus_sayisi'].plot(figsize=(12, 4), title='Tüm Veri: Geri Dönüş Zaman Serisi')
plt.grid(True)
plt.show()


#eğitim ve test ayrımı
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

#ETS Modeli Eğitimi ve Tahmini

#model_ets = ExponentialSmoothing(train["geri_donus_sayisi"], trend='add', seasonal='add', seasonal_periods=7)
#model_fit = model_ets.fit()
#forecast = model_fit.forecast(len(test))

best_model = None
best_score = float('inf')

for trend in ['add', 'mul']:
    for seasonal in ['add', 'mul']:
        try:
            model = ExponentialSmoothing(train["geri_donus_sayisi"],
                                         trend=trend,
                                         seasonal=seasonal,
                                         seasonal_periods=7)
            fit = model.fit(optimized=True)
            forecast = fit.forecast(len(test))
            mse = mean_squared_error(test["geri_donus_sayisi"], forecast)

            print(f"Trend: {trend}, Seasonal: {seasonal}, MSE: {mse:.2f}")

            if mse < best_score:
                best_score = mse
                best_model = (trend, seasonal, fit, forecast)
        except Exception as e:
            print(f"{trend}-{seasonal} model başarısız: {e}")


forecast = best_model[3]  # en iyi modelin tahmin sonuçları

mse = mean_squared_error(test["geri_donus_sayisi"], forecast)
mae = mean_absolute_error(test["geri_donus_sayisi"], forecast)
print("En İyi Model → Trend:", best_model[0], "| Seasonal:", best_model[1])
print("MSE:", mse)
print("MAE:", mae)

r2 = r2_score(test["geri_donus_sayisi"], forecast)
print("R² Skoru:", r2)


# Grafikle Görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(train.index, train["geri_donus_sayisi"], label='Modelin Öğrendiği')
plt.plot(test.index, test["geri_donus_sayisi"], label='Gerçek Veri')
plt.plot(test.index, forecast, label='ETS Tahmini(En İyi Model', linestyle='--')
plt.legend()
#plt.title("ETS ile Gönüllü Geri Dönüş Tahmini") titleda en iyi ets modelini yazdırması için kaldırıldı.
plt.title(f"ETS Tahmini | Trend: {best_model[0]}, Seasonal: {best_model[1]}")
plt.xlabel("Tarih")
plt.ylabel("Geri Dönüş Sayısı")
plt.grid(True)
plt.tight_layout()
plt.show()
    
