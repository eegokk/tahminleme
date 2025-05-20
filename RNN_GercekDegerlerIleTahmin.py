import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
import os   #random seed sabitleme için eklendi
import random  #random seed sabitleme için eklendi
import tensorflow as tf  #random seed sabitleme için eklendi
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Tekrarlanabilirlik için sabit tohum değerleri
os.environ['PYTHONHASHSEED'] = '0'  #random seed sabitleme için eklendi
np.random.seed(42)  #random seed sabitleme için eklendi
random.seed(42) #random seed sabitleme için eklendi
tf.random.set_seed(42) #random seed sabitleme için eklendi


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


    

# veri hazırlama
# Sadece hedef kolonu al
data_values = df['geri_donus_sayisi'].values.reshape(-1, 1)


# Veriyi ölçekle (0-1 arası)
scaler = MinMaxScaler()
data_values = df['geri_donus_sayisi'].values.reshape(-1, 1) #gerçek değer tahmini için eklendi.
scaled_data = scaler.fit_transform(data_values)


# Hedef veri sütununu al ve normalleştir
data_values = df['geri_donus_sayisi'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_values)

# Lookback (kaç gün geçmişi kullanacağımız)
look_back = 15

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, look_back)

X = X.reshape((X.shape[0], look_back, 1)) #gerçek değer tahmini için eklendi.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False) #gerçek değer tahmini için eklendi.


# RNN için reshape: [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))


# Eğitim ve doğrulama verilerini %80-%20 oranında ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Modeli eğit
#history = model.fit(X, y, epochs=50, batch_size=16, verbose=1)


model = Sequential()
model.add(SimpleRNN(50, activation='tanh', input_shape=(look_back, 1)))
model.add(Dropout(0.2))  # %20 oranında dropout için eklendi
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  #earlystopping için eklendi

history = model.fit(X_train, y_train,
    # epochs=50,
    epochs=50, #earlystopping için eklendi
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# 5. Geçmiş Veriler Üzerinden Tahmin
X_past = []
tahmin_tarihleri_past = []
for i in range(look_back, len(scaled_data)):
    X_past.append(scaled_data[i - look_back:i, 0])
    tahmin_tarihleri_past.append(df.index[i])
X_past = np.array(X_past).reshape(-1, look_back, 1)
y_past_scaled = model.predict(X_past)
y_past = scaler.inverse_transform(y_past_scaled)
gercek_past = df['geri_donus_sayisi'].values[look_back:]

past_df = pd.DataFrame({
    'tarih': tahmin_tarihleri_past,
    'gercek': gercek_past,
    'tahmin': y_past.flatten()
})

# Gelecek günler için tahmin 
# Son 'look_back' günle başlayarak tahmin yapıyoruz
future_steps = 10  #döngünün kaç kez çalışacağını belirlemek için ekledik.
last_sequence = scaled_data[-look_back:]
#predictions = []
future_predictions = []
for _ in range(future_steps):
    input_seq = last_sequence.reshape((1, look_back, 1))
    next_pred = model.predict(input_seq)[0][0]
    #predictions.append(next_pred)
    future_predictions.append(next_pred)
    last_sequence = np.append(last_sequence[1:], [[next_pred]], axis=0)
future_pred = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
forecast_df = pd.DataFrame({
    'tarih': future_dates,
    'tahmin': future_pred.flatten()
})
forecast_df['gercek'] = df['geri_donus_sayisi'].reindex(forecast_df['tarih']).values  # güvenli eşleşme    
    

# Tahminleri ölçekten çıkartıp orijinal ölçeğe geri dönüoruz.
predicted_values = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Tahminleri tarihlerle eşleştiriyoruz.
last_date = df.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_steps)

# Sonuçları DataFrame olarak göster
forecast_df = pd.DataFrame({
    'tarih': future_dates,
    'tahmin': future_pred.flatten()
})
forecast_df['gercek'] = df['geri_donus_sayisi'].reindex(forecast_df['tarih']).values



# Görselleştirme
plt.figure(figsize=(10,5))
plt.plot(df.index[-50:], df['geri_donus_sayisi'].values[-50:], label='Gerçek Değerler')
plt.plot(forecast_df['tarih'], forecast_df['tahmin'], label='RNN Tahminleri', linestyle='--', marker='o')
plt.legend()
plt.grid(True)
plt.title('RNN ile Geri Dönüş Tahmini')
plt.xlabel('Tarih')
plt.ylabel('Geri Dönüş Sayısı')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Overfitting için Görselleştirme
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Loss (Eğitim vs Doğrulama)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 7. Birleştir ve Grafik (#gerçek değer tahmini için eklendi. )
final_df = pd.concat([past_df, forecast_df], ignore_index=True)

plt.figure(figsize=(14, 5))
plt.plot(final_df['tarih'], final_df['gercek'], label='Gerçek Değerler', linewidth=2)
plt.plot(final_df['tarih'], final_df['tahmin'], label='RNN Tahmini', linestyle='--', linewidth=2)
plt.title('Geçmiş + Gelecek Tahmin')
plt.xlabel('Tarih')
plt.ylabel('Geri Dönüş Sayısı')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Hata Metrikleri (Sadece geçmiş için)
mae = mean_absolute_error(past_df['gercek'], past_df['tahmin'])
rmse = np.sqrt(mean_squared_error(past_df['gercek'], past_df['tahmin']))
print(f"Geçmiş için MAE: {mae:.2f}")
print(f"Geçmiş için RMSE: {rmse:.2f}")

# 9. Gelecek veride olmayan tarihleri yazdır
print("Eşleşmeyen tarih(ler):")
print(forecast_df[forecast_df['gercek'].isna()][['tarih']])


# MAPE hesaplaması
print("Ortalama gerçek değer:", df['geri_donus_sayisi'].mean())
print("Standart sapma:", df['geri_donus_sayisi'].std())
mape = np.mean(np.abs((past_df['gercek'] - past_df['tahmin']) / past_df['gercek'])) * 100
print(f"MAPE: {mape:.2f}%")


# SMAPE Hesaplaması
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0  # 0 bölü 0 hatası engellenir
    return np.mean(diff) * 100

smape_score = smape(past_df['gercek'].values, past_df['tahmin'].values)
print(f"SMAPE: {smape_score:.2f}%")


# MASE Hesaplaması
naive_forecast = past_df['gercek'].shift(1).dropna()
actual_values = past_df['gercek'][1:]  # aynı uzunlukta
mae_naive = mean_absolute_error(actual_values, naive_forecast)
mae_model = mean_absolute_error(past_df['gercek'], past_df['tahmin'])
mase_score = mae_model / mae_naive
print(f"MASE: {mase_score:.2f}")
