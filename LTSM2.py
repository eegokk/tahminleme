import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# Veriyi Hazırla (Scaler ve Sekans Oluşturma)
# Sadece hedef sütunu al (geri dönüş sayısı)
values = df['geri_donus_sayisi'].values.reshape(-1, 1)

# Normalizasyon (LSTM için önemlidir)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)

# LSTM girişi için veri oluştur (look_back: kaç gün geriye bakılacak)
def create_sequences(data, look_back=7):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 7  # örnek olarak 7 gün
X, y = create_sequences(scaled_values, look_back)

# Eğitim ve test veri seti (örneğin %80 eğitim)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM Modelini Kur
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Eğit
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Tahmin ve Ters Dönüşüm
# Tahmin yap
y_pred = model.predict(X_test)

# Ters scale işlemi
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# Hata metrikleri
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")


# Görselleştirme
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Gerçek')
plt.plot(y_pred_inv, label='Tahmin')
plt.title('Gerçek vs Tahmin - Geri Dönüş Sayısı')
plt.legend()
plt.show()


# Geleceğe Dönük Tahmin (İleriye Tahmin)
future_steps = 7  # 7 gün ileriye tahmin

last_sequence = scaled_values[-look_back:]
future_predictions = []

for _ in range(future_steps):
    pred = model.predict(last_sequence.reshape(1, look_back, 1))
    future_predictions.append(pred[0])
    last_sequence = np.append(last_sequence[1:], pred, axis=0)

# Ters scale
future_predictions = scaler.inverse_transform(future_predictions)

# Tarih etiketleri
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_steps)

# Sonuçları göster
plt.figure(figsize=(10,5))
plt.plot(df.index[-30:], df['geri_donus_sayisi'].values[-30:], label='Son 30 Gün')
plt.plot(future_dates, future_predictions, label='7 Günlük Tahmin')
plt.title("Geleceğe Dönük Geri Dönüş Tahmini")
plt.legend()
plt.show()
