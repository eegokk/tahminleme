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
import numpy as np  #random seed sabitleme için eklendi
import tensorflow as tf  #random seed sabitleme için eklendi

# Tekrarlanabilirlik için sabit tohum değerleri
os.environ['PYTHONHASHSEED'] = '0'  #random seed sabitleme için eklendi
np.random.seed(42)  #random seed sabitleme için eklendi
random.seed(42) #random seed sabitleme için eklendi
tf.random.set_seed(42) #random seed sabitleme için eklendi


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
    print(df.columns)        
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
    

# veri hazırlama
# NumPy dizisini 2 boyutlu hale getirir. -1 değeri satır sayısını otomatik ayarlar.
# 1 değeri her satırda yalnızca bir değer olur.
data_values = df['geri_donus_sayisi'].values.reshape(-1, 1)


# Veriyi ölçekle (0-1 arası)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_values)


# Lookback (kaç gün geçmişi kullanacağımız)
look_back = 10

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, look_back)

# RNN için reshape: [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))


# Eğitim ve doğrulama verilerini %80-%20 oranında ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Modeli eğit
#history = model.fit(X, y, epochs=50, batch_size=16, verbose=1)

#overfitting için eklendi
model = Sequential()
model.add(SimpleRNN(50, activation='tanh', input_shape=(look_back, 1)))
model.add(Dropout(0.2))  # %20 oranında dropout için eklendi
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  #earlystopping için eklendi

#overfitting
history = model.fit(
    X_train, y_train,
    # epochs=50,
    epochs=50, #earlystopping için eklendi
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)


# Gelecek günler için tahmin 
# Son 'look_back' günle başlayarak tahmin yapıyoruz
future_steps = 10  #döngünün kaç kez çalışacağını belirlemek için ekledik.
last_sequence = scaled_data[-look_back:]
predictions = []

for _ in range(future_steps):
    input_seq = last_sequence.reshape((1, look_back, 1))
    next_pred = model.predict(input_seq)[0][0]
    predictions.append(next_pred)
    last_sequence = np.append(last_sequence[1:], [[next_pred]], axis=0)

# Tahminleri ölçekten çıkartıp orijinal ölçeğe geri dönüoruz.
predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Tahminleri tarihlerle eşleştiriyoruz.
last_date = df.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_steps)

# Sonuçları DataFrame olarak göster
forecast_df = pd.DataFrame({'tarih': future_dates, 'tahmin': predicted_values.flatten()})
print(forecast_df)



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


