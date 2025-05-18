import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Oracle veritabanına bağlan
username = 'ECINAR'
password = '123'
dsn = '127.0.0.1:1521/orcl'

try:
    connection = cx_Oracle.connect(username, password, dsn)
    print("Bağlantı başarılı ✅")

    cursor = connection.cursor()
    query = "SELECT * FROM ECINAR.YK_GGD_SAYI"
    cursor.execute(query)

    columns = [col[0] for col in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)

    cursor.close()
    connection.close()
except cx_Oracle.DatabaseError as e:
    print("Veritabanı bağlantı hatası:", e)

# 2. Veri ön işleme
df['tarih'] = pd.to_datetime(df['TARIH'])
df.set_index('tarih', inplace=True)
df.rename(columns={'SAYI': 'geri_donus_sayisi'}, inplace=True)
df = df.sort_index()

# 3. Özellik çıkarımı
df['month'] = df.index.month
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# 4. Özellik ve hedef ayrımı
feature_cols = ['month', 'dayofweek', 'is_weekend']
target_col = ['geri_donus_sayisi']

X_raw = df[feature_cols]
y_raw = df[target_col]

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(X_raw)
y_scaled = target_scaler.fit_transform(y_raw)

# 5. Sequence oluştur
window_size = 5
X, y = [], []

for i in range(len(y_scaled) - window_size):
    seq_x = np.hstack([
        y_scaled[i:i+window_size],
        X_scaled[i:i+window_size, :]
    ])
    X.append(seq_x)
    y.append(y_scaled[i + window_size])

X = np.array(X)
y = np.array(y)

# 6. LSTM Modeli
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

# 7. Model eğitimi
model.fit(X, y, epochs=50, batch_size=1, verbose=1)

# 8. Tahmin ve inverse transform
y_pred = model.predict(X)
y_pred_inverse = target_scaler.inverse_transform(y_pred)
y_true_inverse = target_scaler.inverse_transform(y)

# 9. Değerlendirme metrikleri
mae = mean_absolute_error(y_true_inverse, y_pred_inverse)
rmse = np.sqrt(mean_squared_error(y_true_inverse, y_pred_inverse))
mape = np.mean(np.abs((y_true_inverse - y_pred_inverse) / y_true_inverse)) * 100
smape = 100 * np.mean(2 * np.abs(y_pred_inverse - y_true_inverse) / (np.abs(y_pred_inverse) + np.abs(y_true_inverse)))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"SMAPE: {smape:.2f}%")

# 10. Grafik
plt.figure(figsize=(10,6))
plt.plot(y_true_inverse, label='Gerçek')
plt.plot(y_pred_inverse, label='Tahmin')
plt.legend()
plt.title("Gerçek vs Tahmin Edilen Geri Dönüş Sayısı")
plt.show()
