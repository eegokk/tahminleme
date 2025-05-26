import cx_Oracle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

# ---------------------------
# 1. ORACLE'DAN VERİYİ AL
# ---------------------------
username = 'ECINAR'
password = '123'
dsn = '127.0.0.1:1521/orcl'

try:
    connection = cx_Oracle.connect(username, password, dsn)
    cursor = connection.cursor()
    query = "SELECT * FROM ECINAR.YK_GGD_HAFTALIK"
    cursor.execute(query)
    columns = [col[0] for col in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)
    cursor.close()
    connection.close()
    print("✅ Veritabanı bağlantısı ve veri çekme başarılı.")
except cx_Oracle.DatabaseError as e:
    print("❌ Veritabanı bağlantı hatası:", e)

# ---------------------------
# 2. VERİ ÖN İŞLEME
# ---------------------------
df['tarih'] = pd.to_datetime(df['TARIH'])
df.set_index('tarih', inplace=True)
df.rename(columns={'SAYI': 'geri_donus_sayisi'}, inplace=True)
df = df.sort_index()
df['geri_donus_sayisi'] = df['geri_donus_sayisi'].fillna(method='ffill')

# Rejim değişikliği ve ek özellikler
rejim_degisimi_tarihi = pd.to_datetime('2025-08-12')
df['rejim_degisti'] = (df.index >= rejim_degisimi_tarihi).astype(int)
df['hafta_no'] = df.index.isocalendar().week.astype(int)
df['yil'] = df.index.year

# ---------------------------
# 3. SCALE
# ---------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['geri_donus_sayisi', 'rejim_degisti','hafta_no', 'yil']])

# ---------------------------
# 4. SEQUENCE OLUŞTUR
# ---------------------------
def create_multivariate_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

look_back = 22

# ---------------------------
# 5. MODELİ EĞİT (Tüm veriyle)
# ---------------------------
X_all, y_all = create_multivariate_sequences(scaled_data, look_back)

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(look_back, 4)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
model.fit(X_all, y_all, epochs=50, batch_size=16, verbose=1)

# ---------------------------
# 6. TÜM GEÇMİŞ VERİDE TAHMİN
# ---------------------------
rolling_predictions = []

for i in range(look_back, len(scaled_data)):
    input_seq = scaled_data[i - look_back:i].reshape(1, look_back, 4)
    pred = model.predict(input_seq, verbose=0)[0][0]
    rolling_predictions.append(pred)

# Ters ölçekleme için eksik sütunları tamamla
rejim_all = scaled_data[look_back:, 1].reshape(-1, 1)
dummy_hafta = np.zeros_like(rejim_all)
dummy_yil = np.zeros_like(rejim_all)
combined_all = np.hstack((np.array(rolling_predictions).reshape(-1, 1), rejim_all, dummy_hafta, dummy_yil))
predicted_all_inv = scaler.inverse_transform(combined_all)[:, 0]

# Gerçek değerler
true_all_inv = df['geri_donus_sayisi'].values[look_back:]
dates_all = df.index[look_back:]

# ---------------------------
# 7. GELECEĞE TAHMİN
# ---------------------------
future_weeks = 10
last_sequence = scaled_data[-look_back:]
future_predictions = []

for _ in range(future_weeks):
    input_seq = last_sequence.reshape((1, look_back, 4))
    next_pred = model.predict(input_seq, verbose=0)[0][0]
    future_predictions.append(next_pred)
    # 4 özellikli yeni adım
    new_step = np.array([next_pred, 1.0, 0.0, 0.0])
    last_sequence = np.vstack([last_sequence[1:], new_step])

# Ters ölçekleme için 4 sütunlu array oluştur
future_scaled = np.array(future_predictions).reshape(-1, 1)
dummy_rejim = np.ones((future_weeks, 1))
dummy_hafta_future = np.zeros((future_weeks, 1))
dummy_yil_future = np.zeros((future_weeks, 1))
future_combined = np.hstack((future_scaled, dummy_rejim, dummy_hafta_future, dummy_yil_future))
future_values = scaler.inverse_transform(future_combined)[:, 0]

# Tahmin tarihleri
last_known_date = df.index[-1]
future_dates = pd.date_range(start=last_known_date + pd.Timedelta(weeks=1),
                             periods=future_weeks, freq='W-MON')
forecast_df = pd.DataFrame(future_values, index=future_dates, columns=['Tahmin'])

# ---------------------------
# 8. GRAFİK (Tüm Geçmiş + Gelecek)
# ---------------------------
plt.figure(figsize=(14, 6))
plt.plot(df['geri_donus_sayisi'], label='Gerçek Veriler', color='blue')
plt.plot(dates_all, predicted_all_inv, label='Geçmiş Tahmin (Tümü)', color='orange')
plt.plot(forecast_df.index, forecast_df['Tahmin'], label='Gelecek Tahmin', color='red')
plt.title("LSTM Tahmin Performansı – Geçmiş ve Gelecek")
plt.xlabel("Tarih")
plt.ylabel("Gönüllü Geri Dönüş Sayısı")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# 9. HATA METRİKLERİ
# ---------------------------
mae_all = mean_absolute_error(true_all_inv, predicted_all_inv)
rmse_all = np.sqrt(mean_squared_error(true_all_inv, predicted_all_inv))
mape_all = np.mean(np.abs((true_all_inv - predicted_all_inv) / true_all_inv)) * 100

naive_pred = pd.Series(true_all_inv).shift(1).dropna()
mase_all = mae_all / np.mean(np.abs(true_all_inv[1:] - naive_pred))

print("\n📊 HATA METRİKLERİ")
print(f"MAE  : {mae_all:.2f}")
print(f"RMSE : {rmse_all:.2f}")
print(f"MAPE : {mape_all:.2f}%")
print(f"MASE : {mase_all:.2f}")
