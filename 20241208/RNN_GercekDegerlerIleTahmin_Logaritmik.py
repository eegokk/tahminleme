

import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import os
import random
import tensorflow as tf

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

username = 'ECINAR'
password = '123'
dsn = '127.0.0.1:1521/orcl'

try:
    connection = cx_Oracle.connect(username, password, dsn)
    print("Bağlantı başarılı ✅")
    cursor = connection.cursor()
    query = "SELECT * FROM ECINAR.YK_GGD_SAYI "
    cursor.execute(query)
    columns = [col[0] for col in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)
    cursor.close()
    connection.close()
except cx_Oracle.DatabaseError as e:
    print("Veritabanı bağlantı hatası:", e)

# Veri hazırlık
df['tarih'] = pd.to_datetime(df['TARIH'])
df.set_index('TARIH', inplace=True)
df.rename(columns={'SAYI': 'geri_donus_sayisi'}, inplace=True)
df = df.sort_index()

# Log dönüşüm
df['log_sayi'] = np.log1p(df['geri_donus_sayisi'])
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['log_sayi'].values.reshape(-1, 1))

look_back = 15

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, look_back)
X = X.reshape((X.shape[0], look_back, 1))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

model = Sequential()
model.add(SimpleRNN(50, activation='tanh', input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stop], verbose=1)

# Geçmiş tahmin
X_past = []
tahmin_tarihleri_past = []
for i in range(look_back, len(scaled_data)):
    X_past.append(scaled_data[i - look_back:i, 0])
    tahmin_tarihleri_past.append(df.index[i])
X_past = np.array(X_past).reshape(-1, look_back, 1)
y_past_scaled = model.predict(X_past)
log_y_past = scaler.inverse_transform(y_past_scaled)
y_past = np.expm1(log_y_past)
gercek_past = df['geri_donus_sayisi'].values[look_back:]

past_df = pd.DataFrame({
    'tarih': tahmin_tarihleri_past,
    'gercek': gercek_past,
    'tahmin': y_past.flatten()
})

# Gelecek tahmin
future_steps = 10
last_sequence = scaled_data[-look_back:]
future_predictions = []
for _ in range(future_steps):
    input_seq = last_sequence.reshape((1, look_back, 1))
    next_pred = model.predict(input_seq)[0][0]
    future_predictions.append(next_pred)
    last_sequence = np.append(last_sequence[1:], [[next_pred]], axis=0)

log_future = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_pred = np.expm1(log_future)
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)

forecast_df = pd.DataFrame({
    'tarih': future_dates,
    'tahmin': future_pred.flatten()
})
forecast_df['gercek'] = df['geri_donus_sayisi'].reindex(forecast_df['tarih']).values

# Birleştir
final_df = pd.concat([past_df, forecast_df], ignore_index=True)

# Grafik
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

# Metrikler
mae = mean_absolute_error(past_df['gercek'], past_df['tahmin'])
rmse = np.sqrt(mean_squared_error(past_df['gercek'], past_df['tahmin']))
print(f"Geçmiş için MAE: {mae:.2f}")
print(f"Geçmiş için RMSE: {rmse:.2f}")

# MAPE
mape = np.mean(np.abs((past_df['gercek'] - past_df['tahmin']) / past_df['gercek'])) * 100
print(f"MAPE: {mape:.2f}%")

# SMAPE
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0
    return np.mean(diff) * 100
smape_score = smape(past_df['gercek'].values, past_df['tahmin'].values)
print(f"SMAPE: {smape_score:.2f}%")

# MASE
naive_forecast = past_df['gercek'].shift(1).dropna()
actual_values = past_df['gercek'][1:]
mae_naive = mean_absolute_error(actual_values, naive_forecast)
mae_model = mean_absolute_error(past_df['gercek'], past_df['tahmin'])
mase_score = mae_model / mae_naive
print(f"MASE: {mase_score:.2f}")