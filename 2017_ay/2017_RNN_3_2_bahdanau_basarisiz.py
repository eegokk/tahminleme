import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Concatenate, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Huber
import os
import random

# Sabit seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------- Oracle bağlantısı -------------------- #
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

# -------------------- Veri hazırlama -------------------- #
df['TARIH'] = pd.to_datetime(df['TARIH'])  
df.set_index('TARIH', inplace=True)
series = df['SAYI'].sort_index()

scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.values.reshape(-1, 1))

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 18
X, y = create_sequences(series_scaled, look_back)
X = X.reshape((X.shape[0], look_back, 1))

# -------------------- Bahdanau Attention -------------------- #
class BahdanauAttention(Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, hidden_states, last_hidden_state):
        last_hidden_state = tf.expand_dims(last_hidden_state, 1)
        score = tf.nn.tanh(self.W1(hidden_states) + self.W2(last_hidden_state))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# -------------------- Model -------------------- #
inputs = Input(shape=(look_back, 1))
rnn_outputs = SimpleRNN(32, return_sequences=True)(inputs)
last_hidden = SimpleRNN(16)(rnn_outputs)

context_vector, _ = BahdanauAttention(16)(rnn_outputs, last_hidden)
combined = Concatenate()([context_vector, last_hidden])
outputs = Dense(1)(combined)

model_rnn = Model(inputs=inputs, outputs=outputs)
model_rnn.compile(optimizer='adam', loss='Huber(delta=1.0)')
model_rnn.fit(X, y, epochs=150, verbose=1)

# -------------------- Geçmiş tahmin -------------------- #
rnn_train_pred_scaled = model_rnn.predict(X, verbose=0)
rnn_train_pred = scaler.inverse_transform(rnn_train_pred_scaled).flatten()
actual_past = series[look_back:]

# -------------------- Gelecek tahmin -------------------- #
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

print("\nGelecek Tahminleri (RNN):")
for date, value in zip(forecast_dates, rnn_forecast):
    print(f"{date.strftime('%Y-%m')} → {int(value)}")

# -------------------- Hata metrikleri -------------------- #
mae = mean_absolute_error(actual_past, rnn_train_pred)
rmse = np.sqrt(mean_squared_error(actual_past, rnn_train_pred))
mape = np.mean(np.abs((actual_past - rnn_train_pred) / actual_past)) * 100
naive = actual_past.shift(1).dropna()
mase = mae / np.mean(np.abs(actual_past[1:] - naive))

print("\nHATA METRİKLERİ")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"MASE : {mase:.2f}")

# -------------------- Grafik -------------------- #
plt.figure(figsize=(12, 6))
plt.plot(series, label='Gerçek Veri', linewidth=2)
plt.plot(actual_past.index, rnn_train_pred, label='Geçmiş Tahmin (RNN)', color='orange')
plt.plot(forecast_dates, rnn_forecast, label='Gelecek Tahmin (RNN)', color='red', linewidth=2)
plt.title('Attention-RNN Tahmini (Geçmiş + Gelecek)')
plt.xlabel("Tarih")
plt.ylabel("Geri Dönüş Sayısı")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
