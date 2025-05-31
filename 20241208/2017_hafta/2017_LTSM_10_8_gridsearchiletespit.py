import cx_Oracle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ---------------------------
# 1. ORACLE'DAN VERİYİ AL
# ---------------------------
username = 'ECINAR'
password = '123'
dsn = '127.0.0.1:1521/orcl'

connection = cx_Oracle.connect(username, password, dsn)
cursor = connection.cursor()
cursor.execute("SELECT * FROM ECINAR.YK_GGD_HAFTALIK")
columns = [col[0] for col in cursor.description]
data = cursor.fetchall()
df = pd.DataFrame(data, columns=columns)
cursor.close()
connection.close()

# ---------------------------
# 2. VERİ ÖN İŞLEME
# ---------------------------
df['tarih'] = pd.to_datetime(df['TARIH'])
df.set_index('tarih', inplace=True)
df.rename(columns={'SAYI': 'geri_donus_sayisi'}, inplace=True)
df = df.sort_index()
df['geri_donus_sayisi'] = df['geri_donus_sayisi'].fillna(method='ffill')

rejim_degisimi_tarihi = pd.to_datetime('2025-08-12')
df['rejim_degisti'] = (df.index >= rejim_degisimi_tarihi).astype(int)
df['geri_donus_fark'] = df['geri_donus_sayisi'].diff().fillna(0)
df['geri_donus_3hafta'] = df['geri_donus_sayisi'].rolling(window=3).mean().fillna(method='bfill')
df['geri_donus_orani'] = df['geri_donus_sayisi'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

# ---------------------------
# 3. SCALE
# ---------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['geri_donus_sayisi', 'rejim_degisti',
                                       'geri_donus_fark', 'geri_donus_3hafta', 'geri_donus_orani']])

# ---------------------------
# 4. GRID SEARCH (LOOK_BACK)
# ---------------------------

def create_seq2seq_data(data, look_back, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back:i + look_back + forecast_horizon, 0].reshape(-1, 1))
    return np.array(X), np.array(y)

def evaluate_model(X_all, y_all, look_back, forecast_horizon):
    input_layer = Input(shape=(look_back, X_all.shape[2]))
    encoder = LSTM(64, return_sequences=False)(input_layer)
    repeat = RepeatVector(forecast_horizon)(encoder)
    decoder = LSTM(64, return_sequences=True)(repeat)
    output = TimeDistributed(Dense(1))(decoder)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss=Huber())
    model.fit(X_all, y_all, epochs=20, batch_size=16, verbose=0)

    predictions = model.predict(X_all, verbose=0).reshape(-1, forecast_horizon)[:, 0]
    y_true = y_all.reshape(-1, forecast_horizon)[:, 0]

    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mape = np.mean(np.abs((y_true - predictions) / (y_true + 1e-6))) * 100
    mase = mae / np.mean(np.abs(np.diff(y_true)))

    return mae, rmse, mape, mase

# ---------------------------
# 5. ÇALIŞTIR
# ---------------------------
look_back_options = [7, 14, 21, 28, 35]
forecast_horizon = 10
results = []

for lb in look_back_options:
    try:
        X_all, y_all = create_seq2seq_data(scaled_data, lb, forecast_horizon)
        mae, rmse, mape, mase = evaluate_model(X_all, y_all, lb, forecast_horizon)
        results.append((lb, mae, rmse, mape, mase))
        print(f"✅ Look_back={lb} | MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}% | MASE={mase:.2f}")
    except Exception as e:
        print(f"❌ Look_back={lb} başarısız: {e}")

# DataFrame olarak sonuçları göster
results_df = pd.DataFrame(results, columns=['look_back', 'MAE', 'RMSE', 'MAPE', 'MASE'])
print("\n🔍 En iyi sonuçlar:")
print(results_df.sort_values('RMSE'))
