import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import random
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Oracle bağlantısı
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

# Zaman serisi hazırlanması
df['TARIH'] = pd.to_datetime(df['TARIH'])  
df.set_index('TARIH', inplace=True)
series = df['SAYI'].sort_index()

# auto_arima ile en iyi parametreyi bul
auto_model = auto_arima(series, seasonal=True, m=12, stepwise=True, suppress_warnings=True)
print("Bulunan ARIMA order:", auto_model.order)

# ARIMA modeli kur
model_arima = ARIMA(series, order=auto_model.order)
model_arima_fit = model_arima.fit()

# Kalıntılar (residuals) alınır
residuals = model_arima_fit.resid

# Normalize et
scaler = MinMaxScaler()
res_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

# LSTM için veri oluştur
def create_sequences(data, look_back=5):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

look_back = 5
X, y = create_sequences(res_scaled, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# LSTM modeli kur
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X, y, epochs=20, verbose=1)

# ARIMA tahmini (gelecek 10 ay)
future_arima = model_arima_fit.forecast(steps=10)

# LSTM ile artık tahmini
last_res = res_scaled[-look_back:]
input_seq = last_res.reshape((1, look_back, 1))

future_residuals = []
for _ in range(10):
    pred = model_lstm.predict(input_seq, verbose=0)
    future_residuals.append(pred[0][0])
    last_res = np.append(last_res[1:], pred[0][0])
    input_seq = last_res.reshape((1, look_back, 1))

# Artıkları eski ölçekteki değerlere dönüştür
arr = np.array(future_residuals).reshape(-1, 1)
arr = np.nan_to_num(arr)
future_residuals = scaler.inverse_transform(arr).flatten()

# ARIMA + LSTM hibrit tahmin
hybrid_forecast = future_arima + future_residuals
hybrid_forecast = np.round(hybrid_forecast).astype(int)

#Aylık tahmin tarihleri oluştur
forecast_dates = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=10, freq='MS')

# Gerçek geçmiş tahmini yap
fitted_arima = model_arima_fit.fittedvalues
residuals_pred_scaled = model_lstm.predict(X, verbose=0)
residuals_pred = scaler.inverse_transform(residuals_pred_scaled).flatten()
hybrid_past = fitted_arima[look_back:] + residuals_pred
actual_past = series[look_back:]

# Sonuçları yazdır
print("Hybrid Forecast:", hybrid_forecast)
print("Forecast Dates:", forecast_dates)
print("ARIMA Forecast:\n", future_arima)
print("LSTM Residuals:\n", future_residuals)
# Sonuçları yazdır (tam sayı olarak)
print("TAHMİNLER :")
for date, value in zip(forecast_dates, hybrid_forecast):
    print(f"{date.strftime('%Y-%m')} → {value}")

# Hata metriklerini hesapla
mae = mean_absolute_error(actual_past, hybrid_past)
rmse = np.sqrt(mean_squared_error(actual_past, hybrid_past))
mape = np.mean(np.abs((actual_past - hybrid_past) / actual_past)) * 100
naive = actual_past.shift(1).dropna()
mase = mae / np.mean(np.abs(actual_past[1:] - naive))

print(f"\n MAE  : {mae:.2f}")
print(f" RMSE : {rmse:.2f}")
print(f" MAPE : {mape:.2f}%")
print(f" MASE : {mase:.2f}")


# Grafikle göster
plt.figure(figsize=(12, 6))
plt.plot(series, label='Gerçek Veri', linewidth=2)
plt.plot(actual_past.index, hybrid_past, label='Geçmiş Tahmin (ARIMA+LSTM)', color='orange')  # 
plt.plot(forecast_dates, hybrid_forecast, label='Gelecek Tahmin (ARIMA+LSTM)', color='red', linewidth=2)
plt.title('ARIMA + LSTM Hibrit Tahmin (Geçmiş ve 10 Aylık Gelecek)')
plt.xlabel("Tarih")
plt.ylabel("Sayı")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()