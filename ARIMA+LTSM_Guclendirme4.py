import os
import random
import cx_Oracle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima

# Sabit tohumlar (reproducibility)
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED) #hash tabanlı işlemlerindeki rastgeleliği sabitler.
np.random.seed(SEED) #NumPy kütüphanesindeki tüm rastgele işlemleri sabitler
random.seed(SEED) #Python’un yerleşik random modülündeki rastgele işlemleri sabitler.
tf.random.set_seed(SEED) #TensorFlow içindeki rastgeleliği kontrol eder.

# Oracle bağlantısı
username, password, dsn = 'ECINAR', '123', '127.0.0.1:1521/orcl'
try:
    with cx_Oracle.connect(username, password, dsn) as connection:
        df = pd.read_sql("SELECT * FROM ECINAR.YK_GGD_SAYI", con=connection)
        print("Veri çekildi ✅")
except Exception as e:
    print("Veritabanı bağlantı hatası:", e)
    raise

# Zaman serisi hazırlığı
df['TARIH'] = pd.to_datetime(df['TARIH'])
df.set_index('TARIH', inplace=True)
series = df['SAYI'].sort_index()

# Grid search
results = []
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

for m in [4, 5, 6, 7]:
    for look_back in [5, 10, 15, 30, 60]:
        try:
            model_order = auto_arima(series, d=1, seasonal=True, m=m, stepwise=True, suppress_warnings=True).order
            model_fit = ARIMA(series, order=model_order).fit()
            residuals = (series[model_fit.loglikelihood_burn:] - model_fit.fittedvalues).dropna()

            scaler = StandardScaler()
            res_scaled = np.nan_to_num(scaler.fit_transform(residuals.values.reshape(-1, 1)))
            X, y = create_sequences(res_scaled, look_back)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            lstm_model = Sequential([
                LSTM(50, activation='tanh', input_shape=(look_back, 1)),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X, y, epochs=20, verbose=0, shuffle=False)

            lstm_pred = lstm_model.predict(X, verbose=0).flatten()
            lstm_pred_inv = scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()

            fitted_cut = model_fit.fittedvalues[model_fit.loglikelihood_burn:model_fit.loglikelihood_burn + len(lstm_pred_inv)]
            hybrid = fitted_cut + lstm_pred_inv
            actual = series[model_fit.loglikelihood_burn + look_back:model_fit.loglikelihood_burn + look_back + len(hybrid)]

            mae = mean_absolute_error(actual, hybrid)
            rmse = np.sqrt(mean_squared_error(actual, hybrid))
            mape = np.mean(np.abs((actual - hybrid) / actual)) * 100
            smape = 100 * np.mean(2 * np.abs(hybrid - actual) / (np.abs(actual) + np.abs(hybrid)))
            mase = mae / mean_absolute_error(actual, actual.shift(1).bfill())

            results.append({"m": m, "look_back": look_back, "MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape, "MASE": mase})

        except Exception as e:
            print(f"Hata (m={m}, look_back={look_back}): {e}")

# 🔽 Grid Search sonuçlarına göre en iyi modeli al
results_df = pd.DataFrame(results).sort_values(by='MAE')
best_model = results_df.loc[results_df['MAE'].idxmin()]
best_m = int(best_model['m'])
best_look_back = int(best_model['look_back'])

print(f"\n🎯 En İyi Model - m={best_m}, look_back={best_look_back}")

# 🔁 ARIMA + LSTM modelini best parametrelerle tekrar eğit
auto_model = auto_arima(series, d=1, seasonal=True, m=best_m, stepwise=True, suppress_warnings=True)
model_fit = ARIMA(series, order=auto_model.order).fit()

# Artıkları çıkar ve ölçekle
residuals = (series[model_fit.loglikelihood_burn:] - model_fit.fittedvalues).dropna()
scaler = StandardScaler()
res_scaled = np.nan_to_num(scaler.fit_transform(residuals.values.reshape(-1, 1)))

# LSTM input dizilerini oluştur
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

X, y = create_sequences(res_scaled, best_look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# LSTM modelini eğit
model_lstm = Sequential([
    LSTM(50, activation='tanh', input_shape=(best_look_back, 1)),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X, y, epochs=20, verbose=0)

# Tahmin üret
lstm_pred = model_lstm.predict(X, verbose=0).flatten()
lstm_pred_inv = scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()

# ARIMA + LSTM Hibrit tahmin üret
fitted_cut = model_fit.fittedvalues[model_fit.loglikelihood_burn:model_fit.loglikelihood_burn + len(lstm_pred_inv)]
hybrid_in_sample = fitted_cut + lstm_pred_inv
hybrid_index = fitted_cut.index

# Gerçek değerlerle karşılaştırma
true_values = series[hybrid_index]

# 🎨 Grafik çizimi
plt.figure(figsize=(12, 6))
plt.plot(true_values.index, true_values.values, label='Gerçek', linewidth=2)
plt.plot(hybrid_index, hybrid_in_sample, label='Best Model (ARIMA + LSTM)',  linewidth=2, color='red')
plt.title(f'Best ARIMA+LSTM Tahmini (m={best_m}, look_back={best_look_back})')
plt.xlabel('Tarih')
plt.ylabel('Sayı')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 🔮 Gelecek 10 gün için tahmin (forecast)
future_arima = model_fit.forecast(steps=10)

# LSTM future tahminleri (artıklar için)
last_res = res_scaled[-best_look_back:]
input_seq = last_res.reshape((1, best_look_back, 1))

future_residuals = []
for _ in range(10):
    pred = model_lstm.predict(input_seq, verbose=0)
    future_residuals.append(pred[0][0])
    last_res = np.append(last_res[1:], pred[0][0])
    input_seq = last_res.reshape((1, best_look_back, 1))

# Artıkları inverse et
arr = np.array(future_residuals).reshape(-1, 1)
arr = np.nan_to_num(arr)
future_residuals = scaler.inverse_transform(arr).flatten()

# Hibrit gelecek tahmini (ARIMA + LSTM artık)
hybrid_forecast = future_arima + future_residuals

# Tarihler
forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=10)

# 🔗 Tüm tahminleri birleştir (in-sample + out-of-sample)
all_dates = hybrid_index.append(forecast_dates)
all_forecasts = np.concatenate([hybrid_in_sample, hybrid_forecast])

# 🎨 GRAFİK: Gerçek + Tahmin (hem geçmiş hem gelecek)
plt.figure(figsize=(14, 6))
plt.plot(series, label='Gerçek Veri', linewidth=2)
plt.plot(all_dates, all_forecasts, label='ARIMA+LSTM Tahmin', color='red', linewidth=2)
plt.axvline(x=series.index[-1], color='gray', linestyle='--', label='Gelecek Başlangıcı')
plt.title(f'ARIMA + LSTM Hibrit Tahmin (m={best_m}, look_back={best_look_back})')
plt.xlabel("Tarih")
plt.ylabel("Sayı")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 📊 Performans metrikleri (geçmiş veriye göre)
mae = mean_absolute_error(true_values, hybrid_in_sample)
rmse = np.sqrt(mean_squared_error(true_values, hybrid_in_sample))
mape = np.mean(np.abs((true_values - hybrid_in_sample) / true_values)) * 100
smape = 100 * np.mean(2 * np.abs(hybrid_in_sample - true_values) / (np.abs(true_values) + np.abs(hybrid_in_sample)))
mase = mae / mean_absolute_error(true_values, true_values.shift(1).bfill())

# 📢 Yazdır
print("\n📈 Best Model Performans Metrikleri (In-Sample):")
print(f"MAE   : {mae:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"MAPE  : {mape:.2f}%")
print(f"SMAPE : {smape:.2f}%")
print(f"MASE  : {mase:.2f}")

# 🔮 Gelecek tahmin çıktısını yazdır
forecast_df = pd.DataFrame({
    'Tarih': forecast_dates,
    'Tahmin': hybrid_forecast
})
print("\n📅 Gelecek 10 Günlük Tahmin:")
print(forecast_df.to_string(index=False))

# 📊 Performans metriklerini hesapla
mae = mean_absolute_error(true_values, hybrid_in_sample)
rmse = np.sqrt(mean_squared_error(true_values, hybrid_in_sample))
mape = np.mean(np.abs((true_values - hybrid_in_sample) / true_values)) * 100
smape = 100 * np.mean(2 * np.abs(hybrid_in_sample - true_values) / (np.abs(true_values) + np.abs(hybrid_in_sample)))
mase = mae / mean_absolute_error(true_values, true_values.shift(1).bfill())

# 📢 Ekrana yazdır
print("\n📈 Best Model Performans Metrikleri:")
print(f"MAE   : {mae:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"MAPE  : {mape:.2f}%")
print(f"SMAPE : {smape:.2f}%")
print(f"MASE  : {mase:.2f}")
