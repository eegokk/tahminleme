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

# Son model ve tahmin
auto_model = auto_arima(series, d=1, seasonal=True, m=5, stepwise=True, trace=True)
model_fit = ARIMA(series, order=auto_model.order).fit()
residuals = (series[model_fit.loglikelihood_burn:] - model_fit.fittedvalues).dropna()
scaler = StandardScaler()
res_scaled = np.nan_to_num(scaler.fit_transform(residuals.values.reshape(-1, 1)))
look_back = 5
X, y = create_sequences(res_scaled, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

model_lstm = Sequential([
    LSTM(50, activation='tanh', input_shape=(look_back, 1)),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X, y, epochs=20, verbose=0)

future_arima = model_fit.forecast(steps=10)
last_res = res_scaled[-look_back:]
input_seq = last_res.reshape((1, look_back, 1))

future_residuals = []
for _ in range(10):
    pred = model_lstm.predict(input_seq, verbose=0)
    future_residuals.append(pred[0][0])
    last_res = np.append(last_res[1:], pred[0][0])
    input_seq = last_res.reshape((1, look_back, 1))

future_residuals = scaler.inverse_transform(np.array(future_residuals).reshape(-1, 1)).flatten()
hybrid_forecast = future_arima + future_residuals

forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=10)
lstm_pred = model_lstm.predict(X, verbose=0).flatten()
lstm_pred_inv = scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
fitted_cut = model_fit.fittedvalues[model_fit.loglikelihood_burn:model_fit.loglikelihood_burn + len(lstm_pred_inv)]

hybrid_in_sample = fitted_cut + lstm_pred_inv
hybrid_index = fitted_cut.index

all_dates = hybrid_index.append(forecast_dates)
all_forecasts = np.concatenate([hybrid_in_sample, hybrid_forecast])

plt.figure(figsize=(14, 6))
plt.plot(series, label='Gerçek Veri', linewidth=2)
plt.plot(all_dates, all_forecasts, label='ARIMA+LSTM Tahmin', color='red', linewidth=2)
plt.title('ARIMA + LSTM Hibrit Tahmin')
plt.xlabel("Tarih")
plt.ylabel("SAYI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Performans metrikleri
actual = series[-10:].values
mae = mean_absolute_error(actual, hybrid_forecast)
rmse = np.sqrt(mean_squared_error(actual, hybrid_forecast))
mape = np.mean(np.abs((actual - hybrid_forecast) / actual)) * 100
smape = 100 * np.mean(2 * np.abs(hybrid_forecast - actual) / (np.abs(hybrid_forecast) + np.abs(actual)))
mase = mae / mean_absolute_error(series[1:], series.shift(1).dropna())

print(f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%\nSMAPE: {smape:.2f}%\nMASE: {mase:.2f}")

results_df = pd.DataFrame(results).sort_values(by='MAE')
print("\n🔍 En iyi sonuçlar:")
print(results_df.head())