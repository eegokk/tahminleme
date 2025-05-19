import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import random
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# Bağlantı bilgileri
username = 'ECINAR'
password = '123'
dsn = '127.0.0.1:1521/orcl'

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


# Zaman serisi verinizi hazırlayın
df['TARIH'] = pd.to_datetime(df['TARIH'])
df.set_index('TARIH', inplace=True)
series = df['SAYI'].sort_index()

# Grid Search 
results = []

for m in [4, 5, 6, 7]:
    for look_back in [5, 10, 15, 30, 60]:
        print(f"\n Testing m={m}, look_back={look_back}")

        try:
            # ARIMA modelini kur
            stepwise_model = auto_arima(series,
                                         start_p=0, start_q=0,
                                         max_p=5, max_q=5,
                                         d=1,
                                         seasonal=True,
                                         m=m,
                                         suppress_warnings=True,
                                         error_action='ignore',
                                         stepwise=True)

            model_arima = ARIMA(series, order=stepwise_model.order)
            model_arima_fit = model_arima.fit()

            fitted_values = model_arima_fit.fittedvalues
            burn_in = model_arima_fit.loglikelihood_burn
            residuals = (series[burn_in:] - fitted_values).dropna()

            scaler = StandardScaler()
            res_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))
            res_scaled = np.nan_to_num(res_scaled)

            def create_sequences(data, look_back=5):
                X, y = [], []
                for i in range(len(data) - look_back):
                    X.append(data[i:i+look_back])
                    y.append(data[i+look_back])
                return np.array(X), np.array(y)

            X, y = create_sequences(res_scaled, look_back)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            model_lstm = Sequential()
            model_lstm.add(LSTM(50, activation='tanh', input_shape=(look_back, 1)))
            model_lstm.add(Dense(1))
            model_lstm.compile(optimizer='adam', loss='mse')
            model_lstm.fit(X, y, epochs=20, verbose=0, shuffle=False)

            lstm_pred_train = model_lstm.predict(X, verbose=0).flatten()
            lstm_pred_train_inv = scaler.inverse_transform(lstm_pred_train.reshape(-1, 1)).flatten()
            fitted_cut = fitted_values[burn_in:burn_in + len(lstm_pred_train_inv)]
            hybrid_in_sample = fitted_cut + lstm_pred_train_inv

            # Gerçek ve tahmin
            actual = series[burn_in + look_back:burn_in + look_back + len(hybrid_in_sample)]
            predicted = hybrid_in_sample

            # Metrikler
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            smape = 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))

            naive_forecast = actual.shift(1).bfill()
            mase = mae / mean_absolute_error(actual, naive_forecast)

            print(f"✅ MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%, MASE: {mase:.2f}")

            results.append({
                'm': m,
                'look_back': look_back,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'SMAPE': smape,
                'MASE': mase
            })

        except Exception as e:
            print(f"❌ Hata oluştu: {e}")



# Otomatik model seçimi auto_arima
stepwise_model = auto_arima(series,
                             start_p=0, start_q=0,
                             max_p=5, max_q=5,
                             d=1,
                             seasonal=True,
                             m=5,
                             trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)
print("Otomatik seçilen ARIMA parametreleri:", stepwise_model.order)

# ARIMA modeli kurulum
model_arima = ARIMA(series, order=stepwise_model.order)
model_arima_fit = model_arima.fit()

# ARIMA tahmini ve fark (residuals) hesaplama
fitted_values = model_arima_fit.fittedvalues
burn_in = model_arima_fit.loglikelihood_burn
residuals = (series[burn_in:] - fitted_values).dropna()




# LSTM hazırlık
scaler = StandardScaler()
res_scaled = scaler.fit_transform(residuals.fillna(0).values.reshape(-1, 1))
res_scaled = np.nan_to_num(res_scaled)


def create_sequences(data, look_back=30):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

# %%
look_back = 5
X, y = create_sequences(res_scaled, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# LSTM modeli
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='tanh', input_shape=(look_back, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X, y, epochs=20, verbose=1)

# Gelecek tahmin
future_arima = model_arima_fit.forecast(steps=10)
last_res = res_scaled[-look_back:]
input_seq = last_res.reshape((1, look_back, 1))

future_residuals = []
for _ in range(10):
    pred = model_lstm.predict(input_seq, verbose=0)
    future_residuals.append(pred[0][0])
    last_res = np.append(last_res[1:], pred[0][0])
    input_seq = last_res.reshape((1, look_back, 1))

arr = np.array(future_residuals).reshape(-1, 1)
arr = np.nan_to_num(arr)
future_residuals = scaler.inverse_transform(arr).flatten()

# Geçmiş tahmin (ARIMA + LSTM)
lstm_pred_train = model_lstm.predict(X, verbose=0).flatten()
lstm_pred_train_inv = scaler.inverse_transform(lstm_pred_train.reshape(-1, 1)).flatten()
fitted_cut = fitted_values[burn_in:burn_in + len(lstm_pred_train_inv)]
hybrid_in_sample = fitted_cut[:len(lstm_pred_train_inv)] + lstm_pred_train_inv
hybrid_in_sample_index = fitted_values.index[burn_in:burn_in + len(hybrid_in_sample)]


# Gelecek hibrit tahmin
hybrid_forecast = future_arima + future_residuals
forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=10)

# Tahminleri birleştir
all_dates = hybrid_in_sample_index.append(forecast_dates)
all_forecasts = np.concatenate([hybrid_in_sample, hybrid_forecast])
all_forecasts = np.nan_to_num(all_forecasts)




# Hibrit Tahmin Grafiği
plt.figure(figsize=(14, 6))
plt.plot(series, label='Gerçek Veri', linewidth=2)
plt.plot(all_dates, all_forecasts, label='ARIMA+LSTM Tahmin (Geçmiş + Gelecek)', color='red', linewidth=2)
plt.title('ARIMA + LSTM Hibrit Tahmin')
plt.xlabel("Tarih")
plt.ylabel("SAYI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Grafik
#plt.figure(figsize=(14, 6))
#plt.plot(series, label='Gerçek Veri', linewidth=2)
#plt.plot(all_dates, all_forecasts, label='ARIMA+LSTM Tahmin (Geçmiş + Gelecek)', color='red', linewidth=2)
#plt.title('ARIMA + LSTM Hibrit Tahmin')
#plt.xlabel("Tarih")
#plt.ylabel("SAYI")
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.show()


# Gerçek ve tahmin değerlerini al (10 gün için)
gercek = series[-10:].values
tahmin = hybrid_forecast

# MAE
mae = mean_absolute_error(gercek, tahmin)

# RMSE
rmse = np.sqrt(mean_squared_error(gercek, tahmin))

# MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero = y_true != 0
    return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

mape = mean_absolute_percentage_error(gercek, tahmin)

# SMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator[denominator == 0] = 1  # sıfıra bölme engellenir
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100

smape = symmetric_mean_absolute_percentage_error(gercek, tahmin)

# MASE (Referans olarak naive tahmin: bir önceki gün = tahmin)
naive_forecast = series.shift(1).dropna()
naive_error = np.mean(np.abs(series[1:] - naive_forecast))

mase = mae / naive_error

#  Metikleri yazdır
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"SMAPE: {smape:.2f}%")
print(f"MASE: {mase:.2f}")


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='MAE')
print("\n🔍 En iyi sonuçlar:")
print(results_df.head())