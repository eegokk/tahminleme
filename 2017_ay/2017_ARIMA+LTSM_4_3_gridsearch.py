import cx_Oracle
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import random
import tensorflow as tf
#from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

series = pd.Series(np.random.randint(8000, 50000, size=100))
residuals = series.diff().dropna()

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
scaler = StandardScaler()
res_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

# LSTM için veri oluştur
def create_sequences(data, look_back=5):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)


"""look_back_list = [5, 10, 15, 30]
epoch_list = [20, 50, 100]"""
look_back_list = [75, 80, 85]
epoch_list = [1750,2000,2250, 2500,2600]
results = []
for look_back in look_back_list:
    X, y = create_sequences(res_scaled, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    for epoch in epoch_list:
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(32))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(X, y, epochs=epoch, verbose=0, shuffle=False)

y_pred = model.predict(X, verbose=0)
y_pred_inv = scaler.inverse_transform(y_pred).flatten()
y_true = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

# Hata metrikleri
mae = mean_absolute_error(y_true, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_true, y_pred_inv))
mape = np.mean(np.abs((y_true - y_pred_inv) / y_true)) * 100
naive = y_true[:-1]
mase = mae / np.mean(np.abs(y_true[1:] - naive))

results.append((look_back, epoch, mae, rmse, mape, mase))

# Sonuçları göster
df_results = pd.DataFrame(results, columns=['look_back', 'epochs', 'MAE', 'RMSE', 'MAPE', 'MASE'])
print(df_results.sort_values(by='MAE'))
