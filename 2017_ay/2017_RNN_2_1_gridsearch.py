import cx_Oracle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

# Oracle bağlantısı
username = 'ECINAR'
password = '123'
dsn = '127.0.0.1:1521/orcl'

try:
    connection = cx_Oracle.connect(username, password, dsn)
    cursor = connection.cursor()
    query = "SELECT * FROM ECINAR.YK_GGD_AYLIK"
    cursor.execute(query)
    columns = [col[0] for col in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)
    cursor.close()
    connection.close()
    print("✅ Veritabanı bağlantısı başarılı.")
except cx_Oracle.DatabaseError as e:
    print("❌ Veritabanı bağlantı hatası:", e)

# Zaman serisi hazırlığı
df['TARIH'] = pd.to_datetime(df['TARIH'])
df.set_index('TARIH', inplace=True)
series = df['SAYI'].sort_index()

# Eğitim/Test ayrımı (örneğin son 12 ay test)
test_size = 12
train_series = series[:-test_size]
test_series = series[-test_size:]

# Normalizasyon
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
test_values = test_series.values

# Sequence oluşturucu
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Grid Search
#look_back_options = [6, 12, 18, 24]   --18
#epoch_options = [50, 100, 150]   --100
#look_back_options = [18, 20, 30, 40]  #18
#epoch_options = [100, 150, 200]  #200
#look_back_options = [15, 17,18, 20]  #17
#epoch_options = [200, 225, 250]  #225
#look_back_options = [18, 30,36]  #36
#epoch_options = [150,200, 225, 250]  #150
look_back_options = [18, 20, 30, 40]  #18
epoch_options = [150,200, 225, 250]  #225
results = []

for look_back in look_back_options:
    X_train, y_train = create_sequences(train_scaled, look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))

    for epoch in epoch_options:
        model = Sequential()
        model.add(SimpleRNN(50, activation='tanh', input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epoch, verbose=0, callbacks=[early_stop])

        # Tahmin
        last_seq = train_scaled[-look_back:]
        forecast_scaled = []

        for _ in range(test_size):
            input_seq = last_seq.reshape(1, look_back, 1)
            pred = model.predict(input_seq, verbose=0)[0][0]
            forecast_scaled.append(pred)
            last_seq = np.append(last_seq[1:], [[pred]], axis=0)

        forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

        # Hata metrikleri
        mae = mean_absolute_error(test_values, forecast)
        rmse = np.sqrt(mean_squared_error(test_values, forecast))
        mape = np.mean(np.abs((test_values - forecast) / test_values)) * 100
        naive = test_series.shift(1).dropna()
        mase = mae / np.mean(np.abs(test_values[1:] - naive.values))

        results.append({
            'look_back': look_back,
            'epoch': epoch,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'MASE': mase
        })
        print(f" look_back={look_back}, epoch={epoch} | MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, MASE={mase:.2f}")

# En iyi sonucu yazdır
best_result = min(results, key=lambda x: x['MAE'])
print("\nEn iyi kombinasyon (MAE bazlı):")
print(f"look_back = {best_result['look_back']}, epoch = {best_result['epoch']}")
print(f"MAE  : {best_result['MAE']:.2f}")
print(f"RMSE : {best_result['RMSE']:.2f}")
print(f"MAPE : {best_result['MAPE']:.2f}%")
print(f"MASE : {best_result['MASE']:.2f}")
