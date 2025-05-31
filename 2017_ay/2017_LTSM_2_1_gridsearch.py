import cx_Oracle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import random
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Bağlantı bilgileri
username = 'ECINAR' 
password = '123'    
dsn = '127.0.0.1:1521/orcl'  

try:
    # Oracle veritabanına bağlantı
    connection = cx_Oracle.connect(username, password, dsn)
    print("Bağlantı başarılı ✅")

    cursor = connection.cursor()
    query = "SELECT * FROM ECINAR.YK_GGD_AYLIK"
    cursor.execute(query)

    columns = [col[0] for col in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)

    print(df.columns)         
    print(df.iloc[:, :2])  

    cursor.close()
    connection.close()

except cx_Oracle.DatabaseError as e:
    print("Veritabanı bağlantı hatası:", e)

# Zaman serisi verinizi hazırlayın
df['TARIH'] = pd.to_datetime(df['TARIH'])  
df.set_index('TARIH', inplace=True)     
series = df['SAYI'].sort_index()
df.rename(columns={'SAYI': 'geri_donus_sayisi'}, inplace=True)   

# Veriyi normalize et
scaler = MinMaxScaler()
scaled_series = scaler.fit_transform(series.values.reshape(-1, 1))


# LSTM için veri oluştur
def create_sequences(data, look_back=5):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)


#$look_back_list = [3, 6, 12]
#epoch_list = [10, 20, 50]
#look_back_list = [12, 18, 24]
#epoch_list = [50, 75, 100]
#look_back_list = [24,36, 48]
#epoch_list = [50, 75, 100]
#look_back_list = [30,36, 48]
#epoch_list = [100,125,150]
#look_back_list = [25,30, 35]
#epoch_list = [100,125,150]
#look_back_list = [30,31,32,33,34,35]
#epoch_list = [100,125,150]
look_back_list = [35,40,45]
epoch_list = [150,175,200]


results = []

for look_back in look_back_list:
    X, y = create_sequences(scaled_series, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    for epochs in epoch_list:
        model_lstm = Sequential()
        model_lstm.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer='adam', loss='mse')
        model_lstm.fit(X, y, epochs=epochs, verbose=1)

        # GEçmiş tahmini
        train_predictions = model_lstm.predict(X, verbose=0)
        train_predictions = scaler.inverse_transform(train_predictions)
        y_true = scaler.inverse_transform(y.reshape(-1, 1))
        
        # Hata Metrikleri
        mae = mean_absolute_error(y_true, train_predictions)
        rmse = np.sqrt(mean_squared_error(y_true, train_predictions))
        mape = np.mean(np.abs((y_true - train_predictions) / y_true)) * 100
        mase = mae / np.mean(np.abs(np.diff(y_true.flatten())))  
        
        results.append({
            'look_back': look_back,
            'epochs': epochs,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'MASE': mase
        })

# Sonuçları DataFrame olarak yazdır
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='MASE')
print("\nGrid Search Sonuçları (look_back vs epochs):")
print(results_df)