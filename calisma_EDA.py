# -*- coding: utf-8 -*-
"""
Created on Mon May 12 22:47:12 2025

@author: cinar
"""

import cx_Oracle
import pandas as pd

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
    query = "SELECT * FROM ECINAR.YK_GGD_DM"
    cursor.execute(query)

    # Sütun adlarını al
    columns = [col[0] for col in cursor.description]

    # Verileri al
    data = cursor.fetchall()

    # DataFrame'e dönüştür
    df = pd.DataFrame(data, columns=columns)

    # Sonuçları yazdır
    #print("Veriler DataFrame olarak alındı:")
    print(df)
    #print(df.head())  # İlk 5 satır
    #print(df.columns)         # Sütun isimlerini göster
    print(df.iloc[:, :2])  
    print(df.dtypes)
    # Eksik değer kontrolü
    print(df.isnull().sum())

    # Bağlantıyı kapat
    cursor.close()
    connection.close()

except cx_Oracle.DatabaseError as e:
    print("Veritabanı bağlantı hatası:", e)
    
    
    
    
import matplotlib.pyplot as plt
import seaborn as sns

df['TARIH'] = pd.to_datetime(df['TARIH'])
df.set_index('TARIH', inplace=True)


# Günlük dönüş grafiği
plt.figure(figsize=(12,6))
sns.lineplot(data=df, x=df.index, y='SAYI')
plt.title("Günlük Gönüllü Geri Dönüş Sayısı")
plt.ylabel("Geri Dönüş")
plt.xlabel("TARIH")
plt.show()

# Haftalık
haftalik = df['SAYI'].resample('W').sum()

# Aylık
aylik = df['SAYI'].resample('M').sum()

# Görselleştir
plt.figure(figsize=(12,6))
aylik.plot()
plt.title("Aylık Geri Dönüş Sayısı")
plt.xlabel("Ay")
plt.ylabel("Toplam Geri Dönüş")
plt.show()


#YAŞ
plt.figure(figsize=(10,5))
sns.histplot(df['YAS'], kde=True)
plt.title("Yaş Dağılımı")
plt.xlabel("YAS")
plt.ylabel("Frekans")
plt.show()


#CINSIYET
sns.countplot(x='CINSIYET', data=df)
plt.title("Cinsiyet Dağılımı")
plt.show()


#ÖĞRENİM DURUMU
sns.countplot(y='OGRENIMDURUM', data=df, order=df['OGRENIMDURUM'].value_counts().index)
plt.title("Öğrenim Durumu Dağılımı")
plt.show()


#YAŞ-GERİ DÖNÜŞ İLİŞKİSİ
sns.scatterplot(x='YAS', y='SAYI', data=df)
plt.title("Yaş ve Geri Dönüş Sayısı İlişkisi")
plt.show()

#CINSIYET KIRILIMI
sns.boxplot(x='CINSIYET', y='SAYI', data=df)
plt.title("Cinsiyete Göre Geri Dönüş Sayısı Dağılımı")
plt.show()

#KORELASYON MATRİSİ
sayisal_df = df[['yas', 'dogum_yili', 'geri_donus_sayisi']]
corr = sayisal_df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()


#ÖZELLİK DÖNÜŞÜMLERİ
sayisal_df = df[['yas', 'dogum_yili', 'SAYI']]
corr = sayisal_df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()

