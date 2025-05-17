# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cx_Oracle
import pandas as pd

# Bağlantı bilgileri
username = 'ECINAR'  # Veritabanı kullanıcı adınız
password = '123'  # Veritabanı şifreniz
dsn = '127.0.0.1:1521/orcl'  # Veritabanı bağlantı adresi (localhost, port ve service name)

try:
    # Oracle veritabanına bağlantı
    connection = cx_Oracle.connect(username, password, dsn)
    print("Bağlantı başarılı ✅")

    # Bağlantıyı kontrol etmek için bir sorgu çalıştıralım
    cursor = connection.cursor()
    cursor.execute("SELECT sysdate  FROM dual")  # Burada tablo_adi yerine veritabanınızdaki bir tabloyu yazın
    for row in cursor:
        print(row)

    # Bağlantıyı kapat
    cursor.close()
    connection.close()
    
except cx_Oracle.DatabaseError as e:
    print("Veritabanı bağlantı hatası:", e)
    
    
    
    
