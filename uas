from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import euclidean_distances

data = pd.read_excel('/content/drive/MyDrive/Semester 4/Machine Learning/Project Akhir/data_Rekrutmen.xlsx')

# Tampilkan data
print("Data:")
print(data)

# Mengubah label 'Prediksi' menjadi numerik
le = LabelEncoder()
data['Prediksi'] = le.fit_transform(data['Prediksi'])
# Mapping hasil LabelEncoder
data['Prediksi'] = data['Prediksi'].map({0: 1, 1: 0})
print(data)

# Input pengguna
k = int(input("Masukkan jumlah K: "))
Usia = int(input("Masukkan usia: "))
IPK = float(input("Masukkan IPK terakhir: "))
Skor_wawancara = float(input("Masukkan skor wawancara: "))
Pengalaman_kerja = float(input("Masukan lama pengalaman kerja: "))

# Fungsi untuk KNN
def knn_predict_stock(data, k, features):
    print("\nLangkah 1: Hitung jarak Euclidean antara data baru dan data yang ada")
    data['Distance'] = euclidean_distances(data[['Usia', 'IPK', 'Skor_Wawancara','Pengalaman_Kerja']], [features]).reshape(-1)
    print(data[['Usia', 'IPK', 'Skor_Wawancara','Pengalaman_Kerja', 'Distance']])

    print("\nLangkah 2: Urutkan data berdasarkan jarak terkecil")
    sorted_data = data.sort_values(by='Distance')
    print(sorted_data[['Usia', 'IPK', 'Skor_Wawancara','Pengalaman_Kerja', 'Distance']])

    print(f"\nLangkah 3: Ambil {k} data terdekat")
    nearest_neighbors = sorted_data.head(k)
    print(nearest_neighbors[['Usia', 'IPK', 'Skor_Wawancara','Pengalaman_Kerja','Prediksi', 'Distance']])

    print("\nLangkah 4: Prediksi berdasarkan suara terbanyak")
    prediction = nearest_neighbors['Prediksi'].mode()[0]
    print(f"Hasil prediksi: {prediction}")

    return prediction


# Prediksi menggunakan KNN
features = [Usia, IPK, Skor_wawancara,Pengalaman_kerja]
result = knn_predict_stock(data, k, features)
print(f"\nPrediksi keputusan perekrutan karyawan berdasarkan data yang diberikan adalah: {result}")
