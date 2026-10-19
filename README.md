# Prediksi Pendapatan Penjualan Es Krim berdasarkan Suhu

## Latar Belakang Masalah

Penjualan es krim sangat dipengaruhi oleh kondisi cuaca, terutama suhu udara. Saat suhu meningkat, permintaan terhadap produk dingin biasanya juga meningkat. Namun, penjual seringkali kesulitan memperkirakan pendapatan harian karena mereka tidak memiliki model kuantitatif yang dapat mendukung keputusan pembelian stok dan persediaan.

Proyek ini bertujuan untuk membangun model pembelajaran mesin berbasis regresi linear sederhana untuk memprediksi pendapatan penjualan es krim berdasarkan suhu udara. Dengan model ini, penjual dapat memperkirakan pendapatan harian, menyiapkan stok yang lebih realistis, dan membuat keputusan operasional yang lebih efisien.

## Ringkasan Dataset

Dataset yang digunakan adalah dataset publik dari Kaggle dengan dua kolom utama: `Temperature` (°C) dan `Revenue` (USD). Dataset berisi 500 baris data numerik yang representasikan hubungan antara suhu udara dan pendapatan penjualan es krim.

- Sumber dataset: [Kaggle placeholder](https://www.kaggle.com/)
- Jumlah baris: 500
- Kolom utama: Temperature, Revenue
- Tipe data: numerik

## Justifikasi Pemilihan Studi Kasus dan Dataset

Studi kasus ini dipilih karena hubungan sebab-akibat antara cuaca panas dan konsumsi es krim cukup jelas dan relevan secara bisnis. Suhu udara merupakan variabel yang mudah diukur dan memiliki hubungan linear yang kuat dengan pendapatan penjualan. Dataset yang dipilih memiliki dua kolom numerik, dan pola tren umumnya bersifat linear. Oleh karena itu, regresi linear sederhana menjadi pendekatan yang tepat dan mudah diinterpretasikan untuk tugas kuliah modul Pembelajaran Mesin.

## Hasil EDA

Pada tahap eksplorasi data (EDA), dilakukan langkah-langkah berikut:

- Memuat dataset dengan pandas
- Menampilkan `head()`, `columns`, `info()`, dan `describe()`
- Mengecek missing values dan baris duplikat
- Menghapus duplikat jika ditemukan
- Mendeteksi outlier menggunakan metode IQR
- Menganalisis hubungan linear menggunakan scatter plot dan korelasi Pearson

Hasil analisis menunjukkan bahwa hubungan antara suhu dan pendapatan bersifat linear dan positif. Data yang digunakan juga berada dalam rentang yang masuk akal secara bisnis, sehingga tidak diperlukan pembuangan data secara berlebihan.

## Hasil Model

Model yang digunakan adalah `LinearRegression` dengan satu fitur, yaitu `Temperature`, dan target `Revenue`.

- Variabel X: suhu udara (°C)
- Variabel y: pendapatan penjualan (USD)
- Pembagian data: train-test split dengan `test_size=0.2` dan `random_state=42`
- Evaluasi: R² dan RMSE

Hasil model yang diperoleh akan disimpan ke file `model/metrics.json` dan dapat dibaca oleh aplikasi web. Nilai aktual ditentukan dari proses pelatihan di `train_model.py`.

### Interpretasi model

Setiap kenaikan 1°C diperkirakan meningkatkan pendapatan sekitar nilai koefisien model, yang dapat dibaca dari `metrics.json` dan ditampilkan di halaman web. Intercept merepresentasikan estimasi pendapatan saat suhu 0°C, sedangkan RMSE menunjukkan rata-rata selisih antara prediksi dan data aktual.

## Struktur Proyek

- `data/Ice_Cream.csv`
- `notebooks/eda_modeling.ipynb`
- `train_model.py`
- `model/ice_cream_model.joblib`
- `model/metrics.json`
- `static/img/`
- `templates/index.html`
- `app.py`
- `requirements.txt`
- `Procfile`
- `.gitignore`
- `README.md`

## Cara Menjalankan Lokal

1. Buat lingkungan virtual Python (opsional tetapi disarankan).
2. Instal dependensi:

```bash
pip install -r requirements.txt
```

3. Jalankan pelatihan model:

```bash
python train_model.py
```

4. Jalankan aplikasi web:

```bash
python app.py
```

5. Buka browser ke alamat:

```text
http://127.0.0.1:5000
```

## Cara Deploy ke Render

1. Buat repository GitHub baru dan upload seluruh isi proyek.
2. Masuk ke dashboard Render.
3. Pilih opsi `New` → `Web Service`.
4. Hubungkan repository GitHub.
5. Pastikan build command dan start command sudah sesuai.
6. Gunakan `Procfile` yang sudah disediakan:

```text
web: gunicorn app:app
```

7. Set `PORT` otomatis akan dibaca oleh aplikasi Flask melalui environment variable pada hosting.

## Placeholder

- Link GitHub: [Tambahkan link repository GitHub Anda](#)
- Link aplikasi: [Tambahkan link deployment aplikasi](#)
- Link video YouTube: [Tambahkan link video presentasi atau demo](#)

## Catatan Pengembangan

Proyek ini dibuat untuk tugas kuliah Pembelajaran Mesin pada modul regresi linear sederhana. Fokus utama adalah membuat solusi end-to-end, mulai dari EDA, pembuatan model, evaluasi, penyimpanan artefak, hingga deployment aplikasi web.
