# Alur Aplikasi Machine Learning Antropometri (Prediksi Stunting)

Aplikasi ini adalah sebuah REST API berbasis Flask yang bertugas untuk memprediksi tinggi badan balita di masa depan dan menentukan status stunting (berdasarkan standar WHO) dari data historis tinggi badannya.

Berikut adalah alur singkat cara kerja aplikasi ini:

## 1. Menerima Request (Entry Point & Route)
- **`app.py`**: Berfungsi sebagai _entry point_ (titik awal) yang menginisialisasi server Flask dan mendaftarkan *blueprint* routing.
- **`routes/prediction_route.py`**: Menyediakan endpoint `POST /api/predict`. Endpoint ini menerima data JSON dari pengguna berupa:
  - `sex` (jenis kelamin)
  - `history` (data historis usia dan tinggi badan balita)
  - `horizon` (jumlah bulan ke depan yang ingin diprediksi, default 6).

## 2. Preprocessing Data
- Memvalidasi input dari user.
- **`services/preprocessing_service.py`**: Mengubah data `history` menjadi fitur _X_ (usia) dan target _y_ (tinggi badan) yang siap digunakan untuk melatih model Machine Learning.

## 3. Training & Evaluasi Multi-Model
- **`services/model_service.py`**: Aplikasi tidak menggunakan satu model statis, melainkan melakukan _training_ **4 model sekaligus** secara *on-the-fly* menggunakan data historis balita tersebut:
  1. Linear Regression
  2. Polynomial Regression Degree 2
  3. Polynomial Regression Degree 3
  4. Gompertz Growth Model
- Model dievaluasi menggunakan metode *Leave-One-Out Cross-Validation (LOOCV)* untuk mendapatkan metrik *error* (RMSE, MAE, R²).
- Model-model tersebut kemudian diurutkan dari yang memiliki tingkat akurasi paling tinggi (RMSE terkecil) hingga yang terendah.

## 4. Prediksi & Pengecekan Kewajaran (Biological Realism)
- **`services/prediction_service.py`**: Aplikasi akan mencoba melakukan prediksi secara rekursif (`recursive_predict`) menggunakan **model terbaik**.
- Hasil prediksi kemudian digabungkan dengan tabel standar pertumbuhan WHO (`who_lms.csv`) melalui **`services/who_service.py`** untuk menghitung **Z-score (HAZ)** dan menentukan klasifikasi status (misal: "Normal", "Severely Stunted").
- **Validasi Biologis**: Prediksi diuji kewajarannya (misal: tinggi tidak boleh menurun, Z-score tidak boleh ekstrem, kurva tidak berbalik arah). Jika prediksi dari model terbaik dianggap "tidak realistis", aplikasi akan turun (fallback) menggunakan model terbaik kedua, dan seterusnya. Jika semuanya tidak wajar, digunakan regresi linear sebagai fallback terakhir.

## 5. Mengembalikan Response (Output)
- Setelah model terbaik dan paling masuk akal terpilih, aplikasi mengembalikan respons JSON berisi:
  - Model yang digunakan (`selected_model`)
  - Metrik evaluasi semua model (`metrics`)
  - Hasil prediksi lengkap tiap bulannya (usia, estimasi tinggi, HAZ, dan status stunting).
