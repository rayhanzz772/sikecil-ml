### 4.4.1 Implementasi Evaluasi Model Menggunakan LOOCV

Evaluasi model merupakan tahap penting dalam sistem prediksi pertumbuhan anak untuk menentukan model terbaik yang akan digunakan dalam menghasilkan prediksi. Pada penelitian ini, metode evaluasi yang digunakan adalah **Leave-One-Out Cross-Validation (LOOCV)**. Pemilihan LOOCV didasarkan pada karakteristik data yang dimiliki, yaitu jumlah data poin yang relatif kecil (umumnya 2–12 data poin per anak). Dengan jumlah data yang terbatas, metode evaluasi seperti *k-fold cross-validation* konvensional tidak optimal karena setiap fold akan memiliki sampel yang sangat sedikit. LOOCV menjadi pilihan yang tepat karena memaksimalkan penggunaan data latih pada setiap iterasi.

#### A. Implementasi Kode LOOCV

Backend menggunakan `LeaveOneOut` dari library scikit-learn. Pada setiap iterasi, sistem membentuk data latih dan data uji, melatih model menggunakan data latih, kemudian menghasilkan prediksi terhadap data uji. Nilai prediksi dari seluruh iterasi dikumpulkan sebagai dasar perhitungan metrik evaluasi. Berikut adalah potongan kode implementasi LOOCV pada file `services/model_service.py`:

```python
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_models(trained_models, X, y):
    metrics = {}
    loo = LeaveOneOut()

    for m in trained_models:
        if m is None:
            continue

        name   = m["name"]
        m_type = m["type"]

        y_true = []
        y_pred = []
        loocv_failed = False

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            try:
                # Latih ulang model pada setiap fold
                model = BayesianRidge()
                model.fit(X_train, y_train)
                pred = model.predict(X_test)[0]
            except Exception:
                loocv_failed = True
                break

            y_true.append(y_test[0])
            y_pred.append(pred)

        # Hitung metrik agregat setelah seluruh iterasi selesai
        if not loocv_failed and len(y_true) > 1:
            mae  = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2   = r2_score(y_true, y_pred)

            metrics[name] = {
                "mae":  round(mae, 6),
                "rmse": round(rmse, 6),
                "r2":   round(r2, 6)
            }

    return metrics
```

Penjelasan alur kode di atas:

1. **Inisialisasi LOOCV** — Objek `LeaveOneOut()` dibuat untuk membagi data menjadi *n* fold, di mana *n* adalah jumlah data poin.
2. **Iterasi per model** — Sistem melakukan loop terhadap seluruh model yang berhasil dilatih. Model yang bernilai `None` (gagal *fitting*) dilewati.
3. **Pembagian data per fold** — Method `loo.split(X)` menghasilkan indeks `train_index` dan `test_index` untuk setiap iterasi. Pada setiap fold, *n*-1 data poin menjadi data latih dan 1 data poin menjadi data uji.
4. **Pelatihan ulang model** — Model dilatih ulang (*re-train*) dari awal pada setiap fold menggunakan data latih yang baru. Hal ini memastikan evaluasi yang objektif tanpa *data leakage*.
5. **Pengumpulan prediksi** — Nilai prediksi (`pred`) dan nilai aktual (`y_test[0]`) dikumpulkan ke dalam list `y_pred` dan `y_true`.
6. **Perhitungan metrik** — Setelah seluruh *n* iterasi selesai, tiga metrik (MAE, RMSE, R²) dihitung secara agregat dari seluruh pasangan prediksi-aktual.
7. **Penanganan kegagalan** — Jika model gagal pada salah satu fold (misalnya *curve_fit* tidak konvergen), flag `loocv_failed` diset `True` dan model tersebut tidak diikutsertakan dalam hasil evaluasi.

#### B. Mekanisme LOOCV

LOOCV bekerja dengan mekanisme sebagai berikut: pada setiap iterasi, satu data poin dikeluarkan sebagai data uji (*test set*), sementara sisanya digunakan sebagai data latih (*training set*). Proses ini diulang sebanyak *n* kali, di mana *n* adalah jumlah total data poin. Dengan demikian, setiap data poin mendapat kesempatan tepat satu kali menjadi data uji. Tabel 4.10 mengilustrasikan mekanisme pembagian data LOOCV pada contoh data dengan 5 data poin.

**Tabel 4.10** Ilustrasi Pembagian Data LOOCV (*n* = 5)

| Iterasi | Data Latih (Training Set)                          | Data Uji (Test Set)   |
|:-------:|:---------------------------------------------------|:---------------------:|
| 1       | (2, 52.1), (3, 55.0), (4, 57.8), (5, 60.5)        | (1, 50.2)             |
| 2       | (1, 50.2), (3, 55.0), (4, 57.8), (5, 60.5)        | (2, 52.1)             |
| 3       | (1, 50.2), (2, 52.1), (4, 57.8), (5, 60.5)        | (3, 55.0)             |
| 4       | (1, 50.2), (2, 52.1), (3, 55.0), (5, 60.5)        | (4, 57.8)             |
| 5       | (1, 50.2), (2, 52.1), (3, 55.0), (4, 57.8)        | (5, 60.5)             |

Pada Tabel 4.10, setiap baris menunjukkan satu iterasi LOOCV. Format data ditampilkan sebagai pasangan (usia dalam bulan, tinggi badan dalam cm). Pada setiap iterasi, model dilatih ulang menggunakan *n* - 1 data poin dan diuji pada 1 data poin yang tersisa.

#### C. Metrik Evaluasi

Setelah seluruh iterasi LOOCV selesai, tiga metrik evaluasi dihitung secara agregat. Tabel 4.11 menjelaskan metrik yang digunakan beserta formulanya.

**Tabel 4.11** Metrik Evaluasi Model

| No | Metrik | Nama Lengkap                    | Formula                                                        | Interpretasi                          |
|:--:|:------:|:--------------------------------|:---------------------------------------------------------------|:--------------------------------------|
| 1  | MAE    | Mean Absolute Error             | $MAE = \frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$         | Rata-rata kesalahan absolut (cm)      |
| 2  | RMSE   | Root Mean Squared Error         | $RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Akar rata-rata kesalahan kuadrat (cm) |
| 3  | R²     | Coefficient of Determination    | $R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Proporsi variansi yang dijelaskan     |

Keterangan:
- $y_i$ = nilai aktual pada iterasi ke-*i*
- $\hat{y}_i$ = nilai prediksi pada iterasi ke-*i*
- $\bar{y}$ = rata-rata dari seluruh nilai aktual
- *n* = jumlah total iterasi (sama dengan jumlah data poin)

Berikut adalah potongan kode perhitungan metrik pada sistem:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
```

MAE dan RMSE memiliki satuan yang sama dengan variabel target (cm untuk tinggi badan, kg untuk berat badan), sehingga dapat diinterpretasikan langsung sebagai besaran kesalahan prediksi. RMSE memberikan penalti lebih besar terhadap error yang besar dibandingkan MAE karena penggunaan kuadrat pada perhitungannya. Nilai R² berkisar antara -∞ hingga 1, di mana semakin mendekati 1 menunjukkan model semakin baik dalam menjelaskan variansi data. Nilai R² negatif mengindikasikan bahwa model memiliki performa lebih buruk daripada model konstan yang hanya memprediksi rata-rata.

#### D. Pemilihan Model Terbaik

Setelah metrik evaluasi diperoleh untuk seluruh model, sistem melakukan pemilihan model terbaik. Berikut adalah potongan kode fungsi perankingan model pada file `services/model_service.py`:

```python
def get_sorted_models(trained_models, metrics):
    valid_models = [
        m for m in trained_models
        if m is not None and m["name"] in metrics
    ]

    if not valid_models:
        raise ValueError("Tidak ada model yang berhasil dilatih.")

    # Urutkan: RMSE ascending, lalu R² descending sebagai tie-breaker
    valid_models.sort(
        key=lambda m: (
            metrics[m["name"]]["rmse"],
            -metrics[m["name"]]["r2"]
        )
    )

    return valid_models
```

Kriteria perankingan yang digunakan:

1. **Kriteria utama**: RMSE terendah (*ascending*) — dipilih karena sensitif terhadap error besar yang tidak diinginkan pada konteks prediksi pertumbuhan anak.
2. **Kriteria sekunder** (*tie-breaker*): R² tertinggi (*descending*) — digunakan ketika dua model memiliki RMSE yang sama.

Model dengan RMSE terendah dianggap sebagai model terbaik karena mengindikasikan prediksi yang paling mendekati nilai aktual. Selain berdasarkan metrik evaluasi, sistem juga melakukan validasi tambahan berupa pengecekan realisme pertumbuhan (*growth realism check*). Jika prediksi dari model peringkat pertama dinilai tidak realistis secara biologis (misalnya penurunan tinggi badan atau pertumbuhan yang terlalu ekstrem), sistem akan beralih ke model peringkat selanjutnya.

#### E. Alur Evaluasi Secara Keseluruhan

Secara ringkas, alur evaluasi model pada sistem ini dapat dirangkum pada Tabel 4.12.

**Tabel 4.12** Ringkasan Alur Evaluasi Model

| Langkah | Proses                                  | Output                                      |
|:-------:|:----------------------------------------|:--------------------------------------------|
| 1       | Terima array X dan y dari tahap preprocessing | Data siap evaluasi                     |
| 2       | Inisialisasi `LeaveOneOut()` sebanyak *n* iterasi | Pembagian indeks train/test per iterasi |
| 3       | Pada setiap iterasi, latih ulang model menggunakan *n*-1 data poin | Model terlatih per iterasi |
| 4       | Prediksi nilai target pada 1 data uji per iterasi | List pasangan (y_aktual, y_prediksi) |
| 5       | Hitung metrik agregat (MAE, RMSE, R²) | Skor evaluasi per model                     |
| 6       | Ranking model via `get_sorted_models()` | Daftar model terurut dari terbaik           |
| 7       | Validasi realisme prediksi (*growth realism check*) | Model final untuk prediksi        |

#### F. Kelebihan dan Keterbatasan LOOCV

Tabel 4.13 menyajikan kelebihan dan keterbatasan metode LOOCV yang diterapkan pada sistem ini.

**Tabel 4.13** Kelebihan dan Keterbatasan LOOCV

| Aspek        | Kelebihan                                                                 | Keterbatasan                                                              |
|:-------------|:--------------------------------------------------------------------------|:--------------------------------------------------------------------------|
| Bias         | Bias estimasi rendah karena data latih hampir selengkap dataset asli (*n*-1 dari *n*) | — |
| Penggunaan data | Memaksimalkan penggunaan data; cocok untuk dataset kecil | — |
| Determinisme | Hasil bersifat deterministik (tidak ada random split) | — |
| Komputasi    | — | Memerlukan *n* kali pelatihan model per model yang dievaluasi |
| Variansi     | — | Variansi estimasi cenderung lebih tinggi karena overlap *n*-2 data poin antar fold |

Pada konteks penelitian ini, keterbatasan komputasi LOOCV tidak menjadi masalah karena jumlah data poin per anak sangat kecil (maksimal 12 poin), sehingga evaluasi dapat dilakukan secara instan pada setiap request API.
