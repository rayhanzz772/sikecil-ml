### 4.4.2 Perbandingan Model Prediksi

Untuk memvalidasi pemilihan model pada sistem, dilakukan evaluasi perbandingan (*head-to-head comparison*) terhadap beberapa algoritma machine learning. Evaluasi ini bertujuan untuk membandingkan performa model konvensional (Linear Regression dan Polynomial Regression) dengan model yang diintegrasikan pada sistem akhir (Gaussian Process Regression dengan WHO Prior). Pengujian dilakukan pada 20 skenario kasus yang mencakup variasi jenis kelamin (Laki-laki dan Perempuan), jumlah data historis (*early months* ≤ 4 bulan dan *normal* ≥ 6 bulan), serta status pertumbuhan (Normal, At Risk, Stunted, dan Severely Stunted).

#### A. Model yang Dibandingkan

Tabel 4.17 menyajikan daftar model yang dilibatkan dalam evaluasi perbandingan beserta deskripsi singkatnya.

**Tabel 4.17** Daftar Model dalam Evaluasi Perbandingan

| No | Model                    | Kategori          | Deskripsi                                                                                     |
|:--:|:-------------------------|:------------------|:----------------------------------------------------------------------------------------------|
| 1  | Linear Regression        | Regresi Linear    | Model linear sederhana tanpa regularisasi; memodelkan hubungan linear antara usia dan tinggi badan |
| 2  | Polynomial Degree 2      | Regresi Polinomial | Regresi polinomial orde 2 (kuadratik); menangkap pola non-linear sederhana                    |
| 3  | Polynomial Degree 3      | Regresi Polinomial | Regresi polinomial orde 3 (kubik); menangkap pola non-linear yang lebih kompleks              |
| 4  | Bayesian Ridge           | Regresi Linear    | Regresi linear dengan regularisasi Bayesian; lebih stabil pada data terbatas                  |
| 5  | Gompertz                 | Kurva Pertumbuhan | Model pertumbuhan non-linear berbentuk sigmoid; memodelkan pola pertumbuhan biologis          |
| 6  | GPR WHO Prior            | Gaussian Process  | Gaussian Process Regression dengan WHO Median sebagai prior mean; memanfaatkan pengetahuan domain pertumbuhan anak |

#### B. Hasil Evaluasi Keseluruhan

Evaluasi dilakukan menggunakan metode LOOCV pada seluruh 20 skenario kasus. Tabel 4.18 menampilkan rata-rata metrik evaluasi per model dari semua kasus yang diujikan.

**Tabel 4.18** Rata-rata Metrik Evaluasi Per Model (Seluruh Kasus)

| Peringkat | Model                | MAE (cm) | RMSE (cm) | R²       |
|:---------:|:---------------------|:--------:|:---------:|:--------:|
| 1         | GPR WHO Prior        | 0.1457   | 0.1795    | 0.9914   |
| 2         | Gompertz             | 1.7673   | 2.1059    | 0.2568   |
| 3         | Polynomial Degree 3  | 4.1140   | 5.5151    | -4.9981  |
| 4         | Polynomial Degree 2  | 5.3663   | 6.7350    | -6.4398  |
| 5         | Bayesian Ridge       | 6.3356   | 7.0719    | -7.1952  |
| 6         | Linear Regression    | 6.4187   | 7.1581    | -7.3984  |

Berdasarkan Tabel 4.18, terlihat perbedaan performa yang sangat signifikan antar model. GPR WHO Prior menempati peringkat pertama dengan rata-rata MAE sebesar 0.1457 cm dan RMSE 0.1795 cm, menunjukkan akurasi prediksi yang sangat tinggi. Sebaliknya, Linear Regression menempati posisi terakhir dengan MAE 6.4187 cm dan RMSE 7.1581 cm. Nilai R² negatif pada Linear Regression (-7.3984) dan Polynomial Regression (-4.9981 hingga -6.4398) mengindikasikan bahwa model-model tersebut memiliki performa lebih buruk daripada model konstan yang hanya memprediksi rata-rata.

#### C. Perbandingan pada Mode Early Months (Riwayat ≤ 4 Bulan)

Pada kasus dengan data historis yang sangat terbatas (≤ 4 bulan), perbedaan performa antar model menjadi lebih terlihat. Tabel 4.19 menampilkan hasil evaluasi khusus untuk mode early months.

**Tabel 4.19** Rata-rata Metrik Evaluasi — Mode Early Months (Riwayat ≤ 4 Bulan)

| Peringkat | Model                | MAE (cm) | RMSE (cm) | R²       |
|:---------:|:---------------------|:--------:|:---------:|:--------:|
| 1         | GPR WHO Prior        | 0.1840   | 0.2241    | 0.9886   |
| 2         | Gompertz             | 1.6943   | 2.0355    | 0.4081   |
| 3         | Polynomial Degree 2  | 5.4153   | 6.9019    | -5.7307  |
| 4         | Bayesian Ridge       | 6.5150   | 7.3934    | -6.7092  |
| 5         | Linear Regression    | 6.5909   | 7.4726    | -6.8747  |

Pada Tabel 4.19, GPR WHO Prior tetap unggul dengan RMSE hanya 0.2241 cm, bahkan ketika data yang tersedia sangat sedikit. Hal ini disebabkan oleh kemampuan GPR dalam memanfaatkan WHO Median sebagai *prior mean*, sehingga model sudah memiliki pengetahuan awal tentang pola pertumbuhan normal anak. Linear Regression dan Polynomial Degree 2 menunjukkan performa yang sangat buruk dengan R² negatif, menunjukkan ketidakmampuan model-model tersebut dalam menangkap pola pertumbuhan non-linear pada data yang terbatas.

#### D. Perbandingan pada Mode Normal (Riwayat ≥ 6 Bulan)

Pada kasus dengan data historis yang lebih banyak (≥ 6 bulan), seluruh model memiliki data latih yang lebih memadai. Tabel 4.20 menampilkan hasil evaluasi untuk mode normal.

**Tabel 4.20** Rata-rata Metrik Evaluasi — Mode Normal (Riwayat ≥ 6 Bulan)

| Peringkat | Model                | MAE (cm) | RMSE (cm) | R²       |
|:---------:|:---------------------|:--------:|:---------:|:--------:|
| 1         | GPR WHO Prior        | 0.1073   | 0.1350    | 0.9943   |
| 2         | Gompertz             | 1.8403   | 2.1763    | 0.1055   |
| 3         | Polynomial Degree 3  | 4.1140   | 5.5151    | -4.9981  |
| 4         | Polynomial Degree 2  | 5.3173   | 6.5681    | -7.1489  |
| 5         | Bayesian Ridge       | 6.1562   | 6.7505    | -7.6811  |
| 6         | Linear Regression    | 6.2465   | 6.8436    | -7.9220  |

Meskipun data latih lebih banyak pada mode normal, Linear Regression dan Polynomial Regression tetap menunjukkan performa yang buruk. Bahkan Polynomial Degree 3 yang memiliki fleksibilitas lebih tinggi masih menghasilkan RMSE 5.5151 cm, jauh tertinggal dari GPR WHO Prior (0.1350 cm). Hal ini mengkonfirmasi bahwa pola pertumbuhan anak bersifat non-linear dan memerlukan model yang memiliki pengetahuan domain (*domain knowledge*) untuk menghasilkan prediksi yang akurat.

#### E. Analisis Perbandingan Linear Regression vs Polynomial Regression vs GPR

Tabel 4.21 menyajikan perbandingan langsung ketiga model utama yang menjadi fokus evaluasi ini, beserta analisis kelebihan dan kekurangannya.

**Tabel 4.21** Analisis Perbandingan Tiga Model Utama

| Aspek                    | Linear Regression     | Polynomial Regression (Deg 2–3) | GPR WHO Prior          |
|:-------------------------|:----------------------|:--------------------------------|:-----------------------|
| **MAE rata-rata**        | 6.4187 cm             | 4.1140 – 5.3663 cm             | 0.1457 cm              |
| **RMSE rata-rata**       | 7.1581 cm             | 5.5151 – 6.7350 cm             | 0.1795 cm              |
| **R² rata-rata**         | -7.3984               | -4.9981 – -6.4398              | 0.9914                 |
| **Asumsi**               | Hubungan linear       | Hubungan polinomial             | Non-parametrik + prior WHO |
| **Kebutuhan data minimum** | 2 poin              | > degree + 1 poin              | 2 poin                 |
| **Pengetahuan domain**   | Tidak ada             | Tidak ada                       | WHO Growth Standards   |
| **Handling data sedikit**| Sangat buruk          | Buruk (overfitting)             | Sangat baik            |
| **Kompleksitas**         | Rendah                | Sedang                          | Tinggi                 |
| **Risiko overfitting**   | Rendah (underfitting) | Tinggi pada data sedikit        | Rendah (dikontrol kernel) |

#### F. Detail Perbandingan Per Skenario Kasus

Untuk memberikan gambaran yang lebih komprehensif, Tabel 4.22 menampilkan perbandingan performa ketiga model pada beberapa skenario kasus representatif.

**Tabel 4.22** Perbandingan Detail Per Skenario — Linear Regression vs Polynomial vs GPR

| Skenario Kasus           | Model               | MAE (cm) | RMSE (cm) | R²       |
|:-------------------------|:--------------------|:--------:|:---------:|:--------:|
| EARLY-L-Normal           | Linear Regression   | 7.0967   | 8.0604    | -7.6441  |
|                          | Polynomial Degree 2 | 5.1033   | 6.5525    | -4.7125  |
|                          | GPR WHO Prior       | 0.0017   | 0.0041    | 1.0000   |
| EARLY-L-Stunted          | Linear Regression   | 6.9550   | 7.9028    | -8.0043  |
|                          | Polynomial Degree 2 | 5.1117   | 6.5450    | -5.1760  |
|                          | GPR WHO Prior       | 0.2533   | 0.3054    | 0.9866   |
| EARLY-P-SeverelyStunted  | Linear Regression   | 5.9567   | 6.7404    | -6.3853  |
|                          | Polynomial Degree 2 | 5.5750   | 7.0628    | -7.1088  |
|                          | GPR WHO Prior       | 0.3433   | 0.4309    | 0.9698   |
| NORM-L-Normal            | Linear Regression   | 6.8050   | 7.4530    | -8.9748  |
|                          | Polynomial Degree 3 | 3.0583   | 4.1374    | -2.0740  |
|                          | GPR WHO Prior       | 0.0000   | 0.0000    | 1.0000   |
| NORM-L-SeverelyStunted   | Linear Regression   | 6.6667   | 7.3156    | -10.4041 |
|                          | Polynomial Degree 3 | 3.5867   | 4.8141    | -3.9384  |
|                          | GPR WHO Prior       | 0.2750   | 0.3409    | 0.9752   |
| NORM-P-AtRisk            | Linear Regression   | 5.7367   | 6.2824    | -6.3623  |
|                          | Polynomial Degree 3 | 4.7250   | 6.3241    | -6.4603  |
|                          | GPR WHO Prior       | 0.0567   | 0.0611    | 0.9993   |

Berdasarkan Tabel 4.22, pada seluruh skenario kasus yang diujikan, GPR WHO Prior secara konsisten menghasilkan MAE dan RMSE yang jauh lebih kecil dibandingkan Linear Regression dan Polynomial Regression. Beberapa temuan penting dari perbandingan ini:

1. **Linear Regression** menghasilkan error terbesar di hampir semua kasus (MAE 5.7–7.1 cm), yang menunjukkan bahwa asumsi linearitas tidak sesuai dengan pola pertumbuhan anak yang bersifat non-linear dan deselerasi (melambat seiring usia).

2. **Polynomial Regression** sedikit lebih baik dari Linear Regression, namun tetap menunjukkan R² negatif di semua kasus. Polynomial Degree 3 lebih baik dari Degree 2 pada mode normal, tetapi tetap menghasilkan RMSE 4–7 cm yang tidak dapat diterima secara klinis.

3. **GPR WHO Prior** mengungguli kedua model konvensional dengan margin yang sangat besar. Pada kasus NORM-L-Normal, GPR mencapai RMSE 0.0000 cm (prediksi sempurna), sementara Linear Regression menghasilkan RMSE 7.4530 cm pada kasus yang sama.

#### G. Visualisasi Perbandingan Performa

Tabel 4.23 menyajikan rasio peningkatan performa GPR WHO Prior dibandingkan model konvensional untuk memperjelas besarnya perbedaan.

**Tabel 4.23** Rasio Peningkatan Performa GPR WHO Prior vs Model Konvensional

| Perbandingan                   | Penurunan MAE | Penurunan RMSE | Peningkatan |
|:-------------------------------|:-------------:|:--------------:|:-----------:|
| GPR vs Linear Regression       | 97.73%        | 97.49%         | ~40x lebih akurat |
| GPR vs Polynomial Degree 2    | 97.28%        | 97.33%         | ~37x lebih akurat |
| GPR vs Polynomial Degree 3    | 96.46%        | 96.74%         | ~31x lebih akurat |

#### H. Kesimpulan Perbandingan

Berdasarkan hasil evaluasi perbandingan yang telah dilakukan, dapat disimpulkan bahwa:

1. **Model konvensional (Linear Regression dan Polynomial Regression) tidak cocok** untuk prediksi pertumbuhan anak. Kedua model menghasilkan R² negatif yang konsisten, menandakan kegagalan sistematis dalam memodelkan pola pertumbuhan.

2. **Penyebab kegagalan model konvensional** terletak pada dua faktor utama:
   - *Data yang sangat terbatas* (2–12 poin per anak) tidak memberikan cukup informasi untuk model tanpa pengetahuan domain.
   - *Pola pertumbuhan yang non-linear dan deselerasi* tidak dapat ditangkap oleh model linear, dan model polinomial cenderung overfitting pada data yang sedikit.

3. **GPR WHO Prior menjadi model terbaik** karena:
   - Memanfaatkan WHO Growth Standards sebagai *prior mean*, memberikan model pengetahuan tentang pola pertumbuhan normal sebelum melihat data.
   - Bersifat non-parametrik sehingga fleksibel menangkap variasi individual.
   - Kernel RBF + WhiteKernel memberikan regularisasi yang tepat untuk menghindari overfitting.

4. **Justifikasi pemilihan model pada sistem**: Hasil perbandingan ini memvalidasi keputusan arsitektur sistem yang menggunakan GPR WHO Prior sebagai model utama dalam mode normal, dengan Bayesian Ridge sebagai *fallback* konservatif ketika GPR gagal konvergen.
