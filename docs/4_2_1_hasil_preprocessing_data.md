### 4.2.1 Hasil Preprocessing Data

Tahap preprocessing merupakan langkah awal yang krusial dalam sistem prediksi pertumbuhan anak. Pada tahap ini, data riwayat pengukuran antropometri yang dikirimkan oleh pengguna melalui API akan divalidasi dan ditransformasi menjadi format numerik yang dapat diproses oleh model machine learning. Proses preprocessing pada sistem ini ditangani oleh modul `preprocessing_service.py` yang mendukung tiga jenis pengukuran, yaitu tinggi badan (*height*), berat badan (*weight*), dan lingkar kepala (*head_circ*).

Secara keseluruhan, proses preprocessing terdiri dari empat tahapan utama: (1) validasi kelengkapan dan kecukupan data, (2) validasi tipe data dan rentang nilai, (3) pengurutan data berdasarkan usia secara *ascending*, serta (4) ekstraksi fitur ke dalam bentuk array NumPy. Keempat tahapan tersebut berjalan secara sekuensial, di mana kegagalan pada satu tahap akan menghentikan proses dan mengembalikan pesan error yang informatif kepada pengguna.

Sebagai ilustrasi, berikut disajikan contoh hasil preprocessing pada data pengukuran tinggi badan seorang anak laki-laki dengan 5 data poin historis. Tabel 4.5 menunjukkan data mentah yang diterima oleh sistem sebelum melalui proses preprocessing.

**Tabel 4.5** Data Mentah Riwayat Pengukuran (Sebelum Preprocessing)

| No | Usia (bulan) | Tinggi Badan (cm) |
|:--:|:------------:|:------------------:|
| 1  | 3            | 55.0               |
| 2  | 1            | 50.2               |
| 3  | 5            | 60.5               |
| 4  | 2            | 52.1               |
| 5  | 4            | 57.8               |

Pada Tabel 4.5 terlihat bahwa data mentah yang diterima dari pengguna belum terurut berdasarkan usia. Data masih dalam format *list of dictionary* dengan key `age` dan `height` pada setiap entry-nya. Jumlah total data poin yang dikirimkan adalah 5 pengukuran.

Selanjutnya, sistem melakukan serangkaian validasi terhadap data tersebut. Tabel 4.6 merangkum hasil dari setiap tahap validasi yang dilakukan.

**Tabel 4.6** Hasil Validasi Data Preprocessing

| No | Tahap Validasi                          | Status | Keterangan                                 |
|:--:|:----------------------------------------|:------:|:-------------------------------------------|
| 1  | Kelengkapan data (*history* tidak kosong) | Lolos  | Data berisi 5 entry                        |
| 2  | Kecukupan data (minimal 2 data poin)   | Lolos  | Terdapat 5 data poin (syarat: >= 2)        |
| 3  | Ketersediaan key `age` pada setiap entry | Lolos  | Semua entry memiliki key `age`             |
| 4  | Ketersediaan key `height` pada setiap entry | Lolos | Semua entry memiliki key `height`        |
| 5  | Validasi tipe dan rentang `age` (numerik, >= 0) | Lolos | Semua nilai `age` bertipe numerik dan non-negatif |
| 6  | Validasi tipe dan rentang `height` (numerik, > 0) | Lolos | Semua nilai `height` bertipe numerik dan positif |
| 7  | Deteksi duplikasi nilai `age`           | Lolos  | Tidak ditemukan duplikasi usia             |

Berdasarkan Tabel 4.6, seluruh data berhasil melewati ketujuh tahap validasi tanpa ditemukan kesalahan. Apabila terdapat kegagalan pada salah satu tahap, sistem akan mengembalikan response error dengan kode HTTP 400 beserta pesan yang menjelaskan letak kesalahannya, misalnya: *"Entry ke-2 tidak memiliki key 'height'"* atau *"Terdapat duplikat nilai 'age' di dalam history"*.

Setelah proses validasi selesai, data diurutkan secara *ascending* berdasarkan nilai `age`. Pengurutan ini penting agar model dapat mempelajari pola pertumbuhan secara kronologis. Tabel 4.7 menunjukkan data yang telah diurutkan.

**Tabel 4.7** Data Setelah Pengurutan Berdasarkan Usia (*Ascending*)

| No | Usia (bulan) | Tinggi Badan (cm) |
|:--:|:------------:|:------------------:|
| 1  | 1            | 50.2               |
| 2  | 2            | 52.1               |
| 3  | 3            | 55.0               |
| 4  | 4            | 57.8               |
| 5  | 5            | 60.5               |

Pada Tabel 4.7 dapat dilihat bahwa data telah terurut berdasarkan usia dari yang termuda (1 bulan) hingga yang tertua (5 bulan). Pola pertumbuhan terlihat konsisten dengan peningkatan tinggi badan seiring bertambahnya usia.

Tahap terakhir dari preprocessing adalah ekstraksi fitur, yaitu mengubah data yang sudah tervalidasi dan terurut menjadi array NumPy yang siap digunakan oleh model. Data dibagi menjadi dua komponen: array fitur **X** yang berisi nilai usia dalam bulan dengan dimensi (*n*, 1), dan array target **y** yang berisi nilai tinggi badan dalam cm dengan dimensi (*n*,). Tabel 4.8 menampilkan hasil ekstraksi fitur tersebut.

**Tabel 4.8** Hasil Ekstraksi Fitur (Output Preprocessing)

| Komponen          | Nilai                            | Dimensi (*Shape*) | Tipe Data     |
|:------------------|:---------------------------------|:------------------:|:-------------:|
| X (fitur usia)    | [[1], [2], [3], [4], [5]]        | (5, 1)             | float64       |
| y (target tinggi) | [50.2, 52.1, 55.0, 57.8, 60.5]  | (5,)               | float64       |

Array **X** memiliki dimensi (5, 1) karena model scikit-learn memerlukan input berupa matriks dua dimensi, meskipun hanya terdapat satu fitur yaitu usia. Sementara itu, array **y** memiliki dimensi (5,) sebagai vektor satu dimensi yang merepresentasikan variabel target. Kedua array ini selanjutnya akan digunakan sebagai input pada tahap pelatihan model, yaitu Bayesian Ridge dan Gompertz.

Tabel 4.9 menyajikan ringkasan statistik deskriptif dari data hasil preprocessing yang memberikan gambaran umum mengenai karakteristik data.

**Tabel 4.9** Ringkasan Statistik Deskriptif Data Hasil Preprocessing

| Parameter                        | Nilai     |
|:---------------------------------|:---------:|
| Jumlah sampel (*n*)              | 5         |
| Usia minimum                     | 1 bulan   |
| Usia maksimum                    | 5 bulan   |
| Rentang usia                     | 4 bulan   |
| Tinggi badan minimum             | 50.2 cm   |
| Tinggi badan maksimum            | 60.5 cm   |
| Rata-rata tinggi badan           | 55.12 cm  |
| Standar deviasi tinggi badan     | 4.07 cm   |
| Usia terakhir (*last_age*)       | 5 bulan   |

Nilai *last_age* pada Tabel 4.9 merupakan parameter penting karena menjadi titik acuan untuk prediksi ke depan. Sistem akan memprediksi pertumbuhan mulai dari usia *last_age* + 1 hingga *last_age* + *horizon*, di mana *horizon* merupakan jumlah bulan ke depan yang ingin diprediksi (default: 6 bulan, maksimal: 24 bulan).

Perlu dicatat bahwa proses preprocessing pada sistem ini tidak melakukan normalisasi atau standarisasi data. Hal ini karena model Bayesian Ridge dan Gompertz yang digunakan bekerja langsung pada skala asli data (usia dalam bulan dan tinggi badan dalam cm). Keputusan ini memudahkan interpretasi hasil prediksi tanpa perlu melakukan transformasi balik (*inverse transform*) pada output model.
