## Tabel 4.5 Hasil Preprocessing Data

Preprocessing dilakukan untuk memvalidasi dan mentransformasi data riwayat pengukuran antropometri anak menjadi format yang siap digunakan oleh model prediksi. Berikut adalah contoh hasil preprocessing pada data tinggi badan:

### A. Data Mentah (Input)

| No | age (bulan) | height (cm) |
|----|-------------|-------------|
| 1  | 3           | 55.0        |
| 2  | 1           | 50.2        |
| 3  | 5           | 60.5        |
| 4  | 2           | 52.1        |
| 5  | 4           | 57.8        |

**Jumlah data poin:** 5

### B. Hasil Validasi

| Tahap Validasi              | Status | Keterangan                          |
|-----------------------------|--------|-------------------------------------|
| Kelengkapan history         | Lolos  | Data tidak kosong (5 entry)         |
| Minimum data poin (≥ 2)    | Lolos  | Terdapat 5 data poin               |
| Ketersediaan key `age`      | Lolos  | Semua entry memiliki key `age`      |
| Ketersediaan key `height`   | Lolos  | Semua entry memiliki key `height`   |
| Tipe data `age` (numerik ≥ 0) | Lolos | Semua nilai age valid            |
| Tipe data `height` (numerik > 0) | Lolos | Semua nilai height valid      |
| Duplikasi nilai `age`       | Lolos  | Tidak ditemukan duplikat            |

### C. Data Setelah Sorting (Ascending by Age)

| No | age (bulan) | height (cm) |
|----|-------------|-------------|
| 1  | 1           | 50.2        |
| 2  | 2           | 52.1        |
| 3  | 3           | 55.0        |
| 4  | 4           | 57.8        |
| 5  | 5           | 60.5        |

### D. Hasil Ekstraksi Fitur

| Komponen | Nilai                              | Shape  |
|----------|-------------------------------------|--------|
| X (fitur usia) | [[1], [2], [3], [4], [5]]     | (5, 1) |
| y (target tinggi) | [50.2, 52.1, 55.0, 57.8, 60.5] | (5,) |

### E. Ringkasan Statistik

| Parameter            | Nilai  |
|----------------------|--------|
| Jumlah sampel (n)   | 5      |
| Usia minimum         | 1 bulan |
| Usia maksimum        | 5 bulan |
| Tinggi minimum       | 50.2 cm |
| Tinggi maksimum      | 60.5 cm |
| Rata-rata tinggi     | 55.12 cm |
| Last age (acuan prediksi) | 5 bulan |
