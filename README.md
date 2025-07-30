<p align="right" style="font-size: 12px;">
<i>Disclaimer: Referensi yang digunakan dapat dilihat pada bagian paling bawah tulisan. Setiap tulisan berwarna <span style="color:blue;">biru</span> merupakan tautan yang dapat diklik menuju sumber aslinya.</i>
</p>

# Nama : Ahmad Raihan
# Project : Diamond Price Prediction

# 1. Domain Proyek
Adapun proyek ini membahas tentang predictive analysis dalam membangun sistem regressi prediksi harga berlian. 

## 1.1 Latar Belakang
<img width="865" height="304" alt="225086030-29852651-c4c3-451b-bf42-19385410343c" src="https://github.com/user-attachments/assets/ea95731c-fbd1-4b41-bf86-849749ab0961" />

Berlian merupakan salah satu batu mulia paling bernilai tinggi yang telah lama digunakan sebagai simbol kemewahan, status sosial, dan investasi. Karakteristik berlian ditentukan oleh standar internasional yang dikenal sebagai 4C, yaitu Carat (berat), Cut (potongan), Color (warna), dan Clarity (kejernihan), yang secara langsung mempengaruhi nilai jualnya. Selain keindahan visual, berlian juga memiliki tingkat kekerasan tertinggi dalam skala Mohs, menjadikannya pilihan utama untuk perhiasan tahan lama dan eksklusif [[1](https://4cs.gia.edu/en-us/4cs-of-diamond-quality/)].

Dalam beberapa tahun terakhir, pasar berlian mengalami dinamika yang signifikan. Sejak tahun 2022, harga berlian alami (natural diamond) mengalami tren penurunan akibat meningkatnya popularitas berlian hasil laboratorium (lab-grown diamond). Berdasarkan laporan dari berbagai sumber industri, harga berlian alami turun hingga 20–26% dari puncaknya pada tahun 2022. Sementara itu, berlian sintetis yang memiliki bentuk dan kualitas serupa dengan harga yang jauh lebih murah dan diproduksi secara berkelanjutan menjadi pilihan alternatif bagi banyak konsumen [[2](https://www.rmol.id/bisnis/read/2023/10/03/591584/harga-berlian-merosot-konsumen-lebih-suka-habiskan-uang-untuk-makan-dan-traveling)]. Pada tahun 2024, pangsa pasar berlian sintetis telah mencapai lebih dari 20% dari total pasar berlian global, yang menunjukkan adanya perubahan besar dalam perilaku konsumen [[3](https://diamondion.com/global-diamond-market-to-reach-4149-billion-in-2024-asia-pacific-demand-surges/)].

Selain persaingan dari produk alternatif, perubahan harga berlian juga dipengaruhi oleh kondisi ekonomi global dan pergeseran budaya konsumen. Generasi muda seperti Gen Z dan milenial lebih memprioritaskan pengeluaran untuk pengalaman dan gaya hidup dibandingkan barang mewah fisik, termasuk perhiasan berlian. Di sisi lain, ketidakpastian ekonomi di negara-negara besar seperti Amerika Serikat dan Tiongkok, serta sanksi terhadap ekspor berlian dari Rusia, turut memberikan tekanan terhadap permintaan dan distribusi berlian secara global [[4](https://www.wsj.com/business/retail/de-beers-diamonds-price-lab-grown-468b33ab)].

Kondisi tersebut menjadi tantangan tersendiri bagi pelaku industri berlian, terutama dalam menentukan harga yang kompetitif dan akurat. Penetapan harga secara manual yang bergantung pada pengalaman dan intuisi manusia memiliki kelemahan dari sisi objektivitas dan efisiensi. Oleh karena itu, diperlukan pendekatan yang lebih modern dan adaptif, salah satunya dengan memanfaatkan teknologi kecerdasan buatan. Dengan membangun model prediksi berbasis algoritma regresi, harga berlian dapat diperkirakan secara otomatis berdasarkan data karakteristik fisik dan kualitasnya.

Penerapan teknologi ini diharapkan dapat membantu pelaku industri seperti toko perhiasan, investor, dan pelaku e-commerce dalam menetapkan harga secara lebih cepat, transparan, dan berbasis data. Selain itu, konsumen juga dapat lebih memahami faktor-faktor yang memengaruhi nilai sebuah berlian, sehingga mendorong terciptanya ekosistem perdagangan yang lebih terbuka dan adil [[5](https://www.globenewswire.com/news-release/2025/05/22/3086367/28124/en/Diamond-Jewelry-Market-Forecast-2025-2030-Lab-Grown-Diamonds-Propel-Sustainability-and-Affordability.html)].

# 2. Business Understanding
![1_nYVObp29l1_9hPNJKhQvGA-1](https://github.com/user-attachments/assets/bc3fc484-02be-40d6-8f9a-570f3d4d9fac)

Dalam menghadapi fluktuasi harga berlian dan perubahan preferensi konsumen global, pelaku industri perhiasan membutuhkan solusi yang adaptif dan berbasis data. Pengembangan model prediksi harga berlian menjadi penting sebagai alat bantu dalam menetapkan nilai jual secara lebih akurat, efisien, dan kompetitif. Model ini memungkinkan estimasi harga dilakukan secara objektif berdasarkan karakteristik fisik berlian seperti carat, cut, color, dan clarity, tanpa harus selalu bergantung pada penilaian manual.

Penerapan sistem ini berpotensi besar untuk mendukung pengambilan keputusan strategis di berbagai lini bisnis, mulai dari produsen, pedagang, hingga platform jual beli online. Selain mempermudah proses penetapan harga, model prediktif ini juga meningkatkan transparansi pasar dan kepercayaan konsumen terhadap produk yang ditawarkan. Dengan demikian, proyek ini tidak hanya menjadi solusi teknis, tetapi juga mendukung efisiensi dan keberlanjutan industri berlian di tengah tantangan global.

## 2.1 Problem Statements
Berdasarkan latar belakang yang telah dipaparkan, dapat dirumuskan beberapa permasalahan utama yang menjadi fokus dalam proyek ini, yaitu:

1. Bagaimana membangun sebuah model machine learning yang mampu memprediksi harga berlian secara akurat berdasarkan atribut fisik seperti `carat`, `cut`, `color`, `clarity`, dan dimensi (`x`, `y`, `z`).
2. Apa saja metrik evaluasi yang tepat untuk mengukur kinerja dari model regresi harga berlian, sehingga hasil prediksi dapat digunakan secara andal dalam pengambilan keputusan bisnis.
3. Bagaimana model regresi ini dapat memberikan manfaat nyata bagi sektor perdagangan berlian, khususnya dalam membantu pelaku industri dalam penentuan harga yang efisien dan berbasis data, serta meningkatkan transparansi pasar.

## 2.2 Goals
Tujuan dari pembuatan proyek ini adalah untuk:

- Mengembangkan model prediksi harga berlian berbasis machine learning yang dapat mengestimasi nilai jual berlian berdasarkan fitur-fitur relevan seperti `carat`, `clarity`, `color`, `cut`, dan dimensi fisik.
- Mengevaluasi performa model menggunakan metrik regresi yang sesuai, seperti **R-squared (R²)**, **Root Mean Squared Error (RMSE)**, dan **Mean Absolute Error (MAE)**, untuk memastikan keandalan hasil prediksi.
- Memberikan solusi berbasis data untuk mendukung efisiensi dan akurasi dalam proses penetapan harga berlian, baik untuk produsen, pedagang, maupun platform jual beli.

## 2.3 Solution Statements
Adapun solusi yang akan diterapkan untuk menyelesaikan proyek ini adalah:

1. **Eksplorasi Data (EDA)**  
   Melakukan eksplorasi terhadap dataset harga berlian untuk memahami distribusi fitur seperti `carat`, `clarity`, `color`, `cut`, `x`, `y`, `z`, serta hubungan antara masing-masing fitur terhadap variabel target `price`.

2. **Pembersihan Data**  
   Menangani missing values, menghapus outlier, serta mengolah variabel kategorikal seperti `cut`, `color`, dan `clarity` dengan teknik encoding yang sesuai. Proses normalisasi atau standarisasi juga diterapkan untuk mempersiapkan data sebelum pelatihan model.

3. **Pemodelan dan Evaluasi**  
   Menerapkan 10 algoritma regresi untuk membangun dan membandingkan performa model prediksi harga berlian:

   ### Model yang Digunakan:
     **a. Linear Regression**  
     Model linier sederhana sebagai baseline, cocok untuk hubungan fitur yang bersifat linier.

     **b. ElasticNet**  
     Kombinasi Lasso dan Ridge, efektif menangani multikolinearitas dan fitur berkorelasi.

     **c. K-Nearest Neighbors Regressor (KNN)**  
     Memprediksi berdasarkan rata-rata `k` tetangga terdekat. Cocok untuk pola lokal.

     **d. Support Vector Regressor (SVR)**  
     Mencari margin optimal untuk prediksi nilai harga dengan toleransi error tertentu.

     **e. Decision Tree Regressor**  
     Model pohon yang membagi data berdasarkan nilai fitur. Mudah diinterpretasi dan tidak perlu normalisasi.

     **f. Random Forest Regressor**  
     Algoritma ensemble berbasis banyak decision tree. Stabil terhadap noise dan overfitting.

     **g. Gradient Boosting Regressor**  
     Menggabungkan weak learner secara bertahap dengan fokus memperbaiki kesalahan model sebelumnya.

     **h. XGBoost Regressor**  
     Versi lebih efisien dari Gradient Boosting. Tahan terhadap overfitting dan sangat populer.

     **i. LightGBM Regressor**  
     Alternatif Gradient Boosting yang ringan dan cepat. Cocok untuk dataset besar dengan banyak fitur kategorikal.

     **j. Multi-layer Perceptron Regressor (MLPRegressor)**  
     Model jaringan saraf yang mampu menangkap hubungan non-linear. Membutuhkan tuning dan preprocessing khusus.

4. **Optimasi Model**  
   Membangun model baseline dari semua algoritma, lalu melakukan optimasi hyperparameter menggunakan **Grid Search** atau **Randomized Search**.

5. **Evaluasi dan Seleksi Model Terbaik**  
   Menilai performa setiap model berdasarkan metrik:
   - **MAE (Mean Absolute Error)**
   - **RMSE (Root Mean Squared Error)**
   - **R² (R-squared Score)**

   Model dengan hasil terbaik akan dipilih sebagai model akhir untuk prediksi harga berlian.

# 3. Data Understanding
## 3.1 EDA - Deskripsi Variabel
**Informasi Datasets**
| Jenis | Keterangan |
| ------ | ------ |
| Title | Data Analysis on Diamonds Dataset |
| Source | [Kaggle](https://www.kaggle.com/datasets/swatikhedekar/price-prediction-of-diamond) |
| Maintainer | [Swati Khedekar ⚡](https://www.kaggle.com/swatikhedekar) |
| License | Other (specified in description) |
| Visibility | Publik |
| Tags | Religion and Belief System, Beginner, Pandas, Matplotlib, Data Visualization, Regression, Exploratory Data Analysis_ |
| Usability | 10.00 |

Berikut informasi pada dataset: 

Data ini disediakan secara publik di kaggle.

Tabel 1. Gambaran Data yang digunakan

| Unnamed: 0 | carat |   cut   | color | clarity | depth | table | price |   x   |   y   |   z   |
|------------|-------|---------|-------|---------|-------|-------|-------|-------|-------|-------|
|     1      | 0.23  | Ideal   |   E   |  SI2    | 61.5  | 55.0  |  326  | 3.95  | 3.98  | 2.43  |
|     2      | 0.21  | Premium |   E   |  SI1    | 59.8  | 61.0  |  326  | 3.89  | 3.84  | 2.31  |
|     3      | 0.23  | Good    |   E   |  VS1    | 56.9  | 65.0  |  327  | 4.05  | 4.07  | 2.31  |
|     4      | 0.29  | Premium |   I   |  VS2    | 62.4  | 58.0  |  334  | 4.20  | 4.23  | 2.63  |
|     5      | 0.31  | Good    |   J   |  SI2    | 63.3  | 58.0  |  335  | 4.34  | 4.35  | 2.75  |

- Dari Tabel 1 di atas terlihat bahwa ada 11 buah kolom/variabel yang digunakan:
  - `Unnamed: 0` : Identifier unik untuk setiap entri data berlian
  - `carat` : Berat berlian dalam satuan karat
  - `cut` : Kualitas potongan berlian (misalnya Ideal, Premium, Good)
  - `color` : Warna berlian, dinyatakan dalam skala huruf dari D (terjernih) hingga J (kurang jernih)
  - `clarity` : Kejernihan berlian, mencerminkan jumlah dan ukuran inklusi atau cacat
  - `depth` : Kedalaman total berlian sebagai persentase dari lebar rata-rata
  - `table` : Lebar bagian atas berlian sebagai persentase dari lebar total
  - `price` : Harga berlian dalam USD (merupakan target variabel atau label untuk regresi)
  - `x` : Panjang dimensi berlian dalam milimeter
  - `y` : Lebar dimensi berlian dalam milimeter
  - `z` : Kedalaman dimensi berlian dalam milimeter

Tabel 2. Informasi Umum dari Dataset 
| No | Kolom        | Non-Null Count | Dtype   |
|----|--------------|----------------|---------|
| 0  | Unnamed: 0         | 53940           | int64 |
| 1  | carat       | 53940           | float64 |
| 2  | cut    | 53940           | object |
| 3  | color  | 53940           | object |
| 4  | clarity    | 53940           | object |
| 5  | depth     | 53940           | float64 |
| 6  | table      | 53940           | float64  |
| 7  | price      | 53940           | int64  |
| 8  | x      | 53940           | float64  |
| 9  | y      | 53940           | float64  |
| 10  | z      | 53940           | float64  |

Tipe data: float64 (6), int64 (2), object (3)  
Jumlah entri: 53941 (index dari 0 sampai 53940)

- Dari Tabel 2 juga dapat dilihat informasi umum dari dataset yang digunakan:
  - Terdapat **6 kolom numerik bertipe `float64`**, yaitu: **carat, depth, table, x, y**, dan **z**.
  - Terdapat **1 kolom numerik bertipe `int64`**, yaitu: **price**.
  - Terdapat **3 kolom bertipe data `object`**, yaitu: **cut, color**, dan **clarity**.
  - Terdapat **1 kolom identifier (`Unnamed: 0`)** yang hanya berfungsi sebagai penomoran atau ID, sehingga perlu dihapus karena tidak memberikan informasi analitis.

Tabel 3 Informasi Statistik dari Dataset

| Kolom  | Count   | Mean     | Std       | Min  | 25%  | 50%   | 75%   | Max     |
|--------|---------|----------|-----------|------|------|-------|-------|---------|
| carat  | 53940.0 | 0.797940 | 0.474011  | 0.2  | 0.40 | 0.70  | 1.04  | 5.01    |
| depth  | 53940.0 | 61.749405| 1.432621  | 43.0 | 61.0 | 61.80 | 62.50 | 79.00   |
| table  | 53940.0 | 57.457184| 2.234491  | 43.0 | 56.0 | 57.00 | 59.00 | 95.00   |
| price  | 53940.0 | 3932.799722 | 3989.439738 | 326.0 | 950.0 | 2401.00 | 5324.25 | 18823.00 |
| x      | 53940.0 | 5.731157 | 1.121761  | 0.0  | 4.71 | 5.70  | 6.54  | 10.74   |
| y      | 53940.0 | 5.734526 | 1.142135  | 0.0  | 4.72 | 5.71  | 6.54  | 58.90   |
| z      | 53940.0 | 3.538734 | 0.705699  | 0.0  | 2.91 | 3.53  | 4.04  | 31.80   |

- Insight yang didapatkan dari informasi statistik adalah sebagai berikut:
   - Ditemukan kejanggalan pada kolom x, y, dan z, di mana terdapat nilai minimum sebesar 0.
   - Nilai 0 pada panjang, lebar, dan kedalaman berlian mustahil ada karena ukuran fisik berlian tidak mungkin nol.
   - Kemungkinan besar, nilai 0 tersebut merupakan representasi dari data yang hilang (missing value) yang disamarkan.
 
## 3.2 EDA - Pengecekan Missing Value 
Pengecekan ini diperlukan agar memastikan bahwa tidak ada nilai yang hilang (`missing value`). 

```python
# Cek kolom dengan data yang hilang
diamonds_df_eda.isnull().sum()
diamonds_df_eda[diamonds_df_eda.isnull.any(axis=1)]
```
- Hasilnya:

| Column  | Missing Values |
|---------|----------------|
| carat   | 0              |
| cut     | 0              |
| color   | 0              |
| clarity | 0              |
| depth   | 0              |
| table   | 0              |
| price   | 0              |
| x       | 8              |
| y       | 7              |
| z       | 20             |

- Dari hasil diatas ditemukan **8 data yang missing pada kolom x, 7 pada kolom y, dan 20 kolom z** pada yang perlu ditangani dimana salah satu caranya dengan **menghapus data tersebut**.

## 3.3 EDA - Pengecekan Data Duplikat
Pengecekan data yang berulang (duplicated data) dilakukan agar memastikan data yang digunakan tidak duplikat, karena penggunaan data yang sama berulang kali dapat menyebabkan model mempelajari informasi yang tidak representatif, serta dapat menyebabkan bias pada hasil analisis dan model yang dibangun.

```python
# Cek jumlah data yang duplikat
print(f'Jumlah data yang duplikat : {apple_df.duplicated().sum()}')
```
- Hasilnya:

Jumlah Data yang Duplikat : 145

- Dari hasil pengecekan yang dilakukan ditemtukan **145 data yang duplikat (sama)** yang perlu ditangani dengan **menghapus data tersebut**.
 
## 3.4 EDA - Pengecekan Outlier
Adapun outlier ini akan dicek dengan menggunakan boxplot yang ditampilkan pada Gambar 1
<img width="1490" height="1181" alt="image" src="https://github.com/user-attachments/assets/c743cf65-60c0-456b-8dc5-27ab44d4aaa2" />

- Insight yang didaptkan dari Gambar 1:
  - Berdasarkan grafik boxplot yang ditampilkan terlihat bahwa hanya semua fitur numerik yang memiliki outlier (ditandai dengan simbol bulat).
  - Adapun untuk outlier yang ditemukan akan dilakukan penanganan dengan dihapus menggunakan metode IQR.

## 3.5 EDA - Univariate Analysis
### 3.5.1 Categorical Column
Pada tahapan ini, akan dibuat barplot untuk melihat distribusi jumlah data berdasarkan kategori pada setiap fitur yang ditampilkan.
<img width="868" height="547" alt="image" src="https://github.com/user-attachments/assets/9997cc41-4117-43bd-a351-365660e66b94" />

- Insight yang didapatkan dari Gambar 2:
  - Berdasarkan visualisasi, mayoritas cut dari diamond berada pada kategori yang baik, yaitu Ideal dan Premium, yang secara keseluruhan mencakup sekitar 65% dari total data.
  - Sementara itu, cut dengan kualitas terendah yaitu Fair, hanya mencakup sekitar 1% dari keseluruhan sampel.

<img width="868" height="547" alt="image" src="https://github.com/user-attachments/assets/e79a0b3a-bf38-4e6e-8448-8e9866961d41" />

- Insight yang didapatkan dari Gambar 3:
  - Berdasarkan grafik di atas, sebagian besar diamond berada pada kualitas color (warna) yang tergolong menengah hingga agak baik, yaitu pada color G, E, dan F, yang mencakup sekitar 57,8% dari total data.

 <img width="868" height="547" alt="image" src="https://github.com/user-attachments/assets/b710435e-e054-4cbd-b57c-9d61ba5bb8b1" />

- Insight yang didapatkan dari Gambar 4:
   - Berdasarkan grafik di atas, sebagian besar diamond berkualitas kejernihan (clarity) yang rendah hingga menengah, dengan sekitar 64% di antaranya berada pada kategori SI2, SI1, dan VS2.
   - Di sisi lain, berlian dengan kualitas kejernihan terendah (I1) hanya mencakup sekitar 1%, sementara yang memiliki kualitas terbaiknya (IF) mencapai sekitar 3% dari total data.
 
### 3.5.2 Numerical Column
- Untuk kolom numerik dilakukan analisis persebaran data menggunakan histogram untuk melihat apakah datanya terdistribusi normal, skewed, ataupun disribusi lainnya.
<img width="1989" height="1964" alt="image" src="https://github.com/user-attachments/assets/a9931f84-19a4-4014-8dc9-aab026d71cc5" />

- Insight yang didapatkan dari Gambar 5:
 1. carat
      - Distribusi nilai pada kolom carat bersifat right-skewed sedang. Hal ini ditunjukkan oleh skewness sebesar 1.116, serta pola histogram di mana sebagian besar data berada pada nilai kecil dengan ekor yang memanjang ke kanan. Rata-rata (mean) lebih besar dari median dan modus.

2. depth
    - Distribusi nilai pada kolom depth mendekati normal (simetris). Skewness-nya sangat kecil, yaitu -0.081, dan mean, median, serta modus hampir berhimpitan. Bentuk histogram pun memperlihatkan sebaran yang simetris di sekitar pusat data.

3. table
    - Kolom table menunjukkan distribusi sedikit right-skewed, dengan nilai skewness sebesar 0.797. Histogram menggambarkan puncak distribusi di tengah dengan sedikit pergeseran nilai ke kanan, serta mean yang sedikit lebih besar dari median dan modus.

4. price
    - Distribusi nilai pada kolom price bersifat sangat right-skewed, dengan skewness sebesar 1.618. Histogram menunjukkan konsentrasi data pada nilai rendah dan ekor panjang ke kanan. Perbedaan antara mean, median, dan modus cukup mencolok.

5. x
    - Kolom x memiliki distribusi sedikit right-skewed, ditunjukkan oleh skewness sebesar 0.398. Histogram memperlihatkan sebaran yang relatif seimbang namun tetap memiliki kecenderungan ke kanan.

6. y
   - Distribusi kolom y bersifat sangat right-skewed, dengan nilai skewness tertinggi yaitu 2.462. Histogram menunjukkan banyak nilai kecil di sisi kiri dengan ekor distribusi yang sangat panjang ke kanan. Mean, median, dan modus berhimpitan di bagian kiri.

7. z
   - Distribusi nilai pada kolom z juga sangat right-skewed, dengan skewness sebesar 1.585. Polanya serupa dengan kolom y, di mana mayoritas data terkumpul pada nilai kecil dan sebagian kecil tersebar jauh di nilai tinggi. Mean, median, dan modus saling berdekatan di bagian kiri distribusi.

- Fitur-fitur yang memiliki distribusi right-skewed disarankan untuk ditransformasi menggunakan logarithmic transformation atau power transformation guna mendekatkan distribusi ke bentuk normal. dan meningkatkan performa model.

## 3.6 EDA - Multivariate Analysis
### 3.6.1 Numeric Columns
Pada tahap ini, dilakukan analisis multivariat untuk memahami hubungan antar fitur numerik dan keterkaitannya dengan label.
<img width="1989" height="1475" alt="image" src="https://github.com/user-attachments/assets/0033f581-1480-4b41-9578-b66cc42c7c06" />

<img width="801" height="665" alt="image" src="https://github.com/user-attachments/assets/d48c6bd9-be72-4b93-ba07-f4af97044241" />

- Insight yang didapatkan dari Gambar 6:
  - Berdasarkan dua visualisasi yang diberikan, terlihat hubungan antara masing-masing fitur dengan label (price) sebagai berikut:
  1. Tidak Ada Korelasi
      - Fitur depth tidak menunjukkan korelasi dengan price. Hal ini terlihat dari regplot yang garis regresinya yang sejajar, serta tidak ada pola yang jelas antara depth dan price. Hal ini menyebabkan fitur **depth** dapat dihapus saat proses preprocessing.

  2. Korelasi Sangat Lemah
      - Fitur table memiliki korelasi positif yang sangat lemah dengan price. Ditunjukkan oleh regplot yang hanya menunjukkan kenaikan yang sangat kecil, serta nilai korelasi yang rendah, yaitu sekitar 0.18.

  3. Korelasi Sangat Kuat
      - Fitur carat, x, y, dan z menunjukkan korelasi positif yang sangat kuat dengan price. Artinya, semakin besar ukuran dan berat diamond, maka harga cenderung semakin tinggi.

  4. Korelasi Antar Fitur
    - Terdapat korelasi sangat tinggi antar fitur carat, x, y, dan z (corr ≈ 0.96-1), yang mengindikasikan adanya multikolinearitas. Hal ini dapat memengaruhi hasil pelatihan model regresi. Disarankan untuk mereduksi fitur-fitur ini, misalnya dengan PCA atau membuat satu fitur gabungan (seperti volume atau ukuran total) pada saat preprocessing.
 
### 3.6.2 Categorical Columns
Pada bagian ini, dilakukan pembuatan tabel dan barplot untuk melihat hubungan atau korelasi antara fitur kategorikal dengan label atau target, yaitu price.
<img width="1490" height="1181" alt="image" src="https://github.com/user-attachments/assets/0f0e72d2-fe10-4545-ba92-50221d87607e" />

- Insight yang didapatkan dari Gambar 7:
  - Meskipun terdapat perbedaan kualitas dalam setiap fitur kategorikal (cut, color, clarity), harga rata-rata diamond tetap berada dalam kisaran yang relatif mirip, tanpa fluktuasi yang signifikan. Ini menunjukkan bahwa kualitas kategori tidak secara langsung menentukan harga. Berikut detailnya:

    1. Fitur cut:
    - Rata-rata harga diamond berada dalam rentang 3500 hingga 4500 untuk semua kategori cut. Bahkan, grade tertinggi seperti Ideal justru memiliki harga rata-rata yang lebih rendah dibandingkan beberapa grade lain. Ini menunjukkan bahwa cut memiliki pengaruh yang relatif kecil terhadap harga.

    2. Fitur color:
    - Warna terbaik dimulai dari grade D (paling bening), dan menurun ke E, F, hingga J. Namun, data menunjukkan bahwa harga rata-rata diamond tidak selalu lebih tinggi pada grade warna yang lebih baik. Bahkan, grade D tidak memiliki rata-rata harga tertinggi. Ini mengindikasikan bahwa fitur color memiliki pengaruh yang rendah terhadap harga.

    3. Fitur clarity:
    - Meskipun grade clarity yang lebih tinggi (misalnya IF atau VVS1) secara teknis lebih baik, rata-rata harganya tidak selalu lebih tinggi. Beberapa grade menengah seperti VS2 atau SI1 justru menunjukkan harga rata-rata yang cukup tinggi, menandakan bahwa fitur clarity juga memiliki pengaruh yang terbatas terhadap harga.

  - Ketiga fitur kategorikal (cut, color, dan clarity) memiliki pengaruh yang rendah terhadap perubahan rata-rata harga diamond.

<img width="645" height="528" alt="image" src="https://github.com/user-attachments/assets/2c41ebfb-94d3-4c59-a5bd-7f2cd56997e4" />

- Insight yang didapatkan dari Gambar 8:
  - Berdasarkan uji Chi-Square terhadap fitur kategorikal cut, color, dan clarity, diperoleh bahwa semua pasangan fitur menghasilkan p-value di atas 0.05. Hal ini menunjukkan bahwa:
    - Tidak ada hubungan yang signifikan antara kombinasi cut dan color, cut dan clarity.
    - Secara statistik, kita menerima hipotesis nol yang menyatakan bahwa fitur-fitur ini tidak saling berhubungan (independen).
