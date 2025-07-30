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


