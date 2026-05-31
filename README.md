# Cholestify

Cholestify adalah sebuah dashboard interaktif berbasis Streamlit yang dirancang untuk menganalisis data terkait kolesterol, menyajikan informasi nutrisi makanan, dan menyediakan fitur prediksi risiko kolesterol menggunakan model _deep learning_ (EfficientNetB0) berdasarkan gambar mata.

## 🌟 Fitur Utama

- **📊 Cholesterol Data**: Visualisasi dan analisis data eksploratif (EDA) dari dataset, mencakup prevalensi, analisis gender, usia, hipertensi, obesitas, dan korelasi antar variabel.
- **🍔 Food Table**: Menampilkan dataset informasi nutrisi untuk berbagai jenis makanan.
- **🔍 Prediction**: Memprediksi tingkat risiko kolesterol (Normal, Berisiko, Kolesterol) dengan mengunggah gambar mata. Gambar akan diproses menggunakan OpenCV dan diprediksi dengan model Keras (EfficientNetB0).

## 🚀 Setup Environment & Instalasi

Ikuti langkah-langkah berikut untuk menjalankan aplikasi secara lokal. Disarankan menggunakan lingkungan virtual (seperti Conda atau venv).

### 1. Buat Virtual Environment
Jika menggunakan conda:
```bash
conda create --name cholestify python=3.10 -y
conda activate cholestify
```

### 2. Instal Dependensi
Pastikan Anda berada di direktori utama proyek (`Cholestify/`) lalu jalankan perintah berikut untuk menginstal semua library yang dibutuhkan:

Upgrade pip terlebih dahulu:
```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
Pastikan Anda berada di root direktori proyek (`Cholestify/`) lalu jalankan aplikasi menggunakan Streamlit:
```bash
python -m streamlit run dashboard/cholestify_streamlit.py
```

## 📂 Struktur Repository

```text
.
├── analisis data/
│   ├── cholestify_cholesterol-notebook.ipynb
│   └── cholestify_food-table-notebook.ipynb
├── dashboard/
│   └── cholestify_streamlit.py     # Script utama Streamlit dashboard
├── model/
│   ├── cholestify_efficientb0.h5
│   └── cholestify_efficientb0.keras
├── data/
│   ├── df_cholesterol.csv
│   ├── df_cholesterol_cleaned.csv
│   ├── df_nutrition.csv
│   ├── df_nutrition_cleaned.csv
│   └── df_food_status_LDL145_HDL42.csv
├── README.md                       # Dokumentasi utama proyek
└── requirements.txt                # Daftar dependensi yang diperlukan
```

## 🛠️ Teknologi yang Digunakan
- [Streamlit](https://streamlit.io/) - Framework untuk antarmuka web dashboard interaktif.
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) - Manipulasi dan analisis data tabular.
- [Plotly](https://plotly.com/) & [Matplotlib](https://matplotlib.org/) - Pembuatan visualisasi data dan grafik.
- [SciPy](https://scipy.org/) - Perhitungan statistika (A/B Testing, korelasi, dll).
- [OpenCV](https://opencv.org/) - Pemrosesan gambar untuk ekstraksi area mata dan peningkatan kontras.
- [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/) - Pembuatan dan implementasi model _deep learning_ untuk klasifikasi gambar.
