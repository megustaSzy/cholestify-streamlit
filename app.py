import streamlit as st
import os
from PIL import Image

# Fungsi untuk membuat logo menjadi persegi (agar tidak peyang di tab browser)
def get_square_logo(path):
    if not os.path.exists(path):
        return "🫀"
    try:
        img = Image.open(path).convert("RGBA")
        width, height = img.size
        if width == height:
            return img
        
        # Buat background transparan berbentuk persegi (mengikuti sisi terpanjang)
        max_dim = max(width, height)
        square_img = Image.new('RGBA', (max_dim, max_dim), (0, 0, 0, 0))
        
        # Tempelkan logo di tengah-tengah
        offset_x = (max_dim - width) // 2
        offset_y = (max_dim - height) // 2
        square_img.paste(img, (offset_x, offset_y), img)
        return square_img
    except Exception:
        return "🫀"

logo_icon = get_square_logo("assets/logo.png")

st.set_page_config(
    page_title="Cholestify App",
    page_icon=logo_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Menambahkan logo asli (tanpa diubah) di bagian atas sidebar
if os.path.exists("assets/logo.png"):
    st.logo("assets/logo.png")

# Konfigurasi halaman menggunakan st.navigation (Tersedia mulai Streamlit 1.36)
pg = st.navigation([
    st.Page("dashboard/cholestify_dashboard.py", title="🍎 Analisis Makanan & Nutrisi", default=True),
    st.Page("dashboard/cholestify_streamlit.py", title="🩸 Analisis & Prediksi Kolesterol")
])

pg.run()
