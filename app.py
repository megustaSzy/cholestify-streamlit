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

# Definisi Halaman
page1 = st.Page("dashboard/cholestify_dashboard.py", title="Dashboard Nutrisi Makanan", icon="🍎", default=True)
page2 = st.Page("dashboard/cholestify_streamlit.py", title="Prediksi & Data Kolesterol", icon="🩸")

# Sembunyikan navigasi bawaan dari sidebar
pg = st.navigation([page1, page2], position="hidden")

# Buat Custom Top Menu (Menu Atas Horizontal)
st.markdown("<div style='margin-top: -30px;'></div>", unsafe_allow_html=True)
st.markdown("### 🫀 Cholestify App Menu")
col1, col2 = st.columns(2)
with col1:
    st.page_link(page1, use_container_width=True)
with col2:
    st.page_link(page2, use_container_width=True)
st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'/>", unsafe_allow_html=True)

pg.run()
