import streamlit as st

st.set_page_config(
    page_title="Cholestify App",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Menambahkan logo di bagian atas sidebar
import os
if os.path.exists("assets/logo.png"):
    st.logo("assets/logo.png")

# Konfigurasi halaman menggunakan st.navigation (Tersedia mulai Streamlit 1.36)
pg = st.navigation([
    st.Page("dashboard/cholestify_dashboard.py", title="🍎 Analisis Makanan & Nutrisi", default=True),
    st.Page("dashboard/cholestify_streamlit.py", title="🩸 Analisis & Prediksi Kolesterol")
])

pg.run()
