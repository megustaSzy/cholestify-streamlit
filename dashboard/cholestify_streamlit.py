# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, spearmanr
# pyrefly: ignore [missing-import]
from scipy import stats

import io
import time
# pyrefly: ignore [missing-import]
import cv2
import keras
# pyrefly: ignore [missing-import]
from PIL import Image

# ═══ Konfigurasi Halaman ═════════════════════════════════════════════════════════
# (Diatur di app.py utama)

# ═══ Konstanta ═══════════════════════════════════════════════════════════════════
ALPHA           = 0.05
MAX_DIM         = 800
MAX_FILE_SIZE   = 10 * 1024 * 1024

CAT_ORDER       = ["Normal", "Berisiko", "Kolesterol"]
IMG_SIZE        = (240, 240)
MODEL_PATH      = "model/cholestify_efficientb0.h5"
CLASS_NAMES     = ["normal", "beresiko", "kolesterol"]

C_NORMAL        = "#22C55E"
C_BERISIKO      = "#F59E0B"
C_KOLESTEROL    = "#EF4444"

# ═══ CSS ═════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #080c14; color: #e2e8f0; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #080c14 100%);
    border-right: 1px solid rgba(96,165,250,0.12);
}
section[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
section[data-testid="stSidebar"] .stRadio label {
    background: rgba(30,41,59,0.5);
    border: 1px solid rgba(51,65,85,0.6);
    border-radius: 8px;
    padding: 8px 14px;
    transition: all 0.2s ease;
    width: 100%;
    cursor: pointer;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(96,165,250,0.12);
    border-color: rgba(96,165,250,0.35);
}
[data-testid="stSidebarNavItems"] li > div > a > span { color: #e2e8f0 !important; }
[data-testid="stSidebarNavItems"] li > div > a:hover { background-color: rgba(96,165,250,0.12) !important; }

/* ── Headings ── */
h1 {
    font-size: 2.1rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem !important;
}
h2 {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #94a3b8 !important;
}
h3 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
}

/* ── KPI Metric Cards ── */
.kpi-card {
    background: linear-gradient(145deg, #111827 0%, #0f172a 100%);
    border: 1px solid rgba(51,65,85,0.8);
    border-radius: 16px;
    padding: 22px 20px;
    margin: 6px 0;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    height: 190px;
    display: flex;
    flex-direction: column;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #60a5fa, #34d399);
    opacity: 0.7;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(96,165,250,0.12);
    border-color: rgba(96,165,250,0.3);
}
.kpi-icon { font-size: 1.5rem; margin-bottom: 8px; display: block; }
.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a5f3fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    white-space: nowrap;
}
.kpi-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 4px;
    font-weight: 600;
}
.kpi-delta {
    font-size: 0.75rem;
    color: #34d399;
    margin-top: auto;
    padding-top: 8px;
    border-top: 1px solid rgba(51,65,85,0.5);
    word-break: break-word;
    line-height: 1.3;
}

/* ── Pipeline Steps ── */
.pipeline-step {
    text-align: center;
    background: linear-gradient(145deg, #111827, #0f172a);
    border: 1px solid rgba(51,65,85,0.7);
    border-radius: 12px;
    padding: 16px 10px;
    min-height: 125px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    transition: all 0.25s ease;
    overflow: hidden;
}
.pipeline-step::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #60a5fa, #34d399);
    opacity: 0;
    transition: opacity 0.25s ease;
}
.pipeline-step:hover { 
    border-color: rgba(96,165,250,0.4); 
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(96,165,250,0.1);
}
.pipeline-step:hover::after { opacity: 1; }
.pipeline-num {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    color: #60a5fa;
    font-weight: 700;
    line-height: 1;
}
.pipeline-title {
    font-size: 0.68rem;
    font-weight: 700;
    color: #cbd5e1;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.pipeline-desc {
    font-size: 0.58rem;
    color: #475569;
    margin-top: 4px;
    line-height: 1.3;
}

/* ── Insight Box ── */
.insight-box {
    background: linear-gradient(135deg, #0a1628 0%, #0d1b2a 100%);
    border: 1px solid rgba(30,58,95,0.8);
    border-left: 3px solid #34d399;
    border-radius: 0 12px 12px 0;
    padding: 16px 18px;
    margin: 14px 0;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #94a3b8;
    box-shadow: 0 0 20px rgba(52,211,153,0.04);
}
.insight-box b, .insight-box strong { color: #34d399; }
.insight-box code {
    background: rgba(96,165,250,0.1);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #93c5fd;
}

/* ── Badges ── */
.badge-safe   { background: rgba(20,83,45,0.7);  color: #4ade80; padding: 4px 12px; border-radius: 20px; font-size: 0.73rem; font-weight: 700; border: 1px solid rgba(74,222,128,0.25); }
.badge-neutral{ background: rgba(113,63,18,0.7); color: #fbbf24; padding: 4px 12px; border-radius: 20px; font-size: 0.73rem; font-weight: 700; border: 1px solid rgba(251,191,36,0.25); }
.badge-limit  { background: rgba(124,45,18,0.7); color: #fb923c; padding: 4px 12px; border-radius: 20px; font-size: 0.73rem; font-weight: 700; border: 1px solid rgba(251,146,60,0.25); }
.badge-avoid  { background: rgba(69,10,10,0.7);  color: #f87171; padding: 4px 12px; border-radius: 20px; font-size: 0.73rem; font-weight: 700; border: 1px solid rgba(248,113,113,0.25); }

/* ── Sidebar Brand ── */
.sidebar-brand {
    background: linear-gradient(135deg, rgba(96,165,250,0.08), rgba(52,211,153,0.08));
    border: 1px solid rgba(96,165,250,0.15);
    border-radius: 14px;
    padding: 18px 16px;
    margin-bottom: 4px;
    text-align: center;
}
.sidebar-brand-icon { font-size: 2.2rem; display: block; margin-bottom: 6px; }
.sidebar-brand-title {
    font-size: 1.25rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}
.sidebar-brand-sub { font-size: 0.72rem; color: #475569; margin-top: 3px; }

/* ── Dataset Info Card ── */
.info-card {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(51,65,85,0.5);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 8px 0;
}
.info-card-label { font-size: 0.68rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; }
.info-card-value { font-family: 'Space Mono', monospace; font-size: 1.1rem; color: #60a5fa; font-weight: 700; }

/* ── Dividers ── */
hr { border-color: rgba(51,65,85,0.4) !important; margin: 12px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(51,65,85,0.8); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(96,165,250,0.4); }

/* ── Streamlit overrides ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(51,65,85,0.5);
    border-radius: 8px 8px 0 0;
    padding: 8px 18px;
    font-size: 0.83rem;
    font-weight: 600;
    color: #64748b;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(96,165,250,0.08); color: #94a3b8; }
.stTabs [aria-selected="true"] { 
    background: rgba(96,165,250,0.12) !important; 
    border-color: rgba(96,165,250,0.3) !important;
    color: #60a5fa !important;
}
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ═══ Custom Keras Layers ═══════════════════════════════════════════════════════════════
@keras.saving.register_keras_serializable(package="compat")
class CompatBatchNorm(keras.layers.BatchNormalization):
    """Buang parameter renorm* yang dihapus di Keras 3."""
    def __init__(self, **kwargs):
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        super().__init__(**kwargs)

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="compat")
class CompatInputLayer(keras.layers.InputLayer):
    def __init__(self, **kwargs):
        kwargs.pop("optional", None)
        batch_shape = kwargs.pop("batch_shape", None)
        input_shape = kwargs.pop("input_shape", None)
        given_shape = kwargs.pop("shape", None)

        if given_shape is not None:
            final_shape = given_shape
        elif input_shape is not None:
            final_shape = input_shape
        elif batch_shape is not None:
            final_shape = batch_shape[1:]
        else:
            final_shape = list(IMG_SIZE) + [3]

        kwargs["shape"] = final_shape
        super().__init__(**kwargs)

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="compat")
class CompatDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {
    "BatchNormalization": CompatBatchNorm,
    "InputLayer":        CompatInputLayer,
    "Dense":             CompatDense,
}


# ═══ Loaders ════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Memuat data...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load kedua dataset CSV. Mengembalikan (df_cholesterol, df_nutrition)."""
    try:
        df_cholesterol = pd.read_csv("data/df_cholesterol_cleaned.csv")
        df_nutrition   = pd.read_csv("data/df_nutrition_cleaned.csv")
        return df_cholesterol, df_nutrition
    except FileNotFoundError as e:
        st.error(f"❌ File data tidak ditemukan: {e}")
        st.stop()


@st.cache_resource(show_spinner="Memuat model...")
def load_model(path: str):
    """Load model Keras dengan custom objects. Mengembalikan (model, error_str | None)."""
    try:
        model = keras.saving.load_model(
            path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )
        return model, None
    except Exception as e:
        return None, str(e)


# ═══ Helper ════════════════════════════════════════════════════════════════════════
def ab_test_totchol(group_a: pd.Series, group_b: pd.Series):
    """Pilih uji statistik yang tepat. Mengembalikan (test_name, stat, p_value)."""
    def is_normal(s):
        if len(s) <= 5000:
            return shapiro(s)[1] > ALPHA
        return stats.kstest(s, "norm", args=(s.mean(), s.std()))[1] > ALPHA

    if is_normal(group_a) and is_normal(group_b):
        equal_var = levene(group_a, group_b)[1] > ALPHA
        stat, p   = ttest_ind(group_a, group_b, equal_var=equal_var)
        name      = "Independent t-test" if equal_var else "Welch's t-test"
    else:
        stat, p = mannwhitneyu(group_a, group_b, alternative="two-sided")
        name    = "Mann-Whitney U"

    return name, stat, p

def process_eye_image(
    image_input,
    target_size: tuple[int, int] = IMG_SIZE,
    clip_limit: float = 2.0,
) -> tuple:
    """
    Proses gambar mata untuk prediksi kolesterol.

    Parameters
    ----------
    image_input : file-like object (st.file_uploader) atau numpy array RGB uint8
    target_size : (width, height) output
    clip_limit  : nilai CLAHE clip limit

    Returns
    -------
    (processed_rgb, original_rgb, crop_method, debug_info | error_str)
    """
    if isinstance(image_input, np.ndarray):
        rgb_original = image_input.copy()
    else:
        image_input.seek(0)
        file_bytes   = np.frombuffer(image_input.read(), np.uint8)
        bgr          = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            return None, None, None, "Gagal membaca gambar"
        rgb_original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    img    = rgb_original.copy()
    h0, w0 = img.shape[:2]

    if max(h0, w0) > MAX_DIM:
        scale = MAX_DIM / max(h0, w0)
        img   = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    gray         = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w         = gray.shape
    circles      = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1.2, 100,
        param1=50, param2=30,
        minRadius=int(h * 0.1),
        maxRadius=int(h * 0.5),
    )

    debug_info = {}
    if circles is not None:
        x, y, r = np.round(circles[0, 0]).astype(int)
        r        = int(r * 1.1)
        crop     = img[max(0, y - r):min(h, y + r), max(0, x - r):min(w, x + r)]
        method   = "Hough Circle"
        debug_info["circle"] = (int(x), int(y), int(r))
    else:
        ch, cw = int(h * 0.8), int(w * 0.8)
        y1, x1 = (h - ch) // 2, (w - cw) // 2
        crop   = img[y1:y1 + ch, x1:x1 + cw]
        method = "Center Crop"

    if crop.size == 0:
        return None, rgb_original, None, "Crop kosong — coba gambar lain"

    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

    # CLAHE pada channel L
    lab         = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    l, a, b     = cv2.split(lab)
    clahe       = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    final       = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)

    return final, rgb_original, method, debug_info


def prepare_input(processed_img: np.ndarray) -> np.ndarray:
    """
    Kirim nilai [0, 255] sebagai float32.
    Layer Rescaling di dalam model (scale=1/255) yang akan menormalisasi.
    """
    arr = processed_img.astype(np.float32)
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def predict(model, img_array: np.ndarray, class_names: list[str]) -> dict[str, float]:
    """Mengembalikan dict {class_name: probability}."""
    probs = model.predict(img_array, verbose=0)[0]
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}


def show_comparison(original_rgb: np.ndarray, processed_rgb: np.ndarray, method: str):
    """Tampilkan gambar asli vs processed secara berdampingan."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Gambar Asli", fontsize=13)
    axes[0].axis("off")
    axes[1].imshow(processed_rgb)
    axes[1].set_title(
        f"Processed {processed_rgb.shape[1]}×{processed_rgb.shape[0]}\n({method})",
        fontsize=13,
    )
    axes[1].axis("off")
    plt.tight_layout()
    return fig


# ═══ Session State Init ═════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "📊 Cholesterol Data"

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ═══ Sidebar ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Menu")

    pages = ["📊 Cholesterol Data", "🍔 Food Table", "🔍 Prediction"]
    for page in pages:
        is_active = st.session_state.page == page
        if st.button(page, use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.page = page
            if page == "🍔 Food Table":
                st.session_state.food_tab = "🏠 Overview"
            st.rerun()
            
        # Tampilkan Sub-Menu di bawah "Food Table" jika sedang aktif
        if page == "🍔 Food Table" and st.session_state.page == "🍔 Food Table":
            sub_pages = [
                "🏠 Overview", "📊 EDA & Distribusi", "🔬 Business Questions",
                "⚗️ A/B Testing", "🤖 Feature Engineering", "🎯 Rekomendasi Personal", "📚 Data Dictionary"
            ]
            if "food_tab" not in st.session_state:
                st.session_state.food_tab = "🏠 Overview"
                
            for sp in sub_pages:
                sp_active = st.session_state.food_tab == sp
                # Gunakan markdown column layout kecil atau tombol agar terlihat seperti sub-menu
                col_spacer, col_btn = st.columns([1, 10])
                with col_btn:
                    if st.button(sp, key=f"sub_{sp}", use_container_width=True, type="primary" if sp_active else "secondary"):
                        st.session_state.food_tab = sp
                        st.rerun()

    st.markdown("---")
    st.info("💡 **Tips:** Gunakan halaman 'Prediction' untuk simulasi risiko kolesterol.")


# ═══ Load Data ═════════════════════════════════════════════════════════════════
df_cholesterol, df_nutrition = load_data()


# ════════════════════════════════════════════════════════════════════════════════
# HALAMAN 1: Cholesterol Data
# ════════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "📊 Cholesterol Data":
    st.title("Cholesterol Dashboard")
    st.markdown("Dashboard ini menampilkan tren data terkait dataset kolesterol.")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview & EDA",
        "1️⃣ · Prevalensi",
        "2️⃣ · Gender",
        "3️⃣ · Usia",
        "4️⃣ · Hipertensi & Obesitas",
        "5️⃣ · Merokok",
        "6️⃣ · Korelasi Spearman",
    ])

    # TAB 1 · OVERVIEW & EDA
    with tab1:
        st.subheader("📋 Struktur & Distribusi Data")
        # KPI cards
        n_total    = len(df_cholesterol)
        n_kolesterol = (df_cholesterol["catChol"] == "Kolesterol").sum()
        n_berisiko   = (df_cholesterol["catChol"] == "Berisiko").sum()
        n_normal     = (df_cholesterol["catChol"] == "Normal").sum()

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Statistik Deskriptif")
            num_cols = ["age", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "totChol"]
            st.dataframe(
                df_cholesterol[num_cols].describe().round(2).T
                .rename(columns={"count": "n", "mean": "mean", "std": "std",
                                "min": "min", "25%": "Q1", "50%": "median",
                                "75%": "Q3", "max": "max"}),
                use_container_width=True, height=290,
            )

        with col_right:
            st.markdown("#### Distribusi Fitur Kategorik")
            cat_cols = ["male", "currentSmoker", "prevalentHyp", "diabetes", "prevalentStroke"]
            cat_labels = {
                "male": "Gender (1=Laki-laki)",
                "currentSmoker": "Perokok",
                "prevalentHyp": "Hipertensi",
                "diabetes": "Diabetes",
                "prevalentStroke": "Stroke",
            }
            rows = []
            for c in cat_cols:
                vc = df_cholesterol[c].value_counts()
                for v, cnt in vc.items():
                    rows.append({"Variabel": cat_labels[c], "Nilai": int(v),
                                "Jumlah": cnt, "Proporsi (%)": f"{cnt/n_total*100:.1f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=290, hide_index=True)

        st.divider()

        # Distribution histograms
        st.markdown("#### Distribusi Variabel Numerik")
        hist_col = st.selectbox("Pilih variabel:", num_cols, index=6)
        fig_hist = px.histogram(
            df_cholesterol, x=hist_col, nbins=30,
            color_discrete_sequence=["#3b82f6"],
            labels={hist_col: hist_col},
            title=f"Distribusi {hist_col}",
            template="plotly_dark",
        )
        fig_hist.update_traces(marker_line_color="#1e293b", marker_line_width=0.8)
        fig_hist.update_layout(height=360, margin=dict(t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"))
        st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        # Pearson heatmap
        st.markdown("#### Heatmap Korelasi Pearson")
        exclude = ["catChol", "age_group", "bmi_category", "smoking_intensity",
                "glucose_risk", "smoker_x_age"]
        num_for_corr = [c for c in df_cholesterol.columns
                        if c not in exclude and df_cholesterol[c].dtype in ["int64", "float64"]]
        corr_mtx = df_cholesterol[num_for_corr].corr()

        fig_heat = go.Figure(go.Heatmap(
            z=corr_mtx.values.round(2),
            x=corr_mtx.columns.tolist(),
            y=corr_mtx.columns.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            text=corr_mtx.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 8},
            hoverongaps=False,
        ))
        fig_heat.update_layout(
            height=560, margin=dict(t=30, b=20),
            xaxis=dict(tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Sumber data:</strong><br>
        <a href="https://www.kaggle.com/datasets/yukeshmarudhasalam/framingham"> Yukesh Marudhasalam - Cholesterol Dataset - <em>Kaggle</em></a>.
        </div>
        """, unsafe_allow_html=True)


    # TAB 2 · PB1 · PREVALENSI
    with tab2:
        st.subheader("PB1 · Seberapa banyak pasien dengan kolesterol tinggi?")

        counts = df_cholesterol["catChol"].value_counts().reindex(CAT_ORDER).fillna(0)

        col1, col2 = st.columns([1, 1])

        with col1:
            fig_pie = go.Figure(go.Pie(
                labels=CAT_ORDER,
                values=counts.values,
                marker_colors=[C_NORMAL, C_BERISIKO, C_KOLESTEROL],
                hole=0.45,
                textinfo="label+percent",
                textfont_size=13,
                hovertemplate="<b>%{label}</b><br>Jumlah: %{value:,}<br>Proporsi: %{percent}<extra></extra>",
            ))
            fig_pie.update_layout(
                title="Proporsi Kategori Kolesterol",
                showlegend=True,
                height=380,
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"),
                margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("#### Ringkasan Jumlah")
            for cat, color_hex, bg in zip(
                CAT_ORDER,
                [C_NORMAL, C_BERISIKO, C_KOLESTEROL],
                ["rgba(34,197,94,0.15)", "rgba(234,179,8,0.15)", "rgba(239,68,68,0.15)"],
            ):
                cnt = int(counts[cat])
                pct = cnt / n_total * 100 if n_total else 0
                st.markdown(f"""
                <div style="background:{bg};border-radius:10px;padding:14px 18px;margin-bottom:10px;">
                    <span style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">{cat}</span><br>
                    <span style="font-size:1.8rem;font-weight:800;color:{color_hex};">{cnt:,}</span>
                    <span style="color:#64748b;font-size:0.9rem;"> pasien &nbsp;·&nbsp; {pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Key Insights:</strong><br>
        • Kategori terbanyak adalah <strong>Kolesterol (≥ 240 mg/dL)</strong> — hampir <strong>4 dari 5</strong> pasien memiliki kadar kolesterol di atas batas normal.<br>
        • Hanya <strong>21.2%</strong> pasien yang berada dalam rentang normal (< 200 mg/dL).<br>
        • Kondisi ini mengindikasikan prevalensi kolesterol yang sangat signifikan pada populasi dataset ini.
        </div>
        """, unsafe_allow_html=True)


    # TAB 3 · PB2 · GENDER
    with tab3:
        st.subheader("PB2 · Perbedaan risiko kolesterol antara laki-laki dan perempuan")

        # Pivot
        pivot_g = (
            df_cholesterol.groupby("male")["catChol"]
            .value_counts(normalize=True).mul(100).rename("Pct")
            .reset_index()
        )
        pivot_g["Gender"] = pivot_g["male"].map({0: "Perempuan", 1: "Laki-laki"})
        pivot_g_wide = (
            pivot_g.pivot(index="Gender", columns="catChol", values="Pct")
            .reindex(columns=CAT_ORDER).fillna(0).round(2)
        )
        pivot_g_cnt = (
            df_cholesterol.groupby("male")["catChol"]
            .value_counts().rename("Cnt").reset_index()
        )
        pivot_g_cnt["Gender"] = pivot_g_cnt["male"].map({0: "Perempuan", 1: "Laki-laki"})
        pivot_g_cnt_wide = (
            pivot_g_cnt.pivot(index="Gender", columns="catChol", values="Cnt")
            .reindex(columns=CAT_ORDER).fillna(0).astype(int)
        )

        col1, col2 = st.columns(2)

        with col1:
            fig_bar_g = go.Figure()
            for cat, color in zip(CAT_ORDER, [C_NORMAL, C_BERISIKO, C_KOLESTEROL]):
                fig_bar_g.add_trace(go.Bar(
                    name=cat,
                    x=pivot_g_wide.index.tolist(),
                    y=pivot_g_cnt_wide[cat].values,
                    marker_color=color,
                    text=pivot_g_cnt_wide[cat].values,
                    textposition="outside",
                ))
            fig_bar_g.update_layout(
                barmode="group",
                title="Frekuensi catChol per Gender",
                yaxis_title="Jumlah Pasien",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"),
                height=380,
                legend=dict(orientation="h", y=1.12),
                margin=dict(t=70, b=20),
            )
            st.plotly_chart(fig_bar_g, use_container_width=True)

        with col2:
            fig_bar_g2 = go.Figure()
            for cat, color in zip(CAT_ORDER, [C_NORMAL, C_BERISIKO, C_KOLESTEROL]):
                fig_bar_g2.add_trace(go.Bar(
                    name=cat,
                    x=pivot_g_wide.index.tolist(),
                    y=pivot_g_wide[cat].values,
                    marker_color=color,
                    text=[f"{v:.1f}%" for v in pivot_g_wide[cat].values],
                    textposition="inside",
                    textfont_color="white",
                ))
            fig_bar_g2.update_layout(
                barmode="stack",
                title="Proporsi (%) catChol per Gender",
                yaxis_title="Proporsi (%)",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"),
                height=380,
                legend=dict(orientation="h", y=1.12),
                margin=dict(t=70, b=20),
            )
            st.plotly_chart(fig_bar_g2, use_container_width=True)

        # A/B Testing
        st.markdown("#### 🧪 A/B Testing – totChol: Perempuan vs Laki-laki")
        perempuan = df_cholesterol[df_cholesterol["male"] == 0]["totChol"].dropna()
        laki_laki = df_cholesterol[df_cholesterol["male"] == 1]["totChol"].dropna()

        if len(perempuan) > 1 and len(laki_laki) > 1:
            test_name, stat, p_val = ab_test_totchol(perempuan, laki_laki)

            ab1, ab2, ab3, ab4 = st.columns(4)
            ab1.metric("Metode Uji",         test_name)
            ab2.metric("Mean Perempuan",     f"{perempuan.mean():.2f}")
            ab3.metric("Mean Laki-laki",     f"{laki_laki.mean():.2f}")
            ab4.metric("p-value",            f"{p_val:.4f}",
                    delta="Tolak H₀ ✅" if p_val < ALPHA else "Gagal Tolak H₀",
                    delta_color="normal" if p_val < ALPHA else "off")

            verdict = "**Tolak H₀** — ada perbedaan signifikan rata-rata totChol antar gender." \
                    if p_val < ALPHA else \
                    "**Gagal Tolak H₀** — tidak ada perbedaan signifikan."
            st.info(f"α = {ALPHA} | {verdict}")
        else:
            st.warning("Data tidak cukup untuk A/B Testing dengan filter saat ini.")

        st.markdown("""
        <div class="insight-box">
        <strong>Key Insights:</strong><br>
        • <strong>Perempuan</strong> mayoritas berada di kategori kolesterol (≈ 46%), distribusi condong ke risiko tinggi.<br>
        • <strong>Laki-laki</strong> memiliki pola lebih merata antara kategori <em>Berisiko</em> (39.99%) dan <em>Kolesterol</em> (38.77%) hampir seimbang.<br>
        • A/B Testing (α = 0.05) membuktikan perbedaan rata-rata totChol antar gender <strong>signifikan secara statistik</strong>.
        </div>
        """, unsafe_allow_html=True)


    # TAB 4 · PB3 · USIA
    with tab4:
        st.subheader("PB3 · Apakah kelompok usia tertentu lebih rentan kolesterol?")

        age_order = ["< 35", "35-55", "> 55"]
        pivot_age = (
            df_cholesterol.groupby("age_group", observed=True)["catChol"]
            .value_counts(normalize=True).mul(100).rename("Proporsi")
            .reset_index()
        )
        pivot_age_wide = (
            pivot_age.pivot(index="age_group", columns="catChol", values="Proporsi")
            .reindex(index=age_order, columns=CAT_ORDER).fillna(0)
        )

        col1, col2 = st.columns([3, 2])

        with col1:
            fig_stacked = go.Figure()
            for cat, color in zip(CAT_ORDER, [C_NORMAL, C_BERISIKO, C_KOLESTEROL]):
                vals = [pivot_age_wide.loc[ag, cat] if ag in pivot_age_wide.index else 0
                        for ag in age_order]
                fig_stacked.add_trace(go.Bar(
                    name=cat, x=age_order, y=vals,
                    marker_color=color,
                    text=[f"{v:.1f}%" for v in vals],
                    textposition="inside",
                    textfont_color="white",
                ))
            fig_stacked.update_layout(
                barmode="stack",
                title="Proporsi catChol per Kelompok Usia",
                xaxis_title="Kelompok Usia",
                yaxis_title="Proporsi (%)",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"),
                height=400,
                legend=dict(orientation="h", y=1.12),
                margin=dict(t=70, b=20),
            )
            st.plotly_chart(fig_stacked, use_container_width=True)

        with col2:
            age_mean = (
                df_cholesterol.groupby("age_group", observed=True)["totChol"]
                .mean().reindex(age_order).round(1).reset_index()
            )
            age_mean.columns = ["Kelompok Usia", "Mean totChol (mg/dL)"]
            fig_line = px.bar(
                age_mean, x="Kelompok Usia", y="Mean totChol (mg/dL)",
                color="Mean totChol (mg/dL)",
                color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
                text="Mean totChol (mg/dL)",
                template="plotly_dark",
                height=400,
            )
            fig_line.update_traces(textposition="outside")
            fig_line.update_layout(
                title="Rata-rata totChol per Kelompok Usia",
                coloraxis_showscale=False, 
                margin=dict(t=70, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8")
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # Scatter age vs totChol
        st.markdown("#### Scatter: Usia vs Total Kolesterol")
        fig_scatter = px.scatter(
            df_cholesterol.sample(min(1500, len(df_cholesterol)), random_state=1),
            x="age", y="totChol", color="catChol",
            color_discrete_map={
                "Normal": C_NORMAL,
                "Berisiko": C_BERISIKO,
                "Kolesterol": C_KOLESTEROL,
            },
            trendline="ols",
            labels={"age": "Usia (tahun)", "totChol": "Total Kolesterol (mg/dL)"},
            template="plotly_dark",
            height=380,
            opacity=0.55,
        )
        fig_scatter.update_layout(margin=dict(t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"))
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Key Insights:</strong><br>
        • Kelompok <strong>&lt; 35 tahun</strong>: 66.67% normal (kelompok paling sehat). Hanya 8.33% yang termasuk kategori kolesterol.<br>
        • Kelompok <strong>35–55 tahun</strong>: terjadi pergeseran signifikan dengan proporsi normal turun ke ≈ 24%, kolesterol naik ke ≈ 38%.<br>
        • Kelompok <strong>&gt; 55 tahun</strong>: &gt; sebagian besar (55.64%) berada di kategori kolesterol, kelompok dengan risiko tertinggi.<br>
        • Tren linier positif antara usia dan totChol terlihat jelas (Spearman r ≈ 0.29).
        </div>
        """, unsafe_allow_html=True)


    # TAB 5 · PB4 · HIPERTENSI & OBESITAS
    with tab5:
        st.subheader("PB4 · Apakah hipertensi dan obesitas memperburuk risiko kolesterol?")

        # Chart 1 – Stacked bar by highBP
        pivot_bp = (
            df_cholesterol.groupby("highBP")["catChol"]
            .value_counts(normalize=True).mul(100).rename("Pct")
            .reset_index()
        )
        pivot_bp["Hipertensi"] = pivot_bp["highBP"].map({0: "Tanpa Hipertensi", 1: "Hipertensi"})
        pivot_bp_wide = (
            pivot_bp.pivot(index="Hipertensi", columns="catChol", values="Pct")
            .reindex(columns=CAT_ORDER)
        )

        col1, col2 = st.columns(2)

        with col1:
            fig_bp = go.Figure()
            for cat, color in zip(CAT_ORDER, [C_NORMAL, C_BERISIKO, C_KOLESTEROL]):
                idx = ["Tanpa Hipertensi", "Hipertensi"]
                vals = [pivot_bp_wide.loc[i, cat] if i in pivot_bp_wide.index else 0 for i in idx]
                fig_bp.add_trace(go.Bar(
                    name=cat, x=idx, y=vals,
                    marker_color=color,
                    text=[f"{v:.1f}%" for v in vals],
                    textposition="inside",
                    textfont_color="white",
                ))
            fig_bp.update_layout(
                barmode="stack",
                title="Proporsi catChol per Status Hipertensi",
                yaxis_title="Proporsi (%)",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"),
                height=360,
                legend=dict(orientation="h", y=1.12),
                margin=dict(t=70, b=20),
            )
            st.plotly_chart(fig_bp, use_container_width=True)

        with col2:
            # Heatmap BP × BMI
            bmi_order = ["Underweight", "Normal", "Overweight", "Obese"]
            heatmap_data = (
                df_cholesterol.groupby(["highBP", "bmi_category"], observed=True)["totChol"]
                .mean().unstack()
                .reindex(columns=bmi_order)
                .round(1)
            )
            heatmap_data.index = ["Tanpa Hipertensi", "Hipertensi"]

            fig_hm = go.Figure(go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns.tolist(),
                y=heatmap_data.index.tolist(),
                colorscale="RdYlGn_r",
                text=heatmap_data.values,
                texttemplate="%{text:.1f}",
                textfont={"size": 13},
                hoverongaps=False,
                colorbar=dict(title="Mean totChol"),
            ))
            fig_hm.update_layout(
                title="Rata-rata totChol: Hipertensi × BMI Category",
                xaxis_title="BMI Category",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"),
                height=360,
                margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        # Chart 3 – Obesitas × Hipertensi
        st.markdown("#### Kombinasi Obesitas & Hipertensi")
        pivot_oxh = (
            df_cholesterol.groupby("obese_x_hyp")["catChol"]
            .value_counts(normalize=True).mul(100).rename("Pct")
            .reset_index()
        )
        pivot_oxh["Grup"] = pivot_oxh["obese_x_hyp"].map({
            0: "Tidak Obesitas\n& Tidak HT",
            1: "Obesitas\n& Hipertensi",
        })
        pivot_oxh_kol = (
            pivot_oxh[pivot_oxh["catChol"] == "Kolesterol"]
            .set_index("Grup")["Pct"]
        )

        fig_combo = go.Figure(go.Bar(
            x=pivot_oxh_kol.index.tolist(),
            y=pivot_oxh_kol.values,
            marker_color=["#f59e0b", "#ef4444"],
            text=[f"{v:.1f}%" for v in pivot_oxh_kol.values],
            textposition="outside",
            width=0.4,
        ))
        fig_combo.update_layout(
            title="% Pasien Kolesterol: Kombinasi Obesitas & Hipertensi",
            yaxis_title="Proporsi Kolesterol (%)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"),
            height=360,
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig_combo, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Key Insights:</strong><br>
        • Penderita <strong>hipertensi</strong> memiliki proporsi kolesterol lebih besar (≈ 52% vs 38%) daripada yang tidak ada riwayat hipertensi.<br>
        • <strong>Heatmap</strong>: Kombinasi terburuk adalah <em>Hipertensi + Overweight</em> (mean totChol ≈ 247 mg/dL); terbaik adalah <em>Tanpa Hipertensi + Normal BMI</em> (≈ 227 mg/dL).<br>
        • Individu dengan <strong>obesitas sekaligus hipertensi</strong> memiliki proporsi kolesterol hingga <strong>55.7%</strong>, jauh lebih tinggi dibanding kelompok tanpa kedua kondisi tersebut.
        </div>
        """, unsafe_allow_html=True)


    # TAB 6 · PB5 · MEROKOK
    with tab6:
        st.subheader("PB5 · Perbedaan risiko kolesterol antara perokok dan non-perokok")

        pivot_smoke_age = (
            df_cholesterol.groupby(["age_group", "currentSmoker"], observed=True)["totChol"]
            .mean().unstack()
        )
        pivot_smoke_age.index = pd.CategoricalIndex(
            pivot_smoke_age.index, categories=["< 35", "35-55", "> 55"], ordered=True
        )
        pivot_smoke_age = pivot_smoke_age.sort_index()
        pivot_smoke_age.columns = ["Non-Smoker", "Smoker"]

        col1, col2 = st.columns(2)

        with col1:
            x_idx = pivot_smoke_age.index.astype(str).tolist()
            fig_smoke = go.Figure()
            for col_name, color in [("Non-Smoker", "#2ecc71"), ("Smoker", "#e74c3c")]:
                if col_name in pivot_smoke_age.columns:
                    fig_smoke.add_trace(go.Bar(
                        name=col_name,
                        x=x_idx,
                        y=pivot_smoke_age[col_name].values,
                        marker_color=color,
                        text=[f"{v:.1f}" for v in pivot_smoke_age[col_name].values],
                        textposition="outside",
                    ))
            fig_smoke.update_layout(
                barmode="group",
                title="Rata-rata totChol per Kelompok Usia",
                xaxis_title="Kelompok Usia",
                yaxis_title="Rata-rata totChol (mg/dL)",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"),
                height=380,
                legend=dict(orientation="h", y=1.12),
                margin=dict(t=80, b=20),
            )
            st.plotly_chart(fig_smoke, use_container_width=True)

        with col2:
            # Age distribution by smoker status
            age_ns = df_cholesterol[df_cholesterol["currentSmoker"] == 0]["age"]
            age_s  = df_cholesterol[df_cholesterol["currentSmoker"] == 1]["age"]

            fig_age_dist = go.Figure()
            fig_age_dist.add_trace(go.Histogram(
                x=age_ns, name=f"Non-Smoker (mean={age_ns.mean():.1f})",
                marker_color="#2ecc71", opacity=0.65, nbinsx=30,
            ))
            fig_age_dist.add_trace(go.Histogram(
                x=age_s, name=f"Smoker (mean={age_s.mean():.1f})",
                marker_color="#e74c3c", opacity=0.65, nbinsx=30,
            ))
            fig_age_dist.add_vline(x=age_ns.mean(), line_dash="dash",
                                line_color="#27ae60", annotation_text="Mean NS")
            fig_age_dist.add_vline(x=age_s.mean(), line_dash="dash",
                                line_color="#c0392b", annotation_text="Mean S")
            fig_age_dist.update_layout(
                barmode="overlay",
                title="Distribusi Usia: Smoker vs Non-Smoker",
                xaxis_title="Usia",
                yaxis_title="Frekuensi",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"),
                height=380,
                legend=dict(orientation="h", y=1.12),
                margin=dict(t=80, b=20),
            )
            st.plotly_chart(fig_age_dist, use_container_width=True)

        

        st.markdown("""
        <div class="insight-box">
        <strong>Key Insights:</strong><br>
        • Tidak terlihat perbedaan signifikan rata-rata totChol antara perokok dan non-perokok <em>dalam kelompok usia yang sama</em>.<br>
        • <strong>Confounding variable terungkap</strong>: perokok pada dataset ini rata-rata <em>lebih muda</em> dibanding non-perokok, sehingga secara keseluruhan kolesterol perokok tampak lebih rendah.<br>
        • Efek merokok terhadap kolesterol tidak dapat disimpulkan secara kausal dari observasi ini tanpa mengontrol variabel usia terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)


    # TAB 7 · PB6 · SPEARMAN CORRELATION
    with tab7:
        st.subheader("PB6 · Faktor yang paling kuat berhubungan dengan total kolesterol")

        exclude_cols = ["catChol", "age_group", "bmi_category", "smoking_intensity",
                        "glucose_risk", "totChol"]
        numeric_cols = [c for c in df_cholesterol.columns
                        if c not in exclude_cols and df_cholesterol[c].dtype in ["int64", "float64"]]

        label_map = {
            "age": "Usia", "sysBP": "sysBP", "diaBP": "diaBP",
            "pulse_pressure": "Pulse Pressure", "bp_ratio": "BP Ratio",
            "BMI": "BMI", "heartRate": "Heart Rate", "glucose": "Glucose",
            "cigsPerDay": "Cigs/Day", "male": "Gender (Male)",
            "currentSmoker": "Merokok", "BPMeds": "BP Meds",
            "prevalentStroke": "Stroke", "prevalentHyp": "Hipertensi",
            "diabetes": "Diabetes", "highBP": "High BP",
            "hyp_x_diabetes": "Hipertensi×Diabetes",
            "smoker_x_age": "Smoker×Usia", "obese_x_hyp": "Obesitas×HT",
        }

        spearman_results = {}
        for col in numeric_cols:
            r, p = spearmanr(df_cholesterol[col], df_cholesterol["totChol"], nan_policy="omit")
            spearman_results[label_map.get(col, col)] = {"r": r, "p": p}

        spearman_df = pd.DataFrame(spearman_results).T.reset_index()
        spearman_df.columns = ["Faktor", "Spearman r", "p-value"]
        spearman_df["abs_r"] = spearman_df["Spearman r"].abs()
        spearman_df = spearman_df.nlargest(15, "abs_r").sort_values("Spearman r")

        fig_spearman = go.Figure(go.Bar(
            x=spearman_df["Spearman r"],
            y=spearman_df["Faktor"],
            orientation="h",
            marker_color=["#f87171" if v >= 0 else "#60a5fa"
                        for v in spearman_df["Spearman r"]],
            text=[f"r = {v:.3f}" for v in spearman_df["Spearman r"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Spearman r: %{x:.3f}<extra></extra>",
        ))
        fig_spearman.add_vline(x=0, line_width=1, line_color="black")
        fig_spearman.add_vline(x=0.3,  line_dash="dot", line_color="gray",
                            annotation_text="r=0.3", annotation_position="top right")
        fig_spearman.add_vline(x=-0.3, line_dash="dot", line_color="gray",
                            annotation_text="r=-0.3", annotation_position="top left")
        fig_spearman.update_layout(
            title="Rangkuman Kekuatan Faktor Risiko (Top 15 Spearman Correlation vs totChol)",
            xaxis_title="Spearman r",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#94a3b8"),
            height=520,
            margin=dict(t=60, b=20, l=180),
        )
        st.plotly_chart(fig_spearman, use_container_width=True)

        # Table
        st.markdown("#### Tabel Lengkap Spearman Correlation")
        display_df = (
            spearman_df[["Faktor", "Spearman r", "p-value"]]
            .sort_values("Spearman r", ascending=False)
            .reset_index(drop=True)
        )
        display_df["Spearman r"] = display_df["Spearman r"].round(4)
        display_df["p-value"]    = display_df["p-value"].apply(lambda x: f"{x:.4f}")
        display_df["Interpretasi"] = display_df["Spearman r"].apply(
            lambda r: "🔴 Lemah-Sedang Positif" if r >= 0.2
            else ("🟡 Lemah Positif" if r > 0
                else ("🔵 Lemah Negatif" if r > -0.2
                        else "🟣 Lemah-Sedang Negatif"))
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=460)

        st.markdown("""
        <div class="insight-box">
        <strong>Key Insights:</strong><br>
        • <strong>Usia</strong> (r ≈ 0.29) adalah faktor paling dominan, semakin tua semakin tinggi pula risiko kolesterol.<br>
        • Variabel <strong>kardiovaskular</strong> (<code>sysBP</code>, <code>diaBP</code>, <code>Pulse Pressure</code>, <code>High BP</code>, <code>Hipertensi</code>) mendominasi daftar korelasi positif.<br>
        • <strong>Gender laki-laki</strong> berkorelasi <em>negatif</em>, sementara perempuan cenderung memiliki totChol lebih tinggi.<br>
        • Semua nilai r < 0.30, mengonfirmasi bahwa kolesterol bersifat <strong>multifaktorial</strong> dan tidak bisa dijelaskan oleh satu faktor tunggal.
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# HALAMAN 2: Food Table
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "🍔 Food Table":
    import runpy
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "cholestify_dashboard.py")
    runpy.run_path(dashboard_path, run_name="__main__")


# ════════════════════════════════════════════════════════════════════════════════
# HALAMAN 3: Prediction
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "🔍 Prediction":
    st.title("Prediksi Risiko Kolesterol")
    st.markdown("Masukkan gambar mata untuk memprediksi tingkat risiko kolesterol secara real-time.")

    col1, col2, col3 = st.columns(3)
    with col1:
        clip_limit = st.slider("CLAHE Clip Limit", 1.0, 4.0, 2.0, 0.5)
    with col2:
        target_w = st.number_input("Target Width",  value=IMG_SIZE[0], step=8)
    with col3:
        target_h = st.number_input("Target Height", value=IMG_SIZE[1], step=8)
    target_size = (int(target_w), int(target_h))

    # Load model
    model, model_err = load_model(MODEL_PATH)

    if model:
        st.success("✅ Model berhasil dimuat")
        with st.expander("Info model"):
            buf = io.StringIO()
            model.summary(print_fn=lambda x: buf.write(x + "\n"))
            st.code(buf.getvalue(), language="text")
    else:
        st.warning(f"⚠️ Model tidak dimuat: {model_err}")

    # Upload gambar
    uploaded_files = st.file_uploader(
        "Upload gambar mata (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("📂 Belum ada gambar diupload")

    for uploaded_file in uploaded_files:
        # Validasi ukuran file
        if uploaded_file.size > MAX_FILE_SIZE:
            st.warning(f"⚠️ `{uploaded_file.name}` melebihi batas 10 MB — dilewati.")
            continue

        st.markdown(f"---\n### 🖼️ `{uploaded_file.name}`")
        col_img, col_result = st.columns([1.3, 1], gap="large")

        with st.spinner("Memproses gambar…"):
            processed, original, method, debug = process_eye_image(
                uploaded_file,
                target_size=target_size,
                clip_limit=clip_limit,
            )

        if processed is None:
            st.error(f"❌ {debug}")
            continue

        with col_img:
            fig = show_comparison(original, processed, method)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            badge = "🟢" if method == "Hough Circle" else "🟡"
            st.caption(f"{badge} Crop method: **{method}**")
            if isinstance(debug, dict) and "circle" in debug:
                x, y, r = debug["circle"]
                st.caption(f"Lingkaran — pusat ({x}, {y}), radius {r}px")


        with col_result:
            st.subheader("📊 Hasil Prediksi")
            img_array = prepare_input(processed)

            if model:
                with st.spinner("Inferensi…"):
                    t0      = time.time()
                    results = predict(model, img_array, CLASS_NAMES)
                    elapsed = time.time() - t0

                top_class = max(results, key=results.get)
                top_conf  = results[top_class]

                badge_map = {"normal": "🟢", "beresiko": "🟡", "kolesterol": "🔴"}
                badge     = badge_map.get(top_class.lower(), "⚪")
                st.metric("Prediksi", f"{badge} {top_class}", f"{top_conf:.1%} confidence")
                st.caption(
                    f"Waktu inferensi: {elapsed * 1000:.1f} ms  |  "
                    f"Params — clip_limit={clip_limit}, size={target_size}"
                )

                st.markdown("**Distribusi probabilitas:**")
                for cls, conf in sorted(results.items(), key=lambda x: -x[1]):
                    b = badge_map.get(cls.lower(), "⚪")
                    st.progress(conf, text=f"{b} {cls}: {conf:.1%}")

            else:
                st.info("Model tidak dimuat — shape tensor input:")
                st.code(
                    f"shape = {img_array.shape}  # (batch, H, W, C)\n"
                    f"dtype = {img_array.dtype}\n"
                    f"range = [{img_array.min():.1f}, {img_array.max():.1f}]"
                )

            # Download hasil preprocessing
            buf = io.BytesIO()
            Image.fromarray(processed).save(buf, format="PNG")
            st.download_button(
                "⬇️ Download Processed Image",
                data=buf.getvalue(),
                file_name=f"processed_{uploaded_file.name}",
                mime="image/png",
            )
