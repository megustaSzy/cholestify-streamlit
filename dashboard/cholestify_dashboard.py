"""
╔══════════════════════════════════════════════════════════════╗
║        CHOLESTIFY — Interactive Data Science Dashboard       ║
║   Analisis Nutrisi Makanan Indonesia untuk Manajemen         ║
║   Kolesterol | Dataset: 1.346 Makanan Indonesia              ║
╚══════════════════════════════════════════════════════════════╝

Jalankan dengan:
    pip install streamlit pandas numpy plotly scipy scikit-learn statsmodels
    streamlit run dashboard/cholestify_dashboard.py

Letakkan file CSV di folder data/ dengan nama:
    df_nutrition_cleaned.csv
"""

# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
from plotly.subplots import make_subplots
# pyrefly: ignore [missing-import]
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
import glob
warnings.filterwarnings("ignore")


# ═══ PAGE CONFIG ════════════════════════════════════════════════════════════════

# (Diatur di app.py utama)


# ═══ CUSTOM CSS ═════════════════════════════════════════════════════════════════

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


# ═══ CONSTANTS ══════════════════════════════════════════════════════════════════

PLOTLY_TEMPLATE = "plotly_dark"
COLORS = {
    "blue":   "#60a5fa", "green":  "#34d399", "orange": "#fb923c",
    "red":    "#f87171", "yellow": "#fbbf24", "purple": "#a78bfa",
    "teal":   "#22d3ee", "gray":   "#64748b",
}
KATEGORI_COLORS = {
    "Tinggi Protein, Rendah Lemak": "#16a34a",
    "Rendah Lemak & Karbo":         "#22c55e",
    "Rendah Lemak":                  "#86efac",
    "Sedang":                        "#fbbf24",
    "Tinggi Protein & Lemak":       "#fb923c",
    "Tinggi Lemak":                  "#ea580c",
}
STATUS_COLORS = {
    "BISA":       "#16a34a",
    "NETRAL":     "#ca8a04",
    "LIMIT":      "#ea580c",
    "TIDAK_BISA": "#dc2626",
}
RISK_THRESHOLDS = {
    0: dict(fat_bisa=20, fat_netral=30, fat_limit=40, cal_limit=500, label="Optimal"),
    1: dict(fat_bisa=15, fat_netral=25, fat_limit=35, cal_limit=450, label="Baik"),
    2: dict(fat_bisa=10, fat_netral=18, fat_limit=25, cal_limit=400, label="Sedang"),
    3: dict(fat_bisa=6,  fat_netral=12, fat_limit=18, cal_limit=350, label="Tinggi"),
    4: dict(fat_bisa=3,  fat_netral=8,  fat_limit=12, cal_limit=300, label="Kritis"),
}


# ═══ LOAD DATA ══════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df, csv_path
    except Exception:
        return None, None


@st.cache_data
def add_rekomendasi_score(df):
    """Tambahkan rekomendasi_score jika belum ada di CSV."""
    df = df.copy()
    if "rekomendasi_score" not in df.columns:
        scaler = MinMaxScaler()
        df["fat_norm"]     = scaler.fit_transform(df[["fat"]])
        df["protein_norm"] = scaler.fit_transform(df[["proteins"]])
        df["cal_norm"]     = scaler.fit_transform(df[["calories"]])
        df["rekomendasi_score"] = (
            0.4 * df["protein_norm"] +
            0.4 * (1 - df["fat_norm"]) +
            0.2 * (1 - abs(df["cal_norm"] - 0.3))
        ).round(4)
    return df


# ── Load Dataset ──────────────────────────────────────────
CSV_CANDIDATES = [
    "df_nutrition_cleaned.csv",
    "data/df_nutrition_cleaned.csv",
    "../data/df_nutrition_cleaned.csv",
]
df_raw = None
csv_path = None
for path in CSV_CANDIDATES:
    if os.path.exists(path):
        df_raw, csv_path = load_data(path)
        if df_raw is not None:
            break

if df_raw is None:
    st.error("⚠️ File dataset tidak ditemukan.\n\nPastikan `df_nutrition_cleaned.csv` ada di folder `data/`.")
    st.stop()

df = add_rekomendasi_score(df_raw)

# Mengambil status halaman aktif dari navigasi utama (app.py / cholestify_streamlit.py)
page = st.session_state.get("food_tab", "🏠 Overview")

with st.sidebar:
    st.markdown("---")
    
    # ── Dataset Info ──────────────────────────────────────────
    st.markdown("<p style='font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px'>📦 Dataset Info</p>", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
        <div class='info-card'>
            <div class='info-card-label'>Makanan</div>
            <div class='info-card-value'>{len(df):,}</div>
        </div>""", unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div class='info-card'>
            <div class='info-card-label'>Kolom</div>
            <div class='info-card-value'>{df.shape[1]}</div>
        </div>""", unsafe_allow_html=True)
    st.caption(f"📁 `{csv_path}`")
    st.markdown("---")
    # Legend badges
    st.markdown("""
    <p style='font-size:0.68rem;color:#475569;margin-bottom:6px'>LEGENDA STATUS</p>
    <div style='display:flex;flex-wrap:wrap;gap:6px'>
        <span class='badge-safe'>BISA</span>
        <span class='badge-neutral'>NETRAL</span>
        <span class='badge-limit'>LIMIT</span>
        <span class='badge-avoid'>HINDARI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.65rem;color:#1e293b;text-align:center'>
        Cholestify v1.0 &nbsp;·&nbsp; Data Science Project
    </div>""", unsafe_allow_html=True)


# ═══ HELPERS ════════════════════════════════════════════════════════════════════

def get_risk_level(ldl: float, hdl: float):
    if ldl < 100:   ldl_score = 0
    elif ldl < 130: ldl_score = 1
    elif ldl < 160: ldl_score = 2
    elif ldl < 190: ldl_score = 3
    else:           ldl_score = 4
    hdl_mod = -1 if hdl >= 60 else (0 if hdl >= 40 else 1)
    risk = max(0, min(4, ldl_score + hdl_mod))
    return risk, RISK_THRESHOLDS[risk]


def classify_food_status(row, ldl, hdl):
    _, t = get_risk_level(ldl, hdl)
    fat = row["fat"]
    if   fat <= t["fat_bisa"]:   status = "BISA"
    elif fat <= t["fat_netral"]: status = "NETRAL"
    elif fat <= t["fat_limit"]:  status = "LIMIT"
    else:                        status = "TIDAK_BISA"
    if row["proteins"] >= 15 and status == "LIMIT":  status = "NETRAL"
    if row["calories"] > t["cal_limit"] and status == "BISA": status = "NETRAL"
    return status


CHART_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8"),
    margin=dict(l=10, r=10, t=30, b=10),
)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown("# 🫀 Cholestify Dashboard")
    st.markdown("<p style='color:#475569;margin-top:-8px;margin-bottom:20px'>Analisis Nutrisi Makanan Indonesia untuk Manajemen Kolesterol</p>", unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────
    total      = len(df)
    mean_cal   = df["calories"].mean()
    median_cal = df["calories"].median()
    mean_fat   = df["fat"].mean()
    mean_prot  = df["proteins"].mean()
    n_safe     = (df["risiko_kolesterol"] == 0).sum()

    kpi_data = [
        ("🍽️", f"{total:,}",            "Total Makanan",   f"Sumber: {csv_path.split('/')[-1]}"),
        ("🔥", f"{mean_cal:.0f} kkal",   "Mean Kalori",     f"Median: {median_cal:.0f} kkal"),
        ("🥑", f"{mean_fat:.1f} g",      "Mean Lemak",      f"Maks: {df['fat'].max():.0f}g"),
        ("💪", f"{mean_prot:.1f} g",     "Mean Protein",    f"Maks: {df['proteins'].max():.0f}g"),
        ("✅", f"{n_safe/total*100:.1f}%","Risiko Rendah",   f"{n_safe} dari {total} makanan"),
    ]
    for col, (icon, val, label, delta) in zip(st.columns(5), kpi_data):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <span class='kpi-icon'>{icon}</span>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-delta'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Distribusi Kategori Nutrisi")
        kat = df["kategori"].value_counts().reset_index()
        kat.columns = ["kategori", "count"]
        kat["pct"] = (kat["count"] / total * 100).round(1)
        fig = px.bar(kat, x="count", y="kategori", orientation="h",
                     color="kategori", color_discrete_map=KATEGORI_COLORS,
                     text=kat.apply(lambda r: f"{r['count']}  ({r['pct']}%)", axis=1))
        fig.update_traces(textposition="outside")
        fig.update_layout(**CHART_LAYOUT, height=340, showlegend=False,
                          yaxis_title="", xaxis_title="Jumlah Makanan")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Risiko Kolesterol")
        risiko_map = {0: "Rendah (Aman)", 1: "Sedang (Batasi)", 2: "Tinggi (Hindari)"}
        rc = df["risiko_kolesterol"].map(risiko_map).value_counts().reset_index()
        rc.columns = ["risiko", "count"]
        fig2 = px.pie(rc, names="risiko", values="count", hole=0.5,
                      color="risiko",
                      color_discrete_map={
                          "Rendah (Aman)":   "#16a34a",
                          "Sedang (Batasi)": "#ca8a04",
                          "Tinggi (Hindari)":"#dc2626",
                      })
        fig2.update_traces(textinfo="label+percent", textfont_size=11)
        fig2.update_layout(**CHART_LAYOUT, height=310,
                           legend=dict(orientation="h", y=-0.15, font=dict(size=10)))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Pipeline ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 Alur Analisis Cholestify")
    steps = [
        ("1", "Gathering",    "1.346 makanan Kaggle"),
        ("2", "Assessing",    "Missing, duplikat, outlier"),
        ("3", "Cleaning",     "Cap karbo >400g"),
        ("4", "EDA",          "5 Business Questions"),
        ("5", "Feature Eng.", "7 fitur baru"),
        ("6", "A/B Testing",  "Uji filter rekomendasi"),
        ("7", "Model Prep",   "Scaler + 80:20 Split"),
        ("8", "Improvement",  "SMOTE + CV"),
        ("9", "Inference",    "Rule-Based LDL/HDL"),
    ]
    steps_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px;'>"
    for (num, title, desc) in steps:
        steps_html += f"""
        <div class='pipeline-step'>
            <div class='pipeline-num'>{num}</div>
            <div class='pipeline-title'>{title}</div>
            <div class='pipeline-desc'>{desc}</div>
        </div>"""
    steps_html += "</div>"
    st.markdown(steps_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: EDA & DISTRIBUSI
# ════════════════════════════════════════════════════════════════════════════════

elif page == "📊 EDA & Distribusi":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Distribusi Nutrisi", "🔗 Korelasi", "🎯 Radar Kategori"])

    with tab1:
        st.markdown("### Distribusi 4 Nutrisi Utama (Data Aktual)")
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=["Kalori (kkal)", "Protein (g)", "Lemak (g)", "Karbohidrat (g)"])
        pairs = [
            ("calories",     COLORS["blue"],   1, 1),
            ("proteins",     COLORS["green"],  1, 2),
            ("fat",          COLORS["orange"], 2, 1),
            ("carbohydrate", COLORS["purple"], 2, 2),
        ]
        for col, color, r, c in pairs:
            data = df[col].dropna()
            fig.add_trace(go.Histogram(x=data, nbinsx=40, marker_color=color,
                                       opacity=0.75, showlegend=False), row=r, col=c)
            fig.add_vline(x=data.mean(),   line_dash="dash", line_color="red",
                          line_width=1.5,  row=r, col=c,
                          annotation_text=f"μ={data.mean():.1f}", annotation_font_size=9)
            fig.add_vline(x=data.median(), line_dash="dot",  line_color="yellow",
                          line_width=1.5,  row=r, col=c,
                          annotation_text=f"m={data.median():.1f}", annotation_font_size=9,
                          annotation_position="bottom right")
        fig.update_layout(**CHART_LAYOUT, height=520)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Statistik Deskriptif")
        desc = df[["calories","proteins","fat","carbohydrate"]].describe().round(2)
        desc.columns = ["Kalori (kkal)","Protein (g)","Lemak (g)","Karbohidrat (g)"]
        st.dataframe(desc, use_container_width=True)

        st.markdown("""
        <div class='insight-box'>
        <b>💡 Insight:</b> Seluruh distribusi bersifat <b>right-skewed</b> — mayoritas makanan
        terkonsentrasi pada nilai rendah. Lemak memiliki gap mean–median paling besar,
        menandakan pengaruh ekstrem dari minyak/santan/mentega.
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### Matriks Korelasi (Data Aktual)")
        num_cols = ["calories","proteins","fat","carbohydrate",
                    "pct_cal_from_fat","pct_cal_from_protein","pct_cal_from_carb"]
        corr = df[num_cols].corr()
        labels = ["Kalori","Protein","Lemak","Karbo",
                  "%Cal Lemak","%Cal Protein","%Cal Karbo"]
        fig_h = go.Figure(go.Heatmap(
            z=corr.values, x=labels, y=labels,
            colorscale="RdYlGn", zmid=0,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont={"size": 11},
        ))
        fig_h.update_layout(**CHART_LAYOUT, height=500)
        st.plotly_chart(fig_h, use_container_width=True)

        r_fat_cal, p_fat_cal = stats.pearsonr(df["fat"], df["calories"])
        st.markdown(f"""
        <div class='insight-box'>
        <b>📌 Korelasi Lemak ↔ Kalori:</b> r = <b>{r_fat_cal:.3f}</b> (p &lt; 0.001) — korelasi kuat positif.
        Lemak adalah penyumbang kalori terbesar; ini menjadi landasan bobot 40% dalam skor Cholestify.
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Scatter: Lemak vs Kalori")
            fig_sc = px.scatter(df.sample(min(500, len(df)), random_state=42),
                                x="fat", y="calories", color="kategori",
                                color_discrete_map=KATEGORI_COLORS,
                                trendline="ols", opacity=0.4,
                                labels={"fat":"Lemak (g)","calories":"Kalori (kkal)"})
            fig_sc.update_layout(**CHART_LAYOUT, height=360,
                                 legend=dict(font=dict(size=9)))
            st.plotly_chart(fig_sc, use_container_width=True)

        with col2:
            st.markdown("#### Scatter: Protein vs Lemak")
            fig_sc2 = px.scatter(df.sample(min(500, len(df)), random_state=7),
                                  x="fat", y="proteins", color="kategori",
                                  color_discrete_map=KATEGORI_COLORS,
                                  opacity=0.4,
                                  labels={"fat":"Lemak (g)","proteins":"Protein (g)"})
            fig_sc2.add_hline(y=10, line_dash="dash", line_color=COLORS["green"],
                               annotation_text="Protein ≥ 10g")
            fig_sc2.add_vline(x=15, line_dash="dash", line_color=COLORS["red"],
                               annotation_text="Lemak > 15g")
            fig_sc2.update_layout(**CHART_LAYOUT, height=360,
                                   legend=dict(font=dict(size=9)))
            st.plotly_chart(fig_sc2, use_container_width=True)

    with tab3:
        st.markdown("### Radar: Rata-rata Nutrisi per Kategori")
        kat_means = df.groupby("kategori")[["calories","proteins","fat","carbohydrate"]].mean().round(1)
        fig_r = go.Figure()
        dims = ["Kalori","Protein","Lemak","Karbohidrat"]
        for kat, color in KATEGORI_COLORS.items():
            if kat in kat_means.index:
                row = kat_means.loc[kat]
                vals = [row["calories"]/10, row["proteins"], row["fat"], row["carbohydrate"]/5]
                fig_r.add_trace(go.Scatterpolar(
                    r=vals+[vals[0]], theta=dims+[dims[0]],
                    name=kat, line_color=color, fill="toself",
                    fillcolor=color, opacity=0.22,
                ))
        fig_r.update_layout(**CHART_LAYOUT, height=500,
                             polar=dict(radialaxis=dict(visible=True, range=[0,35])),
                             legend=dict(font=dict(size=10)))
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption("*Kalori ÷ 10, Karbohidrat ÷ 5 untuk normalisasi skala visual*")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: BUSINESS QUESTIONS
# ════════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Business Questions":
    st.markdown("# 🔬 Business Questions Analysis")
    st.markdown("---")

    bq = st.tabs(["BQ1 — Kalori","BQ2 — Berisiko","BQ3 — Korelasi","BQ4 — Kategori","BQ5 — Top 20"])

    with bq[0]:
        st.markdown("## BQ1: Distribusi Kalori Makanan Indonesia")
        kal_low  = (df["calories"] < 100).sum()
        kal_med  = ((df["calories"] >= 100) & (df["calories"] <= 300)).sum()
        kal_high = (df["calories"] > 300).sum()

        c1, c2 = st.columns([3,2])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df["calories"], nbinsx=45,
                                        marker_color=COLORS["blue"], opacity=0.75, name="Frekuensi"))
            fig.add_vrect(x0=0,   x1=100, fillcolor="#16a34a", opacity=0.07, line_width=0)
            fig.add_vrect(x0=100, x1=300, fillcolor="#ca8a04", opacity=0.07, line_width=0)
            fig.add_vrect(x0=300, x1=df["calories"].max()+50, fillcolor="#dc2626", opacity=0.07, line_width=0)
            fig.add_vline(x=df["calories"].mean(),   line_dash="dash", line_color="red",
                          line_width=2, annotation_text=f"Mean {df['calories'].mean():.0f} kkal",
                          annotation_font_color="red")
            fig.add_vline(x=df["calories"].median(), line_dash="dot",  line_color="yellow",
                          line_width=2, annotation_text=f"Median {df['calories'].median():.0f} kkal",
                          annotation_font_color="yellow")
            fig.update_layout(**CHART_LAYOUT, height=380,
                               xaxis_title="Kalori (kkal/100g)", yaxis_title="Frekuensi")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = go.Figure(go.Pie(
                labels=["Rendah (<100)", "Sedang (100-300)", "Tinggi (>300)"],
                values=[kal_low, kal_med, kal_high],
                marker_colors=["#16a34a","#ca8a04","#dc2626"],
                textinfo="label+percent", hole=0.45,
            ))
            fig2.update_layout(**CHART_LAYOUT, height=300, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            c1b,c2b,c3b = st.columns(3)
            c1b.metric("🟢 Rendah", kal_low,  f"{kal_low/len(df)*100:.1f}%")
            c2b.metric("🟡 Sedang", kal_med,  f"{kal_med/len(df)*100:.1f}%")
            c3b.metric("🔴 Tinggi", kal_high, f"{kal_high/len(df)*100:.1f}%")

        st.markdown(f"""
        <div class='insight-box'>
        <b>📌 Kesimpulan BQ1:</b> Distribusi kalori bersifat <b>right-skewed (bimodal)</b>.
        Median {df['calories'].median():.0f} kkal jauh di bawah mean {df['calories'].mean():.0f} kkal.
        <b>{(kal_low+kal_med)/len(df)*100:.1f}% makanan</b> masuk rentang kalori aman (≤300 kkal/100g).
        </div>""", unsafe_allow_html=True)

    with bq[1]:
        st.markdown("## BQ2: Makanan Tinggi Lemak & Rendah Protein")
        berisiko = df[(df["fat"] > 20) & (df["proteins"] < 5)].copy()
        berisiko["tier"] = pd.cut(berisiko["fat"], bins=[20,40,80,200],
                                   labels=["🟡 Waspada (20-40g)",
                                           "🟠 Berbahaya (40-80g)",
                                           "🔴 Sangat Berbahaya (>80g)"])

        st.markdown(f"**{len(berisiko)} makanan** ({len(berisiko)/len(df)*100:.1f}%) memenuhi kriteria: lemak >20g **dan** protein <5g")

        c1,c2 = st.columns([3,1])
        with c1:
            fig = px.scatter(berisiko, x="fat", y="calories", size=berisiko["fat"],
                             color="tier",
                             color_discrete_map={
                                 "🟡 Waspada (20-40g)":   "#ca8a04",
                                 "🟠 Berbahaya (40-80g)": "#ea580c",
                                 "🔴 Sangat Berbahaya (>80g)":"#dc2626",
                             },
                             hover_data={"name":True,"proteins":True,"fat":True,"calories":True},
                             opacity=0.65, size_max=30,
                             labels={"fat":"Lemak (g)","calories":"Kalori (kkal)"})
            fig.update_layout(**CHART_LAYOUT, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            for tier, color in [("🟡 Waspada (20-40g)","#ca8a04"),
                                  ("🟠 Berbahaya (40-80g)","#ea580c"),
                                  ("🔴 Sangat Berbahaya (>80g)","#dc2626")]:
                n = (berisiko["tier"] == tier).sum()
                st.markdown(f"""
                <div style='background:#111827;border-left:3px solid {color};
                            padding:14px;border-radius:8px;margin:8px 0'>
                    <div style='font-weight:700;font-size:0.8rem;color:#94a3b8'>{tier}</div>
                    <div style='font-size:2rem;font-weight:800;color:{color};font-family:Space Mono,monospace'>{n}</div>
                    <div style='font-size:0.7rem;color:#475569'>makanan</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("#### Top 15 Paling Berisiko")
        top = berisiko.nlargest(15, "fat")[["name","fat","calories","proteins","carbohydrate","tier"]]
        top.columns = ["Nama","Lemak (g)","Kalori (kkal)","Protein (g)","Karbo (g)","Tier"]
        st.dataframe(top, use_container_width=True, hide_index=True)

    with bq[2]:
        st.markdown("## BQ3: Korelasi Lemak dan Kalori")
        r, p = stats.pearsonr(df["fat"], df["calories"])
        c1,c2,c3 = st.columns(3)
        c1.metric("Pearson r", f"{r:.3f}", "Korelasi Kuat Positif")
        c2.metric("P-value",   "< 0.001",  "Signifikan Statistik")
        c3.metric("Interpretasi","Sangat Kuat","r > 0.7")

        fig = px.scatter(df, x="fat", y="calories", color="kategori",
                          color_discrete_map=KATEGORI_COLORS,
                          trendline="ols", opacity=0.35,
                          labels={"fat":"Lemak (g/100g)","calories":"Kalori (kkal/100g)"},
                          title=f"Scatter Lemak vs Kalori — Pearson r = {r:.3f}, p < 0.001")
        fig.update_layout(**CHART_LAYOUT, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with bq[3]:
        st.markdown("## BQ4: Kategorisasi Profil Nutrisi")
        kat = df["kategori"].value_counts().reset_index()
        kat.columns = ["kategori","count"]
        kat["pct"] = (kat["count"]/len(df)*100).round(1)

        c1,c2 = st.columns([2,3])
        with c1:
            thresh_df = pd.DataFrame([
                ["Tinggi Protein, Rendah Lemak","fat≤3g & protein≥10g","✅ Ideal"],
                ["Rendah Lemak & Karbo",         "fat≤3g & karbo≤15g",  "✅ Sangat Baik"],
                ["Rendah Lemak",                  "fat≤3g",              "✅ Baik"],
                ["Sedang",                         "fat 3-15g",           "⚠️ Perhatikan Porsi"],
                ["Tinggi Protein & Lemak",        "fat>15g & prot≥15g",  "⚠️ Batasi"],
                ["Tinggi Lemak",                   "fat>15g",             "❌ Hindari"],
            ], columns=["Kategori","Kondisi","Status"])
            st.dataframe(thresh_df, use_container_width=True, hide_index=True)

        with c2:
            fig = px.bar(kat, x="count", y="kategori", orientation="h",
                          color="kategori", color_discrete_map=KATEGORI_COLORS,
                          text=kat.apply(lambda r: f"{r['count']} ({r['pct']}%)", axis=1))
            fig.update_traces(textposition="outside")
            fig.update_layout(**CHART_LAYOUT, height=340, showlegend=False,
                               yaxis_title="", xaxis_title="Jumlah Makanan")
            st.plotly_chart(fig, use_container_width=True)

        safe_n = df[df['kategori'].isin(['Tinggi Protein, Rendah Lemak','Rendah Lemak & Karbo','Rendah Lemak'])].shape[0]
        st.markdown(f"""
        <div class='insight-box'>
        <b>📌 Kesimpulan BQ4:</b>
        {safe_n} makanan ({safe_n/len(df)*100:.1f}%) tergolong aman.
        Hanya {df[df['kategori']=='Tinggi Lemak'].shape[0]} makanan ({df[df['kategori']=='Tinggi Lemak'].shape[0]/len(df)*100:.1f}%)
        yang perlu benar-benar dihindari.
        </div>""", unsafe_allow_html=True)

    with bq[4]:
        st.markdown("## BQ5: Top 20 Rekomendasi Makanan Cholestify")
        top20 = df.nlargest(20, "rekomendasi_score")[
            ["name","calories","proteins","fat","carbohydrate","rekomendasi_score","kategori"]
        ].reset_index(drop=True)
        top20.index += 1

        fig = go.Figure(go.Bar(
            x=top20["rekomendasi_score"],
            y=top20["name"],
            orientation="h",
            marker=dict(
                color=top20["rekomendasi_score"],
                colorscale=[[0,"#1e4d2b"],[0.5,"#16a34a"],[1.0,"#4ade80"]],
                showscale=True,
                colorbar=dict(title="Skor"),
            ),
            text=[f"P:{r['proteins']:.1f}g  L:{r['fat']:.1f}g  {r['calories']:.0f}kkal"
                  for _,r in top20.iterrows()],
            textposition="outside", textfont=dict(size=10),
        ))
        fig.add_vline(x=0.7, line_dash="dash", line_color=COLORS["yellow"],
                       annotation_text="Threshold Baik (0.7)")
        fig.update_layout(**{**CHART_LAYOUT, 'margin': dict(l=10,r=100,t=30,b=10)},
                           height=650,
                           yaxis=dict(autorange="reversed"),
                           xaxis_title="Skor Rekomendasi Cholestify (0–1)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='insight-box'>
        <b>⚙️ Formula Skor:</b><br>
        <code>skor = 0.4 × protein_norm + 0.4 × (1 − fat_norm) + 0.2 × (1 − |cal_norm − 0.3|)</code><br><br>
        <b>📌 Insight:</b> Produk ikan/udang kering mendominasi karena proses dehidrasi
        mengkonsentrasikan protein sekaligus meminimalkan lemak.
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: A/B TESTING
# ════════════════════════════════════════════════════════════════════════════════

elif page == "⚗️ A/B Testing":
    st.markdown("# ⚗️ A/B Testing")
    st.markdown("**Uji Statistik: Filter Rekomendasi Makanan Aman vs Filter Kalori Saja**")
    st.markdown("---")

    grup_a = df[df["calories"] < 100]["fat"]
    grup_b = df[(df["fat"] < 3) & (df["proteins"] >= 5)]["fat"]
    stat, pval = stats.mannwhitneyu(grup_a, grup_b, alternative="greater")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#111827,#0f172a);border:1px solid #334155;
                    border-left:3px solid #64748b;border-radius:12px;padding:20px;'>
            <div style='font-size:0.68rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em'>GRUP A — Kontrol</div>
            <div style='color:#94a3b8;margin-top:8px;font-weight:700;font-size:1rem'>Filter Kalori Rendah</div>
            <div style='font-size:0.83rem;color:#64748b;margin-top:4px'>Kalori &lt; 100 kkal</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#111827,#0f172a);border:1px solid rgba(52,211,153,0.3);
                    border-left:3px solid #34d399;border-radius:12px;padding:20px;'>
            <div style='font-size:0.68rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em'>GRUP B — Treatment</div>
            <div style='color:#34d399;margin-top:8px;font-weight:700;font-size:1rem'>Filter Lemak + Protein</div>
            <div style='font-size:0.83rem;color:#64748b;margin-top:4px'>Lemak &lt; 3g &amp; Protein ≥ 5g</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("n Grup A", len(grup_a))
    m2.metric("n Grup B", len(grup_b))
    m3.metric("Mean Lemak A", f"{grup_a.mean():.2f}g")
    m4.metric("Mean Lemak B", f"{grup_b.mean():.2f}g")
    m1b,m2b,m3b,m4b = st.columns(4)
    m1b.metric("Median A", f"{grup_a.median():.2f}g")
    m2b.metric("Median B", f"{grup_b.median():.2f}g")
    m3b.metric("U Statistic", f"{stat:.0f}")
    m4b.metric("P-Value", f"{pval:.4f}", "< α(0.05) ✅" if pval < 0.05 else "> α(0.05) ❌")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=grup_a, name="Grup A", nbinsx=30,
                                marker_color=COLORS["gray"], opacity=0.6))
    fig.add_trace(go.Histogram(x=grup_b, name="Grup B", nbinsx=30,
                                marker_color=COLORS["green"], opacity=0.6))
    fig.update_layout(**CHART_LAYOUT, barmode="overlay", height=360,
                       xaxis_title="Lemak (g/100g)", yaxis_title="Frekuensi")
    st.plotly_chart(fig, use_container_width=True)

    if pval < 0.05:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#0a1f12,#0d2618);border:1px solid rgba(22,163,74,0.4);
                    border-radius:12px;padding:22px'>
            <h3 style='color:#4ade80;margin:0'>✅ TOLAK H₀</h3>
            <p style='color:#86efac;margin:10px 0 4px'>Filter kalori (Grup A) terbukti menghasilkan makanan dengan
            lemak lebih tinggi. Filter lemak+protein (Grup B) lebih efektif melindungi dari risiko kolesterol.</p>
            <p style='color:#4ade80;font-weight:700;margin:0'>→ Gunakan filter Lemak+Protein sebagai rekomendasi utama Cholestify.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1a1400,#1a1a0d);border:1px solid rgba(202,138,4,0.4);
                    border-radius:12px;padding:22px'>
            <h3 style='color:#fbbf24;margin:0'>⚠️ GAGAL TOLAK H₀</h3>
            <p style='color:#fde68a;margin:10px 0 0'>Tidak terbukti filter kalori menghasilkan lemak lebih tinggi secara statistik.</p>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Feature Engineering":
    st.markdown("# 🤖 Feature Engineering")
    st.markdown("Fitur-fitur yang dihasilkan dari proses feature engineering notebook.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["🔧 7 Fitur Baru", "⚖️ Distribusi Label"])

    with tab1:
        fe_info = [
            ("kategori",            "Kategori Profil",       "Rule-based (fat & protein)",    "Target kategorisasi 6 kelas"),
            ("protein_density",     "Kepadatan Protein",     "proteins / (calories+1)",       "Efisiensi protein per kalori"),
            ("fat_ratio",           "Rasio Lemak",           "fat / (calories+1)",            "Proporsi lemak terhadap kalori"),
            ("pct_cal_from_fat",    "% Kalori dari Lemak",   "(fat×9)/(cal+1)×100",           "AHA: ≤30% ideal"),
            ("pct_cal_from_protein","% Kalori dari Protein", "(prot×4)/(cal+1)×100",          "Energi dari protein"),
            ("pct_cal_from_carb",   "% Kalori dari Karbo",   "(carb×4)/(cal+1)×100",         "Energi dari karbohidrat"),
            ("risiko_kolesterol",   "Label Risiko (0/1/2)",  "fat & pct_cal_from_fat",        "Target multi-kelas"),
            ("is_recommended",      "Rekomendasi (0/1)",     "fat≤10 & prot≥5 & cal≤300",     "Target binary"),
        ]
        for feat, name, formula, note in fe_info:
            c1,c2,c3,c4 = st.columns([1.5,2,2.5,2.5])
            c1.code(feat)
            c2.write(name)
            c3.code(formula)
            c4.write(f"*{note}*")
            st.markdown("<hr style='margin:4px 0;border-color:rgba(30,41,59,0.8)'>",
                        unsafe_allow_html=True)

        st.markdown("### Distribusi Fitur Baru (Data Aktual)")
        fig = make_subplots(rows=1, cols=3,
                             subplot_titles=["Protein Density","Fat Ratio","% Kalori dari Lemak"])
        for i,(col,color) in enumerate([
            ("protein_density",  COLORS["green"]),
            ("fat_ratio",        COLORS["orange"]),
            ("pct_cal_from_fat", COLORS["red"]),
        ]):
            fig.add_trace(go.Histogram(x=df[col], nbinsx=30,
                                        marker_color=color, opacity=0.75,
                                        showlegend=False), row=1, col=i+1)
        fig.update_layout(**CHART_LAYOUT, height=280)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("### Label Risiko Kolesterol (3 Kelas)")
            risiko_map = {0:"Rendah (Aman)",1:"Sedang (Batasi)",2:"Tinggi (Hindari)"}
            rc = df["risiko_kolesterol"].map(risiko_map).value_counts()
            for label, count in rc.items():
                color = {"Rendah (Aman)":"#16a34a","Sedang (Batasi)":"#ca8a04",
                          "Tinggi (Hindari)":"#dc2626"}[label]
                pct = count/len(df)*100
                bar = "█" * int(pct/2.5)
                st.markdown(f"""
                <div style='background:#111827;border-left:3px solid {color};
                            padding:12px 14px;border-radius:8px;margin:6px 0;font-family:monospace'>
                    <span style='color:{color};font-weight:700'>{label}</span><br>
                    <span style='color:#475569;font-size:0.82rem'>{count} ({pct:.1f}%)  {bar}</span>
                </div>""", unsafe_allow_html=True)

            imb = rc.max()/rc.min()
            if imb > 3:
                st.warning(f"⚠️ Imbalance ratio **{imb:.1f}x** — SMOTE direkomendasikan!")
            else:
                st.success(f"✅ Imbalance ratio {imb:.1f}x — cukup imbang")

        with c2:
            st.markdown("### Label Rekomendasi (Binary)")
            rec_c = df["is_recommended"].value_counts().sort_index()
            fig = go.Figure(go.Bar(
                x=["Tidak Disarankan (0)","Direkomendasikan (1)"],
                y=[rec_c.get(0,0), rec_c.get(1,0)],
                marker_color=["#dc2626","#16a34a"],
                text=[f"{rec_c.get(0,0)} ({rec_c.get(0,0)/len(df)*100:.1f}%)",
                       f"{rec_c.get(1,0)} ({rec_c.get(1,0)/len(df)*100:.1f}%)"],
                textposition="auto",
            ))
            fig.update_layout(**CHART_LAYOUT, height=320, yaxis_title="Jumlah Makanan")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='insight-box'>
        <b>🏆 Pipeline Final yang Direkomendasikan:</b><br>
        <code>ImbPipeline([StandardScaler → SMOTE(k=5) → RandomForest(class_weight='balanced')])</code><br><br>
        Gunakan <b>Balanced Accuracy & F1 Macro</b> sebagai metrik utama — accuracy konvensional
        bias terhadap kelas mayoritas (Rendah/Aman).
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: REKOMENDASI PERSONAL
# ════════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Rekomendasi Personal":
    st.markdown("# 🎯 Sistem Rekomendasi Personal Cholestify")
    st.markdown("<p style='color:#475569;margin-top:-8px'>Masukkan nilai LDL & HDL Anda untuk rekomendasi makanan yang dipersonalisasi.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_in, col_out = st.columns([1,2])

    with col_in:
        st.markdown("### 🩺 Profil Kolesterol")
        ldl_val = st.slider("LDL (mg/dL)", 50, 250, 145, 5,
                             help="<100 optimal | 100-129 near optimal | 130-159 borderline high | ≥190 very high")
        hdl_val = st.slider("HDL (mg/dL)", 20, 100, 42, 2,
                             help="≥60 protektif | 40-59 normal | <40 rendah (risiko lebih tinggi)")

        risk, t = get_risk_level(ldl_val, hdl_val)
        risk_info = {
            0: ("🟢 Optimal", "#16a34a"),
            1: ("🟡 Baik",    "#84cc16"),
            2: ("🟡 Sedang",  "#ca8a04"),
            3: ("🟠 Tinggi",  "#ea580c"),
            4: ("🔴 Kritis",  "#dc2626"),
        }
        rl, rc_col = risk_info[risk]
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#111827,#0f172a);border:2px solid {rc_col};
                    border-radius:14px;padding:20px;margin-top:14px;text-align:center'>
            <div style='font-size:0.68rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em'>Risk Level</div>
            <div style='font-size:2.2rem;font-weight:800;color:{rc_col};margin:6px 0'>{rl}</div>
            <div style='color:#64748b;font-size:0.8rem'>LDL: {ldl_val} mg/dL &nbsp;·&nbsp; HDL: {hdl_val} mg/dL</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Threshold Lemak Personal")
        th_df = pd.DataFrame({
            "Status":["BISA ✅","NETRAL ⚠️","LIMIT 🚫","HINDARI ❌"],
            "Batas Lemak":[f"≤{t['fat_bisa']}g",
                            f"{t['fat_bisa']}–{t['fat_netral']}g",
                            f"{t['fat_netral']}–{t['fat_limit']}g",
                            f">  {t['fat_limit']}g"],
        })
        st.dataframe(th_df, use_container_width=True, hide_index=True)

    with col_out:
        df_p = df.copy()
        df_p["status"] = df_p.apply(lambda r: classify_food_status(r, ldl_val, hdl_val), axis=1)
        sc = df_p["status"].value_counts()

        fig_pie = go.Figure(go.Pie(
            labels=[f"{k} ({sc.get(k,0)})" for k in STATUS_COLORS],
            values=[sc.get(k,0) for k in STATUS_COLORS],
            marker_colors=list(STATUS_COLORS.values()),
            textinfo="label+percent", hole=0.45,
        ))
        fig_pie.update_layout(**{**CHART_LAYOUT, 'margin': dict(l=0,r=0,t=10,b=30)},
                               height=300, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_sc = px.scatter(df_p.sample(min(400,len(df_p)), random_state=42),
                             x="fat", y="calories", color="status",
                             color_discrete_map=STATUS_COLORS,
                             opacity=0.45, hover_data={"name":True},
                             labels={"fat":"Lemak (g)","calories":"Kalori (kkal)"})
        for line_v, col_v, lbl in [
            (t["fat_bisa"],  "#16a34a", f"BISA ≤{t['fat_bisa']}g"),
            (t["fat_netral"],"#ca8a04", f"NETRAL ≤{t['fat_netral']}g"),
            (t["fat_limit"], "#ea580c", f"LIMIT ≤{t['fat_limit']}g"),
        ]:
            fig_sc.add_vline(x=line_v, line_dash="dash", line_color=col_v,
                              line_width=1.5, annotation_text=lbl, annotation_font_size=9)
        fig_sc.update_layout(**CHART_LAYOUT, height=340)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Cari Makanan")
    filter_col, search_col = st.columns([2,3])
    with filter_col:
        sel_status = st.selectbox("Filter Status",
                                   ["BISA ✅","NETRAL ⚠️","LIMIT 🚫","TIDAK_BISA ❌"])
    with search_col:
        keyword = st.text_input("Cari nama makanan", placeholder="contoh: ayam, tempe, ikan...")

    smap = {"BISA ✅":"BISA","NETRAL ⚠️":"NETRAL","LIMIT 🚫":"LIMIT","TIDAK_BISA ❌":"TIDAK_BISA"}
    result = df_p[df_p["status"] == smap[sel_status]]
    if keyword:
        result = result[result["name"].str.contains(keyword, case=False, na=False)]

    show = result[["name","calories","proteins","fat","carbohydrate",
                    "kategori","rekomendasi_score"]].sort_values(
        "rekomendasi_score", ascending=(smap[sel_status] in ["TIDAK_BISA","LIMIT"])
    ).head(40)
    show.columns = ["Nama","Kalori","Protein (g)","Lemak (g)","Karbo (g)","Kategori","Skor"]
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🌡️ Sensitivity Map: LDL × HDL Risk Level")
    ldl_range = np.arange(70, 210, 10)
    hdl_range_arr = np.arange(25, 80, 5)
    risk_mx = np.array([[get_risk_level(ldl, hdl)[0]
                          for ldl in ldl_range]
                          for hdl in hdl_range_arr])

    fig_ht = go.Figure(go.Heatmap(
        z=risk_mx, x=[str(v) for v in ldl_range], y=[str(v) for v in hdl_range_arr],
        colorscale=[[0,"#16a34a"],[0.25,"#84cc16"],[0.5,"#ca8a04"],
                     [0.75,"#ea580c"],[1,"#dc2626"]],
        zmin=0, zmax=4,
        text=np.vectorize(lambda v: ["O","B","S","T","K"][v])(risk_mx),
        texttemplate="%{text}", textfont={"size":10},
        colorbar=dict(title="Risk",tickvals=[0.4,1.2,2.0,2.8,3.6],
                      ticktext=["0 Optimal","1 Baik","2 Sedang","3 Tinggi","4 Kritis"]),
    ))
    li = int(np.argmin(np.abs(ldl_range - ldl_val)))
    hi = int(np.argmin(np.abs(hdl_range_arr - hdl_val)))
    fig_ht.add_shape(type="rect", x0=li-0.5,x1=li+0.5, y0=hi-0.5,y1=hi+0.5,
                      line=dict(color="white",width=3))
    fig_ht.add_annotation(x=str(ldl_range[li]), y=str(hdl_range_arr[hi]),
                            text="← Anda", showarrow=True, arrowhead=2,
                            arrowcolor="white", font=dict(color="white",size=11))
    fig_ht.update_layout(**CHART_LAYOUT, height=400,
                          xaxis_title="LDL (mg/dL)", yaxis_title="HDL (mg/dL)")
    st.plotly_chart(fig_ht, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: DATA DICTIONARY
# ════════════════════════════════════════════════════════════════════════════════

elif page == "📚 Data Dictionary":
    st.markdown("# 📚 Data Dictionary")
    st.markdown("---")

    tab1,tab2,tab3 = st.tabs(["📦 Kolom CSV","📏 Threshold","🔎 Preview Data"])

    with tab1:
        orig = pd.DataFrame([
            ["id",                  "int",    "—",          "ID unik makanan (1–1346)"],
            ["calories",            "float",  "kkal/100g",  "Kandungan kalori"],
            ["proteins",            "float",  "g/100g",     "Kandungan protein total"],
            ["fat",                 "float",  "g/100g",     "Kandungan lemak total"],
            ["carbohydrate",        "float",  "g/100g",     "Kandungan karbohidrat (capped 400g)"],
            ["name",                "string", "—",          "Nama makanan dalam Bahasa Indonesia"],
            ["kategori",            "string", "6 kelas",    "Kategori profil nutrisi"],
            ["protein_density",     "float",  "—",          "proteins / (calories+1)"],
            ["fat_ratio",           "float",  "—",          "fat / (calories+1)"],
            ["pct_cal_from_fat",    "float",  "%",          "(fat×9)/(calories+1)×100"],
            ["pct_cal_from_protein","float",  "%",          "(proteins×4)/(calories+1)×100"],
            ["pct_cal_from_carb",   "float",  "%",          "(carbohydrate×4)/(calories+1)×100"],
            ["risiko_kolesterol",   "int",    "0/1/2",      "0=Rendah, 1=Sedang, 2=Tinggi"],
            ["is_recommended",      "int",    "0/1",        "1 jika fat≤10 & prot≥5 & cal≤300"],
        ], columns=["Kolom","Tipe","Satuan","Deskripsi"])
        st.dataframe(orig, use_container_width=True, hide_index=True)

    with tab2:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("#### Threshold Kategori Nutrisi")
            st.dataframe(pd.DataFrame([
                ["Rendah","<100 kkal","<3g","<5g","<10g"],
                ["Sedang","100-300 kkal","3-15g","5-15g","10-40g"],
                ["Tinggi",">300 kkal",">15g",">15g",">40g"],
            ], columns=["Level","Kalori","Lemak","Protein","Karbo"]),
            use_container_width=True, hide_index=True)

            st.markdown("#### Label Risiko Kolesterol")
            st.dataframe(pd.DataFrame([
                ["0","fat ≤ 8g","Aman dikonsumsi"],
                ["1","fat 8–15g","Perlu dibatasi porsi"],
                ["2","fat>15g & pct_fat>35%","Perlu dihindari"],
            ], columns=["Label","Kondisi","Interpretasi"]),
            use_container_width=True, hide_index=True)

        with c2:
            st.markdown("#### Risk Level System (LDL + HDL)")
            st.dataframe(pd.DataFrame([
                [0,"Optimal","≤20g","≤30g","≤40g"],
                [1,"Baik",   "≤15g","≤25g","≤35g"],
                [2,"Sedang", "≤10g","≤18g","≤25g"],
                [3,"Tinggi", "≤6g", "≤12g","≤18g"],
                [4,"Kritis", "≤3g", "≤8g", "≤12g"],
            ], columns=["Risk","Label","Fat BISA","Fat NETRAL","Fat LIMIT"]),
            use_container_width=True, hide_index=True)

            st.markdown("#### Kategori LDL & HDL (AHA/ACC)")
            st.dataframe(pd.DataFrame([
                ["LDL","<100","Optimal"],["LDL","100-129","Near Optimal"],
                ["LDL","130-159","Borderline High"],["LDL","160-189","High"],
                ["LDL","≥190","Very High"],
                ["HDL","≥60","Protektif (−1 risk)"],
                ["HDL","40-59","Normal (0)"],["HDL","<40","Rendah (+1 risk)"],
            ], columns=["Jenis","Range (mg/dL)","Kategori"]),
            use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### Preview Dataset Aktual")
        
        search_query = st.text_input("🔍 Cari data (berdasarkan kolom teks seperti nama/kategori)", "")
        
        col_opts1, col_opts2 = st.columns(2)
        with col_opts1:
            cols_show = st.multiselect("Pilih Kolom", options=df.columns.tolist(),
                                        default=["name","calories","proteins","fat",
                                                  "carbohydrate","kategori","risiko_kolesterol",
                                                  "is_recommended"])
            n_show = st.slider("Jumlah Baris", 5, 100, 20)
            
        with col_opts2:
            if cols_show:
                sort_col = st.selectbox("Urutkan berdasarkan", options=cols_show,
                                         index=min(2, len(cols_show)-1))
                sort_order = st.radio("Urutan", ["Terkecil ke Terbesar (Ascending)", "Terbesar ke Terkecil (Descending)"])
                asc = True if "Ascending" in sort_order else False

        if cols_show:
            filtered_df = df.copy()
            if search_query:
                # Filter pada kolom yang bertipe object/string
                str_cols = filtered_df.select_dtypes(include=['object', 'string']).columns
                if not str_cols.empty:
                    mask = filtered_df[str_cols].apply(lambda x: x.astype(str).str.contains(search_query, case=False, na=False)).any(axis=1)
                    filtered_df = filtered_df[mask]
            
            st.dataframe(filtered_df[cols_show].sort_values(sort_col, ascending=asc).head(n_show),
                          use_container_width=True)
            st.caption(f"Menampilkan {min(n_show, len(filtered_df))} dari {len(filtered_df):,} baris data yang cocok")


# ════════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(f"""
<div style='text-align:center;padding:16px 0 8px'>
    <span style='font-size:1.3rem'>🫀</span><br>
    <span style='font-size:0.95rem;font-weight:700;
                 background:linear-gradient(135deg,#60a5fa,#34d399);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text'>Cholestify Dashboard</span><br>
    <span style='font-size:0.72rem;color:#334155'>
        {len(df):,} makanan Indonesia &nbsp;·&nbsp; 
        Data: <code style='background:rgba(96,165,250,0.08);padding:2px 6px;border-radius:4px;color:#60a5fa'>{csv_path}</code>
    </span>
</div>""", unsafe_allow_html=True)
