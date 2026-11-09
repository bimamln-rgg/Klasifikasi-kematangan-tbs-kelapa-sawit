import io
import os
import json
import hashlib
from datetime import datetime

import altair as alt
import cv2
import gdown
import keras
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers

# =====================================================================
# 1. KONFIGURASI & DOWLOAD MODEL OTOMATIS
# =====================================================================
st.set_page_config(
    page_title="SawitScan - Klasifikasi Kematangan TBS Sawit",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ID File Google Drive kamu
DRIVE_FILE_ID = "1qBOpW704HxPfnD8QLULZf0uc0GMyZsM4"
MODEL_PATH = "best_efficientnet_b4_cbam_clean.h5"
EVAL_PATH = "evaluasi.json"
IMG_SIZE = 380
CLASS_NAMES = ["Busuk", "Matang", "Mentah", "Terlalu matang"]
PAGES = ["Beranda", "Klasifikasi", "Evaluasi Model", "Riwayat", "Panduan"]

CLASS_INFO = {
    "Mentah": {"color": "#16a34a", "desc": "Belum mencapai kematangan, warna dominan hitam/ungu tua.",
               "note": "Buah belum matang sempurna. Tunda pemanenan."},
    "Matang": {"color": "#d97706", "desc": "Warna merah-oranye merata, berondolan mulai lepas.",
               "note": "Kondisi matang optimal untuk dipanen dan diolah."},
    "Terlalu matang": {"color": "#b45309", "desc": "Melewati batas optimal, banyak berondolan lepas.",
                       "note": "Melewati batas kematangan optimal. Segera olah."},
    "Busuk": {"color": "#dc2626", "desc": "Kualitas menurun, tampak membusuk atau berjamur.",
              "note": "Kualitas menurun, kadar asam lemak bebas (ALB) tinggi."},
}

st.session_state.setdefault("nav", "Beranda")
st.session_state.setdefault("history", [])
st.session_state.setdefault("seen", set())


def go(page):
    st.session_state.nav = page


def html(block):
    return "".join(line.strip() for line in block.splitlines())


# =====================================================================
# 2. CSS CUSTOM
# =====================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: radial-gradient(1100px 480px at 88% -8%, rgba(212,175,55,.16), transparent 60%), #faf8f2; }
#MainMenu, footer, header, [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
.block-container { max-width: 1180px; padding-top: 1.2rem; padding-bottom: 3rem; }
h1, h2, h3, .serif { font-family: 'Fraunces', serif !important; color: #0f3d2a; }

/* brand + nav */
.brand { display:flex; align-items:center; gap:12px; padding:6px 0; }
.brand .logo { width:44px; height:44px; border-radius:12px; background:#0f3d2a; display:flex; align-items:center; justify-content:center; font-size:1.4rem; border:2px solid #d4af37; }
.brand b { font-family:'Fraunces',serif; font-size:1.25rem; color:#0f3d2a; display:block; line-height:1.1; }
.brand small { color:#7a867e; font-size:.72rem; letter-spacing:.4px; }
div[role="radiogroup"] { gap:4px; justify-content:flex-end; flex-wrap:wrap; }
div[role="radiogroup"] label { padding:8px 18px; border-radius:999px; cursor:pointer; transition:.15s; }
div[role="radiogroup"] label > div:first-child { display:none; }
div[role="radiogroup"] label p { color:#3d5247 !important; font-weight:600; font-size:.9rem; }
div[role="radiogroup"] label:hover { background:rgba(15,61,42,.07); }
div[role="radiogroup"] label:has(input:checked) { background:#0f3d2a; }
div[role="radiogroup"] label:has(input:checked) p { color:#f1d77a !important; }
.navline { height:1px; background:linear-gradient(90deg,#d4af37,rgba(212,175,55,.05)); margin:6px 0 26px 0; }

/* hero */
.tag { display:inline-block; font-size:.74rem; font-weight:700; letter-spacing:1.2px; color:#8a6d0f; background:rgba(212,175,55,.18); padding:6px 14px; border-radius:999px; margin-bottom:18px; }
.hero-h { font-family:'Fraunces',serif; font-size:3.1rem; line-height:1.12; color:#0f3d2a; margin:0 0 16px 0; font-weight:700; }
.hero-h em { font-style:normal; background:linear-gradient(transparent 62%, rgba(212,175,55,.45) 62%); }
.lead { color:#4b5d52; font-size:1.05rem; line-height:1.75; max-width:560px; }

/* panels */
.panel { background:#fff; border:1px solid #ebe5d0; border-radius:18px; padding:22px 24px; box-shadow:0 8px 28px rgba(15,61,42,.06); }
.k { font-size:.72rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:#8a7a2e; }
.big { font-family:'Fraunces',serif; font-size:2rem; font-weight:700; color:#0f3d2a; margin:4px 0 8px 0; }
.chip { display:inline-block; color:#fff; font-size:.78rem; font-weight:600; padding:4px 14px; border-radius:999px; }
.crow { display:flex; gap:12px; padding:12px 0; border-bottom:1px solid #f0ebd8; align-items:flex-start; }
.crow:last-child { border-bottom:none; }
.dot { width:12px; height:12px; border-radius:50%; margin-top:6px; flex:none; }
.crow b { color:#0f3d2a; font-size:.95rem; } .crow span { color:#6b7a70; font-size:.84rem; display:block; }

/* kpi strip */
.kpis { display:flex; background:#0f3d2a; border-radius:18px; overflow:hidden; margin:34px 0 8px 0; border-bottom:4px solid #d4af37; }
.kpi { flex:1; padding:22px 18px; text-align:center; border-right:1px solid rgba(255,255,255,.1); }
.kpi:last-child { border-right:none; }
.kpi b { display:block; font-family:'Fraunces',serif; font-size:1.5rem; color:#f1d77a; }
.kpi span { color:#bcd6c4; font-size:.76rem; letter-spacing:.8px; text-transform:uppercase; }

/* sections */
.sec { margin:46px 0 6px 0; font-family:'Fraunces',serif; font-size:1.7rem; font-weight:700; color:#0f3d2a; }
.sub { color:#6b7a70; font-size:.95rem; margin-bottom:20px; max-width:640px; }
.feat { padding:6px 0 18px 0; }
.feat .ic { font-size:1.5rem; }
.feat h4 { margin:6px 0 4px 0; color:#0f3d2a; font-size:1.02rem; font-weight:700; }
.feat p { margin:0; color:#5b6a60; font-size:.88rem; line-height:1.6; }
.tl { display:flex; gap:0; position:relative; margin-top:10px; }
.tl-i { flex:1; padding:0 18px 0 0; position:relative; }
.tl-i::before { content:""; position:absolute; top:13px; left:34px; right:6px; height:2px; background:repeating-linear-gradient(90deg,#d4af37 0 6px,transparent 6px 12px); }
.tl-i:last-child::before { display:none; }
.tl-n { width:28px; height:28px; border-radius:50%; background:#d4af37; color:#0f3d2a; font-weight:800; font-size:.85rem; display:flex; align-items:center; justify-content:center; position:relative; z-index:1; }
.tl-i h4 { margin:12px 0 4px 0; color:#0f3d2a; font-size:1rem; } .tl-i p { margin:0; color:#5b6a60; font-size:.87rem; line-height:1.6; }
.meth { border-left:3px solid #d4af37; padding:2px 0 2px 16px; }
.meth h4 { margin:0 0 4px 0; color:#0f3d2a; } .meth p { margin:0; color:#5b6a60; font-size:.88rem; line-height:1.65; }

/* tombol, tab, uploader */
div.stButton > button, div.stDownloadButton > button { background:#0f3d2a; color:#f1d77a; border:none; border-radius:12px; font-weight:600; padding:.6rem 1.4rem; transition:.15s; }
div.stButton > button:hover, div.stDownloadButton > button:hover { background:#d4af37; color:#0f3d2a; transform:translateY(-1px); }
button[data-baseweb="tab"] p { font-weight:600; color:#3d5247; }
button[data-baseweb="tab"][aria-selected="true"] p { color:#0f3d2a; }
div[data-baseweb="tab-highlight"] { background:#d4af37 !important; height:3px; }
section[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploaderDropzone"] { background:#fff; border:2px dashed #d4af37; border-radius:16px; }
[data-testid="stExpander"] { background:#fff; border:1px solid #ebe5d0 !important; border-radius:14px; }
.footer { text-align:center; color:#8a958d; font-size:.8rem; margin-top:56px; padding-top:18px; border-top:1px solid #ebe5d0; }
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# 3. MODEL (EfficientNet-B4 + CBAM) & GRAD-CAM
# =====================================================================
@keras.saving.register_keras_serializable(package="tbs")
class ChannelMean(layers.Layer):
    def call(self, x):
        return keras.ops.mean(x, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], 1)


@keras.saving.register_keras_serializable(package="tbs")
class ChannelMax(layers.Layer):
    def call(self, x):
        return keras.ops.max(x, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], 1)


def cbam_block(inputs, ratio=16):
    ch = inputs.shape[-1]
    dense1 = layers.Dense(ch // ratio, activation="relu")
    dense2 = layers.Dense(ch)
    avg = dense2(dense1(layers.GlobalAveragePooling2D()(inputs)))
    mx = dense2(dense1(layers.GlobalMaxPooling2D()(inputs)))
    ca = layers.Activation("sigmoid")(layers.Add()([avg, mx]))
    ca = layers.Reshape((1, 1, ch))(ca)
    x = layers.Multiply()([inputs, ca])

    avg_s = ChannelMean()(x)
    max_s = ChannelMax()(x)
    sa = layers.Concatenate(axis=-1)([avg_s, max_s])
    sa = layers.Conv2D(1, 7, padding="same", activation="sigmoid", use_bias=False)(sa)
    return layers.Multiply(name="cbam_out")([x, sa])


def build_model():
    base_model = tf.keras.applications.EfficientNetB4(
        include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    x = cbam_block(base_model.output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return keras.Model(base_model.input, outputs)


@st.cache_resource
def load_tbs_model():
    # Mengunduh otomatis dari Google Drive jika file belum ada
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Mengunduh file bobot model dari Google Drive... (Hanya sekali saat startup awal)"):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            
    model = build_model()
    model.load_weights(MODEL_PATH)
    return model


def generate_gradcam(img_array, model, pred_index=None):
    cbam_layer = model.get_layer("cbam_out")
    grad_model = keras.Model(model.inputs, [cbam_layer.output, model.output])
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.squeeze(conv_outputs @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy(), pred_index.numpy()


# =====================================================================
# 4. FUNGSI BANTU & LAPORAN
# =====================================================================
def prep(image):
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    return resized, np.array(resized).astype(np.float32)


def quality_issues(image):
    issues = []
    if min(image.size) < 224:
        issues.append("Resolusi gambar kecil (< 224 px), detail bisa hilang.")
    
    img_np = np.array(image)
    g = cv2.resize(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), (IMG_SIZE, IMG_SIZE))
    
    if g.mean() < 50:
        issues.append("Gambar terlalu gelap. Gunakan pencahayaan lebih terang.")
    elif g.mean() > 215:
        issues.append("Gambar terlalu terang / overexposed.")
    if cv2.Laplacian(g, cv2.CV_64F).var() < 60:
        issues.append("Gambar terlihat buram. Pastikan fokus kamera tepat pada TBS.")
        
    return issues


def make_heat_images(img_resized, heatmap, cmap_name, alpha):
    cmap = {"JET": cv2.COLORMAP_JET, "VIRIDIS": cv2.COLORMAP_VIRIDIS, "PLASMA": cv2.COLORMAP_PLASMA}[cmap_name]
    h = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    h = cv2.GaussianBlur(h, (15, 15), 0)
    colored = cv2.applyColorMap(np.uint8(255 * np.clip(h, 0, 1)), cmap)
    base = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base, 1 - alpha, colored, alpha, 0)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB), cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def log_history(source, name, label, conf, key):
    if key in st.session_state.seen:
        return
    st.session_state.seen.add(key)
    st.session_state.history.append({
        "Waktu": datetime.now().strftime("%d-%m-%Y %H:%M:%S"), 
        "Sumber": source, 
        "File": name,
        "Kelas": label, 
        "Kepercayaan (%)": "-" if conf is None else f"{round(conf, 2)}%",
    })


def build_pdf(name, label, conf, note, probs, imgs):
    from fpdf import FPDF

    def clean(t):
        return str(t).encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Laporan Klasifikasi Kematangan TBS Sawit", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Tanggal: {datetime.now():%d-%m-%Y %H:%M}   |   File: {clean(name)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, f"Kelas Prediksi: {label}   ({conf:.2f}%)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, clean(f"Catatan: {note}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    for (title, im), x in zip(imgs, (10, 75, 140)):
        pdf.image(Image.fromarray(im), x=x, y=pdf.get_y() + 6, w=58)
        pdf.set_xy(x, pdf.get_y())
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(58, 5, title)
    pdf.set_y(pdf.get_y() + 70)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Probabilitas Tiap Kelas", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for n, p in zip(CLASS_NAMES, probs):
        pdf.cell(0, 6, f"- {n}: {p * 100:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Hasil prediksi bersifat pendukung keputusan. Sistem: EfficientNet-B4 + CBAM + Grad-CAM.")
    return bytes(pdf.output())


def render_single(image, name, key, source, model, cfg):
    cmap_name, alpha, thr = cfg
    issues = quality_issues(image)
    for msg in issues:
        st.warning("📷 " + msg)

    resized, arr = prep(image)
    batch = np.expand_dims(arr, 0)
    with st.spinner("🔍 Menganalisis citra..."):
        probs = model.predict(batch, verbose=0)[0]
    
    # Metrik Deteksi Out of Distribution (OOD)
    sorted_probs = np.sort(probs)[::-1]
    top = float(sorted_probs[0]) * 100
    margin = float(sorted_probs[0] - sorted_probs[1]) * 100

    # Gambar ditolak jika kepercayaan di bawah ambang (55%) atau margin kebingungan terlalu tipis (< 12%)
    is_ood = (top < thr) or (margin < 12.0 and top < 60.0)

    if is_ood:
        a, b = st.columns([1, 2])
        a.image(image, caption="Citra yang diunggah", use_container_width=True)
        b.error("⚠️ **Gambar yang diunggah bukan Tandan Buah Segar (TBS) Kelapa Sawit.**\n\n"
                "Sistem mengidentifikasi objek luar domain (seperti pakaian, kendaraan, atau latar umum).\n"
                "Silakan unggah foto TBS Kelapa Sawit yang jelas, utuh, dan memiliki pencahayaan memadai.")
        
        log_history(source, name, "Bukan TBS", None, key)
        return

    with st.spinner("🔬 Menghitung Grad-CAM..."):
        heatmap, idx = generate_gradcam(batch, model)
    label, conf = CLASS_NAMES[int(idx)], float(probs[int(idx)]) * 100
    info = CLASS_INFO[label]
    heat_rgb, over_rgb = make_heat_images(resized, heatmap, cmap_name, alpha)
    log_history(source, name, label, conf, key)

    st.markdown('<div class="sec" style="margin-top:26px">Hasil Prediksi</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.1, 1, 1.5])
    c1.markdown(html(f'<div class="panel"><div class="k">Kelas prediksi</div><div class="big">{label}</div>'
                      f'<span class="chip" style="background:{info["color"]}">Terdeteksi</span></div>'), unsafe_allow_html=True)
    c2.markdown(html(f'<div class="panel"><div class="k">Tingkat kepercayaan</div><div class="big">{conf:.2f}%</div>'
                      f'<span style="color:#6b7a70;font-size:.82rem">Confidence score</span></div>'), unsafe_allow_html=True)
    c3.markdown(html(f'<div class="panel"><div class="k">Catatan status</div>'
                      f'<div style="margin-top:10px;font-weight:600;color:#0f3d2a;line-height:1.5">{info["note"]}</div></div>'),
                unsafe_allow_html=True)

    st.write("")
    with st.expander("Probabilitas semua kelas", expanded=True):
        for n, p in zip(CLASS_NAMES, probs):
            st.progress(float(p), text=f"{n}: {p * 100:.2f}%")

    st.markdown('<div class="sec" style="margin-top:26px">Visualisasi Grad-CAM</div>'
                '<div class="sub">Area berwarna hangat adalah bagian citra yang paling memengaruhi keputusan model.</div>',
                unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    v1.image(image, caption="Citra asli", use_container_width=True)
    v2.image(heat_rgb, caption=f"Heatmap Grad-CAM ({cmap_name})", use_container_width=True)
    v3.image(over_rgb, caption=f"Overlay fokus ({label})", use_container_width=True)

    st.write("")
    d1, d2, _ = st.columns([1, 1, 2])
    try:
        pdf_bytes = build_pdf(name, label, conf, info["note"], probs,
                              [("Citra asli", np.array(resized)), ("Heatmap", heat_rgb), ("Overlay", over_rgb)])
        d1.download_button("📄 Unduh Laporan PDF", pdf_bytes, f"Laporan_{label}.pdf", "application/pdf", key=f"pdf-{key}")
    except ImportError:
        d1.caption("Pasang `fpdf2` untuk laporan PDF.")
    txt = f"LAPORAN KLASIFIKASI TBS SAWIT\nKelas: {label}\nKepercayaan: {conf:.2f}%\n" + \
          "".join(f"- {n}: {p * 100:.2f}%\n" for n, p in zip(CLASS_NAMES, probs))
    d2.download_button("📥 Unduh Ringkasan (.txt)", txt, "Hasil_Klasifikasi_TBS.txt", key=f"txt-{key}")


def file_key(data):
    return hashlib.md5(data).hexdigest()


# =====================================================================
# 5. HEADER & NAVIGASI
# =====================================================================
hb, hn = st.columns([1, 2.2])
hb.markdown(html('<div class="brand"><div class="logo">🌴</div><div><b>SawitScan</b>'
                  '<small>EfficientNet-B4 · CBAM · Grad-CAM</small></div></div>'), unsafe_allow_html=True)
hn.radio("Navigasi", PAGES, key="nav", horizontal=True, label_visibility="collapsed")
st.markdown('<div class="navline"></div>', unsafe_allow_html=True)

# =====================================================================
# 6. HALAMAN APLIKASI
# =====================================================================
def page_home():
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown(html("""
            <div class="tag">DEEP LEARNING · EXPLAINABLE AI</div>
            <div class="hero-h">Kenali kematangan <em>tandan buah sawit</em> dalam hitungan detik.</div>
            <p class="lead">Sistem berbasis web yang memanfaatkan CNN <b>EfficientNet-B4</b> dengan modul
            perhatian <b>CBAM</b> untuk mengklasifikasikan TBS, lengkap dengan <b>Grad-CAM</b>
            agar setiap keputusan model dapat dipahami.</p>"""), unsafe_allow_html=True)
        st.write("")
        b1, b2, _ = st.columns([1.3, 1.1, 1])
        b1.button("🚀 Mulai Klasifikasi", on_click=go, args=("Klasifikasi",), key="h1")
        b2.button("📖 Lihat Panduan", on_click=go, args=("Panduan",), key="h2")
    with right:
        rows = "".join(
            f'<div class="crow"><div class="dot" style="background:{CLASS_INFO[n]["color"]}"></div>'
            f'<div><b>{n}</b><span>{CLASS_INFO[n]["desc"]}</span></div></div>'
            for n in ["Mentah", "Matang", "Terlalu matang", "Busuk"])
        st.markdown(html(f'<div class="panel"><div class="k" style="margin-bottom:6px">Empat kategori kematangan</div>{rows}</div>'),
                    unsafe_allow_html=True)

    st.markdown(html("""<div class="kpis">
        <div class="kpi"><b>4</b><span>Kelas kematangan</span></div>
        <div class="kpi"><b>B4 + CBAM</b><span>Arsitektur model</span></div>
        <div class="kpi"><b>380 × 380</b><span>Resolusi input</span></div>
        <div class="kpi"><b>Grad-CAM</b><span>Visual explanation</span></div></div>"""), unsafe_allow_html=True)

    st.markdown('<div class="sec">Yang bisa Anda lakukan</div><div class="sub">Fitur lengkap untuk menguji dan mengevaluasi model.</div>',
                unsafe_allow_html=True)
    feats = [("📷", "Unggah atau foto langsung", "Gunakan file JPG/PNG atau ambil foto TBS dari kamera perangkat."),
             ("🗂️", "Proses banyak gambar", "Unggah puluhan citra sekaligus, hasilnya rapi dalam tabel CSV/Excel."),
             ("🔥", "Grad-CAM interaktif", "Pilih skema warna dan transparansi heatmap sesuai kebutuhan."),
             ("📄", "Laporan PDF", "Ringkasan lengkap dengan citra asli, heatmap, dan probabilitas."),
             ("📊", "Evaluasi model", "Akurasi, confusion matrix, precision, recall, F1, dan grafik training."),
             ("🕘", "Riwayat prediksi", "Pantau semua hasil dalam satu sesi beserta ringkasan per kelas.")]
    for row in (feats[:3], feats[3:]):
        for col, (ic, t, d) in zip(st.columns(3, gap="large"), row):
            col.markdown(html(f'<div class="feat"><div class="ic">{ic}</div><h4>{t}</h4><p>{d}</p></div>'), unsafe_allow_html=True)

    st.markdown('<div class="sec">Cara kerja</div>', unsafe_allow_html=True)
    st.markdown(html("""<div class="tl">
        <div class="tl-i"><div class="tl-n">1</div><h4>Unggah citra</h4><p>Pilih atau potret foto TBS dengan objek jelas dan pencahayaan cukup.</p></div>
        <div class="tl-i"><div class="tl-n">2</div><h4>Analisis otomatis</h4><p>Model memprediksi kelas, kepercayaan, dan probabilitas tiap kelas.</p></div>
        <div class="tl-i"><div class="tl-n">3</div><h4>Baca penjelasan</h4><p>Amati Grad-CAM untuk melihat area yang menentukan keputusan.</p></div></div>"""),
                unsafe_allow_html=True)

    st.markdown('<div class="sec">Metode yang digunakan</div>', unsafe_allow_html=True)
    meths = [("EfficientNet-B4", "CNN yang menyeimbangkan kedalaman, lebar, dan resolusi jaringan untuk mengekstraksi fitur warna dan tekstur TBS."),
             ("CBAM", "Convolutional Block Attention Module menambahkan perhatian channel dan spasial agar model fokus pada fitur penting."),
             ("Grad-CAM", "Teknik explainable AI yang menampilkan area citra paling berpengaruh terhadap prediksi.")]
    for col, (t, d) in zip(st.columns(3, gap="large"), meths):
        col.markdown(html(f'<div class="meth"><h4>{t}</h4><p>{d}</p></div>'), unsafe_allow_html=True)


def page_classify():
    st.markdown('<h2 style="margin:0 0 4px 0">Klasifikasi Kematangan TBS</h2>'
                '<div class="sub">Unggah satu gambar, ambil foto dari kamera, atau proses banyak gambar sekaligus.</div>',
                unsafe_allow_html=True)
    
    with st.spinner("⚡ Memuat model EfficientNet-B4 + CBAM..."):
        model = load_tbs_model()

    with st.expander("⚙️ Pengaturan visualisasi & deteksi"):
        s1, s2, s3 = st.columns(3)
        cmap_name = s1.selectbox("Skema warna heatmap", ["JET", "VIRIDIS", "PLASMA"])
        alpha = s2.slider("Transparansi overlay", 0.1, 0.9, 0.4, 0.05)
        # Ambang kepercayaan minimum default set di 55%
        thr = s3.slider("Ambang kepercayaan minimum (%)", 30, 99, 55, 1,
                        help="Di bawah ambang ini atau jika kebingungan model tinggi, gambar dianggap bukan TBS.")
    cfg = (cmap_name, alpha, thr)

    t1, t2, t3 = st.tabs(["📁 Satu gambar", "📷 Kamera", "🗂️ Batch (banyak gambar)"])

    with t1:
        up = st.file_uploader("Unggah citra buah sawit (JPG / PNG)", type=["jpg", "jpeg", "png"], key="single")
        if up is None:
            st.info("Unggah gambar TBS sawit untuk memulai prediksi.")
        else:
            data = up.getvalue()
            render_single(Image.open(io.BytesIO(data)).convert("RGB"), up.name, file_key(data), "Unggah", model, cfg)

    with t2:
        cam = st.camera_input("Arahkan kamera ke TBS lalu ambil foto")
        if cam is not None:
            data = cam.getvalue()
            render_single(Image.open(io.BytesIO(data)).convert("RGB"), "foto_kamera.jpg", file_key(data), "Kamera", model, cfg)

    with t3:
        files = st.file_uploader("Unggah banyak citra sekaligus", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True, key="batch")
        if files and st.button("▶️ Proses semua gambar", key="runbatch"):
            rows, bar = [], st.progress(0.0, text="Memproses...")
            for i, f in enumerate(files):
                data = f.getvalue()
                im = Image.open(io.BytesIO(data)).convert("RGB")
                _, arr = prep(im)
                p = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
                sorted_p = np.sort(p)[::-1]
                top = float(sorted_p[0]) * 100
                margin = float(sorted_p[0] - sorted_p[1]) * 100
                
                is_ood = (top < thr) or (margin < 12.0 and top < 60.0)
                
                if is_ood:
                    label, conf = "Bukan TBS", "-"
                else:
                    label, conf = CLASS_NAMES[int(np.argmax(p))], f"{top:.2f}%"
                rows.append({"No": i + 1, "File": f.name, "Kelas": label, "Kepercayaan (%)": conf,
                             "Catatan": "; ".join(quality_issues(im)) or "-"})
                log_history("Batch", f.name, label, None if is_ood else top, file_key(data) + "-batch")
                bar.progress((i + 1) / len(files), text=f"Memproses {i + 1}/{len(files)}")
            bar.empty()
            st.session_state.batch_df = pd.DataFrame(rows)
        df = st.session_state.get("batch_df")
        if df is not None and not df.empty:
            st.markdown('<div class="sec" style="margin-top:20px">Hasil batch</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.bar_chart(df["Kelas"].value_counts())
            e1, e2, _ = st.columns([1, 1, 2])
            e1.download_button("⬇️ Unduh CSV", df.to_csv(index=False).encode("utf-8"), "hasil_batch.csv", "text/csv")
            try:
                buf = io.BytesIO()
                df.to_excel(buf, index=False, engine="openpyxl")
                e2.download_button("⬇️ Unduh Excel", buf.getvalue(), "hasil_batch.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception:
                e2.caption("Pasang `openpyxl` untuk ekspor Excel.")


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return tp.sum() / cm.sum(), prec, rec, f1


def page_eval():
    st.markdown('<h2 style="margin:0 0 4px 0">Evaluasi Model</h2>'
                '<div class="sub">Kinerja model pada data uji: akurasi, confusion matrix, dan grafik training.</div>',
                unsafe_allow_html=True)
    data = None
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        up = st.file_uploader("Unggah file evaluasi.json", type=["json"])
        if up:
            data = json.load(up)
    with st.expander("Cara membuat evaluasi.json (jalankan di notebook training)", expanded=data is None):
        st.code(
            "import json, numpy as np\n"
            "from sklearn.metrics import confusion_matrix\n\n"
            "# y_true: label asli data uji (urutan kelas: Busuk, Matang, Mentah, Terlalu matang)\n"
            "y_pred = np.argmax(model.predict(test_ds), axis=1)\n"
            "cm = confusion_matrix(y_true, y_pred)\n"
            "hist = {k: [float(v) for v in history.history[k]]\n"
            "        for k in ['accuracy', 'val_accuracy', 'loss', 'val_loss'] if k in history.history}\n"
            "json.dump({'confusion_matrix': cm.tolist(), 'history': hist}, open('evaluasi.json', 'w'))",
            language="python")
        st.caption("Simpan `evaluasi.json` di folder yang sama dengan `app.py`, lalu muat ulang halaman.")
    if data is None:
        return

    cm = np.array(data["confusion_matrix"])
    acc, prec, rec, f1 = metrics_from_cm(cm)
    k = st.columns(4)
    for col, (t, v) in zip(k, [("Akurasi", acc), ("Macro Precision", prec.mean()),
                               ("Macro Recall", rec.mean()), ("Macro F1-score", f1.mean())]):
        col.markdown(html(f'<div class="panel"><div class="k">{t}</div><div class="big">{v * 100:.2f}%</div></div>'),
                     unsafe_allow_html=True)

    st.markdown('<div class="sec" style="margin-top:30px">Confusion Matrix</div>', unsafe_allow_html=True)
    df = pd.DataFrame([(CLASS_NAMES[i], CLASS_NAMES[j], int(cm[i][j])) for i in range(len(CLASS_NAMES))
                       for j in range(len(CLASS_NAMES))], columns=["Aktual", "Prediksi", "Jumlah"])
    base = alt.Chart(df).encode(x=alt.X("Prediksi:N", sort=CLASS_NAMES), y=alt.Y("Aktual:N", sort=CLASS_NAMES))
    heat = base.mark_rect().encode(color=alt.Color("Jumlah:Q", scale=alt.Scale(range=["#faf8f2", "#0f3d2a"]), legend=None))
    text = base.mark_text(fontSize=16, fontWeight="bold").encode(
        text="Jumlah:Q",
        color=alt.condition(alt.datum.Jumlah > df["Jumlah"].max() / 2, alt.value("white"), alt.value("#0f3d2a")))
    st.altair_chart((heat + text).properties(height=360), use_container_width=True)

    st.markdown('<div class="sec" style="margin-top:20px">Metrik per Kelas</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Kelas": CLASS_NAMES, "Precision": np.round(prec, 4), "Recall": np.round(rec, 4),
        "F1-score": np.round(f1, 4), "Jumlah data uji": cm.sum(1)}), use_container_width=True, hide_index=True)

    hist = data.get("history")
    if hist:
        st.markdown('<div class="sec" style="margin-top:20px">Grafik Training</div>', unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        acc_df = pd.DataFrame({k: v for k, v in hist.items() if "acc" in k})
        loss_df = pd.DataFrame({k: v for k, v in hist.items() if "loss" in k})
        if not acc_df.empty:
            g1.caption("Akurasi per epoch")
            g1.line_chart(acc_df)
        if not loss_df.empty:
            g2.caption("Loss per epoch")
            g2.line_chart(loss_df)
        if "accuracy" in hist and "val_accuracy" in hist:
            gap = (hist["accuracy"][-1] - hist["val_accuracy"][-1]) * 100
            st.caption(f"Selisih akurasi training vs validasi pada epoch terakhir: **{gap:.2f}%**")


def page_history():
    st.markdown('<h2 style="margin:0 0 4px 0">Riwayat Prediksi</h2>'
                '<div class="sub">Semua prediksi pada sesi ini. Riwayat hilang jika halaman dimuat ulang.</div>',
                unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("Belum ada prediksi. Mulai dari halaman Klasifikasi.")
        return
    df = pd.DataFrame(st.session_state.history)
    k1, k2, k3 = st.columns(3)
    k1.metric("Total prediksi", len(df))
    k2.metric("Terdeteksi TBS", int((df["Kelas"] != "Bukan TBS").sum()))
    k3.metric("Bukan TBS", int((df["Kelas"] == "Bukan TBS").sum()))
    c1, c2 = st.columns([1.6, 1], gap="large")
    c1.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)
    c2.caption("Jumlah per kelas")
    c2.bar_chart(df["Kelas"].value_counts())
    b1, b2, _ = st.columns([1, 1, 2])
    b1.download_button("⬇️ Unduh CSV", df.to_csv(index=False).encode("utf-8"), "riwayat_prediksi.csv", "text/csv")
    if b2.button("🗑️ Hapus riwayat"):
        st.session_state.history, st.session_state.seen = [], set()
        st.rerun()


def page_guide():
    st.markdown('<h2 style="margin:0 0 4px 0">Panduan & Tentang</h2>'
                '<div class="sub">Tips pengambilan foto, pertanyaan umum, dan batasan sistem.</div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### 📸 Tips memotret TBS")
        st.markdown("- Pastikan TBS terlihat **utuh** dan memenuhi sebagian besar frame.\n"
                    "- Gunakan **cahaya alami** yang merata, hindari bayangan keras dan cahaya belakang.\n"
                    "- Jaga kamera tetap **fokus dan stabil** agar tidak buram.\n"
                    "- Hindari latar yang ramai atau banyak objek lain.\n"
                    "- Gunakan resolusi minimal sekitar **224 × 224 piksel**.")
    with b:
        st.markdown("#### ⚠️ Batasan sistem")
        st.markdown("- Hanya untuk citra **TBS kelapa sawit**; gambar lain akan ditolak.\n"
                    "- Akurasi **tidak 100%**; hasil adalah pendukung keputusan, bukan pengganti penilaian ahli.\n"
                    "- Kondisi pencahayaan dan sudut yang jauh berbeda dari data latih dapat menurunkan akurasi.")
    st.markdown("#### ❓ Pertanyaan umum")
    faqs = [("Apa itu Grad-CAM?", "Teknik yang menampilkan heatmap area citra yang paling berpengaruh pada keputusan model. Warna hangat (merah/kuning) berarti pengaruh tinggi."),
            ("Mengapa gambar saya dibilang bukan TBS?", "Model tidak cukup yakin bahwa gambar tersebut TBS sawit. Coba foto dengan pencahayaan lebih baik, atau turunkan ambang kepercayaan di pengaturan."),
            ("Apa arti tingkat kepercayaan?", "Probabilitas tertinggi yang diberikan model untuk kelas yang dipilih. Nilai tinggi menunjukkan model lebih yakin."),
            ("Apakah data saya disimpan?", "Tidak. Gambar hanya diproses pada sesi berjalan dan riwayat hanya ada di memori sesi.")]
    for q, ans in faqs:
        with st.expander(q):
            st.write(ans)
    st.markdown("#### 🧪 Informasi teknis")
    st.table(pd.DataFrame({"Komponen": ["Arsitektur", "Modul perhatian", "Explainability", "Ukuran input", "Kelas"],
                           "Keterangan": ["EfficientNet-B4", "CBAM (channel + spatial attention)", "Grad-CAM",
                                          f"{IMG_SIZE} × {IMG_SIZE} piksel", ", ".join(CLASS_NAMES)]}))


{"Beranda": page_home, "Klasifikasi": page_classify, "Evaluasi Model": page_eval,
 "Riwayat": page_history, "Panduan": page_guide}[st.session_state.nav]()

st.markdown('<div class="footer">© 2026 SawitScan · Penelitian Skripsi Klasifikasi Kematangan TBS Sawit · EfficientNet-B4, CBAM & Grad-CAM</div>',
            unsafe_allow_html=True)