import streamlit as st
import numpy as np
import joblib
import tempfile
import pandas as pd
from utils.feature_extraction import extract_features
from st_audiorec import st_audiorec
import librosa
import speech_recognition as sr

# ========== KONFIGURASI HALAMAN ==========
st.set_page_config(
    page_title="🎧 Voice Identification System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======== SIDEBAR ========
st.sidebar.title("⚙️ Pengaturan & Informasi")
st.sidebar.info(
    """
    Aplikasi ini menggunakan dua model:
    - **User Model:** mengenali siapa yang berbicara (`user1` atau `user2`)
    - **Status Model:** mengenali apakah kata yang diucapkan adalah *buka* atau *tutup*
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("Dibuat dengan 💙 menggunakan Streamlit + RandomForest + SpeechRecognition")

# ======== HEADER UTAMA ========
st.markdown(
    """
    <div style="text-align:center;">
        <h1>🎙️ Voice Identification System</h1>
        <p>Sistem ini mendeteksi <b>pengguna</b> dan <b>status suara</b> (buka/tutup)
        menggunakan model <b>Random Forest</b> dan analisis fitur MFCC.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ======== LOAD MODEL ========
try:
    model_user = joblib.load("models/user_model.pkl")
    model_status = joblib.load("models/status_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    st.sidebar.success("✅ Model berhasil dimuat")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ======== PILIH INPUT ========
st.markdown("### 🎧 Pilih Metode Input")
input_option = st.radio(
    "",
    ["🎤 Rekam suara langsung", "📁 Upload file (.wav)"],
    horizontal=True
)

# ======== PROSES AUDIO ========
def process_audio(audio_path):
    features = extract_features(audio_path)
    if not features:
        st.error("⚠️ Gagal mengekstraksi fitur dari suara.")
        return

    # ===== DETEKSI SILENCE =====
    y, sr_audio = librosa.load(audio_path, sr=None)
    rms = np.mean(librosa.feature.rms(y=y))
    duration = len(y) / sr_audio
    if duration < 0.1 or rms < 1e-4:
        st.error("🚫 Tidak ada suara yang terdeteksi. Silakan rekam ulang.")
        return

    feature_df = pd.DataFrame([features])
    X = feature_df[feature_cols].to_numpy().reshape(1, -1)

    # ===== PREDIKSI =====
    user_pred_proba = model_user.predict_proba(X)
    status_pred_proba = model_status.predict_proba(X)

    user_pred = np.argmax(user_pred_proba)
    status_pred = np.argmax(status_pred_proba)

    user_conf = np.max(user_pred_proba)
    status_conf = np.max(status_pred_proba)

    user_label = f"user{user_pred + 1}"
    status_label = "buka" if status_pred == 0 else "tutup"

    # ===== SPEECH TO TEXT =====
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="id-ID")
        st.markdown("### 📝 Hasil Speech-to-Text")
        st.info(f"“{text}”")
        text_lower = text.lower()
        if "buka" in text_lower:
            status_label = "buka"
        elif "tutup" in text_lower:
            status_label = "tutup"
    except Exception as e:
        st.warning(f"Teks tidak dapat dikenali: {e}")

    # ===== TAMPILKAN HASIL =====
    st.markdown("### 📊 Hasil Identifikasi")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("🧍 Deteksi Pengguna", user_label, f"Confidence: {user_conf:.2f}")
    with col2:
        st.metric("🔓 Status Suara", status_label, f"Confidence: {status_conf:.2f}")

    if status_label == "buka":
        st.success(f"✅ Teridentifikasi: **{user_label} sedang membuka** sesuatu.")
    else:
        st.warning(f"🔒 Teridentifikasi: **{user_label} sedang menutup** sesuatu.")

    # ===== Fitur yang diekstraksi =====
    with st.expander("📈 Detail Fitur yang Diekstraksi"):
        st.dataframe(feature_df.T, use_container_width=True)

# ======== INPUT OPSI ========
if input_option == "🎤 Rekam suara langsung":
    st.write("Klik tombol di bawah untuk mulai merekam suara:")
    audio_bytes = st_audiorec()
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name
        st.audio(audio_path, format="audio/wav")
        st.info("🎧 Suara berhasil direkam, memproses...")
        process_audio(audio_path)

elif input_option == "📁 Upload file (.wav)":
    uploaded_file = st.file_uploader("Unggah file suara Anda:", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name
        st.info("📂 File berhasil diunggah, memproses...")
        process_audio(audio_path)
