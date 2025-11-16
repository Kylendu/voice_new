import streamlit as st
import numpy as np
import joblib
import tempfile
import pandas as pd
import os
from utils.feature_extraction import extract_features
from st_audiorec import st_audiorec
import librosa
import speech_recognition as sr
from pydub import AudioSegment  # <== Untuk konversi m4a → wav

# ================= KONFIGURASI HALAMAN =================
st.set_page_config(
    page_title="🎧 Voice Identification System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= SIDEBAR =================
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

# ================= LOAD MODEL =================
try:
    model_user = joblib.load("models/user_model.pkl")
    model_status = joblib.load("models/status_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    st.sidebar.success("✅ Model berhasil dimuat")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ================= FUNGSI KONVERSI =================
def convert_m4a_to_wav(input_file):
    """Konversi file .m4a ke .wav dan kembalikan path output"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio = AudioSegment.from_file(input_file, format="m4a")
            audio.export(tmp_wav.name, format="wav")
            return tmp_wav.name
    except Exception as e:
        st.error(f"Gagal mengonversi file M4A: {e}")
        return None

# ================= FUNGSI PROSES AUDIO =================
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

    if user_pred == 0:
        user_label = "user1"
    elif user_pred == 1:
        user_label = "user2"
    elif user_pred == 2:
        user_label = "anomali"
    else:
        user_label = "error user"
    # user_label = f"user{user_pred + 1}"
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

    with st.expander("📈 Detail Fitur yang Diekstraksi"):
        st.dataframe(feature_df.T, use_container_width=True)

# ================= INPUT OPSI =================
st.markdown("### 🎧 Pilih Metode Input")
input_option = st.radio(
    "",
    ["🎤 Rekam suara langsung", "📁 Upload file (.wav / .m4a)"],
    horizontal=True
)

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

elif input_option == "📁 Upload file (.wav / .m4a)":
    uploaded_file = st.file_uploader("Unggah file suara Anda:", type=["wav", "m4a"])
    if uploaded_file:
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            input_path = tmp.name

        # 🔁 Konversi otomatis jika format m4a
        if suffix == ".m4a":
            st.info("🔄 Mengonversi file .m4a ke .wav ...")
            wav_path = convert_m4a_to_wav(input_path)
            if wav_path:
                st.audio(wav_path, format="audio/wav")
                st.info("✅ Konversi selesai, memproses file .wav ...")
                process_audio(wav_path)
        else:
            st.audio(input_path, format="audio/wav")
            st.info("📂 File berhasil diunggah, memproses...")
            process_audio(input_path)
