import os
import numpy as np
import pandas as pd
import librosa

def extract_features(file_path):
    """Ekstraksi fitur MFCC + fitur tambahan dari file WAV"""
    try:
        y, sr = librosa.load(file_path, sr=None)
        y = librosa.util.normalize(y)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        features = np.hstack([
            mfcc,
            chroma,
            spec_contrast,
            [zcr, rms, centroid, bandwidth, rolloff]
        ])

        features = np.nan_to_num(features)
        return {f"mfcc{i}": float(v) for i, v in enumerate(features)}

    except Exception as e:
        print(f"[ERROR] Gagal memproses {file_path}: {e}")
        return None

def create_dataset(data_dir="data", output_csv="data/voice_dataset.csv"):
    rows = []
    for user in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user)
        if not os.path.isdir(user_path):
            continue

        for status in ["buka", "tutup"]:
            status_path = os.path.join(user_path, status)
            if not os.path.exists(status_path):
                continue

            for file in os.listdir(status_path):
                if file.endswith(".wav"):
                    path = os.path.join(status_path, file)
                    feats = extract_features(path)
                    if feats:
                        feats["user"] = user
                        feats["status"] = status
                        feats["filename"] = file
                        rows.append(feats)
                    else:
                        print(f"[SKIP] {path} tidak bisa diproses.")

    df = pd.DataFrame(rows)

    # Konversi user & status ke numerik
    df["user"] = df["user"].replace({"user1": 0, "user2": 1})
    df["status"] = df["status"].replace({"buka": 0, "tutup": 1})

    # Pastikan semua fitur numerik
    for col in df.columns:
        if col not in ["user", "status", "filename"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Hapus baris yang ada NaN
    df = df.dropna().reset_index(drop=True)

    # Simpan CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[INFO] Dataset berhasil dibuat: {output_csv} ({len(df)} sampel)")
    print(f"[DEBUG] Nilai unik user: {df['user'].unique()}")
    print(f"[DEBUG] Nilai unik status: {df['status'].unique()}")
    return df

if __name__ == "__main__":
    create_dataset()
