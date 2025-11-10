import os
import pandas as pd
from utils.feature_extraction import create_dataset

def main():
    output_csv = "data/voice_dataset.csv"
    
    if not os.path.exists(output_csv):
        print("[INFO] Dataset belum ditemukan, sedang dibuat...")
        df = create_dataset(output_csv=output_csv)
    else:
        df = pd.read_csv(output_csv)
        print(f"[INFO] Dataset sudah ada: {output_csv} ({len(df)} sampel)")

    print("[INFO] Contoh 5 baris dataset:")
    print(df.head())

    print("[INFO] Kolom dataset:")
    print(df.columns.tolist())

    # Pastikan kolom numerik untuk fitur
    feature_cols = [col for col in df.columns if col.startswith("mfcc")]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"[INFO] Dataset siap digunakan dengan {len(feature_cols)} fitur.")

if __name__ == "__main__":
    main()
