"""
automate_Andi-Arif-Abdillah.py
================================
Otomatisasi data preprocessing untuk dataset Telco Customer Churn.

File ini merupakan konversi dari notebook eksperimen
`Eksperimen_Andi-Arif-Abdillah.ipynb`. Tahapan preprocessing-nya sama,
tetapi dibungkus dalam fungsi-fungsi agar dapat dijalankan secara otomatis
dan menghasilkan data yang siap dilatih.

Cara pakai:
    # Sebagai modul
    from automate_Andi_Arif_Abdillah import preprocess_data
    df_clean = preprocess_data("telco_churn_raw/telco_churn_raw.csv")

    # Sebagai script (CLI)
    python automate_Andi-Arif-Abdillah.py \
        --input telco_churn_raw/telco_churn_raw.csv \
        --output preprocessing/telco_churn_preprocessing.csv
"""

import argparse
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Kolom kategorikal biner (2 kategori) -> Label Encoding 0/1
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# Kolom kategorikal multi-kelas (>2 kategori) -> One-Hot Encoding
MULTI_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

# Kolom numerik -> Standardisasi
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

TARGET = "Churn"


def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari file CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path)
    print(f"[load] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan data: drop identifier, tangani TotalCharges, hapus duplikat."""
    df = df.copy()

    # 1. Hapus kolom identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 2. TotalCharges: string kosong -> NaN -> numerik -> imputasi 0 (pelanggan baru)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 3. Hapus duplikat
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"[clean] Setelah pembersihan: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encoding target, fitur biner (Label), dan fitur multi-kelas (One-Hot)."""
    df = df.copy()

    # Target: No -> 0, Yes -> 1
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)

    # Fitur biner -> Label Encoding
    le = LabelEncoder()
    for col in BINARY_COLS:
        df[col] = le.fit_transform(df[col])

    # Fitur multi-kelas -> One-Hot Encoding
    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=True)

    # Ubah kolom hasil one-hot (bool) menjadi integer 0/1
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"[encode] Setelah encoding: {df.shape[1]} kolom")
    return df


def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Standardisasi fitur numerik menggunakan StandardScaler."""
    df = df.copy()
    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    print("[scale] Fitur numerik distandardisasi:", NUMERIC_COLS)
    return df


def preprocess_data(input_path: str) -> pd.DataFrame:
    """
    Pipeline preprocessing lengkap.

    Menerima path dataset mentah dan mengembalikan DataFrame yang sudah
    bersih, ter-encode, dan terstandardisasi (siap dilatih).
    """
    df = load_data(input_path)
    df = clean_data(df)
    df = encode_features(df)
    df = scale_numeric(df)
    print(f"[done] Data siap latih: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Otomatisasi preprocessing Telco Customer Churn."
    )
    parser.add_argument(
        "--input", default="telco_churn_raw/telco_churn_raw.csv",
        help="Path file dataset mentah (CSV).",
    )
    parser.add_argument(
        "--output", default="preprocessing/telco_churn_preprocessing.csv",
        help="Path output dataset hasil preprocessing (CSV).",
    )
    args = parser.parse_args()

    df_clean = preprocess_data(args.input)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df_clean.to_csv(args.output, index=False)
    print(f"[save] Dataset hasil preprocessing disimpan di: {args.output}")


if __name__ == "__main__":
    main()
