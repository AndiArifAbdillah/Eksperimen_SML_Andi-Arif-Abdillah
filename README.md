# Eksperimen SML — Andi Arif Abdillah

Repository eksperimen & otomatisasi preprocessing untuk submission kelas
**Membangun Sistem Machine Learning** (Dicoding).

## Dataset
**Telco Customer Churn** — klasifikasi biner untuk memprediksi apakah seorang
pelanggan telekomunikasi akan berhenti berlangganan (`Churn`: Yes/No).
7.043 baris, 21 kolom (campuran fitur numerik & kategorikal).

## Struktur
```
Eksperimen_SML_Andi-Arif-Abdillah
├── .github/workflows/preprocessing.yml     # CI preprocessing otomatis (Advanced)
├── telco_churn_raw/
│   └── telco_churn_raw.csv                  # dataset mentah
└── preprocessing/
    ├── Eksperimen_Andi-Arif-Abdillah.ipynb  # notebook eksperimen (EDA + preprocessing)
    ├── automate_Andi-Arif-Abdillah.py       # otomatisasi preprocessing (Skilled)
    └── telco_churn_preprocessing.csv        # dataset hasil preprocessing
```

## Menjalankan preprocessing otomatis
```bash
python preprocessing/automate_Andi-Arif-Abdillah.py \
    --input telco_churn_raw/telco_churn_raw.csv \
    --output preprocessing/telco_churn_preprocessing.csv
```

## Tahapan preprocessing
1. Hapus kolom identifier `customerID`.
2. Konversi `TotalCharges` ke numerik & imputasi nilai kosong (pelanggan baru, `tenure`=0).
3. Hapus data duplikat.
4. Encoding target `Churn` (No→0, Yes→1).
5. Label Encoding untuk fitur kategorikal biner.
6. One-Hot Encoding untuk fitur kategorikal multi-kelas.
7. Standardisasi fitur numerik (`tenure`, `MonthlyCharges`, `TotalCharges`).

Hasil akhir: **7.021 baris × 31 kolom** siap dilatih.
