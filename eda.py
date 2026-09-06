# ==========================================
# eda.py
# Exploratory Data Analysis & Data Cleaning
# ==========================================

import pandas as pd


def clean_and_analyze(train, test):
    """
    Melakukan data cleaning dan EDA sederhana
    pada dataset train dan test.
    """

    # ==========================================
    # 1. CEK & HAPUS DUPLIKAT
    # ==========================================

    print("\n=== 1. CEK DUPLIKAT ===")

    duplikat_awal = train.duplicated(
        subset=["title", "content"]
    ).sum()

    print(f"Jumlah duplikat di Train : {duplikat_awal}")

    # Hapus duplikat dari train
    train = (
        train
        .drop_duplicates(
            subset=["title", "content"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    duplikat_akhir = train.duplicated(
        subset=["title", "content"]
    ).sum()

    print(f"Sisa duplikat di Train   : {duplikat_akhir}")
    print("-" * 40)


    # ==========================================
    # 2. CEK & TANGANI MISSING VALUES
    # ==========================================

    print("=== 2. CEK MISSING VALUES ===")

    print(
        "Jumlah missing value di Train sebelum dibersihkan:"
    )
    print(
        train[["title", "content", "label"]].isna().sum()
    )

    # Hapus baris train yang memiliki
    # title, content, atau label kosong
    train = (
        train
        .dropna(
            subset=["title", "content", "label"]
        )
        .reset_index(drop=True)
    )

    # Jangan menghapus baris dari test.
    # Missing value pada teks diganti dengan string kosong.
    test["title"] = test["title"].fillna("").astype(str)
    test["content"] = test["content"].fillna("").astype(str)

    print(
        "\nJumlah missing value di Train setelah dibersihkan:"
    )
    print(
        train[["title", "content", "label"]].isna().sum()
    )

    print("-" * 40)


    # ==========================================
    # 3. DISTRIBUSI KELAS
    # ==========================================

    print("=== 3. DISTRIBUSI KELAS (TRAIN) ===")

    label_counts = train["label"].value_counts()

    label_pct = (
        train["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    dist_df = pd.DataFrame({
        "Jumlah Baris": label_counts,
        "Persentase (%)": label_pct
    })

    print(dist_df)
    print("-" * 40)


    # ==========================================
    # 4. ANALISIS PANJANG TEKS
    # ==========================================

    print("=== 4. STATISTIK PANJANG KATA (TRAIN) ===")

    # Jumlah kata pada judul
    train["title_word_count"] = (
        train["title"]
        .astype(str)
        .apply(lambda x: len(x.split()))
    )

    # Jumlah kata pada isi berita
    train["content_word_count"] = (
        train["content"]
        .astype(str)
        .apply(lambda x: len(x.split()))
    )

    print("\nStatistik Judul (Title):")
    print(
        train["title_word_count"]
        .describe()
        .to_frame()
        .T
    )

    print("\nStatistik Isi Berita (Content):")
    print(
        train["content_word_count"]
        .describe()
        .to_frame()
        .T
    )

    print("-" * 40)


    # ==========================================
    # RETURN DATA
    # ==========================================

    return train, test