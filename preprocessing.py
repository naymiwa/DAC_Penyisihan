# ==========================================
# preprocessing.py
# NLP Preprocessing
# ==========================================

import os
import re

import numpy as np
from collections import Counter
from joblib import Parallel, delayed

from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)

from Sastrawi.Stemmer.StemmerFactory import (
    StemmerFactory
)


# ==========================================
# 1. INISIALISASI SASTRAWI
# ==========================================

stopwords_set = set(
    StopWordRemoverFactory().get_stop_words()
)


# ==========================================
# 2. FAST CLEAN
# ==========================================

def fast_clean(text, max_words=None):

    if not isinstance(text, str):
        text = str(text)

    if max_words:
        text = " ".join(
            text.split()[:max_words]
        )

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text,
        flags=re.MULTILINE
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# 3. STEMMING PER CHUNK
# ==========================================

def stem_chunk(words_chunk):

    local_stemmer = (
        StemmerFactory()
        .create_stemmer()
    )

    return {
        word: local_stemmer.stem(word)
        for word in words_chunk
    }


# ==========================================
# 4. APPLY STEM DICTIONARY
# ==========================================

def apply_stem_dict(text, stem_dict):

    return " ".join(
        stem_dict[word]
        for word in text.split()
        if word in stem_dict
    )


# ==========================================
# 5. MAIN PREPROCESSING FUNCTION
# ==========================================

def preprocess_data(train, test):

    print("\n=== NLP PREPROCESSING ===")


    # ------------------------------------------
    # FAST CLEAN
    # ------------------------------------------

    print("1. Melakukan pembersihan dasar...")

    train["temp_title"] = train["title"].apply(
        lambda x: fast_clean(x)
    )

    train["temp_content"] = train["content"].apply(
        lambda x: fast_clean(x, max_words=256)
    )

    test["temp_title"] = test["title"].apply(
        lambda x: fast_clean(x)
    )

    test["temp_content"] = test["content"].apply(
        lambda x: fast_clean(x, max_words=256)
    )


    # ------------------------------------------
    # WORD FREQUENCY
    # ------------------------------------------

    print("2. Menghitung frekuensi kosakata...")

    all_texts = (
        train["temp_title"].tolist()
        + train["temp_content"].tolist()
        + test["temp_title"].tolist()
        + test["temp_content"].tolist()
    )

    word_counter = Counter(
        " ".join(all_texts).split()
    )

    print(
        f"Ditemukan {len(word_counter)} kata unik."
    )


    # ------------------------------------------
    # VOCABULARY PRUNING
    # ------------------------------------------

    MIN_FREQ = 2

    words_to_stem = []
    stem_dict = {}

    for word, freq in word_counter.items():

        if word in stopwords_set:
            continue

        if freq >= MIN_FREQ:
            words_to_stem.append(word)

        else:
            # Kata langka tidak di-stem
            stem_dict[word] = word


    print(
        f"3. {len(words_to_stem)} kata akan di-stem."
    )

    print(
        f"   {len(stem_dict)} kata langka dilewati."
    )


    # ------------------------------------------
    # PARALLEL STEMMING
    # ------------------------------------------

    n_jobs = os.cpu_count() or 1

    print(
        f"4. Menjalankan Sastrawi "
        f"dengan {n_jobs} proses..."
    )

    chunks = np.array_split(
        words_to_stem,
        n_jobs * 4
    )

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=5
    )(
        delayed(stem_chunk)(chunk.tolist())
        for chunk in chunks
        if len(chunk) > 0
    )


    for result in results:
        stem_dict.update(result)


    # ------------------------------------------
    # APPLY STEMMING
    # ------------------------------------------

    print("5. Menerapkan hasil stemming...")

    train["clean_title"] = train["temp_title"].apply(
        lambda x: apply_stem_dict(x, stem_dict)
    )

    train["clean_content"] = train["temp_content"].apply(
        lambda x: apply_stem_dict(x, stem_dict)
    )

    test["clean_title"] = test["temp_title"].apply(
        lambda x: apply_stem_dict(x, stem_dict)
    )

    test["clean_content"] = test["temp_content"].apply(
        lambda x: apply_stem_dict(x, stem_dict)
    )


    # ------------------------------------------
    # HAPUS KOLOM SEMENTARA
    # ------------------------------------------

    train.drop(
        columns=["temp_title", "temp_content"],
        inplace=True
    )

    test.drop(
        columns=["temp_title", "temp_content"],
        inplace=True
    )


    print("Preprocessing selesai!")

    return train, test, stem_dict