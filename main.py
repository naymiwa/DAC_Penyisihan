# ==========================================
# main.py
# Main Pipeline
# ==========================================

from config import *

import os
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp

from modelling import train_model
from eda import clean_and_analyze
from feature_extraction import extract_features
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import lightgbm as lgb


# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================

SEED = 42

MAX_FEATURES = 10_000
NGRAM_RANGE = (1, 2)

VECTOR_SIZE = 100
WINDOW_SIZE = 5

FAST_MODE = True
N_FOLDS = 5


# ==========================================
# 2. SET RANDOM SEED
# ==========================================

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)

print("=" * 50)
print("Python environment berhasil disiapkan.")
print(f"Random seed: {SEED}")
print("=" * 50)


# ==========================================
# 3. LOAD DATASET
# ==========================================

DATA_DIR = "data"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")


# Cek file
assert os.path.exists(train_path), \
    f"File tidak ditemukan: {train_path}"

assert os.path.exists(test_path), \
    f"File tidak ditemukan: {test_path}"

assert os.path.exists(sample_path), \
    f"File tidak ditemukan: {sample_path}"

# Cek isi CSV di sekitar baris yang bermasalah
with open(train_path, "r", encoding="utf-8-sig") as f:
    for i, line in enumerate(f, start=1):
        if 500 <= i <= 510:
            print(f"BARIS {i}:")
            print(repr(line))

# Load dataset
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)


print("\n=== DATASET ===")
print(f"Train              : {train.shape}")
print(f"Test               : {test.shape}")
print(f"Sample Submission  : {sample_sub.shape}")


# ==========================================
# 4. EDA & DATA CLEANING
# ==========================================

train, test = clean_and_analyze(
    train,
    test
)


# ==========================================
# 5. NLP PREPROCESSING
# ==========================================

# Nanti:
from preprocessing import preprocess_data

train, test, stem_dict = preprocess_data(
    train,
    test
)


# ==========================================
# 6. FEATURE EXTRACTION
# ==========================================

# Nanti:
# from feature_extraction import create_features
train, test, X_train, X_test, tfidf, w2v_model = extract_features(
    train,
    test,
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
    vector_size=VECTOR_SIZE,
    window_size=WINDOW_SIZE,
    seed=SEED
)

y_train = train["label"].values

oof_preds_probs, test_preds_probs = train_model(
    X_train,
    X_test,
    y_train,
    seed=SEED,
    n_folds=N_FOLDS
)

# ==========================================
# 6. FEATURE EXTRACTION & MODELLING
# ==========================================
train, test, X_train, X_test, tfidf, w2v_model = extract_features(...)

y_train = train["label"].values

oof_preds_probs, test_preds_probs = train_model(
    X_train, X_test, y_train, seed=SEED, n_folds=N_FOLDS
)

# --- TAMBAHKAN BLOK POST-PROCESSING INI ---
print("\n=== MENERAPKAN RULE-BASED POST-PROCESSING ===")
# Jika judul punya angka (title_has_number == 1) TAPI tidak ada angka yang cocok di isi (number_overlap < 0.5)
# ATAU ada perbedaan kata bantahan/negasi (negation_mismatch == 1)
# Maka: Paksa turunkan probabilitasnya agar model menebak "Tidak Sesuai" (0)

# Untuk Data Train (OOF)
penalty_mask_train = ((train["title_has_number"] == 1) & (train["number_overlap"] < 0.4)) | (train["negation_mismatch"] == 1)
oof_preds_probs[penalty_mask_train] -= 0.35  # Kurangi keyakinan model sebesar 35%

# Pastikan tidak ada nilai di bawah 0.0
oof_preds_probs = np.clip(oof_preds_probs, 0.0, 1.0)

# Untuk Data Test (Submission)
penalty_mask_test = ((test["title_has_number"] == 1) & (test["number_overlap"] < 0.4)) | (test["negation_mismatch"] == 1)
test_preds_probs[penalty_mask_test] -= 0.35

test_preds_probs = np.clip(test_preds_probs, 0.0, 1.0)
# ------------------------------------------

# ==========================================
# 7. EVALUATION & THRESHOLD TUNING
# ==========================================
from evaluation import evaluate_oof
best_th, best_f1, auc = evaluate_oof(y_train, oof_preds_probs)

# --- TAMBAHKAN KODE INI UNTUK ERROR ANALYSIS ---
print("\n=== 7B. ERROR ANALYSIS (FALSE NEGATIVES) ===")
# Masukkan probabilitas dan tebakan ke dataframe
train["oof_prob"] = oof_preds_probs
train["pred"] = (oof_preds_probs >= best_th).astype(int)

# Saring kasus di mana aslinya "Tidak Sesuai" (0) tapi ditebak "Sesuai" (1)
false_negatives = train[(train["label"] == 0) & (train["pred"] == 1)]
print(f"Total False Negative: {len(false_negatives)} kasus.")

if len(false_negatives) > 0:
    print("\nContoh 5 Berita yang Mengecoh Model (Topik Mirip, Tapi Tidak Sesuai):")
    # Ambil 5 sampel acak, tampilkan judul dan skor kemiripannya
    sampel_fn = false_negatives[[
        "title", "jaccard_sim", "tfidf_cosine_sim", "w2v_cosine_sim", "oof_prob"
    ]].sample(min(5, len(false_negatives)), random_state=SEED)
    print(sampel_fn)
# -----------------------------------------------


# ==========================================
# 8. GENERATE SUBMISSION
# ==========================================

test_labels = (test_preds_probs >= best_th).astype(int)
sample_sub["label"] = test_labels
sample_sub.to_csv("submission.csv", index=False)
print("\nSubmission tersimpan di submission.csv")
