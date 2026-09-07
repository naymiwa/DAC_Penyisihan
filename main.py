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
from preprocessing import preprocess_data
from evaluation import evaluate_oof


# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================

SEED = 42
MAX_FEATURES = 20000
NGRAM_RANGE = (1, 3)
VECTOR_SIZE = 100
WINDOW_SIZE = 5
FAST_MODE = True
N_FOLDS = 5
MAX_CONTENT_WORDS = 512


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

assert os.path.exists(train_path), f"File tidak ditemukan: {train_path}"
assert os.path.exists(test_path), f"File tidak ditemukan: {test_path}"
assert os.path.exists(sample_path), f"File tidak ditemukan: {sample_path}"

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

train, test = clean_and_analyze(train, test)


# ==========================================
# 5. NLP PREPROCESSING
# ==========================================

train, test, stem_dict = preprocess_data(
    train, test, max_content_words=MAX_CONTENT_WORDS
)


# ==========================================
# 6. FEATURE EXTRACTION
# ==========================================

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


# ==========================================
# 7. MODELLING
# ==========================================

oof_preds_probs, test_preds_probs = train_model(
    X_train,
    X_test,
    y_train,
    seed=SEED,
    n_folds=N_FOLDS
)


# ==========================================
# 8. POST-PROCESSING RULE-BASED
# ==========================================

print("\n=== MENERAPKAN RULE-BASED POST-PROCESSING ===")

penalty_mask_train = (
    (train["title_has_number"] == 1) &
    (train["number_overlap"] >= 0) &
    (train["number_overlap"] < 0.4) &
    (train["tfidf_cosine_sim"] < 0.3)
) | (
    (train["negation_mismatch"] == 1) &
    (train["tfidf_cosine_sim"] < 0.3)
)

oof_preds_probs[penalty_mask_train] -= 0.2
oof_preds_probs = np.clip(oof_preds_probs, 0.0, 1.0)

penalty_mask_test = (
    (test["title_has_number"] == 1) &
    (test["number_overlap"] >= 0) &
    (test["number_overlap"] < 0.4) &
    (test["tfidf_cosine_sim"] < 0.3)
) | (
    (test["negation_mismatch"] == 1) &
    (test["tfidf_cosine_sim"] < 0.3)
)

test_preds_probs[penalty_mask_test] -= 0.2
test_preds_probs = np.clip(test_preds_probs, 0.0, 1.0)


# ==========================================
# 9. EVALUATION & THRESHOLD TUNING
# ==========================================

best_th, best_f1, auc = evaluate_oof(y_train, oof_preds_probs)


# ==========================================
# 10. ERROR ANALYSIS
# ==========================================

print("\n=== ERROR ANALYSIS ===")
train["oof_prob"] = oof_preds_probs
train["pred"] = (oof_preds_probs >= best_th).astype(int)

false_negatives = train[(train["label"] == 0) & (train["pred"] == 1)]
false_positives = train[(train["label"] == 1) & (train["pred"] == 0)]

print(f"False Negative (Tidak Sesuai diprediksi Sesuai): {len(false_negatives)}")
print(f"False Positive (Sesuai diprediksi Tidak Sesuai): {len(false_positives)}")

if len(false_negatives) > 0:
    print("\nContoh False Negatives:")
    sampel_fn = false_negatives[[
        "title", "jaccard_sim", "tfidf_cosine_sim", "w2v_cosine_sim", "oof_prob"
    ]].sample(min(5, len(false_negatives)), random_state=SEED)
    print(sampel_fn.to_string())


# ==========================================
# 11. GENERATE SUBMISSION
# ==========================================

test_labels = (test_preds_probs >= best_th).astype(int)
sample_sub["label"] = test_labels
sample_sub.to_csv("submission.csv", index=False)
print(f"\nSubmission tersimpan di submission.csv")
print(f"Threshold: {best_th:.2f} | Macro F1: {best_f1:.4f}")
