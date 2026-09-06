# ==========================================
# evaluation.py
# Threshold Tuning & Evaluation Report
# ==========================================

import numpy as np
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, roc_auc_score
)


def find_best_threshold(y_oof, p_oof, th_range=np.arange(0.05, 0.96, 0.01)):
    best_th, best_f1 = 0.5, 0.0
    for th in th_range:
        f1 = f1_score(y_oof, (p_oof >= th).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th, best_f1


def evaluate_oof(y_oof, p_oof, target_names=("Tidak Sesuai (0)", "Sesuai (1)")):
    print("\n=== EVALUASI OOF & THRESHOLD TUNING (MACRO F1) ===")

    # Cek dulu apakah probabilitas ini punya ranking power sama sekali,
    # independen dari threshold berapa pun.
    auc = roc_auc_score(y_oof, p_oof)
    print(f"ROC-AUC (OOF)            : {auc:.4f}")
    if auc < 0.55:
        print("PERINGATAN: AUC mendekati 0.5 -> model kemungkinan belum")
        print("menangkap sinyal apa pun dari fitur saat ini. Ganti threshold")
        print("TIDAK akan banyak membantu -- fokus perbaiki fitur/model dulu.")

    f1_default = f1_score(y_oof, (p_oof >= 0.5).astype(int), average="macro")
    best_th, best_f1 = find_best_threshold(y_oof, p_oof)

    print(f"Macro F1 @0.50           : {f1_default:.4f}")
    print(f"Threshold optimal        : {best_th:.2f}")
    print(f"Macro F1 @threshold opt  : {best_f1:.4f}")

    print("\nClassification report (OOF, threshold optimal):")
    print(classification_report(
        y_oof, (p_oof >= best_th).astype(int),
        target_names=list(target_names)
    ))

    print("Confusion matrix (baris=aktual, kolom=prediksi):")
    print(confusion_matrix(y_oof, (p_oof >= best_th).astype(int)))

    return best_th, best_f1, auc
