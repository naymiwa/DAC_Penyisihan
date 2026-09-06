import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def train_model(X_train, X_test, y_train, seed=42, n_folds=5):
    print("Memulai pelatihan model LightGBM...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds_probs = np.zeros(X_train.shape[0])
    test_preds_probs = np.zeros(X_test.shape[0])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n===== FOLD {fold + 1} =====")
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        # LightGBM dengan Class Weight Manual Agresif (Saran 4)
        model = lgb.LGBMClassifier(
            class_weight={0: 10, 1: 1}, 
            random_state=seed,
            n_estimators=300,
            learning_rate=0.05,
            n_jobs=-1
        )

        model.fit(X_tr, y_tr)
        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds_probs[val_idx] = val_probs

        fold_f1 = f1_score(y_val, (val_probs >= 0.5).astype(int), average="macro")
        print(f"Fold {fold + 1} | Macro F1 (@0.5): {fold_f1:.4f}")

        test_preds_probs += model.predict_proba(X_test)[:, 1] / n_folds

        # --- FITUR IMPORTANCE CHECK DI FOLD TERAKHIR (Saran 2) ---
        if fold == n_folds - 1:
            # Sesuaikan urutan ini dengan urutan di extra_cols feature_extraction
            extra_cols_names = [
                "jaccard_sim", "tfidf_cosine_sim", "length_ratio", 
                "w2v_cosine_sim", "number_overlap", "title_has_number"
            ]
            importances = model.feature_importances_
            print("\n=== FEATURE IMPORTANCE (Fitur Tambahan) ===")
            # Mengambil 6 nilai importance terakhir dari array
            for name, imp in zip(extra_cols_names, importances[-len(extra_cols_names):]):
                print(f"{name}: {imp}")

    print("\nPelatihan model selesai!")
    return oof_preds_probs, test_preds_probs