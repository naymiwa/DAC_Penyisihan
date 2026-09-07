import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


def train_model(X_train, X_test, y_train, seed=42, n_folds=5):
    print("Memulai pelatihan model (LightGBM + LogisticRegression ensemble)...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # LightGBM OOF & test
    oof_lgb = np.zeros(X_train.shape[0])
    test_lgb = np.zeros(X_test.shape[0])

    # LogisticRegression OOF & test
    oof_lr = np.zeros(X_train.shape[0])
    test_lr = np.zeros(X_test.shape[0])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n===== FOLD {fold + 1} =====")
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        # --- LightGBM ---
        model_lgb = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1
        )

        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )

        val_probs_lgb = model_lgb.predict_proba(X_val)[:, 1]
        oof_lgb[val_idx] = val_probs_lgb
        test_lgb += model_lgb.predict_proba(X_test)[:, 1] / n_folds

        fold_f1_lgb = f1_score(y_val, (val_probs_lgb >= 0.5).astype(int), average="macro")
        print(f"  LightGBM  | Fold {fold + 1} | Macro F1 (@0.5): {fold_f1_lgb:.4f}")

        # --- LogisticRegression ---
        model_lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
            n_jobs=-1
        )

        model_lr.fit(X_tr, y_tr)
        val_probs_lr = model_lr.predict_proba(X_val)[:, 1]
        oof_lr[val_idx] = val_probs_lr
        test_lr += model_lr.predict_proba(X_test)[:, 1] / n_folds

        fold_f1_lr = f1_score(y_val, (val_probs_lr >= 0.5).astype(int), average="macro")
        print(f"  LogReg    | Fold {fold + 1} | Macro F1 (@0.5): {fold_f1_lr:.4f}")

        # --- Feature importance (fold terakhir) ---
        if fold == n_folds - 1:
            importances = model_lgb.feature_importances_
            top_idx = np.argsort(importances)[-20:][::-1]
            print(f"\n=== TOP 20 FEATURE IMPORTANCE (LightGBM) ===")
            for idx in top_idx:
                print(f"  Feature {idx}: {importances[idx]}")

    # --- Ensemble: cari bobot terbaik ---
    print("\n=== MENCARI BOBOT ENSEMBLE TERBAIK ===")
    best_w, best_f1 = 0.5, 0.0
    for w in np.arange(0.0, 1.05, 0.05):
        combined = w * oof_lgb + (1 - w) * oof_lr
        f1 = f1_score(y_train, (combined >= 0.5).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_w = f1, w

    print(f"Bobot terbaik: LightGBM={best_w:.2f}, LogReg={1-best_w:.2f}")
    print(f"Macro F1 ensemble (@0.5): {best_f1:.4f}")

    # Final ensemble probabilities
    oof_preds_probs = best_w * oof_lgb + (1 - best_w) * oof_lr
    test_preds_probs = best_w * test_lgb + (1 - best_w) * test_lr

    print("\nPelatihan model selesai!")
    return oof_preds_probs, test_preds_probs
