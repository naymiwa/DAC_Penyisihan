"""
generate_figures.py
Generate semua gambar/visualisasi untuk makalah IEEE DAC IFEST 2026.
Jalankan setelah pipeline selesai (train, test, features, model sudah ada).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# =============================================================
# GAMBAR 1: Distribusi Kelas pada Dataset Latih
# =============================================================
def plot_class_distribution(train_df, save_path="figures/fig1_class_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Bar chart
    counts = train_df["label"].value_counts().sort_index()
    colors = ["#e74c3c", "#2ecc71"]
    bars = axes[0].bar(["Tidak Sesuai (0)", "Sesuai (1)"], counts.values, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Jumlah Sampel")
    axes[0].set_title("(a) Jumlah Sampel per Kelas")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, str(val),
                     ha="center", va="bottom", fontsize=9)

    # Pie chart
    axes[1].pie(counts.values, labels=["Tidak Sesuai", "Sesuai"], autopct="%1.1f%%",
                colors=colors, startangle=90, explode=(0.03, 0.03))
    axes[1].set_title("(b) Proporsi Kelas")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================
# GAMBAR 2: Distribusi Panjang Kata
# =============================================================
def plot_length_distribution(train_df, save_path="figures/fig2_length_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    title_lens = train_df["title"].astype(str).apply(lambda x: len(x.split()))
    content_lens = train_df["content"].astype(str).apply(lambda x: len(x.split()))

    axes[0].hist(title_lens, bins=30, color="#3498db", edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Jumlah Kata")
    axes[0].set_ylabel("Frekuensi")
    axes[0].set_title("(a) Distribusi Panjang Judul")
    axes[0].axvline(title_lens.median(), color="red", linestyle="--", label=f"Median: {title_lens.median():.0f}")
    axes[0].legend()

    axes[1].hist(content_lens, bins=50, color="#e67e22", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Jumlah Kata")
    axes[1].set_ylabel("Frekuensi")
    axes[1].set_title("(b) Distribusi Panjang Isi Berita")
    axes[1].axvline(content_lens.median(), color="red", linestyle="--", label=f"Median: {content_lens.median():.0f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================
# GAMBAR 3: Diagram Alur Pipeline (Flowchart)
# =============================================================
def plot_pipeline_diagram(save_path="figures/fig3_pipeline_diagram.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (1, 4.5, "Dataset\n(Train + Test)"),
        (3, 4.5, "EDA &\nCleaning"),
        (5, 4.5, "Text\nPreprocessing"),
        (7, 4.5, "Feature\nExtraction"),
        (9, 4.5, "Model\nTraining"),
        (9, 2.5, "Post-\nProcessing"),
        (7, 2.5, "Threshold\nTuning"),
        (5, 2.5, "Evaluation\n(Macro F1)"),
        (3, 2.5, "Generate\nSubmission"),
    ]

    box_colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c",
                  "#1abc9c", "#f39c12", "#34495e", "#27ae60"]

    for (x, y, text), color in zip(boxes, box_colors):
        rect = mpatches.FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=7, fontweight="bold", color="white")

    # Arrows
    arrow_style = dict(arrowstyle="->", color="black", lw=1.2)
    for i in range(len(boxes)-1):
        x1, y1, _ = boxes[i]
        x2, y2, _ = boxes[i+1]
        if y1 == y2:
            ax.annotate("", xy=(x2-0.7, y2), xytext=(x1+0.7, y1), arrowprops=arrow_style)
        else:
            ax.annotate("", xy=(x2, y2+0.4), xytext=(x1, y1-0.4), arrowprops=arrow_style)

    # Feature extraction detail
    ax.text(7, 5.3, "TF-IDF (word+char) | Word2Vec | BM25\nJaccard | Dice | Cosine | Negation | Number",
            ha="center", va="bottom", fontsize=6, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="gray"))

    ax.set_title("Diagram Alur Pipeline Penelitian", fontsize=11, fontweight="bold", pad=10)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================
# GAMBAR 4: Arsitektur Model BiGRU Wide and Deep
# =============================================================
def plot_model_architecture(save_path="figures/fig4_model_architecture.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title Input
    rect = mpatches.FancyBboxPatch((0.5, 6.5), 2, 0.6, boxstyle="round,pad=0.1",
                                    facecolor="#3498db", edgecolor="black", alpha=0.85)
    ax.add_patch(rect)
    ax.text(1.5, 6.8, "Title Input", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Content Input
    rect = mpatches.FancyBboxPatch((0.5, 5.2), 2, 0.6, boxstyle="round,pad=0.1",
                                    facecolor="#3498db", edgecolor="black", alpha=0.85)
    ax.add_patch(rect)
    ax.text(1.5, 5.5, "Content Input", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Tabular Input
    rect = mpatches.FancyBboxPatch((0.5, 3.0), 2, 0.6, boxstyle="round,pad=0.1",
                                    facecolor="#f39c12", edgecolor="black", alpha=0.85)
    ax.add_patch(rect)
    ax.text(1.5, 3.3, "Tabular Features", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Shared Embedding
    rect = mpatches.FancyBboxPatch((3.5, 6.0), 2, 1.5, boxstyle="round,pad=0.1",
                                    facecolor="#2ecc71", edgecolor="black", alpha=0.85)
    ax.add_patch(rect)
    ax.text(4.5, 6.75, "Shared\nEmbedding\n(128-dim)", ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    # BiGRU
    rect = mpatches.FancyBboxPatch((6.5, 6.0), 2, 1.5, boxstyle="round,pad=0.1",
                                    facecolor="#9b59b6", edgecolor="black", alpha=0.85)
    ax.add_patch(rect)
    ax.text(7.5, 6.75, "BiGRU\n(Shared)\n+ GlobalMaxPool", ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    # Concatenate
    rect = mpatches.FancyBboxPatch((6.5, 4.0), 2, 0.6, boxstyle="round,pad=0.1",
                                    facecolor="#e74c3c", edgecolor="black", alpha=0.85)
    ax.add_patch(rect)
    ax.text(7.5, 4.3, "Concatenate", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Dense layers
    rect = mpatches.FancyBboxPatch((6.5, 2.5), 2, 0.6, boxstyle="round,pad=0.1",
                                    facecolor="#1abc9c", edgecolor="black", alpha=0.85)
    ax.add_patch(rect)
    ax.text(7.5, 2.8, "Dense 128 + Dropout", ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    rect = mpatches.FancyBboxPatch((6.5, 1.5), 2, 0.6, boxstyle="round,pad=0.1",
                                    facecolor="#1abc9c", edgecolor="black", alpha=0.85)
    ax.add_patch(rect)
    ax.text(7.5, 1.8, "Dense 64", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Output
    rect = mpatches.FancyBboxPatch((6.5, 0.3), 2, 0.6, boxstyle="round,pad=0.1",
                                    facecolor="#27ae60", edgecolor="black", alpha=0.85)
    ax.add_patch(rect)
    ax.text(7.5, 0.6, "Sigmoid Output", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    # Arrows
    arrow = dict(arrowstyle="->", color="black", lw=1.2)
    ax.annotate("", xy=(3.5, 6.8), xytext=(2.5, 6.8), arrowprops=arrow)
    ax.annotate("", xy=(3.5, 6.2), xytext=(2.5, 5.5), arrowprops=arrow)
    ax.annotate("", xy=(6.5, 7.0), xytext=(5.5, 7.0), arrowprops=arrow)
    ax.annotate("", xy=(7.5, 4.6), xytext=(7.5, 6.0), arrowprops=arrow)
    ax.annotate("", xy=(7.5, 3.6), xytext=(2.5, 3.3), arrowprops=arrow)
    ax.annotate("", xy=(7.5, 3.1), xytext=(7.5, 4.0), arrowprops=arrow)
    ax.annotate("", xy=(7.5, 2.5), xytext=(7.5, 2.8), arrowprops=arrow)
    ax.annotate("", xy=(7.5, 1.5), xytext=(7.5, 2.1), arrowprops=arrow)
    ax.annotate("", xy=(7.5, 0.9), xytext=(7.5, 1.5), arrowprops=arrow)

    # Labels
    ax.text(4.5, 5.5, "Wide Features", ha="center", va="center", fontsize=7, style="italic", color="#e67e22")
    ax.text(8.5, 3.3, "Deep Features", ha="center", va="center", fontsize=7, style="italic", color="#9b59b6")

    ax.set_title("Arsitektur Model BiGRU Wide and Deep", fontsize=11, fontweight="bold", pad=10)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================
# GAMBAR 5: Confusion Matrix
# =============================================================
def plot_confusion_matrix(y_true, y_pred, save_path="figures/fig5_confusion_matrix.png"):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Absolute
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Tidak Sesuai", "Sesuai"],
                yticklabels=["Tidak Sesuai", "Sesuai"])
    axes[0].set_xlabel("Prediksi")
    axes[0].set_ylabel("Aktual")
    axes[0].set_title("(a) Confusion Matrix (Absolut)")

    # Normalized
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=axes[1],
                xticklabels=["Tidak Sesuai", "Sesuai"],
                yticklabels=["Tidak Sesuai", "Sesuai"])
    axes[1].set_xlabel("Prediksi")
    axes[1].set_ylabel("Aktual")
    axes[1].set_title("(b) Confusion Matrix (Ternormalisasi)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================
# GAMBAR 6: Feature Importance (LightGBM)
# =============================================================
def plot_feature_importance(model, feature_names, top_n=15,
                            save_path="figures/fig6_feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(top_n), importances[indices], color="#3498db", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(f"Top {top_n} Feature Importance (LightGBM)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================
# GAMBAR 7: Perbandingan Performa Model (Bar Chart)
# =============================================================
def plot_model_comparison(results_dict, save_path="figures/fig7_model_comparison.png"):
    """
    results_dict: {"LightGBM": 0.XX, "LogReg": 0.XX, "Ensemble": 0.XX, "BiGRU": 0.XX}
    """
    models = list(results_dict.keys())
    scores = list(results_dict.values())
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(models, scores, color=colors[:len(models)], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Macro F1-Score")
    ax.set_title("Perbandingan Performa Model")
    ax.set_ylim(0, 1.0)

    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================
# GAMBAR 8: Distribusi Fitur Kemiripan per Label
# =============================================================
def plot_feature_distributions(train_df, save_path="figures/fig8_feature_distributions.png"):
    features = ["jaccard_sim", "tfidf_cosine_sim", "w2v_cosine_sim"]
    titles = ["(a) Jaccard Similarity", "(b) TF-IDF Cosine Similarity", "(c) Word2Vec Cosine Similarity"]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    for ax, feat, title in zip(axes, features, titles):
        if feat not in train_df.columns:
            ax.text(0.5, 0.5, f"{feat}\nnot found", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        for label, color, name in [(0, "#e74c3c", "Tidak Sesuai"), (1, "#2ecc71", "Sesuai")]:
            subset = train_df[train_df["label"] == label][feat]
            ax.hist(subset, bins=30, alpha=0.6, color=color, label=name, edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Skor")
        ax.set_ylabel("Frekuensi")
        ax.set_title(title)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================
# MAIN: Jalankan semua
# =============================================================
if __name__ == "__main__":
    print("=== GENERATING FIGURES ===")
    print()

    # Load data
    train_df = pd.read_csv("data/train.csv")

    # Gambar 1: Distribusi kelas
    plot_class_distribution(train_df)

    # Gambar 2: Distribusi panjang kata
    plot_length_distribution(train_df)

    # Gambar 3: Pipeline diagram
    plot_pipeline_diagram()

    # Gambar 4: Arsitektur model
    plot_model_architecture()

    # Gambar 7: Perbandingan model (isi dengan skor aktual nanti)
    plot_model_comparison({
        "LightGBM": 0.85,
        "LogReg": 0.82,
        "Ensemble": 0.87,
        "BiGRU": 0.88,
    })

    # Gambar 8: Distribusi fitur (perlu kolom fitur sudah ada)
    # Uncomment setelah feature extraction selesai:
    # plot_feature_distributions(train_df)

    # Gambar 5: Confusion matrix (perlu y_true, y_pred)
    # Uncomment setelah model selesai:
    # plot_confusion_matrix(y_val, y_pred)

    # Gambar 6: Feature importance (perlu model)
    # Uncomment setelah model selesai:
    # plot_feature_importance(model_lgb, feature_names)

    print()
    print("=== DONE! Cek folder figures/ ===")
    print("Note: Gambar 5, 6, 8 perlu uncomment setelah model selesai dijalankan.")
