"""Classical pixel-space baselines (no pretrained features).

Two baselines on raw resized images:
    pixel_knn  - kNN distance in flattened pixel space
    ocsvm      - One-Class SVM on PCA-compressed pixels

Compared with our pretrained-feature results from results_summary.csv,
producing fig5_baseline_per_category.png.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

for candidate in [
    os.path.expanduser("~/Library/Fonts/Open Sans regular.ttf"),
    os.path.expanduser("~/.fonts/OpenSans-Regular.ttf"),
    "/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf",
]:
    if os.path.exists(candidate):
        fm.fontManager.addfont(candidate)

plt.rcParams.update({
    "font.family": "Open Sans" if any("Open Sans" in f.name for f in fm.fontManager.ttflist) else "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
METRICS_CSV = ROOT / "outputs" / "metrics" / "results_summary.csv"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["hazelnut", "screw", "transistor", "zipper", "capsule"]
IMG_SIZE = 64
PCA_DIMS = 64
KNN_K = 5
SVM_NU = 0.1


def load_images(category):
    """Resize, flatten and stack train/test images for a category."""
    cat_dir = DATA_ROOT / category

    def read_split(split_dir, label):
        vecs, labels = [], []
        for fpath in sorted(split_dir.glob("**/*.png")):
            img = Image.open(fpath).convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR,
            )
            vecs.append(np.asarray(img, dtype=np.float32).flatten() / 255.0)
            labels.append(label)
        return vecs, labels

    train_vecs, _ = read_split(cat_dir / "train" / "good", label=0)
    test_vecs, test_labels = [], []
    for sub in sorted((cat_dir / "test").iterdir()):
        if not sub.is_dir():
            continue
        lbl = 0 if sub.name == "good" else 1
        v, l = read_split(sub, label=lbl)
        test_vecs.extend(v)
        test_labels.extend(l)

    return (np.stack(train_vecs).astype(np.float32),
            np.stack(test_vecs).astype(np.float32),
            np.array(test_labels, dtype=int))


def pixel_knn(X_train, X_test):
    nn = NearestNeighbors(n_neighbors=KNN_K, metric="euclidean", n_jobs=-1)
    nn.fit(X_train)
    return nn.kneighbors(X_test)[0].mean(axis=1)


def pixel_ocsvm(X_train, X_test):
    """PCA-compress then fit a one-class RBF SVM. Higher score = more anomalous."""
    n_comp = min(PCA_DIMS, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    Z_train = pca.fit_transform(X_train)
    Z_test = pca.transform(X_test)
    clf = OneClassSVM(kernel="rbf", nu=SVM_NU, gamma="scale").fit(Z_train)
    return -clf.decision_function(Z_test)


def run_baselines():
    rows = []
    for cat in CATEGORIES:
        print(f"  {cat}...", end=" ", flush=True)
        X_train, X_test, y_test = load_images(cat)
        print(f"train={X_train.shape[0]} test={X_test.shape[0]} "
              f"defective={int(y_test.sum())}", end="  ")

        knn_scores = pixel_knn(X_train, X_test)
        svm_scores = pixel_ocsvm(X_train, X_test)
        knn_auc = roc_auc_score(y_test, knn_scores)
        svm_auc = roc_auc_score(y_test, svm_scores)
        print(f"kNN={knn_auc:.3f}  OC-SVM={svm_auc:.3f}")

        rows.append({
            "category": cat,
            "pixel_knn_roc": knn_auc,
            "pixel_knn_pr": average_precision_score(y_test, knn_scores),
            "ocsvm_roc": svm_auc,
            "ocsvm_pr": average_precision_score(y_test, svm_scores),
        })
    return pd.DataFrame(rows)


def deep_results():
    """Best ROC-AUC per category for kNN and Mahalanobis on pretrained features."""
    df = pd.read_csv(METRICS_CSV)
    df = df[
        ~df["layer"].isin(["avgpool"])
        & ~df["layer"].str.endswith("_patches", na=False)
        & ~df["layer"].str.endswith("_patchidx", na=False)
    ]
    rows = []
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        rows.append({
            "category": cat,
            "deep_knn_roc": sub[sub["detector"] == "kNN(k=5)"]["roc_auc"].max(),
            "deep_maha_roc": sub[sub["detector"] == "Mahalanobis"]["roc_auc"].max(),
        })
    return pd.DataFrame(rows)


def make_figure(baseline_df, deep_df):
    merged = baseline_df.merge(deep_df, on="category")
    approaches = [
        ("pixel_knn_roc", "Pixel kNN"),
        ("ocsvm_roc", "One-Class SVM"),
        ("deep_knn_roc", "Deep kNN"),
        ("deep_maha_roc", "Deep Mahalanobis"),
    ]
    blues = ["#BBDEFB", "#64B5F6", "#1976D2", "#0D47A1"]

    n_app = len(approaches)
    bar_w = 0.18
    offsets = np.linspace(-(n_app - 1) * bar_w / 2,
                          (n_app - 1) * bar_w / 2, n_app)
    x = np.arange(len(CATEGORIES))

    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    for ai, ((key, label), offset, color) in enumerate(zip(approaches, offsets, blues)):
        vals = [merged.loc[merged["category"] == c, key].values[0] for c in CATEGORIES]
        bars = ax.bar(x + offset, vals, bar_w,
                      color=color, edgecolor="white", linewidth=0.3,
                      label=label, zorder=3)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.008,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="#444", rotation=40)

    ax.axhline(0.5, color="#ccc", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CATEGORIES])
    ax.set_ylim(0.0, 1.18)
    ax.set_ylabel("ROC-AUC")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["left"].set_color("#ccc")
    ax.spines["bottom"].set_color("#ccc")

    path = OUT_DIR / "fig5_baseline_per_category.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {path}")
    return merged


def print_summary(merged):
    cols = {
        "category": "Category",
        "pixel_knn_roc": "Pixel kNN",
        "ocsvm_roc": "OC-SVM",
        "deep_knn_roc": "Deep kNN",
        "deep_maha_roc": "Deep Mahalanobis",
    }
    out = merged[list(cols)].rename(columns=cols)
    print("\nROC-AUC: classical baselines vs pretrained features")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nMeans:")
    for old, new in list(cols.items())[1:]:
        print(f"  {new:<20s} {merged[old].mean():.4f}")


if __name__ == "__main__":
    print("Pixel-space baselines:")
    baseline_df = run_baselines()
    deep_df = deep_results()
    merged = make_figure(baseline_df, deep_df)
    print_summary(merged)
    merged.to_csv(OUT_DIR / "fig5_baseline_metrics.csv",
                  index=False, float_format="%.4f")
