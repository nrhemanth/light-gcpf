"""Generate the four quantitative analysis figures from results_summary.csv.

Outputs go to outputs/figures/:
    fig1_model_comparison.png    - per-category model comparison (ResNet vs DeiT)
    fig2_layer_within_model.png  - best layer inside each model
    fig3_cross_dataset.png       - per-layer ROC-AUC across all 5 datasets
    fig4_baseline_comparison.png - approach complexity vs ROC-AUC
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to load Open Sans if installed, otherwise fall back to default.
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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "outputs", "metrics", "results_summary.csv")
OUT_DIR = os.path.join(BASE, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV)
df = df[
    ~df["layer"].isin(["avgpool"])
    & ~df["layer"].str.endswith("_patches", na=False)
    & ~df["layer"].str.endswith("_patchidx", na=False)
]

CATEGORIES = ["hazelnut", "screw", "transistor", "zipper", "capsule"]
CAT_LABEL = {c: c.capitalize() for c in CATEGORIES}

RESNET_LAYERS = ["layer1", "layer2", "layer3", "layer4"]
DEIT_LAYERS = ["block2", "block5", "block8", "block11"]

LAYER_LABEL = {
    "layer1": "Layer 1\n(c=64)",
    "layer2": "Layer 2\n(c=128)",
    "layer3": "Layer 3\n(c=256)",
    "layer4": "Layer 4\n(c=512)",
    "block2": "Block 2",
    "block5": "Block 5",
    "block8": "Block 8",
    "block11": "Block 11",
}

ORANGE = "#FF8B33"
GREY = "#D3D3D3"
BLUE_BEST = "#1565C0"
GREY_REST = "#C0C0C0"
BLUES_ASC = ["#BBDEFB", "#64B5F6", "#1976D2", "#0D47A1"]


def best_roc(model, layer=None, category=None):
    s = df[df["model"] == model]
    if layer:
        s = s[s["layer"] == layer]
    if category:
        s = s[s["category"] == category]
    return s["roc_auc"].max() if len(s) else np.nan


def mean_std_roc(model, layer):
    """Mean and std of best per-category ROC-AUC across all categories."""
    vals = [
        df[(df["model"] == model) & (df["layer"] == layer) & (df["category"] == c)]["roc_auc"].max()
        for c in CATEGORIES
    ]
    vals = [v for v in vals if not np.isnan(v)]
    return np.mean(vals), np.std(vals)


def fig1():
    """Per-category model comparison: best layer + best detector per backbone."""
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    x = np.arange(len(CATEGORIES))
    bar_w = 0.32

    for mi, (model, color, label) in enumerate([
        ("resnet18", ORANGE, "ResNet-18"),
        ("deit_tiny", GREY, "DeiT-tiny"),
    ]):
        offset = (mi - 0.5) * bar_w
        vals = [best_roc(model, category=c) for c in CATEGORIES]
        bars = ax.bar(x + offset, vals, bar_w, color=color,
                      edgecolor="white", linewidth=0.4, label=label, zorder=3)
        for rect, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 0.007,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8.5, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABEL[c] for c in CATEGORIES])
    ax.set_ylim(0.5, 1.12)
    ax.set_ylabel("ROC-AUC")
    ax.legend(fontsize=9)
    ax.spines["left"].set_color("#ccc")
    ax.spines["bottom"].set_color("#ccc")

    path = os.path.join(OUT_DIR, "fig1_model_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def fig2():
    """Within each backbone, mean ROC-AUC per layer; best layer highlighted."""
    fig, (ax_r, ax_d) = plt.subplots(1, 2, figsize=(11, 4.2),
                                      constrained_layout=True, sharey=True)

    for model, ax, layers, title in [
        ("resnet18", ax_r, RESNET_LAYERS, "ResNet-18"),
        ("deit_tiny", ax_d, DEIT_LAYERS, "DeiT-tiny"),
    ]:
        means, stds = zip(*[mean_std_roc(model, l) for l in layers])
        best_i = int(np.nanargmax(means))

        for xi, m in enumerate(means):
            color = BLUE_BEST if xi == best_i else GREY_REST
            ax.bar(xi, m, color=color, edgecolor="none", width=0.52, zorder=3)
            ax.text(xi, m + 0.013, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=9.5, color="#444")

        ax.set_xticks(range(len(layers)))
        labels = ([LAYER_LABEL[l].replace("(c=", "c = ").replace(")", "") for l in layers]
                  if model == "resnet18" else [LAYER_LABEL[l] for l in layers])
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylim(0.4, 1.1)
        ax.set_title(title, pad=8)
        ax.spines["left"].set_color("#ccc")
        ax.spines["bottom"].set_color("#ccc")

    ax_r.set_ylabel("Mean ROC-AUC")

    path = os.path.join(OUT_DIR, "fig2_layer_within_model.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def fig3():
    """Layer-wise ROC-AUC across all 5 sub-datasets."""
    fig, axes = plt.subplots(
        2, len(CATEGORIES),
        figsize=(3.5 * len(CATEGORIES), 7.5),
        gridspec_kw={"hspace": 0.55, "wspace": 0.38},
    )

    for mi, (model, layers, model_label) in enumerate([
        ("resnet18", RESNET_LAYERS, "ResNet-18"),
        ("deit_tiny", DEIT_LAYERS, "DeiT-tiny"),
    ]):
        for ci, cat in enumerate(CATEGORIES):
            ax = axes[mi, ci]
            sub = df[(df["model"] == model) & (df["category"] == cat)]
            vals = [sub[sub["layer"] == l]["roc_auc"].max() for l in layers]
            best_i = int(np.nanargmax(vals))

            for xi, v in enumerate(vals):
                color = BLUE_BEST if xi == best_i else GREY_REST
                ax.bar(xi, v, color=color, edgecolor="none", width=0.55, zorder=3)
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8, color="#444")

            ax.set_xticks(range(len(layers)))
            labels = ([LAYER_LABEL[l].replace("(c=", "c = ").replace(")", "") for l in layers]
                      if model == "resnet18" else [LAYER_LABEL[l] for l in layers])
            ax.set_xticklabels(labels, fontsize=8.5)
            ax.set_ylim(0.0, 1.20)
            ax.spines["left"].set_color("#ddd")
            ax.spines["bottom"].set_color("#ddd")
            if mi == 0:
                ax.set_title(CAT_LABEL[cat], pad=7)
            if ci == 0:
                ax.set_ylabel(f"{model_label}\nROC-AUC", fontsize=9)

    path = os.path.join(OUT_DIR, "fig3_cross_dataset.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def fig4():
    """Ascending-complexity baseline comparison for both backbones."""
    rows = []
    for cat in CATEGORIES:
        for model, layers in [("resnet18", RESNET_LAYERS), ("deit_tiny", DEIT_LAYERS)]:
            sub = df[(df["model"] == model) & (df["category"] == cat)]
            shallow = layers[0]
            pix = sub[(sub["layer"] == shallow) & (sub["detector"] == "KMeans(k=3)")]["roc_auc"]
            naive = sub[sub["detector"] == "KMeans(k=3)"].groupby("layer")["roc_auc"].max().max()
            knn = sub[sub["detector"] == "kNN(k=5)"].groupby("layer")["roc_auc"].max().max()
            maha = sub[sub["detector"] == "Mahalanobis"].groupby("layer")["roc_auc"].max().max()
            rows.append({
                "category": cat, "model": model,
                "pix": pix.values[0] if len(pix) else np.nan,
                "naive": naive, "knn": knn, "maha": maha,
            })
    bdf = pd.DataFrame(rows)

    approach_labels = [
        "Pixel proxy\n(shallow layer\n+ KMeans)",
        "Naive cluster\n(best layer\n+ KMeans)",
        "kNN\n(best layer)",
        "Mahalanobis\n(best layer)",
    ]
    approach_keys = ["pix", "naive", "knn", "maha"]

    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    x = np.arange(len(approach_keys))
    bar_w = 0.3
    off = [-bar_w / 2, bar_w / 2]

    for mi, (model, m_label) in enumerate([
        ("resnet18", "ResNet-18"),
        ("deit_tiny", "DeiT-tiny"),
    ]):
        sub = bdf[bdf["model"] == model]
        vals = [sub[k].mean() for k in approach_keys]
        bars = ax.bar(
            x + off[mi], vals, bar_w,
            color=BLUES_ASC, edgecolor="white", linewidth=0.4,
            label=m_label, zorder=3, alpha=0.92,
            hatch="" if model == "resnet18" else "///",
        )
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8.5, color="#444")

    ax.axhline(0.5, color="#ccc", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(approach_labels, fontsize=9.5)
    ax.set_ylim(0.3, 1.1)
    ax.set_ylabel("Mean ROC-AUC (across 5 categories)")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["left"].set_color("#ccc")
    ax.spines["bottom"].set_color("#ccc")

    path = os.path.join(OUT_DIR, "fig4_baseline_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def print_summary():
    idx = df.groupby(["category", "model"])["roc_auc"].idxmax()
    top = (
        df.loc[idx][["category", "model", "layer", "detector", "roc_auc", "pr_auc", "best_f1"]]
        .sort_values(["category", "model"])
    )
    top["model"] = top["model"].map({"resnet18": "ResNet-18", "deit_tiny": "DeiT-tiny"})
    top["layer"] = top["layer"].map(LAYER_LABEL).fillna(top["layer"])
    top.columns = ["Category", "Model", "Best Layer", "Best Detector",
                   "ROC-AUC", "PR-AUC", "F1"]
    print(top.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    print(f"Loaded {len(df)} rows across {df['category'].nunique()} categories")
    fig1()
    fig2()
    fig3()
    fig4()
    print("\nBest (layer + detector) per model and category:")
    print_summary()
