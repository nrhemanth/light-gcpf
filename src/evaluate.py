"""Compute anomaly-detection metrics and save a summary CSV.

For each (category, model, layer, detector) combination we report
ROC-AUC, PR-AUC, F1-optimal threshold, and the corresponding confusion
matrix counts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def compute_metrics(labels, scores, threshold=None):
    """Binary classification metrics on (labels, scores).

    If threshold is None, picks the threshold that maximises F1.
    """
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = int(np.argmax(f1s))
    best_f1 = float(f1s[best_idx])
    opt_thresh = float(thresholds[best_idx]) if threshold is None else threshold

    preds = (scores >= opt_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "best_f1": round(best_f1, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "threshold": round(opt_thresh, 6),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_all(score_results, label_map, output_dir):
    """Evaluate all combinations and write a results CSV.

    score_results: {category: {model_layer_key: {detector_name: scores}}}
    label_map:     {category: labels array}
    """
    rows = []
    for category, model_layer_dict in score_results.items():
        labels = label_map[category]
        for model_layer_key, detector_dict in model_layer_dict.items():
            model_name, layer_name = model_layer_key.split("__", 1)
            for detector_name, scores in detector_dict.items():
                try:
                    m = compute_metrics(labels, scores)
                    rows.append({
                        "category": category,
                        "model": model_name,
                        "layer": layer_name,
                        "detector": detector_name,
                        **m,
                    })
                except Exception as exc:
                    print(f"warning: metrics failed for "
                          f"{category}/{model_layer_key}/{detector_name}: {exc}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(
            ["category", "model", "layer", "roc_auc"],
            ascending=False, inplace=True,
        )
        out_dir = Path(output_dir) / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results_summary.csv"
        df.to_csv(out_path, index=False)
        print(f"results saved to {out_path}")

    return df


def print_summary(df, top_n=5):
    """Print top configurations by ROC-AUC for each category."""
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat].nlargest(top_n, "roc_auc")
        print(f"\n{cat}")
        print(sub[["model", "layer", "detector", "roc_auc", "pr_auc", "best_f1"]]
              .to_string(index=False))
