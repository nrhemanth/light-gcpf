"""End-to-end pipeline: extract features, score with all detectors, save metrics.

Usage:
    python src/run_pipeline.py \
        --data_root /path/to/mvtec \
        --categories hazelnut screw transistor zipper capsule \
        --output_dir outputs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from anomaly import fit_and_score
from dataset import build_mvtec_dataframe, dataset_summary
from evaluate import evaluate_all, print_summary
from extract_features import (
    available_feature_keys,
    extract_and_save,
    load_features,
)


def run_pipeline(data_root, categories, output_dir,
                 batch_size=32, num_workers=0,
                 knn_k=5, kmeans_ks=(3, 5, 10),
                 skip_extraction=False, force_extraction=False):
    """Run dataset build, feature extraction, scoring, and evaluation."""
    categories = [c.lower() for c in categories]

    # 1. Build dataset.
    print("\n[1/3] Building dataset")
    df_all = build_mvtec_dataframe(data_root, categories)
    if df_all.empty:
        raise RuntimeError(
            f"No images found under {data_root} for {categories}. "
            "Check that --data_root points at the extracted MVTec AD directory."
        )
    print(dataset_summary(df_all).to_string(index=False))

    # 2. Extract features.
    print("\n[2/3] Extracting features")
    if skip_extraction:
        print("  skipping (using cached features)")
    else:
        extract_and_save(
            data_root=data_root,
            categories=categories,
            output_dir=output_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            force=force_extraction,
        )

    # 3. Score and evaluate.
    print("\n[3/3] Scoring and evaluating")
    score_results = {}
    label_map = {}

    for category in categories:
        keys = available_feature_keys(output_dir, category)
        if not keys:
            print(f"  warning: no cached features for {category}, skipping")
            continue

        score_results[category] = {}
        labels_set = False

        for model_name, layer in keys:
            try:
                train_feats, test_feats, _, df_test = load_features(
                    output_dir, category, model_name, layer,
                )
            except FileNotFoundError as exc:
                print(f"  warning: {exc}")
                continue

            if not labels_set:
                label_map[category] = df_test["label"].values
                labels_set = True

            if train_feats.shape[0] < 5:
                print(f"  skip {category}/{model_name}/{layer}: "
                      f"only {train_feats.shape[0]} training samples")
                continue

            print(f"  {category}/{model_name}/{layer}: "
                  f"train={train_feats.shape} test={test_feats.shape}")
            key = f"{model_name}__{layer}"
            score_results[category][key] = fit_and_score(
                train_feats, test_feats,
                knn_k=knn_k, kmeans_ks=list(kmeans_ks),
            )

    results_df = evaluate_all(score_results, label_map, output_dir)
    print_summary(results_df)
    return results_df


def parse_args():
    p = argparse.ArgumentParser(description="Run the anomaly detection pipeline")
    p.add_argument("--data_root", required=True, help="Path to MVTec AD root")
    p.add_argument("--output_dir", default="outputs", help="Where to write features and metrics")
    p.add_argument("--categories", nargs="+",
                   default=["hazelnut", "screw", "transistor", "zipper", "capsule"],
                   help="MVTec categories to process")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--knn_k", type=int, default=5)
    p.add_argument("--kmeans_ks", nargs="+", type=int, default=[3, 5, 10])
    p.add_argument("--skip_extraction", action="store_true")
    p.add_argument("--force_extraction", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_root=args.data_root,
        categories=args.categories,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        knn_k=args.knn_k,
        kmeans_ks=tuple(args.kmeans_ks),
        skip_extraction=args.skip_extraction,
        force_extraction=args.force_extraction,
    )
