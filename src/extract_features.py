"""Batched feature extraction with disk caching.

Outputs land in <output_dir>/features/:
    {category}_{model}_{layer}_train.npy   (N_train, D)
    {category}_{model}_{layer}_test.npy    (N_test,  D)
    metadata_{category}.csv                image_path, split, label, defect_type
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MVTecDataset, build_mvtec_dataframe, get_transforms
from models import DeiTExtractor, ResNetExtractor


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _extract_split(extractor, df_split, device, batch_size, num_workers):
    """Run a single split through the extractor and stack outputs per layer."""
    dataset = MVTecDataset(df_split, transform=get_transforms())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type != "mps"),
    )

    buffers = {}
    extractor.eval().to(device)

    for images, _, _ in tqdm(loader, leave=False, desc="  batches"):
        images = images.to(device)
        feats = extractor(images)
        for layer, tensor in feats.items():
            buffers.setdefault(layer, []).append(tensor.float().numpy())

    return {layer: np.concatenate(bufs, axis=0) for layer, bufs in buffers.items()}


def extract_and_save(data_root, categories, output_dir,
                     batch_size=32, num_workers=0, force=False):
    """Extract features for every category and both backbones, cache to disk."""
    feat_dir = Path(output_dir) / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"device: {device}")

    categories = [c.lower() for c in categories]
    df_all = build_mvtec_dataframe(data_root, categories)

    extractors = {
        "resnet18": ResNetExtractor(pretrained=True),
        "deit_tiny": DeiTExtractor(pretrained=True),
    }

    for category in categories:
        df_cat = df_all[df_all["category"] == category]
        if df_cat.empty:
            print(f"warning: no images for {category}")
            continue

        df_cat.to_csv(feat_dir / f"metadata_{category}.csv", index=False)
        df_train = df_cat[df_cat["split"] == "train"]
        df_test = df_cat[df_cat["split"] == "test"]
        print(f"\n{category}: train={len(df_train)} test={len(df_test)}")

        for model_name, extractor in extractors.items():
            if not force and _already_cached(feat_dir, category, model_name):
                print(f"  {model_name}: cached, skipping")
                continue

            print(f"  {model_name}: train split")
            train_feats = _extract_split(extractor, df_train, device,
                                         batch_size, num_workers)
            print(f"  {model_name}: test split")
            test_feats = _extract_split(extractor, df_test, device,
                                        batch_size, num_workers)

            for layer in train_feats:
                prefix = f"{category}_{model_name}_{layer}"
                np.save(feat_dir / f"{prefix}_train.npy", train_feats[layer])
                np.save(feat_dir / f"{prefix}_test.npy", test_feats[layer])

    for ex in extractors.values():
        ex.remove_hooks()


def _already_cached(feat_dir, category, model_name):
    pattern = f"{category}_{model_name}_*_train.npy"
    return len(list(feat_dir.glob(pattern))) > 0


def load_features(output_dir, category, model_name, layer):
    """Load cached features and metadata for a given (category, model, layer)."""
    feat_dir = Path(output_dir) / "features"
    prefix = f"{category}_{model_name}_{layer}"

    train_feats = np.load(feat_dir / f"{prefix}_train.npy")
    test_feats = np.load(feat_dir / f"{prefix}_test.npy")

    meta = pd.read_csv(feat_dir / f"metadata_{category}.csv")
    df_train = meta[meta["split"] == "train"].reset_index(drop=True)
    df_test = meta[meta["split"] == "test"].reset_index(drop=True)
    return train_feats, test_feats, df_train, df_test


def available_feature_keys(output_dir, category):
    """List (model_name, layer) pairs that already have cached features."""
    feat_dir = Path(output_dir) / "features"
    keys = []
    for fpath in sorted(feat_dir.glob(f"{category}_*_train.npy")):
        stem = fpath.stem.replace(f"{category}_", "").replace("_train", "")
        for mname in ("resnet18", "deit_tiny"):
            if stem.startswith(mname + "_"):
                layer = stem[len(mname) + 1:]
                if layer in {"avgpool"} or layer.endswith(("_patches", "_patchidx")):
                    continue
                keys.append((mname, layer))
    return keys
