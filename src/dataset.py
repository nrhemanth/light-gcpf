"""Build the MVTec AD DataFrame and dataset classes.

Expected directory layout under data_root:
    <category>/train/good/*.png
    <category>/test/good/*.png
    <category>/test/<defect>/*.png
    <category>/ground_truth/<defect>/*_mask.png
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_mvtec_dataframe(data_root, categories):
    """Walk the MVTec AD tree and return a DataFrame.

    Columns: image_path, category, split, label, defect_type, mask_path.
    label is 0 for normal and 1 for defective.
    """
    records = []

    for category in categories:
        cat = category.lower()
        cat_dir = Path(data_root) / cat

        if not cat_dir.exists():
            print(f"warning: category not found: {cat_dir}")
            continue

        train_good = cat_dir / "train" / "good"
        for img in sorted(train_good.glob("*.png")):
            records.append({
                "image_path": str(img),
                "category": cat,
                "split": "train",
                "label": 0,
                "defect_type": "good",
                "mask_path": None,
            })

        test_dir = cat_dir / "test"
        if not test_dir.exists():
            print(f"warning: test directory not found: {test_dir}")
            continue

        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue

            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            gt_dir = cat_dir / "ground_truth" / defect_type if label == 1 else None

            for img in sorted(defect_dir.glob("*.png")):
                mask_path = None
                if gt_dir is not None:
                    candidate = gt_dir / f"{img.stem}_mask.png"
                    if candidate.exists():
                        mask_path = str(candidate)

                records.append({
                    "image_path": str(img),
                    "category": cat,
                    "split": "test",
                    "label": label,
                    "defect_type": defect_type,
                    "mask_path": mask_path,
                })

    return pd.DataFrame(records)


def get_transforms(image_size=224):
    """Standard ImageNet preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MVTecDataset(Dataset):
    """Wrap a subset of the MVTec DataFrame as a PyTorch Dataset."""

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or get_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"]), row["image_path"]


def dataset_summary(df):
    """Per-category counts of split, label, defect type."""
    return (
        df.groupby(["category", "split", "label", "defect_type"])
        .size()
        .rename("count")
        .reset_index()
    )
