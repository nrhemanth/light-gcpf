"""Make the dataset sample figure.

Top row    - one normal image per category
Bottom row - one defective image per category with the ground-truth
             defect region overlaid in red.

Output: outputs/figures/fig0_dataset_samples.png
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFECT_CHOICE = {
    "hazelnut": "crack",
    "screw": "scratch_head",
    "transistor": "bent_lead",
    "zipper": "broken_teeth",
    "capsule": "crack",
}
CATEGORIES = ["hazelnut", "screw", "transistor", "zipper", "capsule"]
CAT_LABEL = {c: c.capitalize() for c in CATEGORIES}


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path):
    return np.asarray(Image.open(path).convert("L")) > 128


def pick(folder, index=0):
    files = sorted(folder.glob("*.png"))
    return files[index % len(files)]


def overlay(img, mask, color=(220, 40, 40), alpha=0.45):
    out = img.copy().astype(float)
    for ch, c in enumerate(color):
        out[:, :, ch] = np.where(
            mask,
            out[:, :, ch] * (1 - alpha) + c * alpha,
            out[:, :, ch],
        )
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    fig, axes = plt.subplots(
        2, len(CATEGORIES),
        figsize=(3.2 * len(CATEGORIES), 6.8),
        gridspec_kw={"hspace": 0.08, "wspace": 0.06},
    )

    for ci, cat in enumerate(CATEGORIES):
        cat_dir = DATA_ROOT / cat
        defect_type = DEFECT_CHOICE[cat]

        normal_path = pick(cat_dir / "train" / "good", index=4)
        normal_img = load_rgb(normal_path)
        ax_top = axes[0, ci]
        ax_top.imshow(normal_img)
        ax_top.set_axis_off()
        ax_top.set_title(CAT_LABEL[cat], pad=6, color="#222")

        defect_path = pick(cat_dir / "test" / defect_type, index=0)
        mask_path = cat_dir / "ground_truth" / defect_type / f"{defect_path.stem}_mask.png"
        if not mask_path.exists():
            mask_path = pick(cat_dir / "ground_truth" / defect_type, index=0)

        defect_img = load_rgb(defect_path)
        mask = load_mask(mask_path)

        if mask.shape != defect_img.shape[:2]:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (defect_img.shape[1], defect_img.shape[0]), Image.NEAREST,
            )
            mask = np.asarray(mask_pil) > 128

        ax_bot = axes[1, ci]
        ax_bot.imshow(overlay(defect_img, mask))
        ax_bot.contour(mask.astype(float), levels=[0.5],
                       colors=["white"], linewidths=[1.0], alpha=0.9)
        ax_bot.set_axis_off()

    for row_idx, row_label in enumerate(["Normal", "Defective"]):
        axes[row_idx, 0].text(
            -0.06, 0.5, row_label,
            transform=axes[row_idx, 0].transAxes,
            va="center", ha="right", rotation=90, color="#444",
        )

    path = OUT_DIR / "fig0_dataset_samples.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
