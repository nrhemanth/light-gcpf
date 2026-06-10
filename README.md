# Light-GCPF: Localized Anomaly Detection with Compact Pretrained Backbones

Localized industrial anomaly detection on the MVTec AD dataset using
pretrained ResNet-18 and DeiT-tiny as feature extractors, scored against
the distribution of normal patches with Mahalanobis scoring.

This repository builds on the GCPF framework introduced by
Wan *et al.* (IEEE TIE 2022) to evaluate whether the same paradigm works
with backbones that are 5–6× smaller than the WRN-50 used in the original
work.

![Dataset samples](figures/fig0_dataset_samples.png)

## Results

### Per-category model comparison

![Model comparison](figures/fig1_model_comparison.png)

### Classical baseline comparison

![Baseline comparison](figures/fig5_baseline_per_category.png)

## Setup and usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Place MVTec AD under data/<category>/{train,test,ground_truth}/...
python src/run_pipeline.py --data_root data --output_dir outputs
python src/layer_analysis.py
python src/baseline_models.py
```

The MVTec AD dataset is provided by MVTec Software GmbH:
[https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
