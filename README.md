# Retinal Disease Detection on ODIR-5K

This project detects retinal disease in colour fundus photographs from the [ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) dataset. It uses EfficientNet-B0 in PyTorch, labels each eye separately, splits the data by patient so no patient's images leak between training and test, and evaluates with 5-fold cross-validation.

It covers two tasks:

1. **Binary:** is the eye normal or abnormal?
2. **8-label:** which of the eight ODIR categories (N, D, G, C, A, H, M, O) does the eye have? An eye can have more than one.

## Results

Every number below is on the **same held-out test set**: 1,281 eyes from 671 patients that were never used for training or tuning.

### Binary (normal vs abnormal)

| Model | Test ROC-AUC |
|---|---|
| v1: 384px, single model | 0.846 |
| v2: 512px with stronger regularisation, single model | 0.861 |
| 5-fold CV, single models (mean ± sd) | 0.863 ± 0.004 |
| **5-fold CV, 5-model ensemble** | **0.875** (95% CI 0.854–0.896) |

At a screening threshold chosen on validation data only, the ensemble catches **86.7%** of abnormal eyes with **65.0%** specificity.

How well each disease is detected depends on how visible it is in the photo. Severe diabetic retinopathy, cataract, myopia and AMD are all detected with AUC > 0.91. Mild diabetic retinopathy (0.75) and glaucoma (0.78) are the weak spots.

### 8-label (per eye)

| Class | Test eyes | AUC | Average precision |
|---|---|---|---|
| Myopia (M) | 52 | 0.996 | 0.964 |
| Cataract (C) | 60 | 0.983 | 0.841 |
| AMD (A) | 44 | 0.947 | 0.635 |
| Glaucoma (G) | 67 | 0.944 | 0.594 |
| Hypertension (H) | 34 | 0.937 | 0.451 |
| Diabetic retinopathy (D) | 355 | 0.903 | 0.844 |
| Normal (N) | 569 | 0.878 | 0.829 |
| Other (O) | 181 | 0.867 | 0.650 |

- **Overall:** macro AUC **0.932**, micro AUC 0.955.
- **ODIR-style patient score:** 0.835. This is computed on our own test split, so it can't be compared with the challenge leaderboard.
- **Label check:** our per-eye labels, combined for each patient, match ODIR's official patient-level labels **100%**.

## Method

- **Labels for each eye.** Labels come from that eye's own diagnostic keywords. The dataset's patient-level labels describe both eyes together and would mislabel the healthy eye of a patient with one-sided disease.
- **Patient-level split.** A fixed test set of 20% of patients is held out, and the remaining patients are split into 5 folds with `StratifiedGroupKFold`. Both eyes of a patient always land in the same split.
- **Model.** EfficientNet-B0 pretrained on ImageNet, with 512×512 input. Training runs in two stages: first only the new classifier head with the backbone frozen, then the whole network.
- **Training settings.**
  - AdamW with weight decay 0.05, cosine learning-rate schedule and mixed precision.
  - Regularisation: mixup, random erasing, dropout 0.5.
  - Early stopping on validation AUC.
- **Evaluation.**
  - Decision thresholds are tuned on validation data only, never on test.
  - 95% confidence intervals use a patient-level bootstrap.
  - Grad-CAM heat-maps show which parts of the image drive each prediction.

## Notebooks

| Notebook | What it does |
|---|---|
| `01_binary_v1_baseline.ipynb` | Binary baseline at 384px (Kaggle T4) |
| `02_binary_v2_512px.ipynb` | Binary model at 512px with stronger regularisation (Kaggle T4) |
| `03_binary_5fold_cv.ipynb` | Binary 5-fold CV and ensemble; training log in `03_binary_5fold_cv_log.txt`, results in `outputs_cv/` |
| `04_binary_per_disease_analysis.ipynb` | Sensitivity per disease, Grad-CAM, most confident mistakes (`outputs_cv/analysis/`) |
| `05_multilabel_8class.ipynb` | 8-label per-eye classifier, 5-fold ensemble (`outputs_ml/`) |

## Running it

```bash
pip install -r requirements.txt
jupyter notebook
```

- **Dataset.** The notebooks download the dataset automatically with `kagglehub`. You may need a Kaggle API token in `~/.kaggle/kaggle.json`.
- **GPU.**
  - On Kaggle, use **GPU T4**. Kaggle's current PyTorch doesn't support the P100.
  - On a V100, install PyTorch 2.14 built for CUDA 12.6 (`--index-url https://download.pytorch.org/whl/cu126`). Newer builds have dropped support for that GPU.
- **Runtime.** Each 5-fold run took about 1.5 hours on a Tesla V100.
- **Weights.** Trained model weights (`*.pt`) are not stored in the repo.

## Limitations

- Labels come from free-text keywords, so some are noisy.
- The models have only been tested on ODIR-5K, with no external validation set.
- Rare classes (hypertension, AMD, glaucoma) are still detected with low precision.
