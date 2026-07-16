# Chest X-Ray Classification: Normal vs. Viral Pneumonia vs. COVID-19

![Python](https://img.shields.io/badge/Python-3-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![Status](https://img.shields.io/badge/Status-Course%20Project-lightgrey.svg)

A transfer-learning model that classifies chest X-ray images into three categories — Normal, Viral Pneumonia, or COVID-19 — using a fine-tuned MobileNetV2 backbone.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Repository Contents](#repository-contents)
- [Methodology](#methodology)
- [Results](#results)
- [Reproducing This](#reproducing-this)
- [Limitations & Honest Notes](#limitations--honest-notes)

## Overview

Distinguishing COVID-19 from viral pneumonia on a chest X-ray is a task where the visual differences can be subtle even to a trained eye — both present as lung congestion, differing mainly in pattern and severity. This project explores whether a lightweight, pre-trained image classifier (MobileNetV2, chosen for its small size relative to heavier backbones) can learn to tell the three classes apart from a modest set of labeled X-rays, using transfer learning rather than training a CNN from scratch.

## Dataset

- **Source:** COVID_IEEE chest X-ray image set
- **Total images:** 1,823 — 668 Normal, 619 Viral Pneumonia, 536 COVID-19
- **Split:** 70% train / 15% validation / 15% test → 1,276 / 274 / 273 images respectively
- **Image size:** 224×224, 3-channel (RGB)

A quick visual check of samples from each class confirmed the expected pattern before any modeling began: normal X-rays show clear lungs, viral pneumonia shows mild congestion, and COVID-19 cases show more pronounced congestion — motivating the choice of a model architecture with enough capacity to pick up on that gradient rather than a binary present/absent classifier.

## Repository Contents

- **`model.ipynb`** — the complete pipeline: data loading, preprocessing, model definition, training, fine-tuning, and evaluation, in a single notebook

That's the entire repository. There's no separate `src/`, no API layer, no saved model weights committed here — the notebook produces `.h5` model files when run, but those are build artifacts, not checked into version control.

## Methodology

**1. Loading.** Images are loaded via PySpark's image data source (`ImageSchema`) rather than a simpler approach like PIL or OpenCV directly — an unusual choice at this dataset size (1,823 images fits comfortably in memory without needing distributed processing), but that's the actual loading path used.

**2. Image enhancement.** Each image goes through white-balance correction (percentile-based channel clipping) followed by CLAHE (Contrast-Limited Adaptive Histogram Equalization) — a standard pairing for X-ray images specifically, since it boosts local contrast in the lung region without over-amplifying noise the way global histogram equalization would.

**3. Normalization and splitting.** Pixel values are normalized, then the combined dataset is split 70/15/15 into train/validation/test sets.

**4. Augmentation.** Random horizontal flips and small rotations are applied to the training set only, to help the model generalize despite the modest dataset size.

**5. Model: MobileNetV2 transfer learning, in two stages.**
   - **Stage 1 (frozen base):** MobileNetV2 pre-trained on ImageNet, with its convolutional base frozen and only a new classification head trained on top, using Adam (`lr=0.001`), early stopping (patience 20 on validation accuracy), and learning-rate reduction on plateau.
   - **Stage 2 (fine-tuning):** layers from index 120 onward in the 155-layer MobileNetV2 base are unfrozen and trained further at a much lower learning rate, letting the model adapt its higher-level features specifically to X-ray images rather than relying solely on generic ImageNet features.

**6. Final retraining.** After validating the approach on the train/val/test split, the model is fine-tuned once more on the full dataset (train + validation + test merged, 1,823 images) to produce the final saved model — a common step when a dataset is small enough that using every available labeled image for the final model is more valuable than holding a permanent test set in reserve.

## Results

**Held-out test set (273 images, genuinely unseen during training):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.90 | 0.93 | 0.92 | 100 |
| COVID-19 | 0.90 | 0.90 | 0.90 | 93 |
| Viral Pneumonia | 0.99 | 0.95 | 0.97 | 80 |
| **Accuracy** | | | **93%** | 273 |

**Validation set (274 images), for reference:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.94 | 0.95 | 0.95 | 100 |
| COVID-19 | 0.92 | 0.94 | 0.93 | 93 |
| Viral Pneumonia | 0.99 | 0.95 | 0.97 | 81 |
| **Accuracy** | | | **95%** | 274 |

**Final model, evaluated on the full merged dataset (1,823 images) it was subsequently fine-tuned on:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.89 | 0.99 | 0.94 | 668 |
| COVID-19 | 0.99 | 0.88 | 0.93 | 619 |
| Viral Pneumonia | 1.00 | 0.98 | 0.99 | 536 |
| **Accuracy** | | | **95%** | 1,823 |

Viral Pneumonia is consistently the easiest class to identify correctly across every evaluation (highest precision and recall throughout), while Normal and COVID-19 show more confusion with each other — sensible, given that mild-to-moderate lung congestion is where the two categories visually overlap most.

## Reproducing This

```bash
git clone https://github.com/Tanmay-IITDSAI/X-Ray-Disease-Prediction.git
cd X-Ray-Disease-Prediction
jupyter notebook model.ipynb
```

You'll need the COVID_IEEE image set (organized into `covid/`, `normal/`, and `virus/` subfolders) placed alongside the notebook, plus PySpark, TensorFlow/Keras, OpenCV, and standard scientific Python packages — the image dataset itself isn't included in this repository.

## Limitations & Honest Notes

- **Small dataset.** 1,823 images total is modest for deep learning, even with transfer learning and augmentation — results should be read as a promising proof of concept rather than a clinically validated result.
- **The final reported 95% accuracy is not a held-out number.** That figure comes from evaluating the model on the same train+validation+test data it was just fine-tuned on. The one genuinely unseen-data result in this project is the pre-merge test-set pass: **93% accuracy**, which is the number to cite if you need an honest estimate of generalization.
- **Three classes, not a general-purpose diagnostic tool.** This model distinguishes Normal / Viral Pneumonia / COVID-19 specifically — it does not detect the wider range of thoracic pathologies (e.g., cardiomegaly, effusion, pneumothorax) sometimes associated with chest X-ray classification projects.
- **No formal hyperparameter search.** Learning rates, freeze points, and epoch counts were set manually rather than tuned via a systematic search.
