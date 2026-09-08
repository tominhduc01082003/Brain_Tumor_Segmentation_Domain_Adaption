# Brain Tumor Segmentation UDA

Unsupervised Domain Adaptation framework for pediatric brain tumor segmentation on multi-parametric MRI using **DA-nnUNet**.

This project transfers segmentation knowledge from **adult glioma MRI (BraTS Adult)** to **pediatric glioma MRI (BRATS_PEDS)** using **nnU-Net v2**, **Gradient Reversal Layer (GRL)**, and a **Multi-scale Domain Classifier**.

---

# Overview

The project focuses on brain tumor segmentation under a domain shift between adult and pediatric MRI.

The proposed **DA-nnUNet** extends nnU-Net v2 with adversarial domain adaptation to learn domain-invariant features.

```text
BraTS Adult
Source Domain
     │
     ▼
nnU-Net 3D Full Resolution
     │
     ├───────────────┐
     │               │
     ▼               ▼
Segmentation       GRL
     │               │
     │               ▼
     │        Domain Classifier
     │               ▲
     │               │
     └───────┬───────┘
             │
             ▼
     Domain-Invariant
         Features
             │
             ▼
       BRATS_PEDS
     Target Domain
             │
             ▼
      Final Prediction
```

The target domain is used without segmentation labels during the adaptation stage.

---

# Features

- 3D brain tumor segmentation using **nnU-Net v2**
- 3D Full Resolution configuration
- Multi-modal MRI:
  - T1
  - T1CE
  - T2
  - FLAIR

- Source-only baseline
- Unsupervised Domain Adaptation
- Gradient Reversal Layer (GRL)
- Multi-scale Domain Classifier
- Balanced source/target sampling
- Progressive GRL Lambda scheduling
- Mixed Precision / AMP
- Test-Time Augmentation (TTA)
- Connected Component Analysis (CCA)
- ET/WT thresholding
- Morphological post-processing
- Dice and HD95 evaluation

---

# Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Structure Project](#structure-project)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contact](#contact)

---

# Structure Project

```text
Brain-Tumor-Segmentation-UDA/

├── .venv/
│
├── nnUNet_raw/
│   ├── Dataset001_BraTSAdult/
│   │   ├── imagesTr/
│   │   ├── labelsTr/
│   │   └── dataset.json
│   │
│   ├── Dataset002_BraTSPeds/
│   │   ├── imagesTr/
│   │   ├── labelsTr/
│   │   └── dataset.json
│   │
│   └── ...
│
├── nnUNet_preprocessed/
│
├── nnUNet_results/
│
├── nnunetv2/
│
├── scripts/
│   ├── Dataset/
│   ├── Fix/
│   ├── Check/
│   ├── Trainer/
│   └── Postprocess/
│
├── checkpoints/
├── predictions/
├── results/
├── visualization/
│
├── requirements.txt
├── README.md
└── .gitignore
```

> Dataset IDs may vary depending on the local nnU-Net configuration.

---

# Pipeline

```text
┌────────────────── DATA PREPARATION ──────────────────┐
│                                                     │
│ BraTS Adult ─────┐                                  │
│                  ├──► Label Conversion              │
│ BRATS PEDS ──────┘        │                         │
│                           ▼                         │
│                      nnU-Net Format                 │
└───────────────────────────┬─────────────────────────┘
                            │
                            ▼
┌─────────────────── SOURCE PRETRAINING ───────────────┐
│                                                     │
│ BraTS Adult → nnU-Net 3D Full Resolution            │
│                                                     │
└───────────────────────────┬─────────────────────────┘
                            │
                            ▼
┌────────────────── DOMAIN ADAPTATION ─────────────────┐
│                                                     │
│ Source + Target → Feature Encoder → GRL             │
│                                      │              │
│                                      ▼              │
│                               Domain Classifier      │
│                                                     │
└───────────────────────────┬─────────────────────────┘
                            │
                            ▼
┌──────────────────── POST-PROCESSING ─────────────────┐
│                                                     │
│ CCA → ET/WT Threshold → TTA → Morphology            │
│                                                     │
└───────────────────────────┬─────────────────────────┘
                            │
                            ▼
                       Evaluation
                    WT / TC / ET
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/tominhduc01082003/Brain_Tumor_Segmentation_Domain_Adaption

cd Brain-Tumor-Segmentation-UDA
```

## Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

The reported experimental environment uses:

```text
Python 3.9.13
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

The project uses:

```text
Source Domain : BraTS Adult
Target Domain : BRATS_PEDS
```

Each case contains four MRI modalities:

```text
0000 → T1
0001 → T1CE
0002 → T2
0003 → FLAIR
```

The original BraTS labels are converted into the label representation used by the experimental nnU-Net configuration.

Example preparation scripts:

```text
Dataset140_BraTS_PEDs.py
Fix_BraTS_Adult_Labels.py
Fix_BraTS_Peds_Labels.py
```

The nnU-Net environment requires:

```text
nnUNet_raw
nnUNet_preprocessed
nnUNet_results
```

---

# Training

## Source-Only Baseline

The baseline model is trained only on labeled Adult data.

```bash
nnUNetv2_train Dataset001_BraTSAdult 3d_fullres 0 --npz
```

The experiment uses 5-fold cross-validation:

```text
Fold 0
Fold 1
Fold 2
Fold 3
Fold 4
```

---

## DA-nnUNet

The custom trainer integrates domain-adversarial training:

```bash
nnUNetv2_train \
    142 \
    3d_fullres \
    0 \
    tr \
    nnUNetTrainerDA_500ep_noDS_4Convs \
    --npz
```

The dataset ID should be changed to match the local configuration.

### Training Strategy

```text
Epoch 0–100
    │
    ▼
Source Pretraining
λ_domain = 0
    │
    ▼
Epoch 101–500
    │
    ▼
Adversarial Domain Adaptation
Source + Target
    │
    ▼
Final DA-nnUNet
```

Main configuration:

| Parameter         |                  Value |
| ----------------- | ---------------------: |
| Configuration     |           `3d_fullres` |
| Patch Size        |      `128 × 160 × 112` |
| Batch Size        |                    `2` |
| Optimizer         |                    SGD |
| Initial LR        |                 `0.01` |
| LR Schedule       |       Polynomial Decay |
| Epochs            |                  `500` |
| Loss              | Dice + CE + Domain BCE |
| Balanced Sampling |                Enabled |
| Deep Supervision  |               Disabled |

---

# Inference

Generate segmentation predictions using the trained DA-nnUNet model.

The prediction pipeline consists of:

```text
MRI
 │
 ▼
DA-nnUNet
 │
 ▼
Raw Prediction
 │
 ▼
Post-processing
 │
 ▼
Final Segmentation
```

The final prediction includes:

- Connected Component Analysis
- ET/WT thresholding
- Test-Time Augmentation
- Morphological operations

---

# Evaluation

The model is evaluated on three tumor regions:

```text
WT → Whole Tumor
TC → Tumor Core
ET → Enhancing Tumor
```

Main metrics:

- Dice Score
- HD95

```text
Higher Dice → Better
Lower HD95 → Better
```

---

# Results

The reported results on the pediatric target domain are:

| Model         |   WT Dice |   TC Dice |   ET Dice |   WT HD95 |   TC HD95 |    ET HD95 |
| ------------- | --------: | --------: | --------: | --------: | --------: | ---------: |
| Source-Only   |     0.782 |     0.515 | **0.548** |     53.29 |     30.22 |     115.08 |
| **DA-nnUNet** | **0.838** | **0.813** |     0.513 | **29.29** | **26.53** | **103.53** |

### Improvement

```text
WT Dice : 0.782 → 0.838
TC Dice : 0.515 → 0.813
ET Dice : 0.548 → 0.513

WT HD95 : 53.29 → 29.29
TC HD95 : 30.22 → 26.53
ET HD95 : 115.08 → 103.53
```

DA-nnUNet provides significant improvements in **WT** and **TC** segmentation and reduces HD95 across all three regions.

ET Dice remains challenging because the enhancing tumor region is extremely small and highly imbalanced.

---

# Post-processing Results

| Method                          |   WT Dice |   TC Dice |   ET Dice |
| ------------------------------- | --------: | --------: | --------: |
| DA-nnUNet Raw                   |     0.821 |     0.801 |     0.465 |
| **DA-nnUNet + Post-processing** | **0.838** | **0.813** | **0.513** |

Post-processing improves the segmentation quality, particularly for the **ET** region.

The reported post-processing includes:

```text
CCA
+
ET/WT Thresholding
+
TTA
+
Morphological Operations
```

---

# Contact

**Tô Minh Đức**

Email: `ducto020803@gmail.com`
