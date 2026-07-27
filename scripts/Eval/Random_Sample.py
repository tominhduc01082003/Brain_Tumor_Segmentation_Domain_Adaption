import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]

test_images = ROOT / "test_image"
test_gt = ROOT / "test_gt"

pred_baseline = ROOT / "test_predictions_baseline"
pred_da = ROOT / "test_predictions_DA"

baseline_csv = SCRIPT_DIR / "metrics_baseline.csv"
da_csv = SCRIPT_DIR / "metrics_DA.csv"


cases = sorted({
    "_".join(f.name.split("_")[:2])
    for f in test_images.glob("*_0000.nii.gz")
})

if len(cases) == 0:
    raise RuntimeError(
        f"No cases found in:\n{test_images}"
    )

random.seed(42)
case = random.choice(cases)

print("=" * 60)
print("Random case:", case)
print("=" * 60)


for mod in ["0000", "0001", "0002", "0003"]:

    p = test_images / f"{case}_{mod}.nii.gz"

    if not p.exists():
        raise FileNotFoundError(p)

gt_path = test_gt / f"{case}.nii.gz"
baseline_path = pred_baseline / f"{case}.nii.gz"
da_path = pred_da / f"{case}.nii.gz"

for p in [gt_path, baseline_path, da_path]:

    if not p.exists():
        raise FileNotFoundError(p)


modalities = []

for mod in ["0000", "0001", "0002", "0003"]:

    img = nib.load(
        test_images / f"{case}_{mod}.nii.gz"
    ).get_fdata()

    modalities.append(img)


gt = nib.load(
    gt_path
).get_fdata()

baseline = nib.load(
    baseline_path
).get_fdata()

da = nib.load(
    da_path
).get_fdata()

baseline_df = pd.read_csv(
    baseline_csv
)

da_df = pd.read_csv(
    da_csv
)

case_filename = f"{case}.nii.gz"

# ----------------------------------------------------------
# BASELINE
# ----------------------------------------------------------

baseline_case = baseline_df[
    baseline_df["Case"] == case_filename
]

if len(baseline_case) == 0:
    raise RuntimeError(
        f"{case_filename} not found in metrics_baseline.csv"
    )

baseline_wt = float(
    baseline_case[
        baseline_case["Labels"] == "WT"
    ]["LesionWise_Score_Dice"].iloc[0]
)

baseline_tc = float(
    baseline_case[
        baseline_case["Labels"] == "TC"
    ]["LesionWise_Score_Dice"].iloc[0]
)

baseline_et = float(
    baseline_case[
        baseline_case["Labels"] == "ET"
    ]["LesionWise_Score_Dice"].iloc[0]
)

baseline_mean = np.mean([
    baseline_wt,
    baseline_tc,
    baseline_et
])

# ----------------------------------------------------------
# DA
# ----------------------------------------------------------

da_case = da_df[
    da_df["Case"] == case_filename
]

if len(da_case) == 0:
    raise RuntimeError(
        f"{case_filename} not found in metrics_DA.csv"
    )

da_wt = float(
    da_case[
        da_case["Labels"] == "WT"
    ]["LesionWise_Score_Dice"].iloc[0]
)

da_tc = float(
    da_case[
        da_case["Labels"] == "TC"
    ]["LesionWise_Score_Dice"].iloc[0]
)

da_et = float(
    da_case[
        da_case["Labels"] == "ET"
    ]["LesionWise_Score_Dice"].iloc[0]
)

da_mean = np.mean([
    da_wt,
    da_tc,
    da_et
])

print("\nBASELINE")
print(f"WT   : {baseline_wt:.4f}")
print(f"TC   : {baseline_tc:.4f}")
print(f"ET   : {baseline_et:.4f}")
print(f"Mean : {baseline_mean:.4f}")

print("\nDA")
print(f"WT   : {da_wt:.4f}")
print(f"TC   : {da_tc:.4f}")
print(f"ET   : {da_et:.4f}")
print(f"Mean : {da_mean:.4f}")


tumor_area = [
    (gt[:, :, z] > 0).sum()
    for z in range(gt.shape[2])
]

z = int(
    np.argmax(tumor_area)
)

print("\nSelected Slice:", z)

# ==========================================================
# COLORMAP
#
# 0 Background
# 1 NCR/NET
# 2 Edema
# 3 ET
# ==========================================================

cmap = ListedColormap([
    "black",
    "red",
    "lime",
    "blue"
])

# ==========================================================
# FIGURE
# ==========================================================

fig, axes = plt.subplots(
    1,
    9,
    figsize=(34, 7)
)

# ==========================================================
# MRI MODALITIES
# ==========================================================

titles = [
    "T1",
    "T1ce",
    "T2",
    "FLAIR"
]

for i in range(4):

    axes[i].imshow(
        modalities[i][:, :, z],
        cmap="gray"
    )

    axes[i].set_title(
        titles[i],
        fontsize=12,
        fontweight="bold"
    )

    axes[i].axis("off")

# ==========================================================
# GROUND TRUTH
# ==========================================================

axes[4].imshow(
    gt[:, :, z],
    cmap=cmap,
    vmin=0,
    vmax=3
)

axes[4].set_title(
    "Ground Truth",
    fontsize=12,
    fontweight="bold"
)

axes[4].axis("off")

# ==========================================================
# BASELINE
# ==========================================================

axes[5].imshow(
    baseline[:, :, z],
    cmap=cmap,
    vmin=0,
    vmax=3
)

axes[5].set_title(
    (
        "Baseline\n"
        f"WT={baseline_wt:.3f}\n"
        f"TC={baseline_tc:.3f}\n"
        f"ET={baseline_et:.3f}"
    ),
    fontsize=10
)

axes[5].axis("off")

# ==========================================================
# DA
# ==========================================================

axes[6].imshow(
    da[:, :, z],
    cmap=cmap,
    vmin=0,
    vmax=3
)

axes[6].set_title(
    (
        "DA\n"
        f"WT={da_wt:.3f}\n"
        f"TC={da_tc:.3f}\n"
        f"ET={da_et:.3f}"
    ),
    fontsize=10
)

axes[6].axis("off")

# ==========================================================
# OVERLAY BASELINE
# ==========================================================

axes[7].imshow(
    modalities[3][:, :, z],
    cmap="gray"
)

axes[7].imshow(
    baseline[:, :, z],
    cmap=cmap,
    alpha=0.45,
    vmin=0,
    vmax=3
)

axes[7].set_title(
    f"Overlay Baseline\nMean={baseline_mean:.3f}",
    fontsize=10
)

axes[7].axis("off")

# ==========================================================
# OVERLAY DA
# ==========================================================

axes[8].imshow(
    modalities[3][:, :, z],
    cmap="gray"
)

axes[8].imshow(
    da[:, :, z],
    cmap=cmap,
    alpha=0.45,
    vmin=0,
    vmax=3
)

axes[8].set_title(
    f"Overlay DA\nMean={da_mean:.3f}",
    fontsize=10
)

axes[8].axis("off")

# ==========================================================
# LEGEND
# ==========================================================

legend_elements = [

    Patch(
        facecolor="red",
        label="1 NCR / NET"
    ),

    Patch(
        facecolor="lime",
        label="2 Edema"
    ),

    Patch(
        facecolor="blue",
        label="3 Enhancing Tumor"
    )
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.63, 0.05),
    ncol=3,
    fontsize=11,
    frameon=True
)

# ==========================================================
# GLOBAL TITLE
# ==========================================================

delta = (
    da_mean
    - baseline_mean
)

plt.suptitle(
    (
        f"{case} | Slice={z}\n"
        f"Baseline Mean Dice = {baseline_mean:.4f}   |   "
        f"DA Mean Dice = {da_mean:.4f}   |   "
        f"Δ = {delta:+.4f}"
    ),
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0.10, 1, 0.95]
)

plt.show()