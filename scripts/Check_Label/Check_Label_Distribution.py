from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import matplotlib

# ======================================================
# FIX FONT WARNING
# ======================================================

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# ======================================================
# PATH
# ======================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]

# ======================================================
# CHOOSE DATASET
# ======================================================

# DATASET_DIR = ROOT / "nnUNet_raw" / "Dataset003_BraTSAdult"

# DATASET_DIR = ROOT / "nnUNet_raw" / "Dataset002_BraTSPeds"
DATASET_DIR = ROOT / "nnUNet_raw" / "Dataset004_BraTSPeds_Test"
LABELS_DIR = DATASET_DIR / "labelsTr"

# ======================================================
# AUTO REPORT DIRECTORY
# ======================================================

dataset_name = DATASET_DIR.name.lower()

if "adult" in dataset_name:
    REPORT_DIR = SCRIPT_DIR / "plots" / "Adults"

elif "peds" in dataset_name and "test" not in dataset_name :
    REPORT_DIR = SCRIPT_DIR / "plots" / "Peds"/ "Train"
elif "test" in dataset_name:
    REPORT_DIR = SCRIPT_DIR / "plots" / "Peds"/ "Test"

else:
    REPORT_DIR = SCRIPT_DIR / "plots" / "Unknown"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# LABEL MAPPING
# ======================================================

LABEL_NAMES = {
    0: "Background",
    1: "Edema",
    2: "NCR/NET",
    3: "Enhancing Tumor"
}

# ======================================================
# LOAD FILES
# ======================================================

label_files = sorted(LABELS_DIR.glob("*.nii.gz"))

if not label_files:
    raise FileNotFoundError(f"No label files found in: {LABELS_DIR}")

print("=" * 60)
print(f"Dataset: {DATASET_DIR.name}")
print(f"Found {len(label_files)} label files")
print("=" * 60)

# ======================================================
# STATS
# ======================================================

voxel_count = defaultdict(int)
case_count = defaultdict(int)
invalid_labels = set()

# ======================================================
# SCAN ALL LABELS
# ======================================================

for path in tqdm(label_files, desc="Checking labels"):

    nii = nib.load(str(path))

    seg = nii.get_fdata().astype(np.uint8)

    unique, counts = np.unique(seg, return_counts=True)

    # --------------------------------------------
    # Check invalid labels
    # --------------------------------------------

    for lb in unique:

        if lb not in LABEL_NAMES:
            invalid_labels.add(int(lb))

    # --------------------------------------------
    # Count voxels
    # --------------------------------------------

    for lb, ct in zip(unique, counts):

        voxel_count[int(lb)] += int(ct)

    # --------------------------------------------
    # Count cases
    # --------------------------------------------

    for lb in unique:

        case_count[int(lb)] += 1

# ======================================================
# SUMMARY
# ======================================================

total_voxels = sum(voxel_count.values())

print("\n" + "=" * 60)
print("LABEL DISTRIBUTION SUMMARY")
print("=" * 60)

for lb in sorted(voxel_count.keys()):

    vox = voxel_count[lb]

    pct = 100 * vox / total_voxels

    cases = case_count[lb]

    print(
        f"Label {lb} ({LABEL_NAMES.get(lb, 'UNKNOWN')}): "
        f"{vox:,} voxels | "
        f"{pct:.6f}% | "
        f"{cases}/{len(label_files)} cases"
    )

# ======================================================
# CHECK INVALID LABELS
# ======================================================

if invalid_labels:

    print("\nWARNING: Invalid labels found!")

    print(sorted(invalid_labels))

else:

    print("\nNo invalid labels found.")

# ======================================================
# PREPARE DATA FOR PLOTS
# ======================================================

labels = sorted(voxel_count.keys())

names = [LABEL_NAMES.get(x, str(x)) for x in labels]

counts = [voxel_count[x] for x in labels]

case_occ = [case_count[x] for x in labels]

# ======================================================
# PLOT 1: TOTAL VOXELS
# ======================================================

plt.figure(figsize=(10, 6))

plt.bar(names, counts)

plt.title("Total Voxel Count per Label")

plt.ylabel("Voxel Count")

plt.xlabel("Label")

plt.xticks(rotation=10)

plt.tight_layout()

save_path_1 = REPORT_DIR / "total_voxel_distribution.png"

plt.savefig(save_path_1, dpi=300)

plt.close()

# ======================================================
# PLOT 2: CASE OCCURRENCE
# ======================================================

plt.figure(figsize=(10, 6))

plt.bar(names, case_occ)

plt.title("Case Occurrence per Label")

plt.ylabel("Number of Cases")

plt.xlabel("Label")

plt.xticks(rotation=10)

plt.tight_layout()

save_path_2 = REPORT_DIR / "case_occurrence_distribution.png"

plt.savefig(save_path_2, dpi=300)

plt.close()

# ======================================================
# PLOT 3: FOREGROUND ONLY
# ======================================================

fg_labels = [x for x in labels if x != 0]

fg_names = [LABEL_NAMES[x] for x in fg_labels]

fg_counts = [voxel_count[x] for x in fg_labels]

plt.figure(figsize=(10, 6))

plt.bar(fg_names, fg_counts)

plt.title("Foreground Tumor Distribution")

plt.ylabel("Voxel Count")

plt.xlabel("Tumor Class")

plt.xticks(rotation=10)

plt.tight_layout()

save_path_3 = REPORT_DIR / "foreground_distribution.png"

plt.savefig(save_path_3, dpi=300)

plt.close()

# ======================================================
# DONE
# ======================================================

print("\nSaved plots:")

print(save_path_1)

print(save_path_2)

print(save_path_3)

print("\nDone.")