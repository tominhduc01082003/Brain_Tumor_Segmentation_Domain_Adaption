import os
import nibabel as nib
import numpy as np
from pathlib import Path

# ===============================
# PATH
# ===============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# INPUT_DIR = PROJECT_ROOT / "Resource" / "Brats_Peds" / "Training"
# OUTPUT_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset002_BraTSPeds" / "labelsTr"
INPUT_DIR = PROJECT_ROOT / "Resource" / "Brats_Peds" / "Test"
OUTPUT_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset004_BraTSPeds_Test" / "labelsTr"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# LABEL MAPPING
# ===============================

# LABEL_MAP = {
#     0:0,
#     1:1,
#     2:1, 
#     3:2,
#     4:3
# }
# Maping with NC:2 ,ED :1 ,ET :3
LABEL_MAP = {
    0:0,
    1:2,
    2:2,   # CC -> NET
    3:1,
    4:3
}

cases = sorted(os.listdir(INPUT_DIR))

count = 0

for case in cases:

    case_path = INPUT_DIR / case

    if not case_path.is_dir():
        continue

    # BraTS-PED-00001-000 → 00001
    case_id = case.split("-")[2]

    new_name = f"BraTSPeds_{case_id}.nii.gz"

    seg_file = None

    for f in os.listdir(case_path):
        if f.endswith("-seg.nii.gz"):
            seg_file = f
            break

    if seg_file is None:
        continue

    seg_path = case_path / seg_file

    print("Processing:", seg_file)

    nii = nib.load(seg_path)
    seg = nii.get_fdata().astype(np.int32)

    print("Before:", np.unique(seg))

    new_seg = np.zeros_like(seg)

    for old_label, new_label in LABEL_MAP.items():
        new_seg[seg == old_label] = new_label

    print("After:", np.unique(new_seg))

    new_seg = new_seg.astype(np.uint8)

    out_path = OUTPUT_DIR / new_name

    nib.save(
        nib.Nifti1Image(new_seg, nii.affine, nii.header),
        out_path
    )

    print("Saved:", new_name)
    print()

    count += 1

print("Finished converting labels")
print("Total processed:", count)