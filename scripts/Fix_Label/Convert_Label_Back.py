from pathlib import Path
import nibabel as nib
import numpy as np

# ==================================================
# PATH
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# gt
# GT_DIR = PROJECT_ROOT / "test_gt"
# predict
# GT_DIR = PROJECT_ROOT / "test_predictions_baseline"
GT_DIR = PROJECT_ROOT / "test_predictions_DA"
print("=" * 60)
print("Convert Test GT Labels To nnUNet Format")
print("=" * 60)

print("Folder:", GT_DIR)

# ==================================================
# CURRENT LABELS
#
# 0 = Background
# 1 = ED
# 2 = NC
# 3 = ET
#
# TARGET nnUNet LABELS
#
# 0 = Background
# 1 = NC
# 2 = ED
# 3 = ET
#
# Mapping:
# 1 -> 2
# 2 -> 1
# ==================================================

files = sorted(GT_DIR.glob("*.nii.gz"))

print(f"Found {len(files)} files\n")

for file in files:

    nii = nib.load(str(file))

    seg = nii.get_fdata().astype(np.uint8)

    unique_before = np.unique(seg)

    # -----------------------------------------
    # SAFE REMAP
    # -----------------------------------------

    new_seg = np.zeros_like(seg, dtype=np.uint8)

    new_seg[seg == 0] = 0
    new_seg[seg == 1] = 2   # ED -> 2
    new_seg[seg == 2] = 1   # NC -> 1
    new_seg[seg == 3] = 3   # ET -> 3

    unique_after = np.unique(new_seg)

    nib.save(
        nib.Nifti1Image(
            new_seg,
            nii.affine,
            nii.header
        ),
        str(file)
    )

    print(
        f"{file.name}: "
        f"{unique_before.tolist()} -> "
        f"{unique_after.tolist()}"
    )

print("\nDone!")
print("\nFinal labels:")

print("0 = Background")
print("1 = NCR/NET")
print("2 = Edema")
print("3 = Enhancing Tumor")