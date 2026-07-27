from pathlib import Path
import nibabel as nib
import numpy as np

# ==========================================================
# LABEL DIRECTORY
# ==========================================================

LABELS_DIR = (
    Path(__file__).resolve().parents[2]
    / "nnUNet_raw"
    / "Dataset003_BraTSAdult"
    / "labelsTr"
) #

print("=" * 60)
print(" FIX BraTS Adult Labels For nnUNet ")
print("=" * 60)

print(f"Labels directory: {LABELS_DIR}")

# ==========================================================
# BraTS ORIGINAL:
#
# 0 = background
# 1 = NCR/NET
# 2 = edema
# 4 = enhancing tumor
#
# nnUNet INTERNAL:
#
# 0 = background
# 1 = edema
# 2 = NCR/NET
# 3 = enhancing tumor
#
# Mapping:
#
# 1 -> 2
# 2 -> 1
# 4 -> 3
#
# ==========================================================

all_files = sorted(LABELS_DIR.glob("*.nii.gz"))

print(f"Found {len(all_files)} segmentation files")
print()

for seg_file in all_files:

    nii = nib.load(seg_file)

    data = nii.get_fdata().astype(np.int16)

    unique_before = np.unique(data)

    # ======================================================
    # CREATE NEW LABEL ARRAY
    # ======================================================

    new_data = np.zeros_like(data, dtype=np.int16)

    # edema
    new_data[data == 2] = 1

    # NCR/NET
    new_data[data == 1] = 2

    # enhancing tumor
    new_data[data == 4] = 3

    unique_after = np.unique(new_data)

    # ======================================================
    # SAVE
    # ======================================================

    nib.save(
        nib.Nifti1Image(
            new_data,
            nii.affine,
            nii.header
        ),
        seg_file
    )

    print(
        f"{seg_file.name}: "
        f"{unique_before.tolist()} -> "
        f"{unique_after.tolist()}"
    )

print()
print("DONE")
print("All labels remapped successfully")

print()
print("Final nnUNet labels:")
print("0 = background")
print("1 = edema")
print("2 = NCR/NET")
print("3 = enhancing tumor")