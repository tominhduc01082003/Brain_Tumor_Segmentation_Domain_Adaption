import numpy as np
import nibabel as nib
from pathlib import Path

# ================== CONFIG ==================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# LABELS_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset001_BraTSAdult" / "labelsTr"
# LABELS_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset003_BraTSAdult" / "labelsTr"
# LABELS_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset002_BraTSPeds" / "labelsTr"
# LABELS_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset142_BraTS_DA" / "labelsTr"
LABELS_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset004_BraTSPeds_Test" / "labelsTr"
EXPECTED_LABELS = {0, 1, 2, 3}

# ================== MAIN ==================
def main():
    print("=== CHECK nnUNet DATASET LABELS ===")
    print(f"Labels directory: {LABELS_DIR}")

    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"❌ labelsTr not found: {LABELS_DIR}")

    all_labels = set()
    bad_files = []

    files = sorted(LABELS_DIR.glob("*.nii.gz"))
    print(f"🔍 Found {len(files)} label files\n")

    for f in files:
        img = nib.load(f)
        data = img.get_fdata()

        unique_labels = set(np.unique(data).astype(int))
        all_labels |= unique_labels

        unexpected = unique_labels - EXPECTED_LABELS
        if unexpected:
            bad_files.append((f.name, unique_labels))

        print(f"{f.name:25s} -> {sorted(unique_labels)}")

    print("\n================ SUMMARY ================")
    print(f"All labels found in dataset: {sorted(all_labels)}")

    if bad_files:
        print("\n❌ Files with unexpected labels:")
        for name, labels in bad_files:
            print(f" - {name}: {sorted(labels)}")
    else:
        print("\n✅ All files contain only expected labels!")

    print("========================================")

if __name__ == "__main__":
    main()
