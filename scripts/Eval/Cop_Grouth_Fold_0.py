import json
import shutil
from pathlib import Path

# ===== PATH =====
# baseline
splits_file = Path("..\\..\\nnUNet_preprocessed\\Dataset003_BraTSAdult\\splits_final.json")
labelsTr = Path("..\\..\\nnUNet_raw\\Dataset003_BraTSAdult\\labelsTr")
output_dir = Path("..\\..\\val_gt")

# DA
# splits_file = Path("..\\..\\nnUNet_preprocessed\\Dataset142_BraTS_DA\\splits_final.json")
# labelsTr = Path("..\\..\\nnUNet_raw\\Dataset142_BraTS_DA\\labelsTr")
# output_dir = Path("..\\..\\val_gt_DA")
output_dir.mkdir(parents=True, exist_ok=True)

# ===== đọc split =====
with open(splits_file) as f:
    splits = json.load(f)

val_cases = splits[0]["val"]

print(f"Found {len(val_cases)} validation cases")

# ===== copy label =====
for case in val_cases:
    src = labelsTr / f"{case}.nii.gz"
    dst = output_dir / src.name

    if src.exists():
        shutil.copy(src, dst)
    else:
        print("Missing:", src)

print("Done copying ground truth!")