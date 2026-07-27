import json
import shutil
from pathlib import Path

# ===== PATH =====
# baseline
splits_file = Path("..\\..\\nnUNet_preprocessed\\Dataset003_BraTSAdult\\splits_final.json")
imagesTr = Path("..\\..\\nnUNet_raw\\Dataset003_BraTSAdult\\imagesTr")
output_dir = Path("..\\..\\val_images")

# DA
# splits_file = Path("..\\..\\nnUNet_preprocessed\\Dataset142_BraTS_DA\\splits_final.json")
# imagesTr = Path("..\\..\\nnUNet_raw\\Dataset142_BraTS_DA\\imagesTr")
# output_dir = Path("..\\..\\val_images_DA")
# tạo folder nếu chưa có
output_dir.mkdir(parents=True, exist_ok=True)

# ===== đọc split =====
with open(splits_file) as f:
    splits = json.load(f)

val_cases = splits[0]["val"]   # fold 0

print(f"Found {len(val_cases)} validation cases")

# ===== copy ảnh =====
modalities = ["0000", "0001", "0002", "0003"]

for case in val_cases:
    for m in modalities:
        src = imagesTr / f"{case}_{m}.nii.gz"
        dst = output_dir / src.name
        
        if src.exists():
            shutil.copy(src, dst)
        else:
            print("Missing:", src)

print("Done copying validation images!")