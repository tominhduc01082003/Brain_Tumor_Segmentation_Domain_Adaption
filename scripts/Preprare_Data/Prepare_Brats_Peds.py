import os
import shutil
from pathlib import Path
import json

# =========================
# PROJECT ROOT (auto detect)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# use training data
# SOURCE_DIR = PROJECT_ROOT / "Resource" / "Brats_Peds" / "Training"
# TARGET_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset002_BraTSPeds"

# Use test data
SOURCE_DIR = PROJECT_ROOT / "Resource" / "Brats_Peds" / "Test"
TARGET_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset004_BraTSPeds_Test"

IMAGES_TR = TARGET_DIR / "imagesTr"
LABELS_TR = TARGET_DIR / "labelsTr"

IMAGES_TR.mkdir(parents=True, exist_ok=True)
LABELS_TR.mkdir(parents=True, exist_ok=True)

# =========================
# MODALITY MAPPING
# =========================
modality_map = {
    "t1n": "0000",
    "t1c": "0001",
    "t2w": "0002",
    "t2f": "0003"
}

cases = sorted(os.listdir(SOURCE_DIR))
num_cases = 0

print(f"Scanning: {SOURCE_DIR}")
print(f"Output to: {TARGET_DIR}")
print("--------------------------------------------------")

for case in cases:
    case_path = SOURCE_DIR / case
    if not case_path.is_dir():
        continue

    # BraTS-PED-00001-000 → 00001
    case_id = case.split("-")[2]
    new_prefix = f"BraTSPeds_{case_id}"

    print(f"Processing case: {case}")

    for file in os.listdir(case_path):
        file_path = case_path / file

        if not file.endswith(".nii.gz"):
            continue

        # Copy label
        if "seg" in file:
            new_name = f"{new_prefix}.nii.gz"
            shutil.copy(file_path, LABELS_TR / new_name)

        # Copy modalities
        else:
            for key in modality_map:
                if key in file:
                    new_name = f"{new_prefix}_{modality_map[key]}.nii.gz"
                    shutil.copy(file_path, IMAGES_TR / new_name)

    num_cases += 1

print("--------------------------------------------------")
print(f"Converted {num_cases} cases successfully.")

# =========================
# AUTO CREATE dataset.json
# =========================
dataset_json = {

    "channel_names": {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "Flair"
    },

    "labels": {

        "background": 0,

        "whole tumor": [1, 2, 3],

        "tumor core": [2, 3],

        "enhancing tumor": [3]
    },

    "numTraining": num_cases,

    "file_ending": ".nii.gz",

    "regions_class_order": [1, 2, 3],

    "reference":
        "see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863",

    "licence":
        "see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863",

    "dataset_release": "1.0",

    "overwrite_image_reader_writer":
        "SimpleITKDomainIO"
    }

with open(TARGET_DIR / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print("dataset.json created.")
print("DONE ✅")