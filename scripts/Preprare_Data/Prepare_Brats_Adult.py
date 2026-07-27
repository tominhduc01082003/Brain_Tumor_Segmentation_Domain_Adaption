import json
import shutil
from pathlib import Path

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATASET_DIR = PROJECT_ROOT / "nnUNet_raw" / "Dataset003_BraTSAdult"

IMAGES_TR = RAW_DATASET_DIR / "imagesTr"
LABELS_TR = RAW_DATASET_DIR / "labelsTr"

SOURCE_DATA_DIR = PROJECT_ROOT / "Resource" / "Brats_Adults"

# ==========================================================
# MRI MODALITIES
# ==========================================================

MODALITY_MAP = {
    "t1": "0000",
    "t1ce": "0001",
    "t2": "0002",
    "flair": "0003",
}

# ==========================================================
# UTILS
# ==========================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def find_file(case_dir: Path, keyword: str):

    for f in case_dir.iterdir():

        if (
            keyword in f.name.lower()
            and f.suffixes[-2:] == [".nii", ".gz"]
        ):
            return f

    return None

# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print(" PREPARE BraTS ADULT DATASET FOR nnU-Net v2 ")
    print("=" * 60)

    ensure_dir(IMAGES_TR)
    ensure_dir(LABELS_TR)

    cases = sorted([
        d for d in SOURCE_DATA_DIR.iterdir()
        if d.is_dir()
    ])

    print(f"Found {len(cases)} cases")

    converted_cases = 0

    for case_dir in cases:

        case_id = case_dir.name

        print(f"Processing {case_id}")

        # ==================================================
        # COPY MRI MODALITIES
        # ==================================================

        for modality, idx in MODALITY_MAP.items():

            src = find_file(case_dir, modality)

            if src is None:
                raise FileNotFoundError(
                    f"Missing modality '{modality}' in {case_id}"
                )

            dst = IMAGES_TR / f"{case_id}_{idx}.nii.gz"

            shutil.copy(src, dst)

        # ==================================================
        # COPY SEGMENTATION
        # ==================================================

        seg = find_file(case_dir, "seg")

        if seg is None:
            raise FileNotFoundError(
                f"Missing segmentation in {case_id}"
            )

        shutil.copy(
            seg,
            LABELS_TR / f"{case_id}.nii.gz"
        )

        converted_cases += 1

    # ==========================================================
    # dataset.json
    # ==========================================================

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

        "numTraining": converted_cases,

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

    with open(RAW_DATASET_DIR / "dataset.json", "w") as f:

        json.dump(dataset_json, f, indent=4)

    print()
    print("DONE")
    print(f"Converted {converted_cases} cases")

# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":
    main()