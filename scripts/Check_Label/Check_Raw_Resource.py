import os
import nibabel as nib
import numpy as np
from collections import defaultdict

# ===============================
# ĐƯỜNG DẪN DATASET
# ===============================
# DATASET_DIR = r"..\\..\\Resource\\Brats_Adults"
DATASET_DIR = r"..\\..\\Resource\\Brats_Peds\\Training"
# DATASET_DIR = r"..\\..\\Resource\\Brats_Peds\\Test"
# ===============================
# BIẾN THỐNG KÊ
# ===============================
global_label_counts = defaultdict(int)
all_labels = set()
file_count = 0

# ===============================
# DUYỆT TOÀN BỘ THƯ MỤC
# ===============================
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.endswith("_seg.nii.gz") or file.endswith("-seg.nii.gz"):
            file_path = os.path.join(root, file)
            print(f"\nProcessing: {file}")

            seg = nib.load(file_path).get_fdata()
            unique = np.unique(seg)

            print("  Unique labels in file:", unique)

            for u in unique:
                voxel_count = np.sum(seg == u)
                global_label_counts[int(u)] += voxel_count
                all_labels.add(int(u))

                print(f"    Label {int(u)} voxel count: {voxel_count}")

            file_count += 1

# ===============================
# TỔNG KẾT TOÀN DATASET
# ===============================
print("\n==============================")
print("DATASET SUMMARY")
print("==============================")
print("Total files processed:", file_count)
print("All labels found:", sorted(all_labels))
print("Total number of unique labels:", len(all_labels))

print("\nTotal voxel count per label (whole dataset):")
for label in sorted(global_label_counts.keys()):
    print(f"Label {label}: {global_label_counts[label]}")