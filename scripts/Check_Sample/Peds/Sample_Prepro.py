import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# ===============================
# CONFIG
# ===============================
CASE_NAME = "BraTS-PED-00001-000"
DATASET_DIR = r"..\\..\\..\\Resource\\Brats_Peds\\Training"

LABELS = {
    0: "Background",
    1: "Non-Enhancing Tumor (NET)",
    2: "Cystic Component (CC)",
    3: "Edema (ED)",
    4: "Enhancing Tumor (ET)"
}

# ===============================
# COLOR MAP
# ===============================
colors = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
]
cmap = mcolors.ListedColormap(colors)

# ===============================
# LOAD DATA
# ===============================
case_path = os.path.join(DATASET_DIR, CASE_NAME)
img = nib.load(os.path.join(case_path, f"{CASE_NAME}-t1c.nii.gz")).get_fdata()
seg = nib.load(os.path.join(case_path, f"{CASE_NAME}-seg.nii.gz")).get_fdata().astype(np.int32)

# ===============================
# 1️⃣ IN FULL 3D INFO
# ===============================
print("\n==== FULL 3D VOLUME INFO ====")
unique_3d = np.unique(seg)
print("Unique labels in full volume:", unique_3d)

for label in unique_3d:
    print(f"Label {label} total voxels:", np.sum(seg == label))

# ===============================
# 2️⃣ CHỌN SLICE CÓ NHIỀU TUMOR NHẤT
# ===============================
tumor_counts = [np.sum(seg[:, :, i] > 0) for i in range(seg.shape[2])]
slice_idx = np.argmax(tumor_counts)

print("\nSelected BEST slice (max tumor voxels):", slice_idx)
print("Tumor voxel count in that slice:", tumor_counts[slice_idx])

# ===============================
# 3️⃣ SLICE INFO
# ===============================
img_slice = img[:, :, slice_idx]
seg_slice = seg[:, :, slice_idx]
unique_slice = np.unique(seg_slice)

print("\n==== SLICE INFO ====")
for label in unique_slice:
    print(f"Label {label} voxels in slice:",
          np.sum(seg_slice == label))

# ===============================
# 4️⃣ PLOT
# ===============================
plt.figure(figsize=(7,7))
plt.imshow(img_slice, cmap="gray")
plt.imshow(seg_slice, cmap=cmap, alpha=0.5, vmin=0, vmax=4)
plt.title(f"{CASE_NAME} - Slice {slice_idx}")
plt.axis("off")

patches = []
for label in unique_slice:
    patches.append(
        mpatches.Patch(
            color=colors[int(label)],
            label=f"{int(label)}: {LABELS.get(int(label),'Unknown')}"
        )
    )

plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()