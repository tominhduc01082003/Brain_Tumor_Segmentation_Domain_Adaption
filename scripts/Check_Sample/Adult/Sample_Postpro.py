import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==============================
# PATH
# ==============================

BASE = "../../../nnUNet_raw/Dataset001_BraTSAdult"

IMAGES = os.path.join(BASE, "imagesTr")
LABELS = os.path.join(BASE, "labelsTr")

CASE = "BraTS2021_00000"

# modalities
flair_path = os.path.join(IMAGES, f"{CASE}_0000.nii.gz")
t1_path    = os.path.join(IMAGES, f"{CASE}_0001.nii.gz")
t1ce_path  = os.path.join(IMAGES, f"{CASE}_0002.nii.gz")
t2_path    = os.path.join(IMAGES, f"{CASE}_0003.nii.gz")

seg_path   = os.path.join(LABELS, f"{CASE}.nii.gz")


# ==============================
# LOAD DATA
# ==============================

flair = nib.load(flair_path).get_fdata()
t1    = nib.load(t1_path).get_fdata()
t1ce  = nib.load(t1ce_path).get_fdata()
t2    = nib.load(t2_path).get_fdata()
seg   = nib.load(seg_path).get_fdata().astype(np.int32)

print("Shape:", flair.shape)
print("Labels:", np.unique(seg))


# ==============================
# FIND BEST SLICE
# ==============================

best_slice = 0
max_voxels = 0

for i in range(seg.shape[2]):

    tumor_voxels = np.sum(seg[:,:,i] > 0)

    if tumor_voxels > max_voxels:
        max_voxels = tumor_voxels
        best_slice = i

print("Best slice:", best_slice)
print("Tumor voxels:", max_voxels)


# ==============================
# EXTRACT SLICE
# ==============================

flair_s = flair[:,:,best_slice]
t1_s    = t1[:,:,best_slice]
t1ce_s  = t1ce[:,:,best_slice]
t2_s    = t2[:,:,best_slice]
seg_s   = seg[:,:,best_slice]


# ==============================
# COLOR MAP (after preprocessing)
# ==============================

color = np.zeros((seg_s.shape[0], seg_s.shape[1], 3))

color[seg_s == 1] = [1,0,0]   # red
color[seg_s == 2] = [0,1,0]   # green
color[seg_s == 3] = [0,0,1]   # blue


# ==============================
# LEGEND
# ==============================

legend = [
    Patch(facecolor='black', label='0 Background'),
    Patch(facecolor='red', label='1 Tumor Core (NCR/NET)'),
    Patch(facecolor='green', label='2 Edema (ED)'),
    Patch(facecolor='blue', label='3 Enhancing Tumor (ET)')
]


# ==============================
# VISUALIZATION
# ==============================

plt.figure(figsize=(16,8))

plt.subplot(2,3,1)
plt.title("FLAIR")
plt.imshow(flair_s, cmap="gray")
plt.axis("off")

plt.subplot(2,3,2)
plt.title("T1")
plt.imshow(t1_s, cmap="gray")
plt.axis("off")

plt.subplot(2,3,3)
plt.title("T1CE")
plt.imshow(t1ce_s, cmap="gray")
plt.axis("off")

plt.subplot(2,3,4)
plt.title("T2")
plt.imshow(t2_s, cmap="gray")
plt.axis("off")

plt.subplot(2,3,5)
plt.title("Segmentation")
plt.imshow(color)
plt.axis("off")

plt.subplot(2,3,6)
plt.title("Overlay (FLAIR + Seg)")
plt.imshow(flair_s, cmap="gray")
plt.imshow(color, alpha=0.5)
plt.axis("off")


plt.figlegend(
    handles=legend,
    loc="lower center",
    ncol=4,
    fontsize=11
)

plt.tight_layout()
plt.show()