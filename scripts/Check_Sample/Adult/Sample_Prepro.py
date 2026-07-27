import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==============================
# PATH
# ==============================

BASE = "..\\..\\..\\Resource\\Brats_Adults\\BraTS2021_00000"

flair_path = os.path.join(BASE, "BraTS2021_00000_flair.nii.gz")
t1_path    = os.path.join(BASE, "BraTS2021_00000_t1.nii.gz")
t1ce_path  = os.path.join(BASE, "BraTS2021_00000_t1ce.nii.gz")
t2_path    = os.path.join(BASE, "BraTS2021_00000_t2.nii.gz")
seg_path   = os.path.join(BASE, "BraTS2021_00000_seg.nii.gz")


# ==============================
# LOAD DATA
# ==============================

flair = nib.load(flair_path).get_fdata()
t1    = nib.load(t1_path).get_fdata()
t1ce  = nib.load(t1ce_path).get_fdata()
t2    = nib.load(t2_path).get_fdata()
seg   = nib.load(seg_path).get_fdata()

seg = seg.astype(np.int32)

print("Shape:", flair.shape)
print("Unique labels:", np.unique(seg))


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
# GET SLICE
# ==============================

flair_s = flair[:,:,best_slice]
t1_s    = t1[:,:,best_slice]
t1ce_s  = t1ce[:,:,best_slice]
t2_s    = t2[:,:,best_slice]
seg_s   = seg[:,:,best_slice]


# ==============================
# COLOR MAP
# ==============================

color = np.zeros((seg_s.shape[0], seg_s.shape[1], 3))

color[seg_s == 1] = [1,0,0]   # Red
color[seg_s == 2] = [0,1,0]   # Green
color[seg_s == 4] = [0,0,1]   # Blue


# ==============================
# LEGEND
# ==============================

legend_elements = [
    Patch(facecolor='black', label='0  Background'),
    Patch(facecolor='red', label='1  NCR/NET (Tumor Core)'),
    Patch(facecolor='green', label='2  Edema (ED)'),
    Patch(facecolor='blue', label='4  Enhancing Tumor (ET)')
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

# add legend
plt.figlegend(
    handles=legend_elements,
    loc="lower center",
    ncol=4,
    fontsize=11
)

plt.tight_layout()
plt.show()