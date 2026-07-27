import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PATH
# =========================

# BASE = "..\\..\\..\\nnUNet_raw\\Dataset002_BraTSPeds"
BASE = "..\\..\\..\\nnUNet_raw\\Dataset004_BraTSPeds_Test"
IMAGE_PATH = os.path.join(BASE, "imagesTr", "BraTSPeds_00209_0000.nii.gz")
LABEL_PATH = os.path.join(BASE, "labelsTr", "BraTSPeds_00209.nii.gz")


# =========================
# LOAD NIFTI
# =========================

img = nib.load(IMAGE_PATH).get_fdata()
label = nib.load(LABEL_PATH).get_fdata()

img = np.array(img)
label = np.array(label)

print("Image shape:", img.shape)
print("Label shape:", label.shape)

print("Unique labels:", np.unique(label))


# =========================
# FIND BEST SLICE
# =========================

best_slice = 0
max_voxels = 0

for i in range(label.shape[2]):

    tumor_voxels = np.sum(label[:,:,i] > 0)

    if tumor_voxels > max_voxels:
        max_voxels = tumor_voxels
        best_slice = i

print("Best slice:", best_slice)
print("Tumor voxels:", max_voxels)


# =========================
# SELECT SLICE
# =========================

img_slice = img[:,:,best_slice]
lab_slice = label[:,:,best_slice]


# =========================
# COLOR MAP
# =========================

color_map = np.zeros((lab_slice.shape[0], lab_slice.shape[1], 3))

color_map[lab_slice == 1] = [1,0,0]  # Red
color_map[lab_slice == 2] = [0,1,0]  # Green
color_map[lab_slice == 3] = [0,0,1]  # Blue


# =========================
# DISPLAY
# =========================

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("MRI Slice")
plt.imshow(img_slice, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Segmentation")
plt.imshow(color_map)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(img_slice, cmap="gray")
plt.imshow(color_map, alpha=0.5)
plt.axis("off")

plt.show()