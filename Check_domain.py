# import pickle
# import os

# pkl_path = r"nnUNet_preprocessed\\Dataset142_BraTS_DA\\nnUNetPlans_3d_fullres\\BraTSPeds_00122.pkl"

# with open(pkl_path, 'rb') as f:
#     data = pickle.load(f)

# print(data.keys())
# print(data)

from collections import Counter
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

dataset = nnUNetDataset(
    "nnUNet_preprocessed\\Dataset142_BraTS_DA\\nnUNetPlans_3d_fullres"
)

counter = Counter()

for k in dataset.keys():
    domain = dataset[k]['properties']['domain']
    counter[domain] += 1

print("DOMAIN COUNT:", counter)