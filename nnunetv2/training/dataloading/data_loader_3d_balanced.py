
# from typing import Union, Tuple

# import numpy as np
# from batchgenerators.utilities.file_and_folder_operations import *
# from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
# from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D

# class nnUNetDataLoader3D_Balanced(nnUNetDataLoader3D):
#     def __init__(self,
#                  data: nnUNetDataset,
#                  batch_size: int,
#                  patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
#                  final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
#                  label_manager,  # LabelManager,
#                  classes,
#                  oversample_foreground_percent: float = 0.0,
#                  sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
#                  pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
#                  probabilistic_oversampling: bool = False):
#         super().__init__(data, batch_size, patch_size, final_patch_size, label_manager, oversample_foreground_percent,
#                          sampling_probabilities, pad_sides, probabilistic_oversampling)
#         self.indices = []
#         for class_id in classes:
#             class_indices = [data_id for data_id in data.keys() if data_id.startswith(class_id)]
#             # Fix (start) :
#             # if class_id == 'adult':
#             #  class_indices = [k for k in data.keys() if k.startswith('BraTS2021')]
#             # elif class_id == 'peds':
#             #  class_indices = [k for k in data.keys() if k.startswith('BraTSPeds')]
#             # else:
#             #  raise RuntimeError(f"Unknown class_id {class_id}")
#             # end.
#             self.indices.append(class_indices)

#     def get_indices(self):
#         indices = []
#         if self.infinite:
#             for class_indices in self.indices:
#                 indices = [*indices,
#                            *np.random.choice(class_indices, int(self.batch_size / len(self.indices)), replace=True,
#                                              p=self.sampling_probabilities)]
#             np.random.shuffle(indices)
#             return indices

from typing import Union, Tuple, List
import numpy as np
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D


class nnUNetDataLoader3D_Balanced(nnUNetDataLoader3D):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager,
                 classes,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[float], Tuple[float, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False):

        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities,
                         pad_sides, probabilistic_oversampling)

        self.classes = classes

        # Build indices theo domain
        self.indices = [[] for _ in classes]

        for k in data.keys():
            props = data[k]['properties']
            domain = props.get('domain', None)

            if domain is None:
                raise RuntimeError(f"[ERROR] No 'domain' found in properties of {k}")

            if domain not in classes:
                raise RuntimeError(f"[ERROR] Domain '{domain}' not in classes {classes}")

            class_idx = classes.index(domain)
            self.indices[class_idx].append(k)

        # DEBUG
        print("=== BALANCED DATALOADER INIT ===")
        total = 0
        for i, v in enumerate(self.indices):
            print(f"{classes[i]}: {len(v)} samples")
            total += len(v)
        print(f"Total used: {total}")
        print("================================")

        # Safety check
        for i, v in enumerate(self.indices):
            if len(v) == 0:
                raise RuntimeError(f"[FATAL] Class '{classes[i]}' has 0 samples!")

        # batch_size phải chia hết cho số class
        if batch_size % len(classes) != 0:
            raise RuntimeError(
                f"[ERROR] batch_size ({batch_size}) must be divisible by number of classes ({len(classes)})"
            )

    def get_indices(self):
        indices = []

        if self.infinite:
            num_classes = len(self.indices)
            samples_per_class = self.batch_size // num_classes

            for class_indices in self.indices:
                chosen = np.random.choice(
                    class_indices,
                    samples_per_class,
                    replace=True
                )
                indices.extend(chosen)

            np.random.shuffle(indices)
            return indices

        else:
            raise RuntimeError("This DataLoader only supports infinite sampling")