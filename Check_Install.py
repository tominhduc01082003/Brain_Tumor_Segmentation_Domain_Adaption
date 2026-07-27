print("=== KIỂM TRA CÀI ĐẶT THƯ VIỆN ===\n")
try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version (torch): {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA KHÔNG khả dụng (đang chạy CPU mode)")
except ImportError:
    print("❌ LỖI: Chưa cài torch hoặc torch không import được!")

print("-" * 50)

try:
    import torchvision
    print(f"torchvision: {torchvision.__version__}")
except ImportError:
    print("❌ LỖI: Chưa cài torchvision")

try:
    import torchaudio
    print(f"torchaudio: {torchaudio.__version__} (warning có thể bỏ qua)")
except ImportError:
    print("⚠ torchaudio chưa cài (KHÔNG ảnh hưởng BraTS / nnU-Net)")

print("-" * 50)
try:
    import nnunetv2
    print("nnU-Net v2: IMPORT OK")
    print("nnU-Net path:", nnunetv2.__file__)
except ImportError:
    print("❌ LỖI: Chưa cài nnunetv2 (pip install nnunetv2)")

print("-" * 50)

try:
    import nibabel as nib
    print(f"nibabel: {nib.__version__}")
except ImportError:
    print("❌ LỖI: Chưa cài nibabel")

try: 
    import SimpleITK as sitk
    print(f"SimpleITK: {sitk.Version()}")
except ImportError:
    print("⚠ SimpleITK chưa cài (không bắt buộc)")

print("-" * 50)

try:
    import monai
    print(f"MONAI: {monai.__version__}")
except ImportError:
    print("❌ LỖI: Chưa cài monai")

print("-" * 50)

try:
    import numpy as np
    print(f"numpy: {np.__version__}")
except ImportError:
    print("❌ LỖI: Chưa cài numpy")

try:
    import scipy
    print(f"scipy: {scipy.__version__}")
except ImportError:
    print("❌ LỖI: Chưa cài scipy")

try:
    import pandas as pd
    print(f"pandas: {pd.__version__}")
except ImportError:
    print("❌ LỖI: Chưa cài pandas")

try:
    import matplotlib
    print(f"matplotlib: {matplotlib.__version__}")
except ImportError:
    print("❌ LỖI: Chưa cài matplotlib")

print("\n=== KIỂM TRA HOÀN TẤT ===")
