### 1.Structure Pro

```python
BRATS_PEDS/
│
├── .venv/
│
├── nnUNet_raw/
│   └── Dataset001_BraTSAdult/
│       ├── imagesTr/
│       │   ├── BraTS2021_00000_0000.nii.gz
│       │   ├── BraTS2021_00000_0001.nii.gz
│       │   ├── BraTS2021_00000_0002.nii.gz
│       │   ├── BraTS2021_00000_0003.nii.gz
│       │   └── ...
│       ├── labelsTr/
│       │   ├── BraTS2021_00000.nii.gz
│       │   └── ...
│       └── dataset.json
│
├── nnUNet_preprocessed/
├── nnUNet_results/
│── nnunetv2/
├── scripts/
│   │
│   ├── Check_Label/
│   │   ├── Check_Raw_Resource.py
│   │   └── Verify_nnUNet_Integrity.py
│   │
│   ├── Check_Sample/
│   │   ├── Adult/
│   │   │   ├── Sample_Postpro.py
│   │   │   └── Sample_Prepro.py
│   │   └── Peds/
│   │       ├── Sample_Postpro.py
│   │       └── Sample_Prepro.py
│   │
│   ├── Eval/
│   │
│   ├── Fix_Label/
│   │   ├── Fix_BraTS_Adult_Labels.py
│   │   └── Fix_BraTS_Peds_Labels.py
│   │   └── Convert_Label_Back.py
│   │
│   └── Prepare_Data/
│       ├── Prepare_Brats_Adult.py
│       └── Prepare_Brats_Peds.py
│
├── Resource/
│   └── Brats_Adults/
│       ├── BraTS2021_00000/
│       │   ├── *_t1.nii.gz
│       │   ├── *_t1ce.nii.gz
│       │   ├── *_t2.nii.gz
│       │   ├── *_flair.nii.gz
│       │   └── *_seg.nii.gz
│       └── ...
├── test_image/
│── test_gt/
├── test_predictions_baseline/
├── test_predictions_DA/
├── Report/
│
├── Check_Install.py
└── requirements.txt
```

### 2.Chạy kiểm tra (nếu in ra đường dẫn thì ok):

- Mỗi khi chạy xong .venv\Scripts\activate thì chạy lệnh này :

```bash
$env:nnUNet_raw = "your\path\BraTS_Peds\nnUNet_raw"
$env:nnUNet_preprocessed = "your\path\BraTS_Peds\nnUNet_preprocessed"
$env:nnUNet_results = "your\path\BraTS_Peds\nnUNet_results"
echo $env:nnUNet_raw
echo $env:nnUNet_preprocessed
echo $env:nnUNet_results
```

### 3.Chạy lần lượt:

- Kiểm tra các gói đã cài thành công hay chưa :

```python
python Check_Install.py
```

- Chuẩn bị dữ liệu adult và chuyển dữ liệu đúng định dạng nnuNet vào thư mục " nnUNet_raw \ Dataset001_BraTSAdult " :

```python
python Prepare_Brats_Adult.py
```

- Sửa nhãn ở trong tập dữ liệu Adults từ label có chỉ số là (0,1,2,4) -> (0,1,2,3) :

```python
python Fix_BraTS_Labels.py
```

- Check xem đã ánh xạ nhãn thành công (0,1,2,4) -> (0,1,2,3) hay chưa :

```python
python Check_Dataset.py
```

### 4.Preprocess trước khi train

- Sau khi tiền xử lý sẽ lưu data mới ở thư mục "nnUNet_preprocessed\Dataset001_BraTSAdult"

```bash
nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity
```

- Sau khi tiền xử lý thì tiếp theo là gộp data Adult và PEDS để train UDA và lưu data mới ở thư mục "nnUNet_preprocessed\Dataset142_BraTS_DA"

```bash
nnUNetv2_plan_and_preprocess -d 142 -c 3d_fullres --verify_dataset_integrity
```

### 5.Training

- Train adult với 3d_fullres ở fold 0 :

```bash
nnUNetv2_train Dataset001_BraTSAdult 3d_fullres 0 --npz
```

- Train UDA với 3d_fullres_bs4 ở fold 0 :

```bash
nnUNetv2_train 142 3d_fullres 0 -tr nnUNetTrainerDA_500ep_noDS_4Convs --npz
```
