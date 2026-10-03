"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Cài đặt hoàn chỉnh theo quy tắc S1-S6 (README.md mục 2.1 và GUIDE.md mục 1.2).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    train_path = labels_dir / f"train_subset{fold}.csv"
    val_path = labels_dir / f"val_subset{fold}.csv"
    test_path = labels_dir / f"test_subset{fold}.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    required_cols = {"Filename", "Label"}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if not required_cols.issubset(df.columns):
            raise ValueError(f"{name}_df thiếu cột bắt buộc trong {required_cols}")
        if "Species" not in df.columns:
            df["Species"] = df["Label"].map(lambda l: CLASS_NAMES[int(l)] if int(l) < len(CLASS_NAMES) else str(l))

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> Dict[str, Any]:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
    3. hợp ba tập phải bằng đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir`
    """
    images_dir = Path(images_dir)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total_samples = n_train + n_val + n_test

    # 1. Kiểm tra tỉ lệ
    train_pct = n_train / total_samples
    val_pct = n_val / total_samples
    test_pct = n_test / total_samples
    assert abs(train_pct - 0.60) < 0.02, f"Tỉ lệ train lệch chuẩn 60%: {train_pct:.2%}"
    assert abs(val_pct - 0.20) < 0.02, f"Tỉ lệ val lệch chuẩn 20%: {val_pct:.2%}"
    assert abs(test_pct - 0.20) < 0.02, f"Tỉ lệ test lệch chuẩn 20%: {test_pct:.2%}"

    # 2. Giao của từng cặp tập theo Filename phải RỖNG
    train_files = set(train_df["Filename"])
    val_files = set(val_df["Filename"])
    test_files = set(test_df["Filename"])

    tv_overlap = train_files.intersection(val_files)
    tt_overlap = train_files.intersection(test_files)
    vt_overlap = val_files.intersection(test_files)

    assert len(tv_overlap) == 0, f"train và val bị giao nhau: {len(tv_overlap)} ảnh"
    assert len(tt_overlap) == 0, f"train và test bị giao nhau: {len(tt_overlap)} ảnh"
    assert len(vt_overlap) == 0, f"val và test bị giao nhau: {len(vt_overlap)} ảnh"

    # 3. Hợp ba tập bằng đúng 17.509 ảnh
    all_files = train_files.union(val_files).union(test_files)
    assert len(all_files) == 17509, f"Tổng số ảnh duy nhất ({len(all_files)}) != 17509"

    # 4. Kiểm tra file tồn tại trong images_dir
    missing_train = [f for f in train_df["Filename"] if not (images_dir / f).exists()]
    missing_val = [f for f in val_df["Filename"] if not (images_dir / f).exists()]
    missing_test = [f for f in test_df["Filename"] if not (images_dir / f).exists()]
    total_missing = len(missing_train) + len(missing_val) + len(missing_test)
    assert total_missing == 0, f"Thiếu {total_missing} file ảnh trong {images_dir}"

    # Thống kê phân bố theo lớp
    train_counts = train_df["Label"].value_counts().sort_index().to_dict()
    val_counts = val_df["Label"].value_counts().sort_index().to_dict()
    test_counts = test_df["Label"].value_counts().sort_index().to_dict()

    return {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "total": total_samples,
        "train_pct": train_pct,
        "val_pct": val_pct,
        "test_pct": test_pct,
        "overlap_tv": len(tv_overlap),
        "overlap_tt": len(tt_overlap),
        "overlap_vt": len(vt_overlap),
        "train_per_class": train_counts,
        "val_per_class": val_counts,
        "test_per_class": test_counts,
    }


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic") -> T.Compose:
    """Tạo torchvision transform cho train hoặc val/test.

    Train augs:
      - "basic": RandomResizedCrop + RandomHorizontalFlip + Normalize
      - "color": basic + ColorJitter
      - "trivial": TrivialAugmentWide + RandomResizedCrop + RandomHorizontalFlip + Normalize
      - "randaug": RandAugment + RandomResizedCrop + RandomHorizontalFlip + Normalize

    Val/test:
      - Resize(256) + CenterCrop(img_size) (hoặc Resize(img_size) nếu img_size=256) + Normalize.
    """
    norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if not train:
        if img_size == 256:
            return T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                norm,
            ])
        resize_dim = int(img_size * 256 / 224) if img_size < 256 else img_size
        return T.Compose([
            T.Resize(resize_dim),
            T.CenterCrop(img_size),
            T.ToTensor(),
            norm,
        ])

    # Augmentation cho train
    t_list = []
    if aug == "basic":
        t_list = [
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            norm,
        ]
    elif aug == "color":
        t_list = [
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            norm,
        ]
    elif aug == "trivial":
        t_list = [
            T.TrivialAugmentWide(),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            norm,
        ]
    elif aug == "randaug":
        t_list = [
            T.RandAugment(num_ops=2, magnitude=9),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            norm,
        ]
    else:
        # Mặc định basic
        t_list = [
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            norm,
        ]

    return T.Compose(t_list)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh DeepWeeds từ images_dir theo DataFrame (Filename, Label)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].values
        self.labels = self.df["Label"].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        fname = self.filenames[idx]
        img_path = self.images_dir / fname
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = T.ToTensor()(img)

        label = int(self.labels[idx])
        return img_tensor, label, fname


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader cho DeepWeeds."""
    dataset = DeepWeedsDataset(df, images_dir, transform=transform)

    custom_sampler = None
    shuffle = train

    if train and sampler == "balanced":
        # Trọng số tỉ lệ nghịch với tần số xuất hiện của mỗi lớp
        class_counts = df["Label"].value_counts().to_dict()
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in df["Label"]]
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
        custom_sampler = WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(sample_weights_tensor),
            replacement=True
        )
        shuffle = False

    drop_last = train and (len(dataset) % batch_size < 4)
    pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=custom_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
