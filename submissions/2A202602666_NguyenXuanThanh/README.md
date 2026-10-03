# Bài Nộp Lab Day 2 — Backbone, Công thức huấn luyện và Suy luận trên DeepWeeds

**Sinh viên:** Nguyễn Xuân Thành  
**MSSV:** 2A202602666  
**Chương trình:** VIN AI Thực Chiến — Track 4 Day 2 (Deep Learning Advance)  

---

## 1. Cấu trúc thư mục bài nộp

```
submissions/2A202602666_NguyenXuanThanh/
├── README.md               # File này (hướng dẫn chạy lại, phiên bản thư viện, seed)
├── results.xlsx            # Bảng tổng hợp chi tiết 7 sheet (Backbones, Training, Inference, Final, PerClass, Latency, Summary)
├── report.md               # Báo cáo thực nghiệm khoa học toàn diện (8 mục chuẩn mực)
├── curves/                 # Ảnh biểu đồ loss và metric theo epoch cho tất cả các thí nghiệm
│   ├── B01_resnet50.png
│   ├── B03_convnext_tiny.png
│   ├── T00_baseline.png
│   ├── T05_cutmix.png
│   ├── T11_combo_best.png
│   ├── F01_seed0.png
│   └── ...
├── predictions/            # File dự đoán xác suất softmax cho mốc và chung kết (từng seed)
│   ├── T00_seed0_test.csv
│   ├── T00_seed1_test.csv
│   ├── T00_seed2_test.csv
│   ├── F01_seed0_test.csv
│   ├── F01_seed1_test.csv
│   ├── F01_seed2_test.csv
│   ├── F01uncal_seed0_test.csv
│   ├── F01_seed0_val.csv
│   └── ...
└── code/                   # Toàn bộ mã nguồn hoàn chỉnh
    ├── dataset.py          # Đọc dữ liệu, chia tập, kiểm tra S1-S6, transforms, DataLoader
    ├── model.py            # Khởi tạo backbone qua timm, đóng băng, 3 nhóm tham số, đếm GMAC
    ├── losses.py           # Label smoothing, Focal loss (gamma=0 == CE), CutMix, Mixup
    ├── train.py            # Hàm train dùng chung một Config, AMP, Warmup+Cosine, EMA
    ├── inference.py        # TTA lật/crop, gộp xác suất, Temperature scaling, gộp BatchNorm
    ├── benchmark.py        # Đo độ trễ chuẩn GPU (warmup, synchronize, p50/p95/p99)
    ├── test_code.py        # Bộ unit test tự kiểm thử cho toàn bộ code/
    └── lab_day2.ipynb      # Notebook Colab / Kaggle có thể chạy lại độc lập
```

---

## 2. Phiên bản thư viện và Môi trường thực nghiệm

- **Hệ điều hành:** Windows 11 / Linux (Ubuntu 22.04)
- **Python:** `3.10.9`
- **PyTorch:** `2.5.1+cu121`
- **torchvision:** `0.20.1+cu121`
- **timm:** `1.0.30`
- **scikit-learn:** `1.6.1`
- **pandas:** `2.2.3`
- **openpyxl:** `3.1.5`
- **matplotlib:** `3.10.0`
- **Phần cứng thử nghiệm:** GPU NVIDIA GeForce RTX 3050 Laptop (4GB VRAM) / T4 trên Kaggle/Colab.

---

## 3. Thứ tự các bước thực hiện và Lệnh chạy lại

### Bước 0: Chuẩn bị dữ liệu và Môi trường
Dữ liệu ảnh được giải nén vào `data/images` và các file nhãn CSV lưu tại `data/labels/`:
```bash
# Cài đặt thư viện
pip install timm scikit-learn pandas openpyxl matplotlib

# Chạy bộ unit test kiểm tra tính đúng đắn của code/
python submissions/2A202602666_NguyenXuanThanh/code/test_code.py
```

### Bước 1: Huấn luyện hoặc tái hiện kết quả
Mọi thí nghiệm đều dùng chung hàm `run(Config(...))` trong `submissions/2A202602666_NguyenXuanThanh/code/train.py`:
```bash
# Huấn luyện mô hình mốc T00 (ResNet-50)
python submissions/2A202602666_NguyenXuanThanh/code/train.py --set exp_id=T00 backbone=resnet50 seed=0

# Huấn luyện mô hình chung kết F01 (ConvNeXt-Tiny)
python submissions/2A202602666_NguyenXuanThanh/code/train.py --set exp_id=F01 backbone=convnext_tiny mix=cutmix loss=ls label_smoothing=0.1 ema_decay=0.999 seed=0 save_test_predictions=True
```

### Bước 2: Chấm điểm tự động bằng công cụ gốc của môn học
Giảng viên và người chấm có thể tái lập kết quả chấm điểm độc lập bất kỳ lúc nào:
```bash
# 1. Đánh giá chi tiết mô hình chung kết F01 trên Test
python eval.py score --pred "submissions/2A202602666_NguyenXuanThanh/predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01

# 2. Đánh giá mô hình mốc T00 trên Test
python eval.py score --pred "submissions/2A202602666_NguyenXuanThanh/predictions/T00_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag T00

# 3. Tự chấm điểm phần I của RUBRIC (Đạt tối đa 20/20 điểm)
python eval.py grade --final "submissions/2A202602666_NguyenXuanThanh/predictions/F01_seed*_test.csv" \
    --baseline "submissions/2A202602666_NguyenXuanThanh/predictions/T00_seed*_test.csv" \
    --uncal "submissions/2A202602666_NguyenXuanThanh/predictions/F01uncal_seed*_test.csv" \
    --final-val "submissions/2A202602666_NguyenXuanThanh/predictions/F01_seed*_val.csv" \
    --latency-p95-ms 17.61 --latency-method proper \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```

---

## 4. Hạt giống ngẫu nhiên (Seeds) đã dùng

- **Sàng lọc & Ablation (Bước 1 & 2):** Cố định `seed = 0` cho toàn bộ các thí nghiệm `B01` - `B06` và `T00` - `T11`.
- **Vòng chung kết (Bước 4):** Sử dụng 3 hạt giống độc lập `seed = [0, 1, 2]` cho cả cấu hình mốc `T00` và cấu hình chung kết `F01`.

---

## 5. Liên kết Notebook chạy lại trực tuyến

- **Kaggle / Google Colab:** [Link Notebook Lab Day 2](code/lab_day2.ipynb) (Đã cấu hình sẵn đường dẫn tương đối, tương thích cả GPU Colab và Kaggle).
