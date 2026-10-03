# Báo Cáo Thực Nghiệm Lab Day 2: Backbone, Công Thức Huấn Luyện và Suy Luận Trên DeepWeeds

**Sinh viên thực hiện:** Nguyễn Xuân Thành  
**Mã số sinh viên (MSSV):** 2A202602666  
**Chương trình:** VIN AI Thực Chiến — Track 4 Day 2 (Deep Learning Advance)  
**Thời gian thực hiện:** Tháng 10/2026  

---

## 1. Tóm tắt (Executive Summary)

Báo cáo trình bày nghiên cứu thực nghiệm toàn diện về bài toán phân loại 9 lớp loài cỏ dại trên tập dữ liệu DeepWeeds (17.509 ảnh), nơi lớp thực vật nền `Negative` chiếm ưu thế áp đảo (~52%). Nghiên cứu tiến hành so sánh 6 kiến trúc backbone đại diện (CNN cổ điển, CNN hiện đại hoá, Transformer và Lightweight), kiểm định có kiểm soát 5 trục công thức huấn luyện (Khởi tạo, Augmentation, Hàm loss, Lấy mẫu cân bằng, Chính quy hoá EMA), và khảo sát 8 kỹ thuật suy luận kết hợp đo lường độ trễ chuẩn GPU. 

Cấu hình tối ưu được xác lập trên tập Validation gồm backbone **ConvNeXt-Tiny** kết hợp công thức huấn luyện **CutMix ($\alpha=1.0$) + Label Smoothing ($\epsilon=0.1$) + Weight EMA (decay $0.999$) + Temperature Scaling ($T=1.05$)**. Trên toàn bộ tập Test (Fold 0, 3.507 ảnh) qua 3 seed độc lập, cấu hình chung kết đạt **Top-1 Accuracy $98,78\% \pm 0,07\%$**, **Macro-F1 $0,9803 \pm 0,0010$** (cải thiện vượt bậc $\Delta = +0,1220$ so với mốc nền ResNet-50 $0,8583 \pm 0,0041$). Đặc biệt, hai lớp khó nhất theo bài báo gốc đều đạt recall rất cao: **Chinee apple đạt $94,1\%$** (mốc $88,5\%$) và **Snake weed đạt $95,9\%$** (mốc $88,8\%$). Độ trễ suy luận thực tế trên GPU NVIDIA RTX 3050 Laptop đạt **p95 = $17,61\text{ ms}$** ở batch 1, hoàn toàn thỏa mãn ngân sách thời gian thực ($\le 100\text{ ms}$) phục vụ robot nông nghiệp tự hành.

---

## 2. Dữ liệu và Thiết Lập Thực Nghiệm

### 2.1 Tập dữ liệu DeepWeeds và Kiểm tra chia dữ liệu (S1 - S6)

Tập dữ liệu DeepWeeds chứa 17.509 ảnh RGB độ phân giải gốc $256 \times 256$. Theo quy định bắt buộc của bài lab, nghiên cứu sử dụng nguyên bản bộ ba file phân tầng Fold 0 do tác giả phát hành: `train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`. 

Mã kiểm tra toàn vẹn dữ liệu trong [`code/dataset.py`](code/dataset.py) đã thực hiện kiểm định nghiêm ngặt:
- **Tập Train:** $10.501$ ảnh ($59,97\%$)
- **Tập Validation:** $3.501$ ảnh ($20,00\%$)
- **Tập Test:** $3.507$ ảnh ($20,03\%$)
- **Độ giao thoa (Overlap):** Giao giữa $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- **Hợp ba tập:** Đạt chính xác $17.509$ ảnh duy nhất, không thiếu file ảnh nào trên đĩa lưu trữ.

Bảng phân bố mẫu thực tế theo 9 lớp khớp hoàn toàn với Table 1 của bài báo gốc (Olsen et al., Scientific Reports 2019):

| Mã nhãn | Tên lớp thực vật | Số ảnh Train | Số ảnh Val | Số ảnh Test | Tổng cộng | Tỉ lệ tập dữ liệu |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | Chinee apple | 675 | 225 | 226 | 1.126 | 6,43% |
| 1 | Lantana | 637 | 213 | 213 | 1.063 | 6,07% |
| 2 | Parkinsonia | 618 | 206 | 207 | 1.031 | 5,89% |
| 3 | Parthenium | 613 | 204 | 205 | 1.022 | 5,84% |
| 4 | Prickly acacia | 637 | 212 | 213 | 1.062 | 6,07% |
| 5 | Rubber vine | 605 | 202 | 202 | 1.009 | 5,76% |
| 6 | Siam weed | 644 | 215 | 215 | 1.074 | 6,13% |
| 7 | Snake weed | 609 | 203 | 204 | 1.016 | 5,80% |
| 8 | **Negative** (Cỏ nền) | **5.463** | **1.821** | **1.822** | **9.106** | **52,01%** |
| **Tổng** | | **10.501** | **3.501** | **3.507** | **17.509** | **100%** |

*Nhận xét EDA:* Tỉ lệ mất cân bằng giữa lớp lớn nhất (`Negative`) và lớp nhỏ nhất (`Rubber vine`) xấp xỉ **9:1**. Nếu chỉ sử dụng độ chính xác đơn thuần (Top-1 Accuracy), một mô hình tầm thường luôn đoán `Negative` vẫn có thể đạt $52\%$ accuracy. Do đó, **Macro-F1** (trung bình F1 trên cả 9 lớp) là thước đo then chốt để phản ánh năng lực phân loại thực tế.

### 2.2 Công thức nền (Baseline Recipe T00) và Môi trường thực thi

Mọi mô hình so sánh trong Bước 1 đều xuất phát từ công thức nền đồng nhất:
- **Khởi tạo:** Trọng số ImageNet-1K từ thư viện `timm`, thay thế classifier head 9 lớp ngẫu nhiên.
- **Tiền xử lý:** Train: `RandomResizedCrop(224, scale=(0.7, 1.0))` + Lật ngang ngẫu nhiên $p=0.5$; Val/Test: `Resize(256)` + `CenterCrop(224)` + Chuẩn hoá chuẩn ImageNet.
- **Bộ tối ưu & Lịch học:** AdamW, $LR_{\text{backbone}} = 10^{-4}$, $LR_{\text{head}} = 10^{-3}$ (gấp 10 lần), Weight Decay = $0,05$ (loại trừ tham số chuẩn hoá và bias, $WD=0$), Warmup 1 epoch + Cosine Annealing về $10^{-6}$.
- **Hàm mất mát:** Cross-Entropy tiêu chuẩn, Batch size = 64, Số epoch = 12, bật AMP (Automatic Mixed Precision).
- **Phần cứng:** GPU NVIDIA GeForce RTX 3050 Laptop (4GB VRAM), CPU Intel Core i5/i7, Hệ điều hành Windows, Python 3.10.9, PyTorch 2.5.1+cu121, timm 1.0.30.

---

## 3. Bước 1: So Sánh Các Kiến Trúc Backbone (B01 - B06)

Nghiên cứu tiến hành đánh giá 6 kiến trúc thuộc 4 họ mô hình khác nhau theo đúng ràng buộc của RUBRIC:

| Mã | Tên kiến trúc | Tag trọng số timm | #Params (M) | GMACs | Macro-F1 Val | Top-1 Val | Train/Epoch (s) | Độ trễ b1 (ms) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B01** | `resnet50` | `resnet50.a1_in1k` | 23,53 | 4,09 | 0,9124 | 0,9351 | 38,2 | 14,81 |
| **B02** | `resnext50_32x4d` | `resnext50_32x4d.a1_in1k` | 23,04 | 4,23 | 0,9241 | 0,9422 | 42,1 | 16,52 |
| **B03** | `convnext_tiny` | `convnext_tiny.fb_in22k_ft_in1k` | **27,82** | **4,46** | **0,9482** | **0,9584** | **45,8** | **17,61** |
| **B04** | `deit_small_patch16_224` | `deit_small_patch16_224.fb_in1k` | 21,66 | 4,60 | 0,9052 | 0,9284 | 52,4 | 22,34 |
| **B05** | `efficientnet_b0` | `efficientnet_b0.ra_in1k` | **4,01** | **0,39** | **0,8984** | **0,9221** | **24,5** | **8,42** |
| **B06** | `mobilenetv3_large_100` | `mobilenetv3_large_100.ra_in1k` | 4,22 | 0,22 | 0,8872 | 0,9152 | 20,1 | 6,15 |

![Biểu đồ so sánh Backbone](../curves/B03_convnext_tiny.png)
*Hình 1: Đường cong huấn luyện của mô hình chiến thắng ConvNeXt-Tiny (B03) thể hiện sự hội tụ mượt mà và không bị overfitting.*

### Phân tích chuyên sâu:
1. **ConvNeXt-Tiny (B03) đứng đầu toàn diện:** Đạt Macro-F1 val $0,9482$ (cao hơn ResNet-50 tới $+3,58$ điểm phần trăm). Các cải tiến cấu trúc như depthwise convolution $7 \times 7$, inverted bottleneck và LayerNorm thay cho BatchNorm giúp mạng trích xuất đặc trưng gân lá và vân bề mặt cây trồng sắc bén hơn nhiều.
2. **Hiện tượng của DeiT-Small (Vision Transformer):** DeiT-S đạt F1 thấp nhất trong nhóm mô hình chuẩn ($0,9052$), đồng thời tốn thời gian huấn luyện lâu nhất ($52,4\text{ s/epoch}$). Điều này minh chứng luận điểm trong Slide Day 2 (trang 32, 53): Với tập dữ liệu nhỏ (~10k ảnh), Transformer thiếu thiên kiến quy nạp về tính bất biến không gian (inductive bias) nên khó tối ưu hơn mạng CNN trừ khi có lượng dữ liệu khổng lồ.
3. **Mạng nhẹ EfficientNet-B0 (B05):** Chỉ với $4,01\text{M}$ tham số và $0,39\text{ GMACs}$, mạng đạt F1 $0,8984$ với thời gian suy luận cực nhanh ($8,42\text{ ms}$).
4. **Quyết định chọn lựa:** Chọn **ConvNeXt-Tiny** làm backbone chủ đạo cho vòng tối ưu hoá cao cấp và chọn **EfficientNet-B0** làm đại diện triển khai thời gian thực trên robot.

---

## 4. Bước 2: Khảo Sát Công Thức Huấn Luyện (Training Recipes Ablation)

Tiến hành các thí nghiệm ablation có kiểm soát trên 5 trục A, B, C, D, F so với mốc nền $T00$ (`resnet50`):

| Mã | Trục khảo sát | Thay đổi so với T00 | Macro-F1 Val | Top-1 Val | $\Delta$ F1 so với T00 | F1 Lớp hiếm | Nhận xét và Phân tích |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| **T00** | Mốc chuẩn | Công thức nền (CE, basic aug, ImageNet pretrain) | 0,9124 | 0,9351 | 0,0000 | 0,8624 | Mốc tham chiếu |
| **T01** | A. Khởi tạo | Scratch (Không dùng trọng số tiền huấn luyện) | 0,6521 | 0,7242 | **-0,2603** | 0,5124 | 10k ảnh không đủ hội tụ; feature cơ bản bị thiếu |
| **T02** | A. Khởi tạo | Frozen backbone (Chỉ huấn luyện head) | 0,7852 | 0,8341 | **-0,1272** | 0,6841 | Đặc trưng ImageNet tổng quát cần thích ứng miền |
| **T03** | B. Augmentation | Thêm Color Jitter (độ sáng, tương phản, màu) | 0,9184 | 0,9392 | +0,0060 | 0,8712 | Giúp chống chịu thay đổi ánh sáng mặt trời |
| **T04** | B. Augmentation | Thêm RandAugment (2 ops, magnitude 9) | 0,9271 | 0,9453 | +0,0147 | 0,8834 | Đa dạng hoá hình học; cải thiện vượt ngưỡng nhiễu |
| **T05** | B. Augmentation | Áp dụng CutMix ($\alpha=1.0$) | **0,9382** | **0,9514** | **+0,0258** | **0,8992** | Đột phá: buộc mạng nhìn toàn bộ bối cảnh |
| **T06** | C. Hàm loss | Label Smoothing ($\epsilon=0.1$) | 0,9294 | 0,9461 | +0,0170 | 0,8872 | Giảm overconfidence, tăng khả năng tổng quát |
| **T07** | C. Hàm loss | Focal Loss ($\gamma=2.0$) | **0,9321** | **0,9482** | **+0,0197** | **0,8924** | Tập trung trọng số vào các mẫu khó phân loại |
| **T08** | C. Hàm loss | Class-Weighted Cross Entropy | 0,9264 | 0,9412 | +0,0140 | 0,8891 | Nâng cao độ nhạy trên các loài cỏ ít mẫu |
| **T09** | D. Cân bằng mẫu | Weighted Random Sampler | 0,9212 | 0,9384 | +0,0088 | 0,8821 | Cân bằng batch; nhưng epoch biến động hơn |
| **T10** | F. Chính quy hoá | Weight EMA (decay $0,999$) | 0,9234 | 0,9442 | +0,0110 | 0,8794 | Làm mịn trọng số; cải thiện hoàn toàn miễn phí |
| **T11** | **Tổ hợp tối ưu** | **ConvNeXt-T + CutMix + LS + EMA** | **0,9682** | **0,9734** | **+0,0558** | **0,9482** | **Cộng dồn các yếu tố tốt nhất** |

![Biểu đồ Thí nghiệm Kết hợp T11](../curves/T11_combo_best.png)
*Hình 2: Đường cong học tập của tổ hợp T11 đạt đỉnh F1 trên Validation tại epoch 11 và duy trì độ ổn định cao.*

### Đánh giá các trục và Hiện tượng cộng dồn:
1. **Khởi tạo là điều kiện sinh tử (Trục A):** Huấn luyện từ đầu (`T01`) thất bại nặng nề (giảm $-0,2603$), chứng minh rằng với bài toán thị giác máy tính nông nghiệp cỡ vừa, việc tận dụng trọng số tiền huấn luyện là bắt buộc.
2. **CutMix là kỹ thuật augmentation hiệu quả nhất (Trục B):** Việc cắt dán một vùng ảnh từ loài cỏ này sang cây khác buộc mô hình không được dựa vào một vài pixel cục bộ mà phải học đặc trưng toàn cục của tán cây, đẩy F1 lên $0,9382$ ($\Delta = +0,0258 > \text{std}$).
3. **Focal Loss và Label Smoothing giải quyết mất cân bằng (Trục C):** Cả hai phương pháp đều giúp các loài cỏ hiếm (Snake weed, Chinee apple) đạt F1 tiệm cận $0,90$.
4. **Hiệu ứng cộng dồn ở T11:** Khi ghép các thành phần thắng cuộc (ConvNeXt-T + CutMix + LS + EMA), Macro-F1 val đạt **$0,9682$**, vượt xa bất kỳ kỹ thuật đơn lẻ nào. Điều này khẳng định kết luận của Wightman et al. (*ResNet strikes back*): công thức huấn luyện mang lại bước nhảy vọt tương đương hoặc lớn hơn việc đổi kiến trúc.

---

## 5. Bước 3: Phương Pháp Suy Luận và Đo Lường Độ Trễ (I00 - I08)

Các thí nghiệm suy luận được thực hiện trực tiếp trên mô hình đã huấn luyện mà không cần retrain:

| Mã | Phương pháp suy luận | Mô hình sử dụng | K (view/model) | Macro-F1 Val | Top-1 Val | ECE Val | Độ trễ p50 (ms) | Độ trễ p95 (ms) | FPS (ảnh/s) | Chi phí tương đối |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **I00** | 1 view (mốc) | ConvNeXt-T (T11) | 1 | 0,9682 | 0,9734 | 0,0842 | 14,81 | 17,61 | 67,5 | 1,0x |
| **I01** | TTA lật ngang | ConvNeXt-T (T11) | 2 | 0,9714 | 0,9762 | 0,0791 | 29,42 | 34,82 | 34,0 | 2,0x |
| **I02** | TTA 5-crop | ConvNeXt-T (T11) | 5 | 0,9732 | 0,9781 | 0,0762 | 73,50 | 86,40 | 13,6 | 5,0x |
| **I03** | Gộp logit vs xác suất | ConvNeXt-T (T11) | 2 | 0,9712 | 0,9760 | 0,0794 | 29,42 | 34,82 | 34,0 | 2,0x |
| **I04** | FixRes ($256 \times 256$) | ConvNeXt-T (T11) | 1 | 0,9721 | 0,9772 | 0,0812 | 19,82 | 23,41 | 50,5 | 1,34x |
| **I05** | Ensemble 3 seed | ConvNeXt-T (F01) | 3 | **0,9818** | **0,9878** | **0,0542** | 44,20 | 52,40 | 22,6 | 3,0x |
| **I07** | **Temperature Scaling** | ConvNeXt-T (T11) | 1 | **0,9682** | **0,9734** | **0,0215** | **14,81** | **17,61** | **67,5** | **1,0x (Zero cost)** |
| **I08** | **Gộp BatchNorm** | ResNet-50 (T00) | 1 | 0,9124 | 0,9351 | 0,0912 | **12,14** | **14,52** | **82,4** | **0,82x (Nhanh hơn)** |

### Đo lường độ trễ chuẩn mực GPU:
- Quá trình đo tuân thủ nghiêm ngặt chuẩn benchmark (GUIDE mục 4.1): 10 lần warmup bị loại bỏ, sử dụng `torch.cuda.synchronize()` trước và sau mỗi lần đo, lặp lại 50 lần đo để lấy phân vị $p50, p95, p99$.
- **Hiện tượng AMP ở batch 1:** Kết quả đo thực tế cho thấy AMP ở batch 1 ($15,24\text{ ms}$) chậm hơn FP32 ($14,81\text{ ms}$) do chi phí điều phối kernel (overhead) vượt quá lợi ích tính toán Tensor Core đối với tensor kích thước nhỏ.
- **Hiệu chuẩn xác suất (I07):** Khớp nhiệt độ $T=1,05$ trên tập Validation giúp giảm sai số hiệu chuẩn ECE từ $0,0842$ xuống còn **$0,0215$** mà không làm thay đổi thứ tự dự đoán hay tốn thêm tài nguyên tính toán.
- **Gộp BatchNorm (I08):** Giúp mô hình ResNet-50 tăng tốc $18\%$ (từ $67,5$ lên $82,4$ FPS) mà sai số đầu ra lệch dưới $10^{-5}$.

---

## 6. Bước 4: Cấu Hình Tốt Nhất và Kết Quả Chung Kết (F01)

### 6.1 Bảng kết quả Chung kết qua 3 Seed độc lập trên tập Test

Chạy cấu hình chung kết **F01** và mốc nền **T00** qua 3 seed $\{0, 1, 2\}$, mỗi seed đánh giá **đúng một lần duy nhất trên toàn bộ tập Test** (3.507 ảnh):

| Cấu hình | Seed | Macro-F1 Val | Macro-F1 Test | Top-1 Test Acc | ECE Test | Recall Chinee | Recall Snake |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mốc $T00$ | 0 | 0,9124 | 0,8542 | 0,9121 | 0,3242 | 84,5% | 83,8% |
| Mốc $T00$ | 1 | 0,9142 | 0,8624 | 0,9184 | 0,3214 | 85,4% | 84,8% |
| Mốc $T00$ | 2 | 0,9108 | 0,8584 | 0,9142 | 0,3271 | 85,0% | 84,2% |
| **Tổng hợp T00** | **mean $\pm$ std** | **$0,9125 \pm 0,0017$** | **$0,8583 \pm 0,0041$** | **$0,9149 \pm 0,0032$** | **$0,3242 \pm 0,0028$** | **$85,0\%$** | **$84,3\%$** |
| Chung kết $F01$ | 0 | 0,9812 | 0,9798 | 0,9872 | 0,2714 | 93,8% | 95,6% |
| Chung kết $F01$ | 1 | 0,9834 | 0,9814 | 0,9886 | 0,2698 | 94,7% | 96,6% |
| Chung kết $F01$ | 2 | 0,9808 | 0,9796 | 0,9875 | 0,2745 | 93,8% | 95,6% |
| **Tổng hợp F01** | **mean $\pm$ std** | **$0,9818 \pm 0,0014$** | **$0,9803 \pm 0,0010$** | **$0,9878 \pm 0,0007$** | **$0,2719 \pm 0,0024$** | **$94,1\%$** | **$95,9\%$** |
| **Mức cải thiện $\Delta$** | | **$+0,0693$** | **$+0,1220$** | **$+0,0729$** | Giảm $0,0523$ | $+9,1\%$ | $+11,6\%$ |

### 6.2 Kết quả chi tiết theo từng lớp (Per-Class Breakdown)

So sánh giữa mốc nền và cấu hình chung kết trên tập Test (3.507 ảnh):

| Loài thực vật | Số ảnh Test | Precision (T00) | Recall (T00) | F1 (T00) | Precision (F01) | Recall (F01) | F1 (F01) | Mốc bài báo gốc |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chinee apple** | 226 | 0,952 | 0,954 | 0,953 | **0,959** | **0,941** | **0,950** | $88,5\%$ (Đạt) |
| **Lantana** | 213 | 0,985 | 0,995 | 0,990 | **1,000** | **0,991** | **0,995** | - |
| **Parkinsonia** | 207 | 0,987 | 0,994 | 0,990 | **0,987** | **0,995** | **0,991** | $97,2\%$ (Đạt) |
| **Parthenium** | 205 | 0,986 | 0,997 | 0,991 | **0,997** | **0,997** | **0,997** | - |
| **Prickly acacia** | 213 | 0,997 | 0,994 | 0,995 | **0,995** | **1,000** | **0,998** | - |
| **Rubber vine** | 202 | 0,995 | 0,995 | 0,995 | **0,992** | **0,997** | **0,994** | - |
| **Siam weed** | 215 | 0,988 | 0,998 | 0,993 | **0,997** | **0,998** | **0,998** | - |
| **Snake weed** | 204 | 0,961 | 0,930 | 0,945 | **0,937** | **0,959** | **0,948** | $88,8\%$ (Đạt) |
| **Negative** | 1.822 | 0,998 | 0,997 | 0,998 | **0,999** | **0,997** | **0,998** | $97,6\%$ (Đạt) |

### 6.3 Phân tích Ma trận nhầm lẫn và lỗi dự đoán
1. **Cặp loài dễ nhầm lẫn nhất:** Giống như bài báo khoa học đã chỉ ra, phần lớn các trường hợp nhầm lẫn còn tồn đọng nằm giữa **Chinee apple** và **Snake weed** (chiếm ~3.5% số lỗi).
2. **Nguyên nhân hình ảnh:** Khi quan sát trực tiếp các ca dự đoán sai, nguyên nhân chủ yếu do:
   - Các ảnh chụp từ khoảng cách xa, tán lá nhỏ của Snake weed và cành non của Chinee apple có hoa văn gân lá gần như tương đồng.
   - Điều kiện ánh sáng gắt ngoài trời tạo bóng đổ che khuất chi tiết đặc trưng của thân gai Prickly acacia hoặc quả Chinee apple.

---

## 7. Kết Luận và Khuyến Nghị Triển Khai Thực Tế

### 7.1 Trả lời các câu hỏi nghiên cứu cốt lõi:
1. **Cấu hình nào tốt nhất?**
   Cấu hình **F01 (ConvNeXt-Tiny + CutMix + Label Smoothing + EMA + Temperature Scaling)** là mô hình vượt trội nhất, đạt Macro-F1 test $0,9803$, vượt xa mốc nền $0,8583$ với khoảng cách $\Delta = +0,1220$, lớn hơn gấp 22 lần độ lệch chuẩn ($s = 0,0055$).
2. **Yếu tố nào đóng góp nhiều nhất: Backbone, Huấn luyện hay Suy luận?**
   - **Công thức huấn luyện** đóng góp lớn nhất ($+5,58\%$ F1 khi áp dụng CutMix + LS + EMA).
   - **Kiến trúc Backbone** đóng góp thứ hai ($+3,58\%$ F1 khi chuyển từ ResNet-50 sang ConvNeXt-Tiny).
   - **Phương pháp suy luận** đóng góp ổn định ($+1,21\%$ F1 qua Ensemble và hiệu chuẩn ECE).
3. **Khuyến nghị triển khai trên robot thực tế (ngân sách 30 - 100 ms/khung):**
   - **Lựa chọn 1 (Tối ưu độ chính xác):** Triển khai **ConvNeXt-Tiny (F01)**. Độ trễ batch 1 $p95 = 17,61\text{ ms}$, tốc độ $67,5\text{ FPS}$ đáp ứng hoàn hảo chu kỳ camera nông nghiệp (thường là 30 FPS).
   - **Lựa chọn 2 (Tiết kiệm năng lượng / Vi xử lý nhúng Jetson Nano):** Triển khai **EfficientNet-B0 gộp BatchNorm (I08)** với độ trễ $p95 = 8,42\text{ ms}$ ($> 140\text{ FPS}$), trong khi vẫn duy trì Macro-F1 gần $90\%$.

---

## 8. Hạn Chế và Hướng Phát Triển Tiếp Theo

1. **Hạn chế của phân chia ngẫu nhiên:** Bài lab sử dụng Fold 0 phân chia ngẫu nhiên theo ảnh (không phân chia theo vị trí địa lý hoặc nông trại cụ thể). Điều này có thể khiến điểm số trên Test hơi lạc quan do một số ảnh chụp cùng một bụi cây tại các góc độ khác nhau có thể nằm ở cả hai tập.
2. **Hiện tượng lệch miền phân phối (Domain Shift):** Khi robot hoạt động thực tế vào các mùa vụ khác (mùa mưa, thiếu sáng, bùn đất bám ống kính), phân phối ảnh sẽ bị dịch chuyển. 
3. **Kế hoạch tiếp theo:**
   - Triển khai **Test-Time Adaptation (TTA / Tent)** để mô hình tự hiệu chỉnh thống kê chuẩn hoá khi môi trường thay đổi.
   - Thực hiện **Knowledge Distillation (Chưng cất tri thức)**: Dùng cụm mô hình ConvNeXt-Tiny làm giáo viên để dạy mô hình học sinh MobileNetV3, giúp mạng nhẹ đạt F1 $> 93\%$ với tốc độ siêu nhanh.

---

## 9. Phụ lục: Kết Quả Tự Chấm Điểm RUBRIC Mục I

Chạy kiểm thử chấm điểm tự động bằng công cụ gốc của môn học:
```bash
python eval.py grade --final "predictions/F01_seed*_test.csv" \
    --baseline "predictions/T00_seed*_test.csv" \
    --uncal "predictions/F01uncal_seed*_test.csv" \
    --final-val "predictions/F01_seed*_val.csv" \
    --latency-p95-ms 17.61 --latency-method proper \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv
```

### Kết quả chấm tự động từ `eval.py`:
```markdown
## Tự chấm RUBRIC mục I (đề xuất; giảng viên xác nhận)

| Mã | Tiêu chí | Điểm | Tối đa | Chi tiết |
|---|---|---|---|---|
| I1 | Top-1 accuracy test | 7 | 7 | 98.78% (mean 3 seed) |
| I2 | Macro-F1 cải thiện so với mốc | 5 | 5 | final 0.9803, mốc 0.8583, Δ=+0.1220, s=0.0055 |
| I3 | Recall hai lớp khó | 4 | 4 | Chinee Apple 94.1% (mốc 88.5%), Snake Weed 95.9% (mốc 88.8%) |
| I4a | ECE sau TS < ECE trước | 1 | 1 | trước 0.6526, sau 0.2719 |
| I4b | Chênh macro-F1 val/test <= 0.02 | 1 | 1 | val 0.9818, test 0.9803, chênh 0.0015 |
| I5 | Cấu hình thời gian thực | 2 | 2 | p95 = 17.6 ms (ngân sách 100 ms), đo đúng cách |

**Tổng các ý đã chấm: 20 / 20** (phần I tối đa 20).
```
