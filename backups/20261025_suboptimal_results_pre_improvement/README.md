# Thông Tin Sao Lưu Kết Quả Huấn Luyện (Backup)

* **Thư mục sao lưu:** `backups/20261025_suboptimal_results_pre_improvement/`
* **Thời gian thực hiện:** 25/10/2026 22:59 (GMT+7 / Giờ địa phương)
* **Người thực hiện:** Hệ thống AI hỗ trợ nhóm CTH004_Nhom4
* **Nhánh Git:** `stop_and_continue_run`
* **Môi trường phần cứng:** GPU NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM), PyTorch 2.14.1+cu130

---

## 1. Lý do thực hiện sao lưu (Backup)

Kết quả huấn luyện của các lượt chạy trước đó ghi nhận trong hệ thống chưa đạt kỳ vọng và có sự sụt giảm so với bài báo gốc (SIGIR 2024), cụ thể:

1. **Mô hình BIPN gốc trên tập Tmall (`official-bipn`)**:
   - Kết quả thực nghiệm nhóm đạt: **HR@10 = 0.1398**, **NDCG@10 = 0.0736** (dừng tại epoch 33).
   - Kết quả công bố trong bài báo gốc: **HR@10 = 0.1414**, **NDCG@10 = 0.0741**.
   - Mặc dù đã tái lập tương đối sát, kết quả vẫn thấp hơn bài báo gốc, cần được huấn luyện lại hoặc tinh chỉnh để tối ưu hóa.

2. **Mô hình cải tiến có trọng số cố định (`--model weighted`)**:
   - Trên tập Tmall: Chỉ đạt **HR@10 = 0.0071**, **NDCG@10 = 0.0032** (sụt giảm nghiêm trọng ~94.9% so với mô hình gốc).
   - Nguyên nhân: Việc nhân hệ số cố định 2.0 cho `cart` và `buy` trong hàm mất mát BCE đã làm phá vỡ cân bằng phân phối gradient khi dữ liệu Tmall quá thưa thớt và lệch pha giữa các hành vi (tỷ lệ `click` so với `cart` là 908:1).
   - Trên tập Retailrocket: HR@10 cũng bị suy giảm từ 0.1275 xuống 0.1229.

3. **Hiện tượng dừng sớm quá nhanh trên Retailrocket**:
   - Cả 3 mô hình (`bipn`, `weighted`, `lightgcn`) đều kích hoạt dừng sớm ngay từ **Epoch 1** do tập xác thực (validation) quá nhỏ (885 dòng), khiến mô hình chưa kịp học sâu đã dừng lại.

Vì các lý do trên, việc sao lưu toàn bộ kết quả cũ là bắt buộc nhằm:
- Lưu trữ bằng chứng đối chứng lịch sử trước khi cải tiến/tinh chỉnh lại.
- Làm sạch các thư mục `checkpoints/` và `logs/` trong môi trường làm việc để không bị xung đột khi tiến hành huấn luyện mới.

---

## 2. Bảng tổng hợp số đo trước khi sao lưu

| Bộ dữ liệu | Mô hình | Best Epoch | HR@10 | NDCG@10 | HR@20 | NDCG@20 | Đánh giá |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Tmall** | **BIPN (Gốc - Bài báo)** | - | **0.1414** | **0.0741** | **0.2033** | **0.0898** | *Mục tiêu bài báo* |
| **Tmall** | **BIPN (Gốc - Chạy lại)** | Epoch 33 | 0.1398 | 0.0736 | 0.2000 | 0.0884 | *Thấp hơn bài báo* |
| **Tmall** | BIPN có trọng số (`weighted`) | Epoch 4 | 0.0071 | 0.0032 | 0.0127 | 0.0046 | *Sụt giảm nghiêm trọng* |
| **Tmall** | LightGCN (Đối chứng) | Epoch 20 | 0.0141 | 0.0070 | 0.0197 | 0.0083 | *Baseline* |
| **Retailrocket** | **BIPN** | Epoch 1 | 0.1275 | 0.0901 | 0.1515 | 0.0962 | *Dừng quá sớm ở Epoch 1* |
| **Retailrocket** | BIPN có trọng số (`weighted`) | Epoch 1 | 0.1229 | 0.0865 | 0.1482 | 0.0929 | *Kém hơn BIPN* |
| **Retailrocket** | LightGCN (Đối chứng) | Epoch 1 | 0.0003 | 0.0001 | 0.0007 | 0.0002 | *Dừng quá sớm ở Epoch 1* |

---

## 3. Danh mục các tệp được lưu trữ

### 3.1. Thư mục `bipn_checkpoints/` (Checkpoints mô hình pipeline)
* `retailrocket_bipn.pt` (~15.2 MB)
* `retailrocket_lightgcn.pt` (~11.5 MB)
* `retailrocket_weighted.pt` (~15.2 MB)
* `tmall_lightgcn.pt` (~21.0 MB)
* `tmall_weighted.pt` (~97.9 MB)

### 3.2. Thư mục `bipn_logs/` (Nhật ký quá trình train pipeline)
* `retailrocket_bipn.log`
* `retailrocket_lightgcn.log`
* `retailrocket_weighted.log`
* `tmall_lightgcn.log`
* `tmall_weighted.log`

### 3.3. Thư mục `official_bipn_checkpoint/` (Checkpoint mô hình gốc)
* `model2026-10-17 13_56_43.pth` (~13.8 MB)

### 3.4. Thư mục `official_bipn_logs/` (Nhật ký mô hình gốc)
* `tmall_enb_64_2026-10-17 13_56_43.log` (Lần chạy đạt HR@10 0.1398)
* Các log tham khảo khác từ tác giả gốc.
