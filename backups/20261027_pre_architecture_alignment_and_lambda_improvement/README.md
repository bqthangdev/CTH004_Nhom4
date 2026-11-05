# Thông Tin Sao Lưu Kết Quả Huấn Luyện (Đợt 2)

* **Thư mục sao lưu:** `backups/20261027_pre_architecture_alignment_and_lambda_improvement/`
* **Thời gian thực hiện:** 27/10/2026 00:46 (Giờ địa phương)
* **Nhánh Git:** `stop_and_continue_run`
* **Môi trường phần cứng:** GPU NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM), PyTorch 2.14.1+cu130

---

## 1. Lý do thực hiện sao lưu (Backup)

1. **Kết quả chạy mô hình `--model learned` trên khung `bipn/` trước khi đồng bộ kiến trúc:**
   - Đạt **HR@10 = 0.0089**, **NDCG@10 = 0.0045** tại Epoch 1 và dừng sớm tại Epoch 6.
   - Tuy cao hơn 25%–40% so với bản `weighted` cũ (0.0071 / 0.0032), nhưng vẫn thấp hơn nhiều so với bài báo gốc (0.1414 / 0.0741).
   - **Nguyên nhân phát hiện được:** Khung `bipn/` viết lại có 5 sai khác kỹ thuật so với `official-bipn/` (hàm phạt L2 mạnh gấp ~40.000 lần không chia theo số node, thiếu `F.normalize` giữa các tầng LightGCN, thiếu residual connection ở nhánh đồ thị hành vi đích, lấy mẫu BPR chỉ 1 cặp/user/epoch lặp lại 285 lần, và hệ số `N / n_b` phóng đại trọng số `cart` lên 370 lần).
2. **Kết quả chạy thử nghiệm `official-bipn` với `--patience 10`:**
   - Đạt **HR@10 = 0.1410**, **HR@20 = 0.2032** tại Epoch 47 (đỉnh Validation `hit@20 = 0.3111`), và đạt đỉnh Test **HR@10 = 0.1431**, **NDCG@10 = 0.0743** tại Epoch 56.
   - Theo định hướng của nhóm: Giữ nguyên thư mục `official-bipn/` như trạng thái ban đầu (chỉ dùng làm minh chứng tái lập bài báo gốc), nên sao lưu lượt chạy này vào đây và khôi phục `official-bipn/` về nguyên trạng.

---

## 2. Bảng tổng hợp số đo trong bản sao lưu này

| Phân hệ | Bộ dữ liệu | Mô hình | Best Epoch | HR@10 | NDCG@10 | HR@20 | NDCG@20 | Ghi chú |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|
| `bipn/` (Chưa đồng bộ) | **Tmall** | `learned` (Softmax BCE + $N/n_b$) | Epoch 1 (dừng ở 6) | 0.0089 | 0.0045 | 0.0132 | 0.0055 | Cao hơn `weighted` (0.0071), nhưng bị ghìm bởi 5 sai khác kỹ thuật trong `bipn/` |
| `official-bipn/` | **Tmall** | BIPN gốc (`--patience 10`) | Epoch 47 (dừng ở 57) | 0.1410 (Test peak Ep56: 0.1431) | 0.0732 (Test peak Ep56: 0.0743) | 0.2032 (Test peak Ep56: 0.2050) | 0.0884 (Test peak Ep56: 0.0895) | Minh chứng tái lập vượt bài báo gốc; đã khôi phục `official-bipn` về nguyên trạng |

---

## 3. Danh mục các tệp được lưu trữ

* `bipn_checkpoints/`:
  * `tmall_learned.pt` (~97.9 MB) — Checkpoint Epoch 1 của `learned` cũ.
  * `tmall_learned.resume.pt` (~125.6 MB) — Trạng thái resume tại Epoch 6.
* `bipn_logs/`:
  * `tmall_learned.log` — Nhật ký 6 epoch của `tmall_learned`.
* `official_bipn_rerun/`:
  * `model2026-10-26 01_09_33.pth` (~13.8 MB) — Checkpoint Epoch 47 của lần chạy `patience=10`.
  * `tmall_enb_64_2026-10-26 01_09_33.log` — Nhật ký đầy đủ 57 epoch.
