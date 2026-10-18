# Nguồn và các chỗ đã sửa

Mã lấy từ https://github.com/MingshiYan/BIPN, commit `31f3a7c09580440aa55ccefa157c94e1d0fe30a6`, ngày 16/10/2026.

Đây là code gốc để chạy lại trên Tmall. Cách chạy nằm ở [../docs/huong-dan-chay.md](../docs/huong-dan-chay.md), mục 4.

## Đã sửa

| File | Sửa |
|---|---|
| `metrics.py` | `np.float` đổi thành `np.float64` ở năm chỗ. NumPy hiện tại không còn tên `np.float`. Dòng `np.float64(preds)` giữ nguyên. |
| `data_set.py` | `torch.cuda.manual_seed` chỉ chạy khi máy có CUDA. Chuyển `all_dict` sang `set` để tra cứu mẫu âm $O(1)$ thay vì gọi `np.isin` $O(N)$ 11 triệu lần/epoch. |
| `main.py` | Dữ liệu Tmall trỏ tới `../data/Tmall` của repo nhóm, không dùng `./data/Tmall`. Bật `cudnn.benchmark = True`. |
| `b_tmall.sh` | Thiết bị đổi từ `cuda:6` sang `cuda:0`. |
| `lightGCN.py` | `A._update` đổi thành kiểm tra `hasattr(A, '_update')` hoặc gọi `A.update` để tương thích với SciPy mới. Đưa `self.A_adj_matrix` lên GPU một lần duy nhất lúc khởi tạo thay vì gọi `.to(self.device)` 8.668 lần qua bus PCIe mỗi epoch. |
| `trainer.py` | Bật `pin_memory=True` và `non_blocking=True` trong DataLoader để nạp batch trực tiếp vào VRAM GPU. |

## File nhóm thêm

`run_tmall.ps1` gọi `main.py` với đúng tham số của `b_tmall.sh` sau khi sửa thiết bị. Máy nhóm dùng PowerShell, không chạy file `.sh`.

## Không đưa vào bản này

`data.zip` của repo gốc không giữ ở đây. Dữ liệu Tmall nằm ở `data/Tmall` và được tải theo hướng dẫn chạy.

## Log có sẵn

Bốn file trong `log/model/` là log tác giả đã đẩy lên GitHub. Không dùng các file đó để điền báo cáo. Lần chạy của nhóm tạo file log mới trong cùng thư mục.
