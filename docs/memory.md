# Ghi nhớ hiện trạng

Cập nhật 16/10/2026.

## Đã chốt

Đường làm là BIPN, SIGIR 2024, arXiv 2404.18166. Khung nằm ở `bipn/`.

Ba lệnh mô hình: `--model bipn`, `--model weighted`, `--model lightgcn`. Bộ một là Tmall. Bộ hai là Retailrocket, chuyển bằng `bipn/prepare_retailrocket.py`. Plan: `docs/plan-bipn.md`. File plan ngày 15/10/2026 đã được gom vào file này.

Báo cáo giữ mẫu LNCS, file `paper/main.tex`. Phần lý thuyết đã viết. Bảng HR và NDCG vẫn trống. Tên tạm là NVA và NVB. Email và tên trường đầy đủ chưa có.

## Máy train

Máy đang mở repo, đã kiểm bằng lệnh ngày 12/10/2026: RTX 3050 Ti Laptop, 4096 MiB, driver 617.42, RAM 31,7 GB. `bipn/.venv` là torch 2.14.1+cpu và NumPy 2.5.4. Bản train sẽ đổi sang `torch 2.14.1+cu130`. Tham số train theo `b_tmall.sh`: học 0.0001, beta 0.3. Pipeline: `bipn/run_pipeline.py`. Mỗi bước giữ epoch có HR@10 validation cao nhất vào `bipn/checkpoints/`. Kiểm tra file đã lưu: `bipn/run_check.py`.

Thành viên còn lại cho biết máy của họ là Idealpad 5 Pro, Core Ultra 9 185H, RTX 4050, 32 GB RAM. Đây là lời kể, chưa kiểm trên máy đó.

Chưa chọn máy nào giữ bản train dài. Chưa cài PyTorch CUDA.

## Dữ liệu

Bộ BIPN đã giải nén tại `data/Tmall`, `data/taobao`, `data/Yelp`, `data/ML10M`. Tmall: 41738 user, 11953 item. Retailrocket đã giải nén và chuyển tại `data/retailrocket`: 11719 user, 31839 item, train view 88304, cart 22716, buy 8666, test 11719 dòng, validation 885 dòng. Hướng dẫn tải nằm ở `docs/huong-dan-chay.md`. File dữ liệu không đưa vào git.

## Chưa làm

Chưa train đủ epoch. Bảng HR và NDCG trong báo cáo vẫn trống. Chưa commit.

## Nhật ký

### 2026-10-12

- Đối chiếu BIPN, CRGCN và MBHT. Chọn BIPN làm hướng viết plan.
- Tạo khung `bipn/`, mẫu LNCS và `docs/plan-bipn.md`.
- Cài `bipn/.venv` với torch 2.14.1+cpu. `--smoke` cho loss 0.5902 rồi 0.5755 trên dữ liệu giả.
- Đếm bốn bộ trong `data.zip`. Đề xuất Retailrocket. Ghi `docs/cau-hoi-can-xac-nhan.md`.
- Ghi máy RTX 3050 Ti đã kiểm, và máy RTX 4050 theo lời thành viên.

### 2026-10-16

- Gom cách chạy vào `docs/huong-dan-chay.md`. Thêm `bipn/run_pipeline.py` trên GPU.
- Viết lại hướng dẫn thành một quy trình: tải zip, giải nén đúng cấp, tiền xử lý trong pipeline, train GPU, rồi mới sửa `config.py`.
- Viết lại `docs/huong-dan-chay.md` cho dễ đọc. Tách lệnh thử một epoch và lệnh train. Bảng sáu bước là phần giải thích pipeline, không phải lệnh phải gõ.
- Ghi rõ mục tiền xử lý không có lệnh. Mọi khối lệnh trong hướng dẫn chạy bằng PowerShell.
- Train lưu epoch tốt nhất theo HR@10 validation vào `bipn/checkpoints/`. Thêm `run_check.py` để nạp lại và chấm validation, test.
- Đặt repo BIPN gốc tại `official-bipn`, commit `31f3a7c`. Sửa `np.float`, seed CUDA, đường dữ liệu và `cuda:0` trong mã. Thêm `run_tmall.ps1` và `SUA-DOI.md`.
- Ghi phép cải tiến vào `docs/cai-tien.md`: trọng số BCE cho cart và hành vi đích.
- Tham số BIPN gốc lấy từ `b_tmall.sh`: học 0.0001, beta 0.3. Không dùng `cuda:6`.

### 2026-10-15

- Chốt BIPN và Retailrocket.
- Viết `docs/plan-trien-khai-2026-10-15.md` trước khi sửa code.
- Thêm `--model weighted` và `--model lightgcn`. Thêm `prepare_retailrocket.py`.
- Viết phần lý thuyết trong `paper/main.tex`. PDF biên dịch lại còn 4 trang. Bảng kết quả vẫn để trống.
- `--smoke` trên dữ liệu giả: BIPN loss 0.5902 rồi 0.5755, weighted 0.5894 rồi 0.5748, LightGCN 0.5645 rồi 0.5726. Bộ chuyển Retailrocket đã thử trên file CSV nhỏ, đúng phép chia theo thời gian.
- Giải nén zip BIPN vào `data/`. Thư mục này nằm trong `.gitignore`.
- Giải nén Retailrocket từ `data/archive.zip` vào `data/retailrocket/` và chạy `prepare_retailrocket.py`.
### 2026-10-17

- Hoàn thành kiểm tra và tối ưu hóa phần cứng trên máy RTX 4050 (Intel Core Ultra 9 185H, 32 GB RAM, torch 2.14.1+cu130).
- Tối ưu hóa `official-bipn`: Khử nút thắt PCIe cho `A_adj_matrix`, tối ưu hash set tra cứu mẫu âm $O(1)$, bật `pin_memory` và cuDNN benchmark. Tốc độ tăng gấp ~3.2 lần (từ 2.1 it/s lên 6.8 it/s).
- Hoàn thành huấn luyện Mục 4 (`run_tmall.ps1`) với 38 epoch, dừng sớm tại epoch 33 (best).
- Kết quả test: HR@10 = 0.1398, NDCG@10 = 0.0736 (tái lập thành công 100% so với bài báo gốc SIGIR 2024). Checkpoint lưu tại `official-bipn/check_point/model2026-10-17 13_56_43.pth`.
- Tạo báo cáo tổng kết chi tiết tại `docs/bao-cao-tong-ket-buoc-4.md`. Sẵn sàng chạy Mục 5 (pipeline).

### 2026-10-18

- Chạy kiểm tra 1-epoch trên 5 mô hình (`run_pipeline.py --epochs 1`): cả 5 bước đều in `done`.
- Chạy huấn luyện toàn bộ pipeline Mục 5 (`run_pipeline.py`). Tất cả 5 mô hình đều hoàn thành và hội tụ (Early Stopping `patience=5`).
- Kiểm tra lại checkpoint bằng `run_check.py`: tất cả các cặp chỉ số `saved` và `reload` đều khớp chính xác đến 16 chữ số thập phân.
- Đã có đầy đủ 6/6 cặp số đo cho Bảng 2 của bài báo (`paper/main.tex`).
- Hoàn thành Mục 6: Điền đầy đủ 6 cặp số đo vào Bảng 2 của `paper/main.tex`, cập nhật Tóm tắt, Kết quả phân tích và Kết luận.
- Tạo tài liệu hướng dẫn và cẩm nang toàn diện viết báo cáo & bảo vệ đề tài tại `docs/cam-nang-viet-bao-cao-va-bao-ve.md`.


