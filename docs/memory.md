# Ghi nhớ hiện trạng

Cập nhật 24/10/2026.

## Đã chốt

Đường làm là BIPN, SIGIR 2024, arXiv 2404.18166. Khung nằm ở `bipn/`. Code gốc nằm ở `official-bipn/`.

Ba lệnh mô hình: `--model bipn`, `--model weighted`, `--model lightgcn`. Bộ một là Tmall. Bộ hai là Retailrocket, chuyển bằng `bipn/prepare_retailrocket.py`. Plan: `docs/plan-bipn.md`.

Báo cáo là LNCS, `paper/main.tex`. Tác giả trên bài: Huỳnh Chí Dũng và Bành Quốc Thắng, Trường Công nghệ Thông tin, Đại học Cần Thơ. Bảng kết quả đã có sáu dòng. PDF biên dịch lại ngày 23/10/2026: 16 trang, 615544 byte. `README.md` và `docs/cau-hoi-can-xac-nhan.md` vẫn ghi tên tạm NVA, NVB.

## Máy train

Máy đang mở repo, đã kiểm bằng lệnh ngày 12/10/2026: RTX 3050 Ti Laptop, 4096 MiB, driver 617.42, RAM 31,7 GB. `bipn/.venv` trên máy này lúc kiểm là torch 2.14.1+cpu.

Bản train dài do Huỳnh Chí Dũng chạy trên RTX 4050 Laptop, 6 GB, Core Ultra 9 185H, 32 GB RAM, torch 2.14.1+cu130. Đây là cấu hình ghi trong `docs/bao-cao-tong-ket-buoc-4.md`.

Máy đang mở repo đã cài `torch 2.14.1+cu130` vào `bipn/.venv` ngày 24/10/2026. GPU là RTX 3050 Ti Laptop, 4096 MiB. Batch trong code vẫn là 1024. Một epoch `--model learned` trên Tmall mất 29 phút 49 giây. Lần thử batch 4096 hết 14 phút 14 giây và không được giữ. Không còn checkpoint của `--model learned`.

## Dữ liệu

Bộ BIPN đã giải nén tại `data/Tmall`, `data/taobao`, `data/Yelp`, `data/ML10M`. Tmall: 41738 user, 11953 item. Retailrocket đã giải nén và chuyển tại `data/retailrocket`: 11719 user, 31839 item, train view 88304, cart 22716, buy 8666, test 11719 dòng, validation 885 dòng. Hướng dẫn tải nằm ở `docs/huong-dan-chay.md`. File dữ liệu không đưa vào git.

## Số đã ghi

Tmall BIPN gốc, epoch 33: HR@10 0,1398, NDCG@10 0,0736. Bài báo công bố 0,1414 và 0,0741. Gần, không trùng.

Tmall có trọng số, epoch 4: 0,0071 và 0,0032. Tmall LightGCN, epoch 20: 0,0141 và 0,0070.

Retailrocket BIPN, epoch 1: 0,1275 và 0,0901. Có trọng số, epoch 1: 0,1229 và 0,0865. LightGCN, epoch 1: 0,0003 và 0,0001.

Log code gốc nằm trong `official-bipn/log/`. Log và checkpoint của pipeline nằm ngoài git (`bipn/logs/`, `bipn/checkpoints/`).

## Còn lại

`README.md` và `docs/cau-hoi-can-xac-nhan.md` chưa đổi tên tạm. Bảng kết quả trong PDF tràn lề khoảng 61 pt. Ba mô hình Retailrocket giữ epoch 1. Mẫu LNCS hay mẫu tạp chí vẫn là câu mở nếu giảng viên yêu cầu mẫu tạp chí. PDF mới chưa commit.

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
- Kết quả test: HR@10 = 0.1398, NDCG@10 = 0.0736. Bài báo là 0.1414 và 0.0741. Checkpoint lưu tại `official-bipn/check_point/model2026-10-17 13_56_43.pth`.
- Tạo báo cáo tổng kết chi tiết tại `docs/bao-cao-tong-ket-buoc-4.md`. Sẵn sàng chạy Mục 5 (pipeline).

### 2026-10-18

- Chạy kiểm tra 1-epoch trên 5 mô hình (`run_pipeline.py --epochs 1`): cả 5 bước đều in `done`.
- Chạy huấn luyện toàn bộ pipeline Mục 5 (`run_pipeline.py`). Tất cả 5 mô hình đều hoàn thành và hội tụ (Early Stopping `patience=5`).
- Kiểm tra lại checkpoint bằng `run_check.py`: tất cả các cặp chỉ số `saved` và `reload` đều khớp chính xác đến 16 chữ số thập phân.
- Đã có đầy đủ 6/6 cặp số đo cho Bảng 2 của bài báo (`paper/main.tex`).
- Hoàn thành Mục 6: Điền đầy đủ 6 cặp số đo vào Bảng 2 của `paper/main.tex`, cập nhật Tóm tắt, Kết quả phân tích và Kết luận.
- Tạo tài liệu hướng dẫn và cẩm nang toàn diện viết báo cáo và bảo vệ tại `docs/cam-nang-viet-bao-cao-va-bao-ve.md`.
- Commit `c0bec20`: log Tmall của lần chạy nhóm, điền bảng, thêm biểu đồ và `docs/phan-tich-chuyen-sau-truc-quan-hoa.md`.

### 2026-10-20

- Commit `253a451`: viết lại `paper/main.tex`, thêm tên và email hai tác giả, bổ sung `paper/refs.bib`.
- Chỉnh lại bộ hình song ngữ. Chuyển script vẽ sang `visualization/`.

### 2026-10-23

- Đối chiếu bảy commit trên `main`. Cập nhật phần đầu file này cho khớp bản đã train.
- Biên dịch lại `paper/main.pdf`: 16 trang, 615544 byte. Không còn citation undefined. Bảng kết quả vẫn overfull khoảng 61 pt.

### 2026-10-24

- Viết lại `docs/cai-tien.md`. Mỗi hướng có vấn đề, giả thuyết, chỗ sửa và cách kiểm tra. Chỉ hướng trọng số BCE là đã chạy.
- Ghi rõ hướng 1 đến 4 xuất phát từ đếm dòng của bộ đang dùng. Chuẩn giảng viên yêu cầu là tổng hợp từ các bài trước, rồi mới sửa phương pháp.
- Đề xuất lại trong `docs/cai-tien.md`: sửa λ của nhánh GCN theo mục 4.4 và kết luận của BIPN, có đối chiếu MBGCN và CRGCN. Chưa cài, chưa đo.
- Hai câu trong kết luận bài nhóm: trọng số học được có nguồn từ MBGCN nhưng chưa chỉ chỗ sửa; chuỗi thời gian là MBHT, không phải một sửa nhỏ trong BIPN, và file Tmall không còn thời điểm.
- Phân tích ghép trọng số học với hướng 2, 3, 4 và với công thức λ. Hướng 2 và 3 trùng việc. Hướng 4 và λ ghép được. Chưa cài.
- Cài `torch 2.14.1+cu130` trên máy RTX 3050 Ti. Batch trong code vẫn là 1024. Một epoch Tmall `--model learned` mất 29 phút 49 giây. Lần thử batch 4096 không được giữ và checkpoint của lần đó đã xóa.
- Cài `--model learned`: mỗi hành vi một logit học được trong BCE, `alpha = softmax(log n_b + logit)`, `w_b = alpha_b * N / n_b`. Chỉ sửa `_bce_loss`, không thêm vào `run_pipeline.py`, không đổi số đã ghi.
- Kiểm tra: logit 0 cho mọi `w_b` = 1; tăng logit một hành vi thì w của nó tăng và w còn lại giảm; `behavior_logit.grad` khác None. `--smoke` vẫn chạy cho bipn, weighted và learned. Chưa có HR/NDCG của learned.
