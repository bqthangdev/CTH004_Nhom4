# Ghi nhớ hiện trạng

Cập nhật 15/10/2026.

## Đã chốt

Đường làm là BIPN, SIGIR 2024, arXiv 2404.18166. Khung nằm ở `bipn/`.

Ba lệnh mô hình: `--model bipn`, `--model weighted`, `--model lightgcn`. Bộ một là Tmall. Bộ hai là Retailrocket, chuyển bằng `bipn/prepare_retailrocket.py`. Plan: `docs/plan-bipn.md`. File plan ngày 15/10/2026 đã được gom vào file này.

Báo cáo giữ mẫu LNCS, file `paper/main.tex`. Phần lý thuyết đã viết. Bảng HR và NDCG vẫn trống. Tên tạm là NVA và NVB. Email và tên trường đầy đủ chưa có.

## Máy train

Máy đang mở repo, đã kiểm bằng lệnh ngày 12/10/2026: RTX 3050 Ti Laptop, 4096 MiB, driver 617.42, RAM 31,7 GB. `bipn/.venv` là torch 2.14.1+cpu và NumPy 2.5.4.

Thành viên còn lại cho biết máy của họ là Idealpad 5 Pro, Core Ultra 9 185H, RTX 4050, 32 GB RAM. Đây là lời kể, chưa kiểm trên máy đó.

Chưa chọn máy nào giữ bản train dài. Chưa cài PyTorch CUDA.

## Dữ liệu

Bộ BIPN đã giải nén tại `data/Tmall`, `data/taobao`, `data/Yelp`, `data/ML10M`. Tmall: 41738 user, 11953 item. Retailrocket đã giải nén và chuyển tại `data/retailrocket`: 11719 user, 31839 item, train view 88304, cart 22716, buy 8666, test 11719 dòng, validation 885 dòng. Hướng dẫn tải nằm ở `data/README.md`. File dữ liệu không đưa vào git.

## Chưa làm

Chưa train đủ epoch. Bảng HR và NDCG trong báo cáo vẫn trống. Chưa commit.

## Nhật ký

### 2026-10-12

- Đối chiếu BIPN, CRGCN và MBHT. Chọn BIPN làm hướng viết plan.
- Tạo khung `bipn/`, mẫu LNCS và `docs/plan-bipn.md`.
- Cài `bipn/.venv` với torch 2.14.1+cpu. `--smoke` cho loss 0.5902 rồi 0.5755 trên dữ liệu giả.
- Đếm bốn bộ trong `data.zip`. Đề xuất Retailrocket. Ghi `docs/cau-hoi-can-xac-nhan.md`.
- Ghi máy RTX 3050 Ti đã kiểm, và máy RTX 4050 theo lời thành viên.

### 2026-10-15

- Chốt BIPN và Retailrocket.
- Viết `docs/plan-trien-khai-2026-10-15.md` trước khi sửa code.
- Thêm `--model weighted` và `--model lightgcn`. Thêm `prepare_retailrocket.py`.
- Viết phần lý thuyết trong `paper/main.tex`. PDF biên dịch lại còn 4 trang. Bảng kết quả vẫn để trống.
- `--smoke` trên dữ liệu giả: BIPN loss 0.5902 rồi 0.5755, weighted 0.5894 rồi 0.5748, LightGCN 0.5645 rồi 0.5726. Bộ chuyển Retailrocket đã thử trên file CSV nhỏ, đúng phép chia theo thời gian.
- Giải nén zip BIPN vào `data/`. Thư mục này nằm trong `.gitignore`.
- Giải nén Retailrocket từ `data/archive.zip` vào `data/retailrocket/` và chạy `prepare_retailrocket.py`.
- Chưa chạy Tmall hay Retailrocket đủ epoch.
