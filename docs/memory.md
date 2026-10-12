# Ghi nhớ hiện trạng

Cập nhật 12/10/2026.

## Đã chốt

Đường làm là BIPN, SIGIR 2024, arXiv 2404.18166. Khung nằm ở `bipn/`. Plan ở `docs/plan-bipn.md`. So sánh phương án ở `docs/de-xuat-topic5-multi-behavior.md`. Câu chưa xác nhận: `docs/cau-hoi-can-xac-nhan.md`.

Báo cáo giữ mẫu LNCS, lớp `llncs` 2.26, file `paper/main.tex`. Slide chỉ ghi mẫu Springer. Có thể hỏi giảng viên về mẫu tạp chí. Chưa đổi.

Tên tạm trên báo cáo là NVA và NVB, cùng Khoa HTTT. Email và tên trường đầy đủ chưa có.

Hành vi cuối là hành vi đích. Bộ một là Tmall: click, collect, cart, buy. Bộ hai đã ghi nhận là Retailrocket, chờ thành viên còn lại xác nhận. Bảng so với bốn bộ trong zip BIPN nằm ở `docs/plan-bipn.md`.

## Máy train

Máy đang mở repo, đã kiểm bằng lệnh: RTX 3050 Ti Laptop, 4096 MiB, driver 617.42, RAM 31,7 GB. `bipn/.venv` là torch 2.14.1+cpu và NumPy 2.5.4, chưa dùng GPU.

Thành viên còn lại cho biết máy của họ là Idealpad 5 Pro, Core Ultra 9 185H, RTX 4050, 32 GB RAM. Đây là lời kể, chưa kiểm trên máy đó.

Chưa chọn máy nào giữ bản train dài. Colab không còn là phương án chính vì cả hai máy đều có GPU.

## Máy và lệnh

Trong `bipn`:

```text
.venv\Scripts\python.exe main.py --smoke
```

Lần chạy 12/10/2026: epoch 1 loss 0.5902, epoch 2 loss 0.5755. Dữ liệu giả 40 user, không phải số của bài báo.

Loader và metric đã kiểm riêng: đọc một thư mục tạm đúng định dạng, từ chối id ngoài phạm vi, HR@1 của một ma trận biết trước bằng 0.5.

## Dữ liệu

`data.zip` của repo BIPN nằm ở `%TEMP%\bipn_data.zip`, 47.108.508 byte, chưa copy vào git. `count.txt`: Tmall 41738 user và 11953 item, Taobao 48749 và 39493, Yelp 19800 và 22734, ML10M 67788 và 8704. Khung không cần `pos_sampling.txt`. Retailrocket chưa tải.

## Chưa làm

Chưa train Tmall. Chưa tải Retailrocket. Chưa thêm trọng số hành vi. Chưa có script LightGCN đối chứng. Bảng trong báo cáo vẫn để trống. Chưa commit.
