# Hướng dẫn chạy

Chạy trong thư mục `bipn`, sau khi `data/Tmall` và `data/retailrocket` đã có `count.txt`. Cách chuẩn bị dữ liệu nằm ở [data/README.md](../data/README.md).

## Cài đặt

Python 3.12. Môi trường nằm tại `bipn/.venv`.

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Kiểm tra

`--smoke` dùng dữ liệu giả. Embedding bị đặt thành 8, một tầng, batch 32, một mẫu âm, hai epoch, thiết bị CPU. Lệnh không đọc thư mục `data/`.

```text
.venv\Scripts\python.exe main.py --smoke --model bipn
.venv\Scripts\python.exe main.py --smoke --model weighted
.venv\Scripts\python.exe main.py --smoke --model lightgcn
```

## Thử một epoch

```text
.venv\Scripts\python.exe main.py --data ../data/Tmall --dataset tmall --model bipn --epochs 1 --device cpu
```

Với bản PyTorch có CUDA, đổi `--device cuda`. Nếu không truyền `--device` và CUDA dùng được, lệnh chọn `cuda`.

## Sáu lần chạy

Giữ cùng bộ tham số. Chỉ đổi `--data`, `--dataset` và `--model`.

```text
.venv\Scripts\python.exe main.py --data ../data/Tmall --dataset tmall --model bipn --device cuda
.venv\Scripts\python.exe main.py --data ../data/Tmall --dataset tmall --model weighted --device cuda
.venv\Scripts\python.exe main.py --data ../data/Tmall --dataset tmall --model lightgcn --device cuda
.venv\Scripts\python.exe main.py --data ../data/retailrocket --dataset retailrocket --model bipn --device cuda
.venv\Scripts\python.exe main.py --data ../data/retailrocket --dataset retailrocket --model weighted --device cuda
.venv\Scripts\python.exe main.py --data ../data/retailrocket --dataset retailrocket --model lightgcn --device cuda
```

Không truyền `--epochs` thì mỗi lệnh chạy 200 epoch. Mỗi epoch in loss, rồi HR và NDCG của validation và test.

Chương trình không lưu checkpoint và không chọn epoch. Khi điền báo cáo, lấy epoch có HR@10 validation cao nhất. Nếu hai epoch bằng nhau, lấy epoch có NDCG@10 validation cao hơn. Số ghi vào bảng là HR@10 và NDCG@10 trên test của epoch đó.

## Tham số dòng lệnh

`main.py` chỉ đọc các cờ dưới đây.

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--data` | rỗng | Thư mục có `count.txt` và các file hành vi. Bắt buộc khi không dùng `--smoke`. |
| `--dataset` | `tmall` | Danh sách hành vi: `tmall`, `taobao`, `yelp`, `ml`, `retailrocket`. |
| `--model` | `bipn` | `bipn`, `weighted` hoặc `lightgcn`. |
| `--epochs` | 200 | Số epoch. Số lớn hơn 0 thì ghi đè mặc định. |
| `--device` | `cuda` nếu có, không thì `cpu` | Thiết bị train và chấm điểm. |
| `--smoke` | tắt | Chạy dữ liệu giả với cấu hình nhỏ ở mục Kiểm tra. |

## Tham số trong `config.py`

Các trường này nằm trong class `Config` của `bipn/config.py`. Sửa file rồi chạy lại. Không có cờ dòng lệnh tương ứng.

| Trường | Giá trị | Ý nghĩa | Khi đổi |
|---|---|---|---|
| `embedding_size` | 64 | Kích thước embedding user và item. | Giảm để thử nhanh. Bản báo cáo dùng 64. |
| `n_layers` | 2 | Số tầng lan truyền sau tầng gốc. Tầng `l` nhân `1/(l+1)`. | Giữ 2 để theo bài. |
| `beta` | 0.5 | Trọng số BCE so với BPR: `beta * BCE + (1 - beta) * BPR`. | Chỉ `bipn` và `weighted`. `lightgcn` không dùng. |
| `gamma` | 0.0001 | Hệ số L2 trên hai bảng embedding. | Tăng khi embedding tăng quá nhanh. |
| `lr` | 0.001 | Tốc độ học của Adam. | Giảm khi loss dao động mạnh. |
| `weight_decay` | 0 | Phân rã trọng số của Adam, tách khỏi `gamma`. | Giữ 0. Giá trị khác 0 phạt trọng số thêm một lần. |
| `batch_size` | 1024 | Số dòng mỗi bước cập nhật, và số user mỗi lần chấm điểm. | Giảm khi thiếu bộ nhớ. |
| `neg_count` | 4 | Số món âm cho mỗi món dương trong BCE. | Chỉ `bipn` và `weighted`. BPR vẫn lấy một món âm cho mỗi user ở mỗi epoch. |
| `epochs` | 200 | Số epoch khi không truyền `--epochs`. | Sáu lần chạy trong báo cáo dùng cùng một giá trị. |
| `topk` | 10 và 20 | Các mốc HR và NDCG được in. | Bài tập dùng mốc 10. |
| `seed` | 2021 | Hạt giống của `random`, NumPy và PyTorch. | Giữ cố định giữa các lần so sánh. |
| `behavior_boost` | 2 | Alpha của cart và hành vi đích với `--model weighted`. Hành vi khác có alpha 1. | Chỉ `weighted`. |

File `b_tmall.sh` của repo gốc dùng học 0.0001, beta 0.3 và gắn `cuda:6`. Không dùng chỉ số GPU đó. Muốn chạy đúng hai số học và beta của file shell thì sửa `lr` và `beta` trong `config.py`, rồi dùng cùng bộ đó cho cả sáu lệnh.
