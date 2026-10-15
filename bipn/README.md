# BIPN

Khung huấn luyện theo mục 3.2 của *Behavior-Contextualized Item Preference Modeling* (SIGIR 2024).

Ba khối:

1. LightGCN trên đồ thị gộp mọi hành vi.
2. Mạng lọc sở thích theo user, item và hành vi.
3. LightGCN trên hành vi đích. Điểm cuối trộn hai nhánh bằng `lambda = 1 / số lần tương tác đích`.

Hành vi cuối trong danh sách là hành vi đích. Tmall dùng `click`, `collect`, `cart`, `buy`. Retailrocket dùng `view`, `cart`, `buy`.

## Mô hình

| `--model` | Nội dung |
|---|---|
| `bipn` | BIPN theo bài, BCE lấy trung bình trên mọi dòng mẫu. |
| `weighted` | Cùng mạng BIPN. Cart và hành vi đích có hệ số 2 trong BCE. |
| `lightgcn` | LightGCN chỉ trên đồ thị hành vi đích. |

Ba mô hình dùng chung cách tính HR và NDCG.

## Cài đặt

Python 3.12. Môi trường nằm tại `.venv` và được `.gitignore` loại trừ.

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Kiểm tra

Các lệnh dưới đây chạy trên dữ liệu giả, trong thư mục `bipn`:

```text
.venv\Scripts\python.exe main.py --smoke
.venv\Scripts\python.exe main.py --smoke --model weighted
.venv\Scripts\python.exe main.py --smoke --model lightgcn
```

## Huấn luyện

Chuẩn bị dữ liệu theo [data/README.md](../data/README.md). Sáu lần chạy, cách chọn epoch và ý nghĩa từng tham số nằm ở [docs/huong-dan-chay.md](../docs/huong-dan-chay.md).
