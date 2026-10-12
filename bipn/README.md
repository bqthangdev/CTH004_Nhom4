# Khung BIPN

Ba khối theo mục 3.2 của bài SIGIR 2024:

1. LightGCN trên đồ thị gộp mọi hành vi.
2. Mạng lọc sở thích theo user, item và hành vi.
3. LightGCN trên hành vi đích, rồi trộn điểm bằng lambda = 1 / số lần tương tác đích.

Hành vi cuối trong danh sách là hành vi đích. Tmall dùng `click, collect, cart, buy`.

Chưa chạy trên dữ liệu thật và chưa đối chiếu số với repo gốc. Môi trường đã tạo là `.venv`, không đưa vào git. Lệnh dưới đây chỉ kiểm tra khung trên dữ liệu giả. Lần chạy 12/10/2026 cho loss 0.5902 rồi 0.5755.

```text
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py --smoke
```

Khi đã giải nén và xử lý dữ liệu theo định dạng `count.txt`, `<hành vi>.txt`, `test.txt`, `validation.txt`:

```text
python main.py --data đường/dẫn/Tmall --dataset tmall --epochs 1 --device cpu
```

Mỗi dòng tương tác là `user item`, id bắt đầu từ 1.
