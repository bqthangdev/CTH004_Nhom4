# Chạy mô hình cải tiến

Mở PowerShell trong thư mục `bipn`. Đợi dòng `epoch N ...` in xong rồi mới tắt máy. Số `--patience` ở lệnh bắt đầu và lệnh chạy tiếp phải giống nhau.

## Tmall

```text
.venv\Scripts\python.exe main.py --data ..\data\Tmall --dataset tmall --model learned --device cuda --patience 5
```

```text
.venv\Scripts\python.exe main.py --data ..\data\Tmall --dataset tmall --model learned --device cuda --patience 5 --resume
```

```text
.venv\Scripts\python.exe main.py --data ..\data\Tmall --dataset tmall --model learned --device cuda --eval
```

## Retailrocket

```text
.venv\Scripts\python.exe main.py --data ..\data\retailrocket --dataset retailrocket --model learned --device cuda --patience 5
```

```text
.venv\Scripts\python.exe main.py --data ..\data\retailrocket --dataset retailrocket --model learned --device cuda --patience 5 --resume
```

```text
.venv\Scripts\python.exe main.py --data ..\data\retailrocket --dataset retailrocket --model learned --device cuda --eval
```

Mỗi bộ có ba lệnh, theo thứ tự: bắt đầu, chạy tiếp, lấy số. Dòng `saved test` của lệnh thứ ba là số của epoch tốt nhất.
