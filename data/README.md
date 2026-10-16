# Dữ liệu

Thư mục này chứa hai bộ dùng cho thí nghiệm. Cách tải, giải nén và chạy nằm ở [docs/huong-dan-chay.md](../docs/huong-dan-chay.md). File dữ liệu không đưa vào git.

| Thư mục | Vai trò |
|---|---|
| `Tmall/` | Bộ thứ nhất, lấy từ `data.zip` của BIPN. Đã chia sẵn. |
| `taobao/`, `Yelp/`, `ML10M/` | Có trong cùng zip. Không dùng trong thí nghiệm hiện tại. |
| `retailrocket/` | Bộ thứ hai. `events.csv` từ Kaggle. Pipeline tạo các file cặp `user item`. |

Định dạng sau khi chuẩn bị: mỗi dòng là `user item`, id bắt đầu từ 1. Mỗi bộ có `count.txt`, một file cho từng hành vi, `validation.txt` và `test.txt`.

`data/Tmall/count.txt` của bản zip:

```json
{"user": 41738, "item": 11953}
```

Retailrocket sau bước chuyển trên bản `events.csv` hiện có:

| Hạng mục | Số lượng |
|---|---:|
| User | 11.719 |
| Item | 31.839 |
| `view.txt` | 88.304 |
| `cart.txt` | 22.716 |
| `buy.txt` | 8.666 |
| `test.txt` | 11.719 |
| `validation.txt` | 885 |
