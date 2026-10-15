# Dữ liệu

Thư mục này chứa hai bộ dùng cho thí nghiệm BIPN.

| Thư mục | Vai trò |
|---|---|
| `Tmall/` | Bộ thứ nhất. Đã chia sẵn trong bản phát hành của BIPN. |
| `taobao/`, `Yelp/`, `ML10M/` | Có trong cùng bản phát hành. Không dùng trong thí nghiệm hiện tại. |
| `retailrocket/` | Bộ thứ hai. File gốc từ Kaggle, kèm bản đã chuyển sang định dạng cặp `user item`. |

`.gitignore` loại mọi tệp trong `data/` trừ file này.

Định dạng sau khi chuẩn bị: mỗi dòng là `user item`, id bắt đầu từ 1. Mỗi bộ có `count.txt`, một file cho từng hành vi, `validation.txt` và `test.txt`.

## BIPN

Nguồn: https://github.com/MingshiYan/BIPN/blob/main/data.zip

Giải nén `data.zip` vào thư mục này, để bốn thư mục dữ liệu nằm trực tiếp dưới `data/`:

```text
data/
├── Tmall/
├── taobao/
├── Yelp/
└── ML10M/
```

Code gốc của BIPN cần chạy `data_process.py` trong từng thư mục. Khung trong `bipn/` đọc các file thô, nên không cần bước đó.

`data/Tmall/count.txt` trong bản đã giải nén:

```json
{"user": 41738, "item": 11953}
```

## Retailrocket

Nguồn: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

Tải về được `archive.zip`. Giải nén vào `data/retailrocket/` và giữ zip trong cùng thư mục:

```text
data/retailrocket/
├── archive.zip
├── events.csv
├── category_tree.csv
├── item_properties_part1.csv
└── item_properties_part2.csv
```

`events.csv` có các cột `timestamp`, `visitorid`, `event`, `itemid`, `transactionid`. Ba giá trị của `event` là `view`, `addtocart` và `transaction`. Huấn luyện chỉ dùng `events.csv`. `category_tree.csv` và hai file `item_properties` không được mô hình đọc.

Chuyển sang định dạng cặp từ thư mục `bipn`:

```text
.venv\Scripts\python.exe prepare_retailrocket.py --events ../data/retailrocket/events.csv --out ../data/retailrocket
```

Script giữ visitor có ít nhất một `transaction`, sắp các món mua theo thời gian và bỏ trùng:

- Từ ba món mua trở lên: món cuối vào `test.txt`, món kế cuối vào `validation.txt`, phần còn lại vào `buy.txt`.
- Đúng hai món: món đầu vào `buy.txt`, món cuối vào `test.txt`.
- Đúng một món: món đó vào `test.txt`.
- `view` và `addtocart` chỉ vào train khi xảy ra trước món mua bị tách ra sớm nhất.

Các file sinh ra: `count.txt`, `view.txt`, `cart.txt`, `buy.txt`, `validation.txt`, `test.txt`.

Kết quả chuyển trên bản `events.csv` hiện có:

| Hạng mục | Số lượng |
|---|---:|
| User | 11.719 |
| Item | 31.839 |
| `view.txt` | 88.304 |
| `cart.txt` | 22.716 |
| `buy.txt` | 8.666 |
| `test.txt` | 11.719 |
| `validation.txt` | 885 |

`validation.txt` ngắn vì đa số visitor không có đủ ba món mua khác nhau.
