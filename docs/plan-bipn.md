# Plan BIPN

Bài: Behavior-Contextualized Item Preference Modeling, SIGIR 2024.

| | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2404.18166 |
| DOI | https://doi.org/10.1145/3626772.3657696 |
| Code gốc | https://github.com/MingshiYan/BIPN |
| Dữ liệu kèm code | https://github.com/MingshiYan/BIPN/blob/main/data.zip |

Khung nằm ở `bipn/`.

## Phạm vi

1. Chạy BIPN trên Tmall: `--model bipn`.
2. Sửa mô hình: `--model weighted`, tăng phần đóng góp của cart và hành vi đích trong BCE.
3. Chạy cùng ba mô hình trên Retailrocket.
4. So với LightGCN chỉ trên hành vi đích: `--model lightgcn`.
5. Điền HR@10 và NDCG@10 vào `paper/main.tex`. Số của bài báo không ghi thành số của nhóm.

Phần lý thuyết trong báo cáo đã viết. Ba mô hình và bộ chuyển Retailrocket đã có trong code. Chưa chạy đủ epoch, nên bảng kết quả vẫn trống.

## Mô hình

Hành vi cuối là hành vi đích. Tmall: `click`, `collect`, `cart`, `buy`. Retailrocket: `view`, `cart`, `buy`.

1. Gộp mọi hành vi thành một đồ thị. Cạnh bằng 1 nếu user và item từng tương tác. LightGCN nhân tầng `l` với `1/(l+1)`, `l` bắt đầu từ 0. Mặc định 2 tầng.
2. Mạng lọc nhận embedding user, item và one-hot hành vi. Lớp trước dùng sigmoid, lớp giữa dùng tanh, lớp sau dùng sigmoid. Cả ba lớp có bias.
3. Với hành vi đích, cộng các vector sở thích của những món user đã tương tác trong train. Song song, LightGCN chạy trên riêng đồ thị hành vi đích.
4. Điểm là `(1 - lambda) * (sở thích · item) + lambda * (embedding đích của user · embedding đích của item)`. `lambda = 1 / N_u`. Nếu user không có tương tác đích trong train thì `lambda = 1`.
5. Mất mát BIPN là `beta * BCE + (1 - beta) * BPR + gamma * L2` trên hai bảng embedding. BCE dùng dạng logits.

`--model weighted` giữ nguyên mạng và điểm. Hệ số alpha bằng 2 với cart và hành vi đích, bằng 1 với hành vi còn lại. Trọng số từng dòng BCE là `alpha_b * N / (n_b * tổng alpha)`. `n_b` là số dòng BCE của hành vi đó, `N` là tổng số dòng. Phần đóng góp kỳ vọng của mỗi hành vi tỉ lệ với alpha.

`--model lightgcn` chỉ dùng đồ thị hành vi đích, điểm là tích vô hướng sau LightGCN, mất mát là BPR cộng L2. Không có BCE và không dùng `beta`.

## Khác code gốc

Số của khung có thể lệch repo gốc. Các chỗ cố ý khác:

- Code gốc gộp hai lớp lọc vào một ma trận và không có bias. Khung làm ba lớp có bias, theo bài.
- Code gốc chuẩn hóa thêm sau mỗi tầng và cộng tầng đó hai lần. Khung chỉ nhân `1/(l+1)`.
- HR bằng 1 nếu top K có ít nhất một món đúng, rồi lấy trung bình theo user. Cách này theo code đánh giá của repo, không theo docstring của hàm đó.
- NDCG dùng độ liên quan 0 hoặc 1 và `log2(vị trí + 1)`.
- L2 là tổng bình phương hai bảng embedding. `gamma` mặc định `1e-4`. `weight_decay` của Adam bằng 0 để hệ số này không bị tính hai lần.

## Dữ liệu

Cách tải và giải nén: [huong-dan-chay.md](huong-dan-chay.md). Không dùng log Tianchi 140281. Log đó khoảng 1,33 tỷ dòng, khác bộ 41.738 user trong `data.zip`.

Số dòng train không cộng `test.txt` hay `validation.txt`. Tmall, Taobao, Yelp và ML10M đếm từ zip ngày 12/10/2026. Retailrocket là bản đã chuyển từ `events.csv` ngày 15/10/2026.

| Bộ | User | Item | Dòng train | Test | Validation |
|---|---:|---:|---|---:|---:|
| Tmall | 41.738 | 11.953 | click 1.813.498, collect 221.514, cart 1.996, buy 181.428 | 31.572 | 74.158 |
| Taobao | 48.749 | 39.493 | view 1.548.162, cart 193.747, buy 162.273 | 48.749 | 48.749 |
| Yelp | 19.800 | 22.734 | tip 285.673, neg 198.106, neutral 238.880, pos 657.926 | 10.000 | 19.417 |
| ML10M | 67.788 | 8.704 | neg 1.370.897, neutral 3.580.155, pos 4.903.210 | 10.000 | 67.774 |
| Retailrocket | 11.719 | 31.839 | view 88.304, cart 22.716, buy 8.666 | 11.719 | 885 |

Table 1 của bài MBHT ghi một bản Retailrocket khác: 11.649 user, 36.223 item, 87.822 tương tác. Không dùng các số đó làm số của nhóm.

`buy.txt` của Tmall có 181.428 dòng. Đó không phải 255.586 lượt mua trong Table 1 của bài BIPN.

## Máy

Máy đang mở repo: RTX 3050 Ti Laptop, 4096 MiB, RAM 31,7 GB, `bipn/.venv` là torch 2.14.1+cu130. Một epoch Tmall `--model learned` khoảng 30 phút.

Bản train đã ghi trong báo cáo chạy trên RTX 4050 Laptop, 6 GB, Core Ultra 9 185H, 32 GB RAM.
