# Plan BIPN

Bài được chọn để lập plan: Behavior-Contextualized Item Preference Modeling, SIGIR 2024.

- Bài: https://arxiv.org/abs/2404.18166
- DOI: https://doi.org/10.1145/3626772.3657696
- Code gốc: https://github.com/MingshiYan/BIPN
- Dữ liệu đã xử lý kèm code gốc: https://github.com/MingshiYan/BIPN/blob/main/data.zip

Khung trong thư mục `bipn/` bám công thức mục 3.2 của bài. Khung này chưa được chạy trên Tmall và chưa so số với repo gốc.

## Việc bài tập cần có

1. Chạy BIPN trên Tmall.
2. Sửa mô hình: thêm trọng số cho từng hành vi trong hàm mất mát, tăng trọng số cart và buy vì click nhiều hơn rất nhiều.
3. Chạy cùng mô hình trên một bộ thứ hai. Chưa chốt. Đề xuất ở mục dưới. Yelp vẫn là phương án không cần xử lý thêm.
4. So với LightGCN chỉ trên hành vi đích, cùng cách chia tập.
5. Điền số đo vào `paper/main.tex`. Chỉ số dùng HR@10 và NDCG@10.

## Mô hình cần trình bày

Hành vi cuối là hành vi đích: mua với Tmall và Taobao, `pos` với Yelp và ML10M.

1. Gộp mọi hành vi thành một đồ thị, cạnh có trọng số 1 nếu user và item từng tương tác. Chạy LightGCN. Tầng `l` nhân `1/(l+1)`, `l` bắt đầu từ 0. Bài ghi 2 tầng.
2. Mạng lọc nhận embedding user, item và one-hot hành vi. Lớp trước dùng sigmoid để lọc user. Lớp giữa dùng tanh để ra sở thích theo món. Lớp sau dùng sigmoid để lọc lại vector đó.
3. Với hành vi đích, cộng các vector sở thích của những món user đã tương tác. Song song, chạy LightGCN trên riêng đồ thị hành vi đích, lấy embedding đã gộp ở bước 1 làm đầu vào.
4. Điểm dự đoán là `(1-lambda)` nhân tích vô hướng nhánh sở thích cộng `lambda` nhân tích vô hướng nhánh LightGCN đích. `lambda = 1/N_u`. Nếu user không có tương tác đích trong tập train thì `lambda = 1`, chỉ dùng nhánh LightGCN.
5. Mất mát là `beta` nhân BCE, `(1-beta)` nhân BPR, cộng L2 trên hai bảng embedding. BCE dùng dạng logits cho ổn định số, cùng mục tiêu với công thức (9).

Tham số mặc định trong khung, lấy từ `main.py` của repo gốc trừ hệ số L2: embedding 64, 2 tầng, beta 0.5, học 0.001, batch 1024, 4 mẫu âm, 200 epoch. `b_tmall.sh` của repo gốc lại dùng học 0.0001, beta 0.3, L2 0.0001 và gắn `cuda:6`. Khi chạy Tmall thật, dùng bộ trong file `.sh`, không dùng `cuda:6`.

## Khác biệt với code gốc

Cần nhớ khi đối chiếu số, vì hai bên không giống từng dòng.

- Code gốc gộp lớp lọc trước và lớp lọc sau vào một ma trận, không có bias. Bài có bias. Khung này làm ba lớp riêng, có bias, đúng bài.
- Code gốc chuẩn hóa thêm từng vector sau mỗi tầng LightGCN và cộng tầng đó hai lần. Bài chỉ nhân `1/(l+1)`. Khung theo bài.
- HR trong code gốc bằng 1 nếu top K có ít nhất một món đúng, rồi lấy trung bình theo user. Docstring của hàm đó viết một công thức khác. Khung theo code đang chạy.
- NDCG dùng độ liên quan 0 hoặc 1 và mẫu số `log2(vị trí+1)`, rồi chia cho IDCG của `min(số món đúng, K)`.
- L2 trong khung là tổng bình phương hai bảng embedding. Gamma mặc định `1e-4`.

## Thứ tự làm

1. Tạo môi trường có PyTorch. Máy đang dùng Python 3.12, NumPy 2.4.6, SciPy 1.17.1, chưa có PyTorch ở lần kiểm tra này. Không cài bộ `torch==1.12.1+cu113` của repo gốc lên Python 3.12.
2. Tải `data.zip` từ repo BIPN, giải nén. Zip có file thô, chưa có `pos_sampling.txt`. Khung này không cần file đó. Cần `count.txt`, từng file hành vi, `test.txt` và `validation.txt`.
3. Chạy `--smoke` để chắc môi trường. Sau đó chạy 1 epoch Tmall trên CPU hoặc GPU chỉ để thấy loss giảm dần, rồi mới chạy đủ epoch.
4. Ghi HR@10 và NDCG@10 trên validation và test. Chọn epoch theo validation, báo số trên test của epoch đó.
5. Thêm trọng số hành vi vào BCE. Giữ nguyên phần còn lại. Chạy lại Tmall.
6. Viết vòng LightGCN chỉ trên `buy.txt` của cùng thư mục Tmall. Cùng embedding, cùng số tầng, cùng cách chấm HR và NDCG.
7. Lặp ba mô hình trên Retailrocket sau khi thành viên còn lại xác nhận. Hành vi dự kiến là xem, giỏ, mua. File Kaggle chưa tải, nên tên cột và cách chia tập chưa có.
8. Điền bảng trong `paper/main.tex`. Không ghi số của bài báo thành số của nhóm.

## Dữ liệu đã kiểm trong zip

`count.txt` của Tmall là 41.738 user và 11.953 item. Dòng đầu `click.txt` là `1 6674`, dòng đầu `buy.txt` là `1 8`. `buy.txt` có 181.428 dòng, `test.txt` 31.572 dòng, `validation.txt` 74.158 dòng. Đây là phần trong file zip, không phải toàn bộ số mua 255.586 của Table 1.

Yelp trong cùng zip có `tip.txt`, `neg.txt`, `neutral.txt`, `pos.txt`.

Không dùng file Tianchi 140281 cho lần chạy này. File đó là log Tmall khoảng 1,33 tỷ dòng, khác bộ 41.738 user trong code.

## Bộ dữ liệu thứ hai

Đã ghi nhận Retailrocket, https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset. Trang mở được khi kiểm tra ngày 10/10/2026. Thành viên còn lại chưa xác nhận. Câu hỏi nằm ở [docs/cau-hoi-can-xac-nhan.md](cau-hoi-can-xac-nhan.md).

Table 1 của bài MBHT ghi bản đã lọc: 11.649 user, 36.223 item, 87.822 tương tác, hành vi buy, cart, pv. Đó là số của bài MBHT, không phải số đếm của nhóm. File Kaggle chưa tải. Bộ này không có trong `data.zip` của BIPN và không có trong bốn bộ thí nghiệm của bài BIPN.

Bốn bộ trong zip, đếm ngày 12/10/2026. User và item lấy từ `count.txt`. Số dòng là từng file train, không cộng `test.txt` hay `validation.txt`.

| Bộ | User | Item | Dòng train | Test | Validation |
|---|---:|---:|---|---:|---:|
| Tmall | 41.738 | 11.953 | click 1.813.498, collect 221.514, cart 1.996, buy 181.428 | 31.572 | 74.158 |
| Taobao | 48.749 | 39.493 | view 1.548.162, cart 193.747, buy 162.273 | 48.749 | 48.749 |
| Yelp | 19.800 | 22.734 | tip 285.673, neg 198.106, neutral 238.880, pos 657.926 | 10.000 | 19.417 |
| ML10M | 67.788 | 8.704 | neg 1.370.897, neutral 3.580.155, pos 4.903.210 | 10.000 | 67.774 |
| Retailrocket | 11.649 | 36.223 | 87.822 tương tác sau lọc, theo Table 1 bài MBHT | chưa có | chưa có |

Retailrocket nhỏ hơn Tmall và cùng kiểu hành vi xem, giỏ, mua. Yelp và ML10M đã nằm trong bài BIPN, hành vi là mức đánh giá chứ không phải xem–giỏ–mua. Taobao cùng miền mua sắm với Tmall và đã có sẵn trong zip. ML10M có 9.854.262 dòng train, nặng nhất trong bốn bộ.

## Điều kiện máy

Máy đang mở repo đã kiểm bằng lệnh: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB, driver 617.42, RAM 31,7 GB. `bipn/.venv` đang là PyTorch bản CPU, nên GPU này chưa được dùng. Chưa đo thời gian một epoch Tmall.

Thành viên còn lại cho biết máy của họ là Idealpad 5 Pro, Core Ultra 9 185H, RTX 4050, 32 GB RAM. Cấu hình này là lời kể, chưa kiểm trên máy đó.

Bản đủ epoch chạy trên một trong hai GPU. Chưa chọn máy nào giữ bản train dài.

1. GPU máy RTX 3050 Ti. Cài bản PyTorch có CUDA, rồi `--device cuda`. Embedding 64, 2 tầng, khoảng 42 nghìn user và 12 nghìn item vừa bộ nhớ 4 GB. Chưa cài bản CUDA.
2. GPU máy RTX 4050 ở trên. Chưa cài gì trên máy đó.
3. CPU máy RTX 3050 Ti chỉ để thử vài epoch. `--smoke` đã chạy. Tmall với 4 mẫu âm tạo khoảng 11,1 triệu dòng BCE mỗi epoch, cỡ 2 GB RAM. RAM 31,7 GB chứa được. Chưa biết 200 epoch mất bao lâu.

`data_set.py` của repo gốc gọi `torch.cuda.manual_seed` lúc import. Khung này không gọi dòng đó, nên chạy được trên CPU.

## Đã kiểm trên máy này

Ngày 12/10/2026, `bipn/.venv` có PyTorch 2.14.1+cpu và NumPy 2.5.4. `python main.py --smoke` chạy xong 2 epoch trên dữ liệu giả: loss 0.5902 rồi 0.5755, có in HR và NDCG. Số này không phải kết quả Tmall.

Loader và HR/NDCG đã đối chiếu trên một thư mục tạm và một ma trận relevance biết trước. HR@1 của ma trận đó là 0.5.

## Chưa làm trong khung

- Chưa train Tmall. Chưa tải Retailrocket.
- Chưa có trọng số hành vi.
- Chưa có script LightGCN đối chứng.
- Chưa điền kết quả vào báo cáo.
