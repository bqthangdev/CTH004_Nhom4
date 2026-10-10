# CTH004 — Topic 5: Multi-Behavior Recommendations

Bài tập nhóm môn Chuyên đề HTTT Thông minh, Khoa HTTT, 2026. Giảng viên: PGS.TS. Nguyễn Thái Nghe.

Slide đề bài: [docs/CD HTTT TM.pdf](docs/CD%20HTTT%20TM.pdf)

Danh sách bài báo của giảng viên: https://github.com/wubinzzu/Multi-Behavior-Recommendation-Papers

Phân tích chi tiết ba phương án: [docs/de-xuat-topic5-multi-behavior.md](docs/de-xuat-topic5-multi-behavior.md)

## Yêu cầu

Nhóm 2 sinh viên. Chọn một phương pháp trong danh sách trên, chạy lại code, sửa mô hình, đổi sang một bộ dữ liệu khác, rồi so với một mô hình khác.

- Báo cáo theo mẫu Springer: 30%
- Kết quả thí nghiệm: 70%

Tên thành viên nhóm chưa có trong tài liệu hiện tại.

## Ba phương án

Nên làm BIPN. CRGCN dễ giảng hơn nhưng khó cài. MBHT là hướng dự đoán món tiếp theo theo thời gian.

Bản dùng để chạy là file đã xử lý đi kèm code. Link gốc bên dưới là trang dữ liệu công khai mà bài báo hoặc README trỏ tới. Hai nguồn này không cùng kích thước.

### BIPN (nên chọn)

Behavior-Contextualized Item Preference Modeling. SIGIR 2024.

| | URL |
|---|---|
| Bài báo (arXiv) | https://arxiv.org/abs/2404.18166 |
| Bài báo (ACM DOI) | https://doi.org/10.1145/3626772.3657696 |
| Code | https://github.com/MingshiYan/BIPN |
| Dữ liệu đã xử lý, đi kèm code | https://github.com/MingshiYan/BIPN/blob/main/data.zip |

`data.zip` có Tmall, Taobao, Yelp và ML10M. Phải chạy `data_process.py` trong từng thư mục trước khi train.

Nguồn gốc mà bài báo ghi:

| Bộ dữ liệu | URL đã kiểm tra | Ghi chú |
|---|---|---|
| Taobao | https://tianchi.aliyun.com/dataset/649 | Trang đang mở, tiêu đề: 淘宝用户购物行为数据集 |
| Tmall | https://tianchi.aliyun.com/dataset/140281 | Trang Rec-Tmall. Log đầy đủ khoảng 1,33 tỷ dòng, không phải bộ 41.738 user trong code |
| Yelp | https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset | Trang Kaggle mở được. Bài báo ghi URL cũ `kaggle.com/yelp-dataset/yelp-dataset` |
| MovieLens 10M | https://grouplens.org/datasets/movielens/10m/ | Trang GroupLens mở được. File `ml-10m.zip` khoảng 63 MB |

### CRGCN

Cascading Residual Graph Convolutional Network. TOIS 2023.

| | URL |
|---|---|
| Bài báo (arXiv) | https://arxiv.org/abs/2205.13128 |
| Code | https://github.com/MingshiYan/CRGCN |
| Dữ liệu đã xử lý, đi kèm code | https://github.com/MingshiYan/CRGCN/blob/main/data.zip |

Code khai báo ba bộ: Tmall, Beibei, Jdata. File zip khoảng 57 MB. Lần kiểm tra này chưa giải nén, nên chưa liệt kê thư mục bên trong.

Nguồn gốc mà bài báo ghi:

| Bộ dữ liệu | URL | Ghi chú |
|---|---|---|
| Tmall | https://tianchi.aliyun.com/dataset/dataDetail?dataId=649 | Bài ghi link này cho Tmall. Cùng mục 649 ở trên đang là dữ liệu Taobao |
| Beibei | https://www.beibei.com | Trang thương mại, không phải file dữ liệu |
| Jdata | https://jdata.jd.com/html/detail.html?id=8 | Lần kiểm tra này không mở được trang |

### MBHT

Multi-Behavior Hypergraph-Enhanced Transformer. KDD 2022. Dự đoán món tiếp theo.

| | URL |
|---|---|
| Bài báo (arXiv) | https://arxiv.org/abs/2207.05584 |
| Bài báo (ACM DOI) | https://doi.org/10.1145/3534678.3539342 |
| Code | https://github.com/yuh-yang/MBHT-KDD22 |
| Dữ liệu đã xử lý | https://drive.google.com/file/d/1OFT_5Xp_az-GSHIl7QEPB9zhulbooLzE/view |

Thư mục `dataset/` trên GitHub không chứa dữ liệu. Link Drive trả về trang, chưa tải file nên chưa xác nhận file còn tải được.

Nguồn gốc mà README của code ghi:

| Bộ dữ liệu | URL đã kiểm tra | Ghi chú |
|---|---|---|
| Retailrocket | https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset | Nên dùng bộ này nếu chọn MBHT. Bài báo: 11.649 user, 87.822 tương tác sau khi lọc |
| Taobao | https://tianchi.aliyun.com/dataset/649 | Cùng trang Taobao ở trên. Bản trong bài MBHT lớn hơn nhiều: 147.892 user |
| IJCAI-15 | https://tianchi.aliyun.com/dataset/42 | Trang đang mở, tiêu đề: IJCAI-15 Repeat Buyers Prediction Dataset. Bản trong bài: 200.000 user |

## Phạm vi đã kiểm tra

Các link arXiv, GitHub, Tianchi 649, Tianchi 42, Tianchi 140281, Kaggle Yelp, Kaggle Retailrocket, GroupLens MovieLens 10M và hai file `data.zip` đều mở được khi kiểm tra ngày 10/10/2026. DOI của ACM bị chặn khi gọi tự động, nhưng đúng số DOI ghi trong thông tin xuất bản của hai bài BIPN và MBHT.
