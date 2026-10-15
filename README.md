# CTH004 — Topic 5: Multi-Behavior Recommendations

Bài tập nhóm môn Chuyên đề HTTT Thông minh, Khoa HTTT, 2026. Giảng viên: PGS.TS. Nguyễn Thái Nghe.

Phương pháp đang triển khai là BIPN (SIGIR 2024). Bộ dữ liệu là Tmall và Retailrocket. Mô hình đối chứng là LightGCN trên hành vi đích.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| [bipn](bipn) | Huấn luyện và chuyển dữ liệu Retailrocket. |
| [data](data) | Hai bộ dữ liệu. Xem [data/README.md](data/README.md). |
| [paper](paper) | Báo cáo LNCS. |
| [docs](docs) | Đề xuất phương án, plan và câu còn mở. |

Slide đề bài: [docs/CD HTTT TM.pdf](docs/CD%20HTTT%20TM.pdf). Danh sách bài của giảng viên: https://github.com/wubinzzu/Multi-Behavior-Recommendation-Papers

## Yêu cầu bài tập

Nhóm 2 sinh viên: chạy lại một phương pháp trong danh sách trên, sửa mô hình, đánh giá trên một bộ dữ liệu khác, và so với một mô hình khác.

- Báo cáo theo mẫu Springer: 30%
- Kết quả thí nghiệm: 70%

Tên trên báo cáo hiện là NVA và NVB. Email và tên trường đầy đủ chưa điền trong [paper/main.tex](paper/main.tex).

## Chạy

Hướng dẫn cài đặt, sáu lần chạy và ý nghĩa tham số: [docs/huong-dan-chay.md](docs/huong-dan-chay.md).

## Báo cáo

Mẫu là lớp `llncs` 2.26 (25/02/2025), mẫu bài hội nghị LNCS của Springer. Slide ghi “Springer paper template” và không ghi mẫu tạp chí.

Hướng dẫn mẫu: https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines

Nguồn bài: [paper/main.tex](paper/main.tex). Tài liệu tham khảo: [paper/refs.bib](paper/refs.bib). Máy biên dịch cần MiKTeX và lớp `llncs` đã cài. Không chép `llncs.cls` vào thư mục.

Trong thư mục `paper`:

```text
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Tài liệu trong repo

| Tài liệu | Nội dung |
|---|---|
| [docs/de-xuat-topic5-multi-behavior.md](docs/de-xuat-topic5-multi-behavior.md) | So sánh BIPN, CRGCN và MBHT. |
| [docs/plan-bipn.md](docs/plan-bipn.md) | Plan BIPN. |
| [docs/huong-dan-chay.md](docs/huong-dan-chay.md) | Cách chạy và các tham số. |
| [docs/cau-hoi-can-xac-nhan.md](docs/cau-hoi-can-xac-nhan.md) | Mục đã chốt và mục còn mở. |
| [docs/memory.md](docs/memory.md) | Hiện trạng và nhật ký. |

## Nguồn

### BIPN

Behavior-Contextualized Item Preference Modeling. SIGIR 2024.

| | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2404.18166 |
| DOI | https://doi.org/10.1145/3626772.3657696 |
| Code | https://github.com/MingshiYan/BIPN |
| Dữ liệu kèm code | https://github.com/MingshiYan/BIPN/blob/main/data.zip |

`data.zip` gồm Tmall, Taobao, Yelp và ML10M. Cách giải nén nằm ở [data/README.md](data/README.md).

| Bộ dữ liệu | URL | Ghi chú |
|---|---|---|
| Taobao | https://tianchi.aliyun.com/dataset/649 | Tiêu đề trang: 淘宝用户购物行为数据集 |
| Tmall | https://tianchi.aliyun.com/dataset/140281 | Rec-Tmall. Log đầy đủ khoảng 1,33 tỷ dòng, khác bộ 41.738 user trong `data.zip` |
| Yelp | https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset | Bài báo ghi URL cũ `kaggle.com/yelp-dataset/yelp-dataset` |
| MovieLens 10M | https://grouplens.org/datasets/movielens/10m/ | File `ml-10m.zip` khoảng 63 MB |
| Retailrocket | https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset | Bộ thứ hai của thí nghiệm |

Các trang arXiv, GitHub, Tianchi, Kaggle và GroupLens ở bảng trên mở được khi kiểm tra ngày 10/10/2026. DOI ACM trả về 403 với truy cập tự động. Số DOI lấy từ thông tin xuất bản của bài.

CRGCN và MBHT được ghi trong [docs/de-xuat-topic5-multi-behavior.md](docs/de-xuat-topic5-multi-behavior.md).
