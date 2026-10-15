# So sánh phương án Topic 5

Nguồn đề bài: `docs/CD HTTT TM.pdf` (PGS.TS. Nguyễn Thái Nghe, Khoa HTTT, 2026), trang Topic 5 và trang Evaluation.

Danh sách bài: https://github.com/wubinzzu/Multi-Behavior-Recommendation-Papers

Phương án đã chọn là BIPN. CRGCN và MBHT là hai phương án đã xét. Các mục dưới chỉ gồm thông tin đã đối chiếu từ code, requirements, dữ liệu hoặc bài báo.

## Yêu cầu

- Nhóm 2 sinh viên.
- Chọn một phương pháp trong danh sách trên.
- Chạy lại code, sửa mô hình, đổi bộ dữ liệu, so với một mô hình khác.
- Báo cáo mẫu Springer: 30%. Thí nghiệm: 70%.

## BIPN

*Behavior-Contextualized Item Preference Modeling for Multi-Behavior Recommendation*, SIGIR 2024.

| | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2404.18166 |
| Code | https://github.com/MingshiYan/BIPN |

Mô hình có ba phần:

1. Gộp mọi hành vi thành một đồ thị và chạy LightGCN để tạo embedding ban đầu.
2. Học sở thích theo từng món và từng hành vi.
3. Xếp hạng bằng hành vi đích. Hành vi phụ chỉ tham gia lúc huấn luyện.

Table 2 của bài, Tmall, HR@10: BIPN 0.1414, LightGCN 0.0393, CRGCN 0.0840. Đây là số của bài báo, không phải số của nhóm.

Cách triển khai đã chọn:

- Chạy lại trên Tmall.
- Sửa hàm mất mát: tăng trọng số cart và buy. Trong `data.zip`, `click.txt` khoảng 21 MB và `cart.txt` khoảng 23 KB.
- Bộ thứ hai là Retailrocket, không nằm trong bốn bộ của bài BIPN.
- So với LightGCN chỉ trên hành vi đích. Repo có `lightGCN.py` và không có script train LightGCN riêng.

`data.zip` khoảng 47 MB. Bên trong là file thô và `data_process.py`, chưa có `pos_sampling.txt` hay `*_dict.txt`. Code gốc cần chạy `data_process.py`. Khung `bipn/` không cần các file đó.

| Bộ dữ liệu | Hành vi trong code | Quy mô trong Table 1 của bài |
|---|---|---|
| Tmall | click, collect, cart, buy | 41.738 user, 11.953 item, khoảng 2,3 triệu tương tác |
| Taobao | view, cart, buy | 48.749 user, 39.493 item, khoảng 2,0 triệu tương tác |
| Yelp | tip, neg, neutral, pos | 19.800 user, 22.734 item, khoảng 1,4 triệu tương tác |
| ML10M | neg, neutral, pos | 67.788 user, 8.704 item, khoảng 9,9 triệu tương tác |

`count.txt` của Tmall trong zip là 41.738 user và 11.953 item, khớp Table 1.

Điểm cần biết khi dùng code gốc:

- Requirements khóa `numpy==1.21.5`, `scipy==1.10.1`, `torch==1.12.1+cu113`. Bộ này cần Python khoảng 3.10. Máy dự án dùng Python 3.12.
- `b_tmall.sh` gắn `cuda:6`. Lệnh thông thường là `python main.py --data_name tmall --device cuda:0`, hoặc `--device cpu` khi không có GPU.
- `data_set.py` và `main.py` gọi `torch.cuda.manual_seed` lúc import. Bản PyTorch chỉ có CPU dừng tại dòng đó.
- Chưa đo thời gian train Tmall trên máy này.

## CRGCN

*Cascading Residual Graph Convolutional Network for Multi-Behavior Recommendation*, TOIS 2023.

| | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2205.13128 |
| Code | https://github.com/MingshiYan/CRGCN |

Mỗi hành vi là một tầng GCN. Embedding đi từ xem, sang giỏ, rồi đến mua. Mỗi tầng cộng phần dư vào embedding tầng trước. Dự đoán dùng biểu diễn của hành vi mua. `model_cascade.py` khoảng 150 dòng.

`main.py` có comment `from model_cascade_fuse_weight import CRGCN`. File `model_cascade_fuse_weight.py` không có trong repo.

Dữ liệu trong bài: Tmall, Beibei, Jdata. Repo có `data.zip` khoảng 57 MB. README nhắc `data_process.py`, file này không có trong danh sách file của repo. Zip chưa được giải nén trong lần đối chiếu.

Requirements khóa PyTorch 1.9.0, PyTorch Geometric 2.0.2, `torch_scatter==2.0.7`, `torch_sparse==0.6.10`, `numpy==1.16.2`. README ghi Python 3.6. `gcn_conv.py` import trực tiếp các thư viện này.

## MBHT

*Multi-Behavior Hypergraph-Enhanced Transformer for Sequential Recommendation*, KDD 2022.

| | URL |
|---|---|
| arXiv | https://arxiv.org/abs/2207.05584 |
| Code | https://github.com/yuh-yang/MBHT-KDD22 |

Bài dự đoán món tiếp theo theo thời gian. Mô hình có Transformer nhiều tỷ lệ và hypergraph giữa các loại hành vi. Code nằm ở `recbole/model/sequential_recommender/mbht.py`.

`MBHT.yaml` có `enable_hg: 1` và `enable_ms: 1`. Hai cờ được đọc trong `mbht.py`. Tắt một trong hai cờ là một ablation có sẵn.

Baseline chuỗi trong bài gồm SASRec. File `sasrec.py` có trong RecBole kèm repo. GRU4Rec và BERT4Rec cũng có file và có tên trong bài.

Table 1 của bài:

| Bộ dữ liệu | User | Item | Tương tác | Hành vi |
|---|---:|---:|---:|---|
| Retailrocket | 11.649 | 36.223 | 87.822 | buy, cart, pv |
| Taobao | 147.892 | 99.038 | 7.092.362 | buy, cart, fav, pv |
| IJCAI | 200.000 | 808.354 | 13.072.940 | buy, cart, fav, pv |

Số Retailrocket ở bảng này là bản đã lọc trong bài MBHT, khác bản chuyển trong `data/retailrocket/` của dự án này.

Điểm cần biết trên repo:

- Thư mục `dataset/` trên GitHub chỉ có `.DS_Store`. README trỏ một file Google Drive. File Drive chưa được tải, nên tên thư mục bên trong chưa được xác nhận.
- Lệnh mẫu trong README là `--dataset=tmall_beh`. Bài báo dùng Retailrocket, Taobao và IJCAI.
- `requirements.txt` khóa `scipy==1.6.0` và `torch>=1.7.0`. `scipy==1.6.0` không cài được trên Python 3.12.

## Phương án không dùng

| Phương án | Lý do |
|---|---|
| MBCGCN, WWW 2023, https://github.com/SS-00-SS/MBCGCN | README ghi Python 3.6 và TensorFlow 1.14. |
| HEC-GCN, https://github.com/marqu22/HEC-GCN | `requirements.txt` ghi đồng thời `torch==2.5.1`, `numpy==1.16.2` và `torch_scatter==2.0.7`. |
| EAGER-LLM, ReLLa | Cần GPU lớn hoặc API. |

## Phạm vi đã đối chiếu

- Slide, 21 trang.
- Danh sách bài trên repo của giảng viên.
- BIPN: `model.py`, `main.py`, `trainer.py`, `data_set.py`, `lightGCN.py`, `requirements.txt`, `b_tmall.sh`, danh sách file trong `data.zip`, gồm `data_process.py` của Tmall và Yelp.
- CRGCN: danh sách file, `requirements.txt`, `main.py`, `model_cascade.py`, `gcn_conv.py`.
- MB-HGCN và HEC-GCN: requirements và cấu trúc repo.
- MBCGCN: README.
- MBHT: README, `requirements.txt`, `MBHT.yaml`, thư mục `dataset/`, cờ trong `mbht.py`, model trong `recbole/model/sequential_recommender/`, Table 1 và mục Baselines.
