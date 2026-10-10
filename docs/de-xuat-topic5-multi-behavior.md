# Topic 5 — Đề xuất phương pháp Multi-Behavior Recommendation

Nguồn đề bài: slide `docs/CD HTTT TM.pdf` (PGS.TS. Nguyễn Thái Nghe, Khoa HTTT, 2026), trang Topic 5 và trang Evaluation.

Danh sách bài báo của giảng viên: https://github.com/wubinzzu/Multi-Behavior-Recommendation-Papers

## Yêu cầu bài tập

- Nhóm 2 sinh viên.
- Chọn một phương pháp trong danh sách trên.
- Chạy lại code.
- Sửa mô hình.
- Đổi sang một bộ dữ liệu khác.
- So sánh với một mô hình khác.
- Báo cáo theo mẫu Springer: 30% điểm. Kết quả thí nghiệm: 70% điểm.

Nên chọn **BIPN**. Hai phương án thêm là **CRGCN** và **MBHT**. Cả ba đều có trong danh sách bài báo và có code công khai. Nội dung dưới đây chỉ ghi những gì đã mở file code, requirements, dữ liệu hoặc bài báo để kiểm tra.

## 1. BIPN — nên chọn

- Bài báo: *Behavior-Contextualized Item Preference Modeling for Multi-Behavior Recommendation*, SIGIR 2024.
- Code: https://github.com/MingshiYan/BIPN
- Bản preprint đã đọc: https://arxiv.org/abs/2404.18166

Mô hình có 3 phần, đủ kỹ thuật để trình bày và vẫn vẽ được trên slide:

1. Gộp mọi hành vi thành một đồ thị, chạy LightGCN để tạo embedding ban đầu.
2. Mạng BIPN học sở thích theo từng món và từng hành vi.
3. Khi xếp hạng chỉ dùng hành vi đích (mua hoặc like). Hành vi phụ chỉ dùng lúc huấn luyện, để giảm nhiễu.

Trong Table 2 của bài, trên Tmall, HR@10 của BIPN là 0.1414, của LightGCN là 0.0393, của CRGCN là 0.0840. Đây là số của bài báo, dùng để đối chiếu sau khi nhóm tự chạy.

### Cách làm cho đủ điểm

- Chạy lại trên Tmall.
- Sửa mô hình: thêm trọng số theo từng hành vi trong hàm mất mát, tăng trọng số cho cart và buy. Trong `data.zip`, `click.txt` khoảng 21 MB còn `cart.txt` khoảng 23 KB, nên click đang lấn át mẫu huấn luyện. Hướng sửa này khớp luận điểm của bài về nhiễu từ hành vi phụ.
- Đổi bộ dữ liệu: chạy cùng mô hình trên Yelp. Yelp đã có trong code, khác miền với Tmall. Bài báo nhận xét đa hành vi giúp Tmall nhiều hơn Yelp vì hành vi đích của Yelp ít thưa hơn.
- So với LightGCN chỉ trên hành vi mua hoặc like, cùng cách chia train/test. LightGCN là baseline đơn hành vi trong Table 2. File `lightGCN.py` đã có. Repo chưa có script huấn luyện LightGCN riêng, nên cần viết một vòng lặp ngắn trên cùng file dữ liệu.

### Dữ liệu đã kiểm tra trong `data.zip` (44,9 MB)

Zip có file thô và `data_process.py`. Chưa có `pos_sampling.txt` và các file `*_dict.txt`. Phải giải nén rồi chạy `data_process.py` trong thư mục dữ liệu trước khi train.

| Bộ dữ liệu | Hành vi trong code | Quy mô trong Table 1 của bài |
|---|---|---|
| Tmall | click, collect, cart, buy | 41.738 user, 11.953 item, khoảng 2,3 triệu tương tác |
| Taobao | view, cart, buy | 48.749 user, 39.493 item, khoảng 2,0 triệu tương tác |
| Yelp | tip, neg, neutral, pos | 19.800 user, 22.734 item, khoảng 1,4 triệu tương tác |
| ML10M | neg, neutral, pos | 67.788 user, 8.704 item, khoảng 9,9 triệu tương tác |

`count.txt` của Tmall trong zip là 41.738 user và 11.953 item, khớp Table 1.

### Việc cần xử lý khi chạy

- Máy hiện có Python 3.12. Requirements khóa `numpy==1.21.5`, `scipy==1.10.1`, `torch==1.12.1+cu113`. Bộ này cần Python khoảng 3.10.
- `b_tmall.sh` gắn cứng `cuda:6`. Máy thường gọi `python main.py --data_name tmall --device cuda:0`. Không có GPU thì dùng `--device cpu`.
- `data_set.py` và `main.py` gọi `torch.cuda.manual_seed` ngay khi import. Bản PyTorch chỉ có CPU sẽ dừng tại đó. Cần bọc bằng `torch.cuda.is_available()`.
- Tmall khoảng 2,3 triệu tương tác. Nên train trên GPU. Chưa đo thời gian trên máy này.

## 2. CRGCN — dễ giảng hơn, khó cài hơn

- Bài báo: *Cascading Residual Graph Convolutional Network for Multi-Behavior Recommendation*, TOIS 2023.
- Code: https://github.com/MingshiYan/CRGCN
- Bản preprint đã đọc: https://arxiv.org/abs/2205.13128

Ý tưởng: mỗi hành vi là một tầng GCN. Embedding đi từ xem, sang giỏ hàng, rồi đến mua. Mỗi tầng cộng phần dư (residual) vào embedding của tầng trước. Lúc dự đoán dùng biểu diễn của hành vi mua. `model_cascade.py` khoảng 150 dòng.

`main.py` có dòng comment `from model_cascade_fuse_weight import CRGCN`, nhưng file `model_cascade_fuse_weight.py` không có trong repo. Hướng sửa phù hợp là thêm trọng số học được cho phần residual của từng hành vi, thay vì cộng đều.

So sánh với LightGCN chỉ trên hành vi mua, cùng split. Bài BIPN cũng dùng CRGCN làm baseline đa hành vi, nên hai mô hình này cùng họ và dễ đối chiếu.

Dữ liệu trong bài: Tmall, Beibei, Jdata. Repo có `data.zip` khoảng 57 MB. README nhắc `data_process.py`, file này không có trong danh sách file của repo. Bộ dữ liệu thứ hai có thể là Beibei nếu zip đã chứa bản đã xử lý. Chưa giải nén zip của CRGCN trong lần kiểm tra này.

Requirements khóa PyTorch 1.9.0, PyTorch Geometric 2.0.2, `torch_scatter==2.0.7`, `torch_sparse==0.6.10`, `numpy==1.16.2`, và README ghi Python 3.6. `gcn_conv.py` import trực tiếp các thư viện này. Trên Windows với Python 3.12, đây là chỗ dễ kẹt khi cài.

## 3. MBHT — bài tuần tự, nếu nhóm muốn khác hướng đồ thị

- Bài báo: *Multi-Behavior Hypergraph-Enhanced Transformer for Sequential Recommendation*, KDD 2022.
- Code: https://github.com/yuh-yang/MBHT-KDD22
- Bản preprint đã đọc: https://arxiv.org/abs/2207.05584

Khác BIPN và CRGCN: bài này dự đoán món tiếp theo theo thời gian, không chỉ dự đoán món sẽ mua từ đồ thị tĩnh. Mô hình có Transformer nhiều tỷ lệ cho chuỗi hành vi, và hypergraph cho quan hệ dài hạn giữa các loại hành vi. Code mô hình nằm ở `recbole/model/sequential_recommender/mbht.py`.

`MBHT.yaml` có `enable_hg: 1` và `enable_ms: 1`. Trong `mbht.py` hai cờ này được đọc và dùng trong `if`. Sửa mô hình có thể là tắt hypergraph (`enable_hg: 0`) hoặc tắt multi-scale (`enable_ms: 0`), rồi đo chênh lệch. Đây là ablation có sẵn trong code, vẫn là một thay đổi kỹ thuật rõ.

So với SASRec. SASRec là baseline chuỗi trong mục Baselines của bài, và file `sasrec.py` có sẵn trong cùng thư mục RecBole. GRU4Rec và BERT4Rec cũng có file trong repo và có tên trong bài.

Quy mô trong Table 1 của bài:

| Bộ dữ liệu | User | Item | Tương tác | Hành vi |
|---|---:|---:|---:|---|
| Retailrocket | 11.649 | 36.223 | 87.822 | buy, cart, pv |
| Taobao | 147.892 | 99.038 | 7.092.362 | buy, cart, fav, pv |
| IJCAI | 200.000 | 808.354 | 13.072.940 | buy, cart, fav, pv |

Nếu chọn MBHT, chỉ nên chạy Retailrocket. Taobao và IJCAI quá lớn cho bài tập nhóm.

Đổi bộ dữ liệu: bài đã có cả Retailrocket và Taobao. Retailrocket là bản nhỏ. Chạy Taobao đầy đủ là không thực tế với cấu hình máy thường.

### Rủi ro đã thấy trên repo

- Thư mục `dataset/` trên GitHub chỉ có `.DS_Store`. Dữ liệu không nằm trong repo. README trỏ một file Google Drive. Lần kiểm tra này chưa tải file Drive, nên chưa xác nhận link còn sống và chưa biết tên thư mục bên trong.
- Lệnh mẫu trong README là `--dataset=tmall_beh`, trong khi bài báo dùng Retailrocket, Taobao và IJCAI. Tên thư mục thật phải xem trong file Drive.
- `requirements.txt` khóa `scipy==1.6.0` và `torch>=1.7.0`. `scipy==1.6.0` không cài được trên Python 3.12. Cần môi trường Python cũ hơn, khoảng 3.8 hoặc 3.9.
- Mô hình nặng hơn BIPN. Retailrocket là cấu hình vừa sức. Taobao 7 triệu tương tác và IJCAI 13 triệu tương tác thì không nên đưa vào phạm vi bài.

## Hướng không nên lấy

- MBCGCN (WWW 2023, https://github.com/SS-00-SS/MBCGCN): README ghi Python 3.6 và TensorFlow 1.14.
- HEC-GCN (https://github.com/marqu22/HEC-GCN): `requirements.txt` ghi cùng lúc `torch==2.5.1`, `numpy==1.16.2` và `torch_scatter==2.0.7`. Bộ phiên bản này không khớp nhau.
- Các bài LLM trong danh sách (EAGER-LLM, ReLLa): cần GPU lớn hoặc API, vượt phạm vi bài tập nhóm 2 người.

## Việc còn cần nhóm quyết

1. Chốt BIPN, CRGCN hay MBHT.
2. Máy train có GPU, hay dùng Google Colab.
3. Với BIPN, bộ dữ liệu thứ hai dùng Yelp có sẵn trong code, hay tự xử lý thêm một bộ ngoài bài.

## Việc đã kiểm tra

- Đọc text 21 trang slide.
- Đọc danh sách bài trên repo của giảng viên.
- BIPN: `model.py`, `main.py`, `trainer.py`, `data_set.py`, `lightGCN.py`, `requirements.txt`, `b_tmall.sh`, và toàn bộ danh sách file trong `data.zip`, gồm `data_process.py` của Tmall và Yelp.
- CRGCN: danh sách file repo, `requirements.txt`, `main.py`, `model_cascade.py`, `gcn_conv.py`, các issue đang mở.
- MB-HGCN và HEC-GCN: requirements và cấu trúc repo.
- MBCGCN: README môi trường TensorFlow 1.14.
- MBHT: README, `requirements.txt`, `MBHT.yaml`, thư mục `dataset/`, các cờ trong `mbht.py`, danh sách model trong `recbole/model/sequential_recommender/`, Table 1 và mục Baselines của bài báo.
