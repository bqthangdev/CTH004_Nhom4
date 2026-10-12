# Câu cần xác nhận

File này chỉ liệt kê việc chưa chốt, để thành viên còn lại trả lời. Phần đã ghi nhận nằm ở [memory.md](memory.md).

## Cấp 1. Phương pháp

1. Có giữ BIPN làm phương pháp của nhóm, hay chuyển sang CRGCN hoặc MBHT?
   Hiện đang ghi BIPN. [de-xuat-topic5-multi-behavior.md](de-xuat-topic5-multi-behavior.md), [plan-bipn.md](plan-bipn.md). Danh sách bài của giảng viên: https://github.com/wubinzzu/Multi-Behavior-Recommendation-Papers

2. Phần sửa mô hình có phải là tăng trọng số cart và buy trong hàm mất mát, vì click của Tmall nhiều hơn cart rất nhiều?
   Hiện đang ghi hướng đó. [plan-bipn.md](plan-bipn.md), mục việc bài tập cần có. Số dòng cart và click: [plan-bipn.md](plan-bipn.md), bảng bộ dữ liệu.

3. Mô hình dùng để so sánh có phải LightGCN chỉ trên hành vi đích, cùng cách chia tập và cùng HR@10, NDCG@10?
   Hiện đang ghi vậy. Repo BIPN có `lightGCN.py` nhưng không có script train riêng. [plan-bipn.md](plan-bipn.md). Code gốc: https://github.com/MingshiYan/BIPN

## Cấp 2. Bên trong BIPN

1. Bộ dữ liệu thứ hai có phải Retailrocket không?
   Đã ghi nhận Retailrocket. Bốn bộ có sẵn trong zip BIPN là Tmall, Taobao, Yelp và ML10M. Bảng so sánh: [plan-bipn.md](plan-bipn.md), mục bộ dữ liệu thứ hai. Trang dữ liệu: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset. File này chưa tải, nên chưa có số đếm của nhóm. Số 11.649 user, 36.223 item và 87.822 tương tác là Table 1 của bài MBHT, ghi lại trong [de-xuat-topic5-multi-behavior.md](de-xuat-topic5-multi-behavior.md).

2. Máy nào giữ bản train đủ epoch: máy RTX 3050 Ti 4 GB đang mở repo, hay máy Idealpad 5 Pro, Core Ultra 9 185H, RTX 4050, 32 GB RAM?
   Cấu hình máy thứ hai là lời kể, chưa kiểm trên máy đó. Máy đang mở repo đã kiểm bằng lệnh, nhưng PyTorch trong `bipn/.venv` vẫn là bản CPU. [plan-bipn.md](plan-bipn.md), mục điều kiện máy. [memory.md](memory.md).

3. Khi chạy Tmall, dùng tham số trong `b_tmall.sh` của repo gốc hay tham số mặc định trong `main.py` của repo gốc?
   File `.sh` dùng học 0.0001 và beta 0.3. `main.py` dùng học 0.001 và beta 0.5. Plan hiện ghi chạy Tmall theo file `.sh`, không dùng `cuda:6`. [plan-bipn.md](plan-bipn.md), mục mô hình cần trình bày.

4. Nếu số của khung `bipn/` lệch số repo gốc, nhóm có chấp nhận giữ khung theo công thức bài báo không?
   Các chỗ cố ý không chép code gốc nằm ở [plan-bipn.md](plan-bipn.md), mục khác biệt với code gốc. Khung: [../bipn/README.md](../bipn/README.md). Bài: https://arxiv.org/abs/2404.18166

5. Retailrocket sẽ chia train, validation và test theo quy tắc nào?
   Zip BIPN đã chia sẵn, mỗi bộ một kiểu: Taobao có đúng một dòng test và một dòng validation cho mỗi user, Tmall thì không. Chưa có quy tắc cho file Kaggle. [plan-bipn.md](plan-bipn.md), bảng bộ dữ liệu.

## Cấp 3. Bài báo và mẫu

1. Giữ mẫu LNCS như hiện tại, hay hỏi giảng viên vì slide chỉ ghi “Springer paper template”?
   Đang giữ LNCS 2.26. [../paper/main.tex](../paper/main.tex), [../README.md](../README.md), mục báo cáo Springer. [memory.md](memory.md).

2. Tên thật và email của hai thành viên là gì, để thay NVA và NVB?
   Báo cáo đang ghi hai tên tạm, chưa có email. [../paper/main.tex](../paper/main.tex).

3. Dòng cơ quan trên bài có giữ “Khoa HTTT”, hay ghi tên trường đầy đủ?
   [../paper/main.tex](../paper/main.tex).
