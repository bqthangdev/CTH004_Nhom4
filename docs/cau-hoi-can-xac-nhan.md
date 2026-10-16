# Câu cần xác nhận

Trạng thái các quyết định của bài tập. Plan nằm ở [plan-bipn.md](plan-bipn.md).

## Đã chốt

| Mục | Quyết định | Tài liệu |
|---|---|---|
| Phương pháp | BIPN | [de-xuat-topic5-multi-behavior.md](de-xuat-topic5-multi-behavior.md) |
| Sửa mô hình | Trọng số cart và hành vi đích trong BCE, `--model weighted` | [plan-bipn.md](plan-bipn.md) |
| Mô hình đối chứng | LightGCN trên hành vi đích, `--model lightgcn` | https://doi.org/10.1145/3397271.3401063 |
| Bộ dữ liệu thứ hai | Retailrocket | https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset |
| Phép chia Retailrocket | Theo thời gian trong `prepare_retailrocket.py` | [plan-bipn.md](plan-bipn.md) |
| Tham số train | Bộ `b_tmall.sh`: học 0.0001, beta 0.3. Không dùng `cuda:6` | [huong-dan-chay.md](huong-dan-chay.md) |
| Máy train | Cả hai máy, mỗi máy dùng GPU của máy đó | [huong-dan-chay.md](huong-dan-chay.md) |
| Chạy lại code | Repo BIPN gốc trên Tmall. Không dùng `cuda:6` | [huong-dan-chay.md](huong-dan-chay.md) |

## Còn mở

1. Mẫu báo cáo giữ LNCS 2.26, hay đổi sau khi hỏi giảng viên? Slide chỉ ghi “Springer paper template”. [../paper/main.tex](../paper/main.tex), [../README.md](../README.md).

2. Tên thật và email của hai thành viên, để thay NVA và NVB trong [../paper/main.tex](../paper/main.tex).

3. Dòng cơ quan giữ “Khoa HTTT”, hay ghi tên trường đầy đủ? [../paper/main.tex](../paper/main.tex).
