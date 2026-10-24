# Câu cần xác nhận

Trạng thái các quyết định của bài tập. Plan nằm ở [plan-bipn.md](plan-bipn.md).

## Đã chốt

| Mục | Quyết định | Tài liệu |
|---|---|---|
| Phương pháp | BIPN | [de-xuat-topic5-multi-behavior.md](de-xuat-topic5-multi-behavior.md) |
| Sửa mô hình đã train | Trọng số cố định cho cart và buy, `--model weighted` | [cai-tien.md](cai-tien.md) |
| Sửa mô hình đang chạy | Hệ số BCE học được, `--model learned` | [chay-mo-hinh-cai-tien.md](chay-mo-hinh-cai-tien.md) |
| Tác giả | Huỳnh Chí Dũng và Bành Quốc Thắng, Trường Công nghệ Thông tin, Đại học Cần Thơ | [../paper/main.tex](../paper/main.tex) |
| Mô hình đối chứng | LightGCN trên hành vi đích, `--model lightgcn` | https://doi.org/10.1145/3397271.3401063 |
| Bộ dữ liệu thứ hai | Retailrocket | https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset |
| Phép chia Retailrocket | Theo thời gian trong `prepare_retailrocket.py` | [plan-bipn.md](plan-bipn.md) |
| Tham số train | Bộ `b_tmall.sh`: học 0.0001, beta 0.3. Không dùng `cuda:6` | [huong-dan-chay.md](huong-dan-chay.md) |
| Máy train | Cả hai máy, mỗi máy dùng GPU của máy đó | [huong-dan-chay.md](huong-dan-chay.md) |
| Chạy lại code | Repo BIPN gốc trên Tmall. Không dùng `cuda:6` | [huong-dan-chay.md](huong-dan-chay.md) |

## Còn mở

1. Mẫu báo cáo giữ LNCS 2.26, hay đổi sau khi hỏi giảng viên? Slide chỉ ghi “Springer paper template”. [../paper/main.tex](../paper/main.tex).
