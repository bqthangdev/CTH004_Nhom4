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

## Còn mở

1. Máy nào chạy đủ epoch: RTX 3050 Ti 4 GB trên máy đang mở repo, hay Idealpad 5 Pro (Core Ultra 9 185H, RTX 4050, 32 GB RAM)? Cấu hình máy thứ hai là thông tin do thành viên cung cấp, chưa kiểm trên máy đó. PyTorch trong `bipn/.venv` hiện là bản CPU. [plan-bipn.md](plan-bipn.md).

2. Tham số Tmall dùng bộ nào: `b_tmall.sh` của repo gốc (học 0.0001, beta 0.3) hay mặc định `main.py` của repo gốc (học 0.001, beta 0.5)? Không dùng `cuda:6` ghi trong file shell. [plan-bipn.md](plan-bipn.md).

3. Khi số của khung `bipn/` lệch repo gốc, có giữ khung theo công thức bài báo không? Các điểm khác nằm ở [plan-bipn.md](plan-bipn.md). Bài: https://arxiv.org/abs/2404.18166. Khung: [../bipn/README.md](../bipn/README.md).

4. Mẫu báo cáo giữ LNCS 2.26, hay đổi sau khi hỏi giảng viên? Slide chỉ ghi “Springer paper template”. [../paper/main.tex](../paper/main.tex), [../README.md](../README.md).

5. Tên thật và email của hai thành viên, để thay NVA và NVB trong [../paper/main.tex](../paper/main.tex).

6. Dòng cơ quan giữ “Khoa HTTT”, hay ghi tên trường đầy đủ? [../paper/main.tex](../paper/main.tex).
