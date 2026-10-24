# BIPN

Khung huấn luyện theo mục 3.2 của *Behavior-Contextualized Item Preference Modeling* (SIGIR 2024).

Ba khối:

1. LightGCN trên đồ thị gộp mọi hành vi.
2. Mạng lọc sở thích theo user, item và hành vi.
3. LightGCN trên hành vi đích. Điểm cuối trộn hai nhánh bằng `lambda = 1 / số lần tương tác đích`.

Hành vi cuối trong danh sách là hành vi đích. Tmall dùng `click`, `collect`, `cart`, `buy`. Retailrocket dùng `view`, `cart`, `buy`.

| Mô hình | Nội dung |
|---|---|
| `bipn` | BIPN theo lệnh train Tmall của repo gốc. |
| `weighted` | Cùng mạng. Cart và hành vi đích có hệ số 2 trong BCE. |
| `learned` | Cùng mạng. Mỗi hành vi một hệ số BCE được học. |
| `lightgcn` | LightGCN chỉ trên đồ thị hành vi đích. |

Lệnh `learned` nằm ở [docs/chay-mo-hinh-cai-tien.md](../docs/chay-mo-hinh-cai-tien.md). Các lệnh còn lại nằm ở [docs/huong-dan-chay.md](../docs/huong-dan-chay.md).
