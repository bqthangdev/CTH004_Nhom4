# Cải tiến: trọng số hành vi trong BCE

Đây là phần sửa mô hình của bài tập. Mã nằm ở `bipn/`, lệnh `--model weighted`. Code gốc trong `official-bipn/` không bị sửa theo phép này.

## Việc đã sửa

Mạng, đồ thị và cách tính điểm giữ nguyên như BIPN. BPR và L2 cũng giữ nguyên.

Chỗ đổi là BCE. Code gốc lấy trung bình mọi dòng mẫu, mỗi dòng một hệ số. Trên Tmall, train có 1.813.498 dòng click, 221.514 dòng collect, 1.996 dòng cart và 181.428 dòng buy. Click nhiều hơn cart rất nhiều, nên trung bình đó gần như chỉ học từ click.

Bản `weighted` gán alpha cho từng hành vi, rồi nhân vào từng dòng BCE trước khi lấy trung bình:

- Cart và hành vi đích có alpha 2. Trên Tmall và Retailrocket, hành vi đích là buy.
- Hành vi còn lại có alpha 1. Trên Tmall là click và collect. Trên Retailrocket là view.

Số dòng BCE của một hành vi là số dòng dương nhân với `1 + neg_count`. Với `neg_count = 4`, mỗi món dương kéo theo 4 món âm. Gọi `n_b` là số dòng của hành vi `b`, `N` là tổng số dòng, `A` là tổng alpha. Trọng số một dòng của hành vi `b` là

`alpha_b * N / (n_b * A)`

Phần đóng góp kỳ vọng của hành vi `b` khi đó tỉ lệ với `alpha_b`. Cart và buy mỗi hành vi chiếm gấp đôi click hoặc collect, dù cart ít dòng. Tổng trọng số vẫn bằng `N`, nên BCE không bị đổi thang so với bản lấy trung bình thường.

Hệ số 2 nằm ở `behavior_boost` trong `bipn/config.py`.

## Việc không sửa

Không thêm tầng, không đổi kích thước embedding, không đổi tốc độ học, beta, gamma, batch, số mẫu âm hay seed. Các giá trị đó vẫn là bộ của `b_tmall.sh`.

LightGCN đối chứng là mô hình khác, không phải bản cải tiến này.

## Số nào dùng để đánh giá

Bảng báo cáo dùng HR@10 và NDCG@10 trên tập test.

- Dòng Tmall BIPN là lần chạy code gốc ở `official-bipn`.
- Dòng BIPN có trọng số là lần chạy `--model weighted`.
- Hai số này không chỉ khác mỗi trọng số. Khung `bipn/` còn khác code gốc ở lớp lọc và cách cộng tầng LightGCN. Các chỗ đó ghi ở [plan-bipn.md](plan-bipn.md).

Vì vậy hiệu của riêng trọng số được thấy khi so `--model weighted` với `--model bipn` trong cùng khung `bipn/`. Pipeline hiện chạy `bipn` trên Retailrocket, và chạy `weighted` trên cả Tmall lẫn Retailrocket.

## Hướng khác

Chưa có số đo nên chưa kết luận hướng nào tốt hơn bản alpha cố định.

Ba hướng còn cùng ý, chưa được cài:

- Chọn alpha trên validation thay vì cố định bằng 2.
- Cho collect một alpha riêng, vì collect cũng ít hơn click nhưng nhiều hơn cart.
- Lấy số dòng BCE bằng nhau cho mỗi hành vi, thay vì nhân trọng số vào loss.
