# Các hướng cải tiến

Giảng viên yêu cầu cải tiến bằng cách đọc nhiều bài trước, ghi lại chỗ mỗi bài sửa, rồi tổng hợp thành một thay đổi của phương pháp. Lý do sửa phải nêu được trước khi nhìn bảng điểm của nhóm.

Hướng 1 đến hướng 4 bên dưới không đạt chuẩn đó. Cả bốn đều xuất phát từ việc đếm dòng click, collect, cart và buy của bộ đang dùng. Công thức BCE có khác bản gốc, nhưng lý do chọn hệ số nằm ở phân bố của bộ đó. Trên một bộ có phễu khác, cùng hệ số không còn cùng ý nghĩa. Số cao hơn trên một bộ cũng chưa phải một cải tiến phương pháp theo chuẩn này. Trong lần đã chạy, bản có trọng số còn không cao hơn: trên Retailrocket, cùng khung `bipn/`, HR@10 giảm từ 0,1275 xuống 0,1229.

Mỗi hướng dưới đây có bốn ý: vấn đề, giả thuyết, chỗ sửa, và cách kiểm tra. Chỉ hướng 1 đã được cài và đã có số đo.

## Đã cài: `--model learned`

Mỗi hành vi một hệ số BCE được học, theo ý của MBGCN (học trọng số hành vi). Chỉ đổi `BIPN._bce_loss`; `rank`, `_aggregate`, `_bpr_loss` và LightGCN không đổi, nên hành vi phụ vẫn không vào điểm xếp hạng cuối.

- `n_b` = số dòng dương của hành vi b × (1 + `neg_count`), `N` = tổng `n_b`.
- `behavior_logit`: một số mỗi hành vi, `nn.Parameter` khởi tạo 0 để Adam cập nhật.
- `alpha = softmax(log(n_b) + behavior_logit)`, `w_b = alpha_b * N / n_b`. Hành vi có `n_b = 0` bị bỏ khỏi softmax.
- Logit bằng 0 thì `w_b = 1` với mọi hành vi, trùng BCE không trọng số. Softmax giữ tổng alpha bằng 1 nên các hành vi tranh nhau một tổng cố định.

Lệnh chạy một epoch (trong `bipn`, dữ liệu Tmall; đổi `--data`/`--dataset` cho Retailrocket):

```text
.venv\Scripts\python.exe main.py --data ..\data\Tmall --dataset tmall --model learned --device cuda --epochs 1
```

Lệnh train đầy đủ (200 epoch theo `config.py`, dừng sớm theo `--patience`):

```text
.venv\Scripts\python.exe main.py --data ..\data\Tmall --dataset tmall --model learned --device cuda
```

Chưa có HR/NDCG của `learned`. Mới kiểm tra `--smoke` và các tính chất của hệ số. `learned` không nằm trong `STEPS` của `run_pipeline.py`, nên số trong bài báo không đổi.

## Đề xuất lại: đổi cách tính λ của nhánh GCN

Hướng này thay cho hướng 1 khi giải thích phần sửa mô hình. Chưa cài. Chưa có số đo.

### Các bài đã ghi

MBGCN, SIGIR 2020, gộp sở thích của từng hành vi bằng một trọng số riêng cho hành vi đó. Bài chỉ ra rằng cộng các hành vi như nhau sẽ đưa nhiễu vào biểu diễn dùng để xếp hạng. DOI 10.1145/3397271.3401123.

CRGCN, TOIS 2024, cùng nhóm tác giả với BIPN. Embedding đi theo thứ tự hành vi, mỗi tầng cộng phần dư của tầng trước. Lý do của bài là hành vi đứng trước hỗ trợ hành vi mua. arXiv 2205.13128.

BIPN, SIGIR 2024, không đưa sở thích của hành vi phụ vào điểm cuối. Hành vi phụ chỉ để học tham số và tạo embedding ban đầu. Điểm cuối trộn hai nguồn bằng λ. Công thức trong bài và trong `bipn/model.py` là λ = 1/N_u khi người dùng có N_u lần mua trong tập train, và λ = 1 khi N_u = 0. Nguồn thứ nhất là sở thích theo từng món đã mua. Nguồn thứ hai là nhánh GCN trên đồ thị mua. DOI 10.1145/3626772.3657696.

Mục 4.4 của BIPN thử λ cố định 0,3, 0,5 và 0,7. λ càng lớn thì HR@10 và NDCG@10 càng giảm, nên không thay công thức bằng một hằng số. Cùng mục nói heuristic 1/N_u phân bổ chưa tốt với người có ít lần mua: sở thích theo món chưa đủ tin cậy, nhánh GCN cần được tính nhiều hơn. Kết luận của bài ghi hướng tiếp theo là một cách phân bổ trọng số khác cho nhánh này. Bài không đưa công thức thay thế.

### Chỗ tổng hợp

MBGCN và CRGCN đều sửa cách hành vi phụ đi vào điểm xếp hạng. BIPN cố ý không đi theo hướng đó, vì bài coi việc đổ hành vi phụ vào điểm cuối là nguồn nhiễu. Chỗ BIPN để ngỏ nằm sau quyết định đó: hệ số trộn λ vẫn là một heuristic, và chính bài nói heuristic này yếu khi số lần mua còn ít. Sửa λ giữ nguyên lập luận của BIPN. Sửa trọng số BCE theo số dòng click, như hướng 1, không nằm trong lập luận này.

### Giả thuyết

Với người có vài lần mua, 1/N_u đã nhỏ, nhánh GCN gần như không còn trong điểm. Một hàm vẫn phụ thuộc N_u nhưng giảm chậm hơn sẽ để nhánh GCN tham gia nhiều hơn đúng nhóm người mà mục 4.4 nói là đang bị phân bổ kém. HR@10 và NDCG@10 trên tập test khi đó cao hơn BIPN cùng khung code.

### Chỗ sửa

Chỉ đổi λ trong công thức điểm. Không đổi BCE, không thêm tầng, không đổi embedding, tốc độ học, beta, gamma, batch, số mẫu âm hay seed.

Khi N_u = 0, λ = 1, giữ như code hiện tại. Khi N_u ≥ 1:

`λ = 1 / (1 + log2(N_u))`

Công thức này không có trong bài BIPN. Đây là công thức nhóm chọn để giảm chậm hơn 1/N_u, và để λ không thành một hằng số mà mục 4.4 đã thử. Không chỉnh đáy log theo tập test.

| N_u | λ hiện tại, 1/N_u | λ đề xuất |
|---:|---:|---:|
| 1 | 1 | 1 |
| 2 | 0,50 | 0,50 |
| 4 | 0,25 | 0,33 |
| 8 | 0,13 | 0,25 |

Ở một và hai lần mua, hai công thức trùng nhau. Từ bốn lần mua trở đi, nhánh GCN nặng hơn bản hiện tại. Bốn cũng là số lần mua trung bình trên Tmall trong Table 1 của bài BIPN.

### Cách kiểm tra

So `--model bipn` với một lệnh mới cùng khung `bipn/`, trên Tmall và trên Retailrocket. Không so với code gốc trong `official-bipn/`, vì hai khung đã khác nhau ở lớp lọc. Khóa công thức trước khi chạy. Tập test chỉ chấm một lần. Giả thuyết được số ủng hộ khi cả HR@10 và NDCG@10 trên test cao hơn bản λ = 1/N_u ở cùng bộ.

Mã của hướng 1 nằm ở `bipn/`, lệnh `--model weighted`. Code gốc trong `official-bipn/` không bị sửa theo phép này.

## Hướng 1. Đã làm: tăng phần đóng góp của giỏ hàng và mua trong BCE

**Vấn đề.** BCE gốc lấy trung bình đều mọi dòng. Trên Tmall, train có 1.813.498 dòng click, 221.514 dòng collect, 1.996 dòng cart và 181.428 dòng buy. Với 4 mẫu âm, click chiếm khoảng 9,1 triệu trong 11,1 triệu dòng BCE. Lần học vì vậy gần như chỉ thấy click, trong khi bảng báo cáo lại chấm hành vi mua.

**Giả thuyết.** Nếu giỏ hàng và mua chiếm phần lớn hơn trong BCE, mô hình sẽ xếp hạng món mua tốt hơn. Giỏ hàng được đưa vào cùng mua vì đó là hành vi sát quyết định mua, không phải vì giỏ hàng nhiều dòng.

**Vì sao sửa BCE, không sửa mạng.** Ba khối của BIPN, cách tính điểm, BPR và L2 giữ nguyên. Embedding, tốc độ học, beta, gamma, batch, số mẫu âm và seed cũng giữ bộ của `b_tmall.sh`. Sửa một chỗ thì khi HR@10 đổi, có thể nói sự đổi đó gắn với trọng số. Thêm tầng hoặc đổi tốc độ học cùng lúc thì không tách được nguyên nhân.

**Chỗ sửa.** Cart và hành vi đích có alpha 2. Hành vi đích trên cả hai bộ là buy. Hành vi còn lại có alpha 1: click và collect trên Tmall, view trên Retailrocket. Số dòng BCE của một hành vi là số dòng dương nhân `1 + neg_count`. Gọi `n_b` là số dòng của hành vi `b`, `N` là tổng số dòng, `A` là tổng alpha. Trọng số một dòng là

`alpha_b * N / (n_b * A)`

Phần đóng góp của hành vi `b` khi đó tỉ lệ với `alpha_b`. Cart và buy mỗi bên chiếm gấp đôi click hoặc collect. Tổng trọng số vẫn bằng `N`, nên thang BCE không bị phóng lên. Hệ số 2 nằm ở `behavior_boost` trong `bipn/config.py`. Hệ số này được chọn trước khi train, không dò trên tập xác thực.

**Cách kiểm tra.** So HR@10 và NDCG@10 trên tập test. Giả thuyết đúng khi bản có trọng số cao hơn bản BCE đều, trong cùng một khung code. Pipeline chạy `--model bipn` trên Retailrocket và `--model weighted` trên cả hai bộ.

Trên Retailrocket, cùng khung `bipn/`, HR@10 giảm từ 0,1275 xuống 0,1229. Giả thuyết không được số này ủng hộ. Trên Tmall, dòng BIPN trong bảng là code gốc ở `official-bipn` (0,1398), dòng có trọng số là khung `bipn/` (0,0071). Hai dòng này còn khác nhau ở lớp lọc và cách cộng tầng, ghi ở [plan-bipn.md](plan-bipn.md), nên mức giảm trên Tmall không phải hiệu của riêng trọng số.

## Hướng 2. Chưa làm: chọn alpha trên tập xác thực

**Vấn đề.** Hướng 1 khóa alpha bằng 2 trước khi nhìn dữ liệu. Số 2 không có cơ sở đo được. Trên Tmall, trọng số một dòng cart lên tới khoảng 370, và HR@10 của bản này rất thấp.

**Giả thuyết.** Một alpha nhỏ hơn, chọn vì làm HR@10 trên tập xác thực cao nhất, sẽ đỡ nhấn giỏ hàng quá mạnh và giữ được tín hiệu của click.

**Chỗ sẽ sửa.** Vẫn chỉ nhân trọng số BCE. Thử vài giá trị `behavior_boost`, giữ epoch theo HR@10 xác thực, rồi chấm test một lần với giá trị đã chọn. Không dùng tập test để chọn alpha.

## Hướng 3. Chưa làm: cho collect một alpha riêng

**Vấn đề.** Hướng 1 gộp collect cùng click, cùng alpha 1. Trên Tmall, collect có 221.514 dòng, ít hơn click nhiều nhưng nhiều hơn cart rất nhiều. Coi collect như click là một lựa chọn chưa được kiểm tra.

**Giả thuyết.** Collect gần với ý định giữ món hơn là xem lướt. Một alpha nằm giữa 1 và alpha của cart có thể tăng phần đóng góp của collect mà không phóng đại cart.

**Chỗ sẽ sửa.** Trong `_alphas` của `bipn/config.py`, collect nhận một hệ số riêng. Cart và buy giữ hệ số của riêng chúng. Các hành vi xem vẫn là 1.

## Hướng 4. Chưa làm: lấy số dòng BCE bằng nhau

**Vấn đề.** Hướng 1 vẫn tạo đủ dòng cho click, rồi nhân trọng số lớn vào vài dòng cart. Một dòng cart hệ số 370 chi phối gradient của batch có dòng đó.

**Giả thuyết.** Nếu mỗi hành vi đóng góp cùng số dòng BCE, giỏ hàng và mua vẫn được học, nhưng không có dòng nào bị nhân hàng trăm lần.

**Chỗ sẽ sửa.** Khi tạo mẫu, mỗi hành vi lấy cùng một số dòng dương, rồi kèm 4 mẫu âm như cũ. Không nhân `w_b`. Đây là đổi cách lấy mẫu, không đổi công thức mạng.

## Hai câu trong kết luận của bài nhóm

Mục kết luận của `paper/main.tex` viết hai hướng cho lần sau. Cả hai chưa có mã. Không dùng hai câu này để giải thích lần trọng số BCE đã chạy.

**Để mô hình tự chỉnh trọng số hành vi.** Ý này có bài đứng phía trước: MBGCN học một trọng số khi gộp từng hành vi, vì cộng đều đưa nhiễu vào điểm. Giảng viên cũng đề xuất hướng này khi báo cáo tiến độ. Câu trong kết luận thêm "dựa trên entropy thông tin" nhưng không trích bài nào cho entropy, nên không dùng entropy. Chỗ cài thêm nằm ở BCE, trong `BIPN._bce_loss`: mỗi hành vi một hệ số học được, chuẩn hóa để tổng cố định. `behavior_weight` hiện là buffer khóa sẵn, Adam không sửa. Không nhân hệ số của click, collect hay cart vào `rank` hay `_aggregate`, vì đó là điểm cuối và BIPN không đưa hành vi phụ vào điểm đó.

Ghép với ba hướng chưa làm:

- Hướng 2, dò alpha trên tập xác thực, trùng việc. Cùng một hệ số mà BCE đọc. Dò ra một số rồi để mạng học đè lên số đó.
- Hướng 3, alpha riêng cho collect, cũng trùng. Collect đã là một phần tử trong vectơ hệ số học.
- Hướng 4, mỗi hành vi cùng số dòng BCE, ghép được. Một bên đổi số dòng trong `bce_rows`. Một bên đổi hệ số nhân vào dòng. Bỏ thừa số `N/n_b`, nên không còn dòng cart bị nhân khoảng 370.

Công thức λ = 1/(1+log2(N_u)) cũng ghép được, vì nó chỉ thay `1/count` trong `_aggregate`. Học một hệ số thay cho λ thì trùng công thức này, không làm cả hai.

Tổ hợp không trùng là: hệ số học trong BCE, cộng hướng 4, cộng công thức λ. Bỏ hướng 2 và hướng 3. Ba thay đổi một lúc thì bảng điểm không tách được phần nào làm HR@10 đổi. Lần so đầu tiên vẫn là một lệnh chỉ bật hệ số học, cùng khung `bipn/`, đối với `--model bipn`.

**Đưa chuỗi thời gian vào mô hình.** Ý này trùng MBHT, KDD 2022: dự đoán món tiếp theo bằng Transformer và hypergraph đa hành vi, arXiv 2207.05584. Đó là một phương pháp khác trong danh sách bài, không phải một chỗ sửa bên trong BIPN. BIPN xếp hạng món mua từ đồ thị, không dự đoán món kế tiếp. File Tmall trong `data.zip` chỉ còn cặp `user item`, không còn thời điểm. Retailrocket có thời điểm trong `events.csv`, nhưng sau khi chuyển sang khung BIPN cũng chỉ còn cặp `user item`. Muốn làm hướng này phải đổi nhiệm vụ, đổi dữ liệu và đổi mô hình. Nhóm đã xét MBHT trong `docs/de-xuat-topic5-multi-behavior.md` và không chọn làm phương pháp chính.
