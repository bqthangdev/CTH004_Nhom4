# Hướng dẫn chạy

Tài liệu này dành cho người vừa lấy repo về và cần chạy thí nghiệm trên máy của mình. Làm lần lượt từ mục 1 đến mục 6. Mục 6 ghi số vào báo cáo. Mục 7 chỉ dùng khi muốn đổi tham số sau các lần chạy đó. Dữ liệu không được đưa vào git, nên sau khi lấy code, thư mục `data/` chưa có file huấn luyện.

Cả hai máy dùng cùng một quy trình và cùng train bằng GPU của máy đó. Mở PowerShell tại thư mục gốc của repo, là thư mục có file `README.md`. Python dùng bản 3.12.

Mọi khối lệnh bên dưới đều chạy trong PowerShell. Không dán vào cửa sổ cmd. `Expand-Archive`, `New-Item`, `Copy-Item` và `Remove-Item` chỉ có trong PowerShell, dán vào cmd sẽ báo không nhận lệnh. Các lệnh `python` chạy được trong cmd, nhưng vẫn gõ ở cùng cửa sổ PowerShell để khỏi chuyển qua lại. Nếu đang mở cmd, gõ `powershell` rồi Enter để sang PowerShell.

Mục 4 chạy code BIPN gốc trên Tmall. Đó là lần chạy lại code mà bài tập yêu cầu. Mục 5 chạy bản cải tiến, LightGCN, và cả ba mô hình trên Retailrocket. Lần chạy đầu giữ nguyên `bipn/config.py`. Chỉ sửa tham số ở mục 7, sau khi mục 4 và mục 5 đã xong.

## 1. Tải dữ liệu

Thí nghiệm dùng hai bộ. Tmall lấy từ file đính kèm code BIPN. Retailrocket tải từ Kaggle.

### Tmall

Tải `data.zip` từ repo BIPN. File khoảng 47 MB. Bên trong zip đã có sẵn một thư mục tên `data`, nên phải giải nén tại thư mục gốc của repo. Nếu giải nén vào trong thư mục `data/` thì các bộ sẽ nằm ở `data/data/Tmall` và pipeline sẽ không thấy file.

Đứng ở thư mục gốc, chạy khối PowerShell sau:

```text
curl.exe -L -o data.zip https://raw.githubusercontent.com/MingshiYan/BIPN/main/data.zip
Expand-Archive -Path data.zip -DestinationPath . -Force
Remove-Item data.zip
```

Lệnh thứ nhất tải zip. Lệnh thứ hai giải nén đúng vào repo. Lệnh thứ ba xóa zip vì phần cần dùng đã nằm trong `data/`.

Chạy xong, mở `data/Tmall/count.txt`. Nội dung phải là `{"user": 41738, "item": 11953}`. Cùng lúc đó sẽ có thêm `data/taobao`, `data/Yelp` và `data/ML10M`. Ba thư mục này đi kèm trong zip, nhưng thí nghiệm hiện tại không dùng.

Nếu thấy đường dẫn `data/data/Tmall`, zip đã bị giải nén thêm một cấp. Chuyển bốn thư mục `Tmall`, `taobao`, `Yelp` và `ML10M` lên trực tiếp dưới `data/`, rồi xóa thư mục `data` nằm thừa bên trong.

### Retailrocket

Bộ này không có trong `data.zip`. Mở https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset, đăng nhập Kaggle và tải `archive.zip`. Đặt file đó tại `data/retailrocket/archive.zip`. Zip chứa bốn file CSV ngay ở tầng ngoài cùng, không bọc thêm thư mục.

Vẫn đứng ở thư mục gốc, chạy khối PowerShell sau:

```text
New-Item -ItemType Directory -Force data\retailrocket
Expand-Archive -Path data\retailrocket\archive.zip -DestinationPath data\retailrocket -Force
```

Lệnh thứ nhất tạo thư mục nếu chưa có. Lệnh thứ hai giải bốn file CSV vào đúng thư mục đó.

Sau khi giải nén phải có `data/retailrocket/events.csv`. Huấn luyện chỉ đọc file này. `category_tree.csv` và hai file `item_properties` nằm cùng zip, nhưng mô hình không dùng.

## 2. Tiền xử lý

Mục này không có lệnh. File Tmall dùng cho khung `bipn/` không cần xử lý thêm. Lệnh `data_process.py` chỉ dành cho code gốc, và nằm ở mục 4.

Tmall đã nằm sẵn ở dạng từng dòng `user item`, và đã được chia thành train, validation, test. Khung của nhóm đọc trực tiếp các file `.txt`, nên không chạy `data_process.py` cho mục 5. File đó chỉ được chạy trong bản code gốc, ở mục 4.

Retailrocket thì `events.csv` chưa đúng dạng đó. Việc chuyển cũng không có lệnh riêng. Nó nằm trong lệnh `run_pipeline.py` ở mục 5. Khi lệnh đó bắt đầu, chương trình tìm `data/retailrocket/count.txt`. Chưa có file này thì chương trình tự chuyển `events.csv`, in dòng `retailrocket users 11719 items 31839`, rồi mới train. Đã có file này thì chương trình bỏ qua và dùng các file đã tạo.

Khi tự chuyển, chương trình chỉ giữ visitor có ít nhất một giao dịch mua, sắp các món đã mua theo thời gian và bỏ món trùng:

- Từ ba món mua trở lên, món cuối cùng vào test, món kế cuối vào validation, các món còn lại vào train.
- Đúng hai món thì món đầu vào train và món cuối vào test.
- Đúng một món thì món đó vào test.
- Lượt xem và lượt thêm vào giỏ chỉ được đưa vào train khi xảy ra trước món mua bị tách ra sớm nhất.

Trên bản `events.csv` hiện có, bước này ra 11.719 user và 31.839 item. Các file đếm được `view.txt` 88.304 dòng, `cart.txt` 22.716 dòng, `buy.txt` 8.666 dòng, `test.txt` 11.719 dòng và `validation.txt` 885 dòng. Validation ngắn vì đa số visitor không có đủ ba món mua khác nhau.

Trước khi sang mục 4, chỉ cần nhìn xem đủ file Tmall hay chưa. `events.csv` cần có trước mục 5. Danh sách dưới đây là tên file, không phải lệnh. Thiếu file Tmall thì quay lại mục 1.

```text
data/Tmall/count.txt
data/Tmall/click.txt
data/Tmall/collect.txt
data/Tmall/cart.txt
data/Tmall/buy.txt
data/Tmall/test.txt
data/Tmall/validation.txt
data/retailrocket/events.csv
```

## 3. Môi trường GPU

Mỗi máy tự tạo một môi trường trong `bipn/.venv`. Không dùng chung thư mục này giữa hai máy, và không đưa nó vào git. Bản PyTorch mặc định của cả hai máy là `torch 2.14.1+cu130`, vì train chạy trên GPU. Nếu một máy đã tạo venv với bản CPU từ trước, lệnh cài torch bên dưới thay bản đó bằng bản GPU.

Đứng ở thư mục gốc, chạy khối PowerShell sau:

```text
cd bipn
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Năm lệnh này lần lượt chuyển vào thư mục `bipn`, tạo môi trường, cài NumPy cùng các gói trong `requirements.txt`, cài PyTorch bản GPU, rồi in kết quả kiểm tra. Dòng in ra phải bắt đầu bằng `True` và kèm tên GPU của máy. Nếu in `False`, driver NVIDIA chưa dùng được với bản CUDA này. Cập nhật driver, rồi chạy lại hai lệnh cuối. Chưa thấy `True` thì chưa sang mục 4.

## 4. Chạy code gốc

Thư mục `official-bipn` là bản đã lấy từ https://github.com/MingshiYan/BIPN. Các chỗ đã sửa và file log của tác giả được ghi trong [official-bipn/SUA-DOI.md](../official-bipn/SUA-DOI.md). Không clone lại và không sửa tay các file trong thư mục đó.

Code gốc cần thêm vài file sinh từ `data_process.py`. File này có sẵn trong `data/Tmall`. Mục 3 đang đứng trong `bipn`. Nếu chưa có `pos_sampling.txt`, `test_dict.txt` và `validation_dict.txt`, chạy khối này một lần:

```text
cd ..\data\Tmall
..\..\bipn\.venv\Scripts\python.exe data_process.py
cd ..\..\official-bipn
.\run_tmall.ps1
```

Nếu ba file đó đã có, bỏ qua `data_process.py` và từ `bipn` chạy:

```text
cd ..\official-bipn
.\run_tmall.ps1
```

`run_tmall.ps1` đã gắn tham số của `b_tmall.sh`, với GPU `cuda:0`.

Nếu PowerShell báo không cho chạy script, dùng `powershell -ExecutionPolicy Bypass -File .\run_tmall.ps1` thay cho dòng thứ hai.

Cửa sổ và file log in đủ các mốc `hit` và `ndcg`. Log mới nằm trong `official-bipn/log/model/`, tên có ngày giờ của lần chạy. Bốn file đã có sẵn trong thư mục đó là log của tác giả, không dùng để điền báo cáo.

Code gốc giữ epoch có `hit@20` trên validation cao nhất, và có thể dừng trước 200 epoch nếu chỉ số này không tăng. Dòng `final test result is` trong log mới là kết quả test của epoch đó. Dòng này giữ đủ các mốc. Cách chọn số cho bảng nằm ở mục 6.

## 5. Chạy pipeline

Mục này có hai lệnh, và cả hai đều chạy trong thư mục `bipn`. Nếu đang đứng trong `official-bipn`, quay lại bằng `cd ..\bipn`. Lệnh thứ nhất kiểm tra xem dữ liệu, GPU và chương trình có thông nhau hay không. Lệnh thứ hai mới là bản train dùng để ghi vào báo cáo.

### Kiểm tra bằng một epoch

Chạy thử trước để phát hiện lỗi đường dẫn, lỗi GPU hoặc lỗi dữ liệu khi mỗi bước mới đi được một epoch. Kết quả của lần này không đưa vào báo cáo. Lệnh này cũng là lúc Retailrocket được chuyển, nếu mục 2 nói rằng `count.txt` chưa có.

```text
.venv\Scripts\python.exe run_pipeline.py --epochs 1
```

Đợi đến khi cả năm bước đều in `done`. Nếu một bước dừng giữa chừng, mở file log được nêu ở cuối mục này, sửa lỗi, rồi chạy lại lệnh trên. Chỉ khi lệnh thử đi hết năm bước mới chạy lệnh train.

### Train đầy đủ

Lệnh train không truyền `--epochs`. Khi đó mỗi bước lấy số epoch trong `bipn/config.py`, hiện là 200.

```text
.venv\Scripts\python.exe run_pipeline.py
```

### Mô hình cải tiến và chạy tiếp

Lệnh của `--model learned` nằm ở [chay-mo-hinh-cai-tien.md](chay-mo-hinh-cai-tien.md). Mở PowerShell trong thư mục `bipn`, dán lần lượt ba lệnh của từng bộ. Đợi dòng `epoch N ...` in xong rồi mới tắt máy. Lệnh có `--resume` chạy tiếp từ epoch sau. Quên `--resume` thì chương trình từ chối, không ghi đè lần đang dở. `run_pipeline.py` không chạy mô hình này.

### Pipeline tự chạy những gì

Hai lệnh ở trên đều gọi một chương trình. Chương trình đó tự đi hết năm bước trong bảng dưới đây. Không gõ năm lệnh riêng, và không cần đổi mô hình bằng tay giữa các bước. Dòng Tmall BIPN không nằm ở đây, vì dòng đó lấy từ code gốc ở mục 4.

| Bước | Bộ dữ liệu | Mô hình | Việc bước này làm |
|---|---|---|---|
| 1 | Tmall | `weighted` | Chạy bản cải tiến: tăng trọng số của cart và hành vi đích |
| 2 | Tmall | `lightgcn` | Chạy mô hình đối chứng, chỉ dùng hành vi đích |
| 3 | Retailrocket | `bipn` | Chạy BIPN trên bộ dữ liệu thứ hai |
| 4 | Retailrocket | `weighted` | Chạy bản cải tiến trên bộ dữ liệu thứ hai |
| 5 | Retailrocket | `lightgcn` | Chạy mô hình đối chứng trên bộ dữ liệu thứ hai |

Cửa sổ lệnh chỉ in hai loại dòng: `start` khi một bước bắt đầu, và `done` khi bước đó xong. Các dòng loss, HR và NDCG của từng epoch không hiện ở cửa sổ này. Chúng được ghi vào `bipn/logs/`. Bước 1 ghi `tmall_weighted.log`, bước 2 ghi `tmall_lightgcn.log`, và các bước sau ghi theo cùng cách đặt tên. Muốn xem pipeline còn chạy hay đã dừng, mở file log của bước vừa in `start`.

Nếu một bước bị lỗi, pipeline dừng tại bước đó và không chạy các bước phía sau. Thư mục `bipn/logs/` không được đưa vào git.

Mỗi bước còn ghi một file mô hình vào `bipn/checkpoints/`. Bước 1 ghi `tmall_weighted.pt`, bước 2 ghi `tmall_lightgcn.pt`, và các bước sau ghi theo cùng cách đặt tên. Trong một bước, chương trình giữ epoch có HR@10 trên validation cao nhất. Nếu hai epoch bằng nhau ở chỉ số này, nó giữ epoch có NDCG@10 trên validation cao hơn. Log của bước có dòng `save epoch` mỗi khi file được đổi. Lệnh thử một epoch cũng ghi các file này, rồi lệnh train đầy đủ ghi đè bằng epoch tốt nhất của bản 200 epoch. Thư mục `bipn/checkpoints/` không được đưa vào git.

## 6. Ghi số vào báo cáo

Train in đủ các mốc mà từng chương trình vẫn in. Không bỏ bớt mốc nào. Bảng trong `paper/main.tex` chỉ có hai cột, HR@10 và NDCG@10 trên tập test, nên khi điền bảng thì dùng đúng hai số đó. Các số khác vẫn giữ trong log để đối chiếu.

Với code gốc, mở log mới của mục 4. Dòng `final test result is` có `hit@10`, `hit@20`, `hit@50`, `hit@80` và các mốc `ndcg` tương ứng.

- `hit@10` điền vào HR@10 của dòng Tmall BIPN. `ndcg@10` điền vào NDCG@10 của dòng đó.
- `hit@20` là mốc code gốc dùng để chọn epoch. Giữ trong log, không điền vào bảng.
- Các mốc 50 và 80 cũng giữ trong log, không điền vào bảng.

Năm dòng còn lại lấy từ pipeline. Đứng trong thư mục `bipn`, chạy:

```text
.venv\Scripts\python.exe run_check.py
```

Lệnh này không train thêm. Với mỗi mô hình, cửa sổ in ba dòng. Dòng `checkpoint` là epoch đang nằm trong file. Dòng `saved` có cụm `valid` và cụm `test`, mỗi cụm có HR và NDCG ở mốc 10 và 20. Dòng `reload` chấm lại từ file. Khi dữ liệu không đổi, cụm `test` của hai dòng này trùng nhau. Nếu báo chưa có file, bước train tương ứng chưa chạy xong.

- `HR@10` và `NDCG@10` trong cụm `test` là hai số điền vào bảng.
- Cụm `valid` cho biết vì sao epoch đó được giữ. Giữ trong log, không điền vào bảng.
- Mốc 20 giữ trong log, không điền vào bảng.

Chỉ chạy lệnh này sau lệnh train đầy đủ của mục 5. Lệnh thử một epoch cũng tạo file mô hình, nên chạy kiểm tra ngay sau lệnh đó thì các file vẫn chỉ là bản một epoch.

| Dòng `checkpoint` | Dòng trong bảng báo cáo |
|---|---|
| `tmall weighted` | Tmall, BIPN có trọng số |
| `tmall lightgcn` | Tmall, LightGCN |
| `retailrocket bipn` | Retailrocket, BIPN |
| `retailrocket weighted` | Retailrocket, BIPN có trọng số |
| `retailrocket lightgcn` | Retailrocket, LightGCN |

Sau khi điền sáu ô, sửa câu ở mục Kết luận đang nói là chưa có số đo.

## 7. Sửa cấu hình

Mục này chỉ làm sau khi mục 4 và lệnh train của mục 5 đã chạy xong với cấu hình gốc. Sửa tham số trước lần đó thì kết quả không còn là lần chạy để ghi báo cáo.

Khi muốn thử một bộ tham số khác cho pipeline, sửa `bipn/config.py`, rồi chạy lại hai lệnh ở mục 5. Năm bước dùng chung file này. Tham số của code gốc không nằm trong file đó. Muốn đổi lần chạy code gốc thì sửa lệnh ở mục 4 và chạy lại mục 4. Trước khi chạy lại, sao chép `bipn/logs/`, `bipn/checkpoints/` và `official-bipn/log/` ra chỗ khác. Lần chạy sau ghi đè các file này.

Bảng dưới là bộ đang dùng. Các giá trị lấy từ file `b_tmall.sh` của repo BIPN, trừ thiết bị `cuda:6` vì máy của nhóm không dùng GPU đó. Trong repo gốc, `main.py` để sẵn tốc độ học 0.001 và beta 0.5, nhưng lệnh train Tmall ghi đè hai số này bằng 0.0001 và 0.3.

| Trường | Giá trị gốc | Ý nghĩa |
|---|---|---|
| `embedding_size` | 64 | Kích thước vector biểu diễn của user và item. |
| `n_layers` | 2 | Số tầng lan truyền sau tầng gốc. Tầng thứ `l` được nhân với `1/(l+1)`. |
| `lr` | 0.0001 | Tốc độ học của Adam. |
| `beta` | 0.3 | Tỷ lệ của hàm BCE so với BPR. Chỉ hai mô hình `bipn` và `weighted` dùng hệ số này. |
| `gamma` | 0.0001 | Hệ số phạt L2 trên hai bảng embedding. |
| `weight_decay` | 0 | Phân rã trọng số của Adam. Giữ bằng 0 để L2 không bị cộng thêm một lần nữa. |
| `batch_size` | 1024 | Số dòng trong một bước cập nhật, đồng thời là số user được chấm điểm trong một lần. |
| `neg_count` | 4 | Số món âm lấy cho mỗi món dương khi tính BCE. BPR vẫn chỉ lấy một món âm cho mỗi user. |
| `epochs` | 200 | Số epoch của lệnh train. Lệnh kiểm tra truyền `--epochs 1` nên không dùng số này. |
| `topk` | 10 và 20 | Hai mốc HR và NDCG được in ra log. Bảng trong báo cáo dùng mốc 10. |
| `seed` | 2021 | Hạt giống của `random`, NumPy và PyTorch, để các lần chạy cùng cấu hình bám cùng một khởi tạo. |
| `behavior_boost` | 2 | Chỉ mô hình `weighted` dùng. Cart và hành vi đích nhận hệ số 2. Các hành vi còn lại nhận hệ số 1. |
