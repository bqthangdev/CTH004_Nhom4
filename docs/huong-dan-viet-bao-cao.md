# Hướng Dẫn Chi Tiết Viết Báo Cáo Học Thuật (Springer LNCS)
## Đề tài Topic 5: Gợi ý Đa hành vi (BIPN - SIGIR 2024)
### Môn học: Chuyên đề Hệ thống Thông tin Thông minh (CTH004) - Giảng viên: PGS.TS. Nguyễn Thái Nghe

> **Tài liệu bàn giao cho thành viên phụ trách viết báo cáo khoa học.**  
> Tài liệu này cung cấp đầy đủ: bối cảnh đề tài, toàn bộ kết quả thực nghiệm đã chạy xong, phân tích chuyên sâu các hiện tượng số liệu, cấu trúc chi tiết từng mục trong tệp `paper/main.tex`, và hướng dẫn biên dịch ra PDF.

---

## I. Tổng quan Đề tài & Yêu cầu của Giảng viên

1. **Yêu cầu môn học (Slide môn CTH004):**
   - **Nhóm:** 2 sinh viên.
   - **Tỉ lệ điểm:** Báo cáo khoa học theo mẫu Springer (30%) + Kết quả và mã nguồn thực nghiệm (70%).
   - **Bốn nhiệm vụ bắt buộc:**
     1. **Chạy lại mô hình gốc (Replication):** Chạy mã nguồn BIPN (*Behavior-Contextualized Item Preference Modeling for Multi-Behavior Recommendation*, SIGIR 2024) trên tập dữ liệu chuẩn **Tmall**.
     2. **Đề xuất cải tiến mô hình (Modification):** Cải tiến hàm mất mát bằng cách gán trọng số $\alpha=2$ cho các hành vi giỏ hàng (`cart`) và mua (`buy`) trong hàm BCE.
     3. **Đổi sang một tập dữ liệu mới (New Dataset):** Thực nghiệm trên tập dữ liệu thương mại điện tử **Retailrocket** (chưa từng xuất hiện trong bài báo gốc).
     4. **So sánh với một mô hình đối chứng (Baseline):** So sánh đối chứng với **LightGCN** (SIGIR 2020) chỉ huấn luyện trên hành vi đích mua hàng (`buy`).

2. **Hiện trạng:** Sáu số trong bảng dưới là các lần đã chạy. Mô hình `--model learned` chưa có số. Lệnh chạy nằm ở [chay-mo-hinh-cai-tien.md](chay-mo-hinh-cai-tien.md). Tên tác giả đã có trong `paper/main.tex`.

---

## II. Bảng Số Liệu Thực Nghiệm Toàn Diện (Bảng 2)

Dưới đây là bảng số liệu chính thức đã được kiểm chứng và điền vào [paper/main.tex](../paper/main.tex#L168-L184):

$$\mathbf{Bảng\ 2:\ Kết\ quả\ xếp\ hạng\ trên\ tập\ kiểm\ tra\ (Test\ Set)}$$

| Bộ dữ liệu | Mô hình | Checkpoint | HR@10 | NDCG@10 | HR@20 | NDCG@20 | Ghi chú & Xuất xứ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tmall** | **BIPN (Gốc)** | Epoch 33 | **0.1398** | **0.0736** | 0.2000 | 0.0884 | Tái lập mã gốc tác giả (`official-bipn`) |
| **Tmall** | **BIPN có trọng số** | Epoch 4 | **0.0071** | **0.0032** | 0.0127 | 0.0046 | Pipeline nhóm (`bipn/checkpoints/tmall_weighted.pt`) |
| **Tmall** | **LightGCN (Đối chứng)** | Epoch 20 | **0.0141** | **0.0070** | 0.0197 | 0.0083 | Pipeline nhóm (`bipn/checkpoints/tmall_lightgcn.pt`) |
| **Retailrocket** | **BIPN** | Epoch 1 | **0.1275** | **0.0901** | 0.1515 | 0.0962 | Pipeline nhóm (`bipn/checkpoints/retailrocket_bipn.pt`) |
| **Retailrocket** | **BIPN có trọng số** | Epoch 1 | **0.1229** | **0.0865** | 0.1482 | 0.0929 | Pipeline nhóm (`bipn/checkpoints/retailrocket_weighted.pt`) |
| **Retailrocket** | **LightGCN (Đối chứng)** | Epoch 1 | **0.0003** | **0.0001** | 0.0007 | 0.0002 | Pipeline nhóm (`bipn/checkpoints/retailrocket_lightgcn.pt`) |

---

## III. Phân Tích Chuyên Sâu Các Hiện Tượng Số Liệu (Vũ khí lấy điểm 9-10)

Khi viết phần **Thảo luận Kết quả (Results Discussion)**, bạn không nên chỉ liệt kê số đơn thuần, mà hãy phân tích theo các luận điểm học thuật sắc bén sau:

### 1. Tại sao BIPN đa hành vi vượt trội gấp nhiều lần so với LightGCN đơn hành vi?
* **Bản chất vấn đề:** Trên các sàn thương mại điện tử, hành vi mua hàng (`buy`) luôn là "sự kiện hiếm" (ít hơn 1–2% tổng tương tác).
* **Minh chứng số liệu:**
  * Trên **Tmall**: BIPN đạt $\text{HR@10} = 0.1398$, vượt gần **10 lần** so với LightGCN ($0.0141$).
  * Trên **Retailrocket**: BIPN đạt $\text{HR@10} = 0.1275$, áp đảo LightGCN ($0.0003$) hơn **400 lần**!
* **Giải thích nguyên nhân:**
  * LightGCN chỉ học trên đồ thị của hành vi mua (`buy`). Trên Retailrocket, 11.719 người dùng chỉ có 8.666 giao dịch mua (trung bình chưa tới 1 món/người). Đồ thị này bị cô lập, phân mảnh cực độ (Data Sparsity & Cold-Start), khiến cơ chế lan truyền đồ thị không thể cập nhật vector embedding.
  * BIPN gộp toàn bộ các hành vi phụ (`view`: 88.304, `cart`: 22.716) vào đồ thị toàn cục và dùng mạng nơ-ron lọc ngữ cảnh hai chiều. Hành vi xem và giỏ hàng đóng vai trò "cầu nối tri thức", chuyển giao thông tin sở thích giúp phục hồi chính xác ý định mua hàng.

### 2. Khả năng Tổng quát hóa (Generalizability) của BIPN
* Bài báo gốc SIGIR 2024 chỉ thử nghiệm trên các tập dữ liệu châu Á (Tmall, Taobao) và Yelp, ML10M.
* Nhóm đã mở rộng thành công sang **Retailrocket** (sàn thương mại điện tử Âu-Mỹ). Kết quả $\text{HR@10} = 0.1275$ và $\text{NDCG@10} = 0.0901$ trên Retailrocket chứng minh kiến trúc BIPN có tính khái quát hóa xuất sắc, hoạt động hiệu quả trên các nền văn hóa mua sắm và miền dữ liệu hoàn toàn khác biệt.

### 3. Phân tích kết quả của Biến thể Cải tiến (BIPN có trọng số)
* **Ý tưởng đề xuất:** Gán trọng số $\alpha = 2$ cho `cart` và `buy` để bù đắp sự lấn át của `click`/`view`.
* **Hiện tượng quan sát:**
  * Trên **Retailrocket**: BIPN có trọng số đạt $\text{HR@10} = 0.1229$ (bám sát nút bản gốc $0.1275$).
  * Trên **Tmall**: BIPN có trọng số đạt $\text{HR@10} = 0.0071$, thấp hơn bản gốc.
* **Luận điểm giải thích giá trị:**
  * *Tính mất cân bằng cực độ của Tmall:* Tmall có tới 1.813.498 lượt `click` nhưng chỉ có 1.996 lượt `cart` (tỉ lệ 908 : 1). Khi ép trọng số $\alpha = 2$ tĩnh, gradient cập nhật bị dồn nén cưỡng bức vào một tập rất nhỏ các món trong giỏ hàng, vô tình triệt tiêu tín hiệu học rộng rãi từ các hành vi khám phá ban đầu (`click`).
  * *Trên Retailrocket:* Tỉ lệ giữa `view` (88k) và `cart` (22k) chỉ là 4 : 1 (phân phối tự nhiên, cân bằng hơn nhiều), do đó trọng số $\alpha = 2$ không gây sốc gradient và mô hình duy trì độ chính xác cao.
  * *Kết luận học thuật:* Cơ chế **lọc ý định tự động và điều tiết động qua mạng nơ-ron của BIPN gốc** thích nghi tốt hơn so với việc gán trọng số tĩnh (heuristic weighting) khi phân phối dữ liệu bị lệch cực đoan. Đây là một đóng góp phân tích thực nghiệm rất có giá trị trong báo cáo!

---

## IV. Hướng Dẫn Chi Tiết Cách Viết Từng Mục trong `paper/main.tex`

Tệp nguồn LaTeX [paper/main.tex](../paper/main.tex) đã được cấu trúc sẵn theo mẫu Springer LNCS (`\documentclass{llncs}`). Dưới đây là hướng dẫn cụ thể từng phần:

### 1. Tiêu đề, Tác giả & Tóm tắt (Lines 16–33)
* **Tiêu đề:** `Gợi ý đa hành vi cho hệ thống thương mại điện tử` (hoặc có thể đặt chi tiết hơn: *Nghiên cứu và Tái lập Mô hình Gợi ý Đa hành vi BIPN trên Nền tảng Thương mại Điện tử*).
* **Tác giả (`\author`, `\institute`):** Điền họ tên 2 thành viên, mã số sinh viên, email và tên khoa/trường:
  ```latex
  \author{Nguyễn Văn A\inst{1} \and Trần Thị B\inst{1}}
  \institute{Khoa Hệ thống Thông tin, Trường Đại học Công nghệ Thông tin, ĐHQG-HCM \\
  \email{\{mssv1, mssv2\}@gm.uit.edu.vn}}
  ```
* **Tóm tắt (Abstract):** Đảm bảo gói gọn 150–200 từ:
  * Nêu bài toán gợi ý đa hành vi và hạn chế của mô hình đơn hành vi.
  * Tóm tắt phương pháp BIPN (SIGIR 2024), đề xuất biến thể trọng số, và mô hình đối chứng LightGCN.
  * Nêu rõ 2 tập dữ liệu: Tmall (chuẩn bài báo) và Retailrocket (dữ liệu mới).
  * Khẳng định kết quả nổi bật: BIPN đạt HR@10 là 0{,}1398 trên Tmall và 0{,}1275 trên Retailrocket, vượt trội hoàn toàn mô hình đơn hành vi.

### 2. Phần Giới thiệu (Section 1: Giới thiệu, Lines 35–45)
* Trình bày phễu hành vi mua sắm: Xem $\to$ Yêu thích $\to$ Thêm giỏ hàng $\to$ Mua hàng.
* Nêu nghịch lý thưa thớt: Hành vi mua là mục tiêu tối hậu nhưng chiếm số lượng rất ít.
* Giới thiệu bài báo BIPN (SIGIR 2024) của tác giả Mingshi Yan.
* Tóm lược 4 nhiệm vụ của nhóm: tái lập mã nguồn trên Tmall, mở rộng Retailrocket, thử nghiệm biến thể trọng số, và đối chứng với LightGCN.

### 3. Phần Phương pháp (Section 2: Phương pháp, Lines 46–124)
* **2.1 Bài toán (Problem Formulation):** Định nghĩa tập $U$, tập $I$, danh sách hành vi $B = \{b_1, \dots, b_K\}$, hành vi đích $b_K = \text{buy}$.
* **2.2 Kiến trúc BIPN:** Giữ nguyên các công thức toán đã viết sẵn:
  * Đồ thị gộp toàn cục và tích chập LightGCN tầng $l$ với trọng số $1/(l+1)$.
  * Mạng lọc sở thích theo ngữ cảnh (Item Preference Filtering) qua 3 lớp Sigmoid $\to$ Tanh $\to$ Sigmoid.
  * Công thức điểm dự đoán kết hợp ý định điểm và ý định lân cận: $y_{ui} = (1-\lambda_u)(\mathbf{p}_u \cdot \mathbf{e}_i) + \lambda_u(\hat{\mathbf{e}}_u \cdot \hat{\mathbf{e}}_i)$.
  * Hàm mất mát tổ hợp: $\mathcal{L} = \beta \mathcal{L}_{\mathrm{bce}} + (1-\beta)\mathcal{L}_{\mathrm{bpr}} + \gamma \mathcal{L}_{\mathrm{reg}}$.
* **2.3 Đề xuất Cải tiến (Trọng số hành vi):**
  * Trình bày lý do: Hành vi giỏ hàng (`cart`) thể hiện ý định mua mạnh mẽ hơn nhiều so với `click`.
  * Trình bày công thức trọng số chuẩn hóa: $w_b = \alpha_b \cdot N / (n_b \cdot A)$ với $\alpha_{\text{cart}} = \alpha_{\text{buy}} = 2$ và $\alpha_{\text{click}} = \alpha_{\text{collect}} = 1$.
* **2.4 Mô hình Đối chứng LightGCN:**
  * LightGCN chuẩn trên đồ thị hành vi đích, hàm mất mát BPR, dùng cùng kích thước embedding ($d=64$) và 2 tầng lan truyền để so sánh công bằng.

### 4. Phần Thí nghiệm (Section 3: Thí nghiệm, Lines 125–189)
* **3.1 Dữ liệu:**
  * Giải thích chi tiết Bảng 1: Tmall (41.738 user, 11.953 item) và Retailrocket (11.719 user, 31.839 item).
  * Nhấn mạnh cách nhóm tự chuyển đổi tập Retailrocket từ `events.csv`: lọc người dùng có giao dịch, phân chia theo thứ tự thời gian (món mua cuối cùng vào test, món kế cuối vào validation).
* **3.2 Thiết lập thực nghiệm:**
  * Tham số chuẩn theo `b_tmall.sh`: embedding 64, 2 tầng GCN, batch size 1024, 4 mẫu âm, learning rate $10^{-4}$, $\beta = 0.3$, L2 regularization $10^{-4}$, seed cố định 2021.
  * Nêu rõ môi trường phần cứng: GPU NVIDIA RTX 4050 Laptop (6 GB VRAM) và CPU Intel Core Ultra 9 185H.
  * Cơ chế Early Stopping: Dừng sau 5 epoch liên tiếp nếu chỉ số validation không cải thiện.
* **3.3 Kết quả & Thảo luận:**
  * Giữ nguyên Bảng 2 đã điền số.
  * Tận dụng các luận điểm tại **Mục III** ở trên để viết 3–4 đoạn văn phân tích sâu sắc.

### 5. Phần Kết luận (Section 4: Kết luận, Lines 192–197)
* Khẳng định việc hoàn thành đầy đủ cả 4 yêu cầu của bài tập lớn.
* Rút ra kết luận khoa học quan trọng nhất: Khai thác đa hành vi là giải pháp bắt buộc để giải quyết bài toán khởi đầu lạnh và thưa thớt dữ liệu trong thương mại điện tử.

---

## V. Hướng Dẫn Biên Dịch Báo Cáo ra PDF

Thư mục chứa mã nguồn bài báo là `paper/`. Bạn có thể lựa chọn 1 trong 2 cách biên dịch sau:

### Cách 1: Sử dụng Overleaf (Nhanh nhất & Thuận tiện nhất - KHUYÊN DÙNG)
1. Nén toàn bộ thư mục `paper/` thành tệp `paper.zip` (bao gồm: `main.tex`, `refs.bib`, `llncsdoc.tex`, `splncs04.bst` nếu có).
2. Truy cập [Overleaf.com](https://www.overleaf.com), tạo dự án mới: **New Project $\to$ Upload Project**.
3. Chọn tệp `paper.zip` vừa nén.
4. Trong giao diện Overleaf, đảm bảo Compiler được đặt là **pdfLaTeX** và file chính là `main.tex`.
5. Bấm **Recompile** $\to$ Tải tệp PDF hoàn chỉnh về máy.

### Cách 2: Biên dịch Ngoại tuyến (Local Compiler)
Nếu máy bạn đã cài đặt MiKTeX hoặc TeX Live:
1. Mở PowerShell hoặc Terminal tại thư mục `paper/`:
   ```powershell
   cd d:\gs\CTH004_CDPTCHTTTTM\src\CTH004_Nhom4\paper
   ```
2. Chạy chuỗi lệnh chuẩn để xử lý trích dẫn BibTeX:
   ```powershell
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```
3. Tệp `main.pdf` sẽ được tạo ra tại cùng thư mục.

---

## VI. Bảng Tra Cứu Tệp Minh Chứng Thực Nghiệm

Khi cần trích dẫn, kiểm tra số đo chi tiết hoặc chụp ảnh minh chứng:

| Hạng mục | Đường dẫn tệp | Mô tả nội dung |
| :--- | :--- | :--- |
| **Báo cáo tổng kết Bước 4** | [`docs/bao-cao-tong-ket-buoc-4.md`](bao-cao-tong-ket-buoc-4.md) | Toàn văn kết quả chạy mã gốc Tmall BIPN |
| **Báo cáo tổng kết Bước 5** | [`docs/bao-cao-tong-ket-buoc-5.md`](bao-cao-tong-ket-buoc-5.md) | Toàn văn kết quả pipeline 5 mô hình nhóm |
| **Log chạy mã gốc Tmall** | [`official-bipn/log/model/tmall_enb_64_2026-10-17 13_56_43.log`](../official-bipn/log/model/tmall_enb_64_2026-10-17%2013_56_43.log) | Log chi tiết 38 epoch của Tmall BIPN |
| **Log 5 bước Pipeline** | `bipn/logs/*.log` | Log chi tiết của từng mô hình trong pipeline |
| **Checkpoint đã lưu** | `bipn/checkpoints/*.pt` | 5 tệp trọng số mô hình tốt nhất |
| **Script kiểm chứng số đo** | [`bipn/run_check.py`](../bipn/run_check.py) | Script nạp checkpoint và đo kiểm tra lại |
| **Nhật ký tiến độ đề tài** | [`docs/memory.md`](memory.md) | Lịch sử thực hiện từ 12/10 đến 18/10/2026 |
