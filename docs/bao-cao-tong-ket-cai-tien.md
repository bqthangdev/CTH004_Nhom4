# Báo Cáo Tổng Kết Thực Nghiệm và Đối Sánh Hai Hướng Tiếp Cận Cải Tiến Mô Hình BIPN

- **Đề tài:** Gợi ý Đa hành vi (Multi-Behavior Recommendation) – Nghiên cứu Thực nghiệm, Khái quát hóa và Cải tiến Mô hình BIPN (SIGIR 2024).
- **Học phần:** Chuyên đề Phát triển các Hệ thống Thông tin Thông minh (CTH004).
- **Môi trường thực nghiệm:** GPU NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), PyTorch `2.14.1+cu130`, Python `3.12`.
- **Trạng thái thực nghiệm:** Toàn bộ mô hình thuộc hai hướng tiếp cận trên hai bộ dữ liệu **Tmall** và **Retailrocket** đã hội tụ và được xác thực độc lập thông qua [`bipn/run_check.py`](../bipn/run_check.py) kèm kiểm định ý nghĩa thống kê cặp (*paired t-test*).

---

## 1. Định hướng Nghiên cứu và Cấu trúc Thực nghiệm

Nghiên cứu được triển khai thông qua bốn nhiệm vụ học thuật trọng tâm:
1. **Tái lập mô hình gốc (Replication):** Thực thi độc lập mã nguồn chuẩn của mô hình BIPN (*Behavior-Contextualized Item Preference Modeling for Multi-Behavior Recommendation*, SIGIR 2024) trên tập dữ liệu chuẩn **Tmall** tại thư mục [`official-bipn/`](../official-bipn/).
2. **Kiểm chứng năng lực khái quát hóa liên miền (Cross-Domain Generalization):** Thiết lập quy trình tiền xử lý và chia tách theo dòng thời gian chống rò rỉ dữ liệu trên tập dữ liệu thương mại điện tử phương Tây **Retailrocket** ([`bipn/prepare_retailrocket.py`](../bipn/prepare_retailrocket.py)).
3. **Phân tích và đối sánh hai hướng tiếp cận cải tiến (Methodological Improvements):**
   - **Hướng tiếp cận 1 (`BIPN-InvWeight` & `BIPN-InvLearned`):** Tái cân bằng hàm mất mát entropy chéo nhị phân (BCE) bằng trọng số tỷ lệ nghịch với tần suất xuất hiện của hành vi ($w_b^{\mathrm{inv}} = \frac{\alpha_b N_{\mathrm{all}}}{n_b A}$) trên cấu hình cơ sở chưa chuẩn hóa xuyên tầng.
   - **Hướng tiếp cận 2 (`BIPN-Weighted` & `BIPN-Learned`):** Chuẩn hóa biểu diễn đơn vị L2 xuyên tầng kết hợp cơ chế **Điều hòa ý định logarit tách pha giữa huấn luyện và suy luận** ($\lambda_u^{\mathrm{rank}} = [1 + \log_2(N_u + 1)]^{-1}$) cùng **Trọng số hành vi bảo toàn thang đo trung bình** hoặc **Học tự thích nghi qua hàm chặn hyperbolic tangent**.
4. **Đối chứng định lượng với mô hình đơn hành vi (`LightGCN`):** Lượng hóa giá trị gia tăng mà dữ liệu đa hành vi mang lại trên cả hai cấu hình cơ sở (`LightGCN-Base`) và cấu hình chuẩn hóa đồng bộ (`LightGCN-Aligned`).

---

## 2. Cơ sở Lý thuyết của Hai Hướng Tiếp Cận Cải Tiến

### 2.1. Hướng tiếp cận 1: Tái cân bằng Nghịch đảo Tần suất trên Khung Cơ sở (`BIPN-InvWeight` & `BIPN-InvLearned`)
- **Giả thuyết:** Trong hàm mất mát $\mathcal{L}_{\mathrm{bce}}$ nguyên bản, mọi mẫu tương tác có trọng số ngang bằng, dẫn tới việc đạo hàm bị chi phối bởi hành vi xem lướt (`click`/`view`). Hướng tiếp cận thứ nhất gán hệ số ưu tiên $\alpha_b$ và định nghĩa trọng số mẫu tỷ lệ nghịch với tần suất hành vi $n_b$:
  $$w_b^{\mathrm{inv}} = \frac{\alpha_b \cdot N_{\mathrm{all}}}{n_b \cdot A}, \quad N_{\mathrm{all}} = \sum_{b \in B} n_b, \quad A = \sum_{b \in B} \alpha_b.$$
- **Hạn chế toán học được phát hiện qua thực nghiệm:**
  1. **Hiện tượng phóng đại đạo hàm cục bộ:** Khi một hành vi có tần suất rất nhỏ ($n_{\mathrm{cart}} = 1.996 \ll N_{\mathrm{all}} = 2.218.436$ trên Tmall), mẫu số $n_b$ làm trọng số $w_{\mathrm{cart}}^{\mathrm{inv}}$ của một mẫu đơn lẻ tăng gấp **$370{,}5$ lần**, gây xung đạo hàm làm co cụm các vector nhúng trên siêu cầu đơn vị (suy thoái tính đồng đều – *Uniformity collapse*).
  2. **Thiếu chuẩn hóa biểu diễn xuyên tầng:** Cấu hình cơ sở chưa áp dụng chuẩn hóa đơn vị L2 tại từng tầng tích chập đồ thị, thiếu kết nối dư ở nhánh đồ thị hành vi đích và sử dụng điều chuẩn tổng bình phương chưa chia trung bình theo số lượng đỉnh.

### 2.2. Hướng tiếp cận 2: Điều hòa Ý định Logarit Tách pha & Trọng số Hành vi Bảo toàn Thang đo (`BIPN-Weighted` & `BIPN-Learned`)
Hướng tiếp cận thứ hai giải quyết đồng thời bài toán ổn định không gian biểu diễn và cải thiện hệ số trộn heuristic $\lambda_u = 1/N_u$ được nêu tại Mục 4.4 của công bố gốc SIGIR 2024:
1. **Chuẩn hóa biểu diễn xuyên tầng và kết nối dư nhánh đích ([`bipn/model.py`](../bipn/model.py)):**
   $$\tilde{E}^{(l)} = \mathrm{Norm}_2\big(\tilde{A} \tilde{E}^{(l-1)}\big), \quad E_{\mathrm{all}} = E^{(0)} + \sum_{l=1}^L \left(1 + \frac{1}{l}\right)\tilde{E}^{(l)}, \quad \hat{E} = E_{\mathrm{all}} + \mathrm{GCN}_{\mathrm{buy}}(E_{\mathrm{all}}), \quad \mathcal{L}_{\mathrm{reg}} = \frac{\lVert E_u^{(0)} \rVert_F}{M} + \frac{\lVert E_i^{(0)} \rVert_F}{N}.$$
2. **Điều hòa ý định logarit tách pha Huấn luyện – Suy luận ($\lambda_u$):**
   - Trong **pha huấn luyện tương phản BPR (`is_eval=False`)**, mục tiêu dương $i \in \mathcal{N}_u^{(b_K)}$ được lấy mẫu từ chính tập lịch sử $\mathcal{N}_u^{(b_K)}$ dùng để tính tổng sở thích $\mathbf{p}_u = \sum_{k \in \mathcal{N}_u^{(b_K)}} \mathbf{p}_{uk b_K}$. Để ngăn ngừa hiện tượng rò rỉ tự tương quan mục tiêu (*target self-correlation leakage*) khi $N_u = 1$, hệ số điều hòa huấn luyện được duy trì ở mức $\lambda_u^{\mathrm{train}} = 1/N_u$ (tức trọng số nhánh sở thích điểm là $1 - \lambda_u^{\mathrm{train}} = \frac{N_u - 1}{N_u}$, tương ứng chính xác với tỷ lệ ngữ cảnh độc lập).
   - Trong **pha suy luận xếp hạng (`is_eval=True`)**, mọi mục ứng viên $i \notin \mathcal{N}_u^{(b_K)}$ đều độc lập với tập lịch sử huấn luyện ($N_{\mathrm{ctx}}^{\mathrm{rank}} = N_u$). Hệ số điều hòa được xác lập theo hàm suy giảm logarit:
     $$\lambda_u^{\mathrm{rank}} = \begin{cases} 1, & \text{nếu } N_u = 0, \\ \dfrac{1}{1 + \log_2(N_u + 1)}, & \text{nếu } N_u \ge 1. \end{cases}$$
     Công thức này kích hoạt nhánh sở thích mục tiêu cho người dùng có một giao dịch ($N_u = 1 \implies \lambda_u^{\mathrm{rank}} = 0{,}50$) và duy trì đóng góp của nhánh đồ thị cộng tác khi $N_u \ge 2$.
3. **Trọng số hành vi bảo toàn thang đo (`BIPN-Weighted`) và Học tự thích nghi có miền chặn (`BIPN-Learned`):**
   $$w_b^{\mathrm{norm}} = \frac{|B_{\mathrm{act}}| \cdot \alpha_b}{\sum_{k \in B_{\mathrm{act}}} \alpha_k}, \qquad w_b^{\mathrm{learn}} = |B_{\mathrm{act}}| \cdot \frac{\exp\big(\ln \alpha_b + \delta \tanh(\theta_b)\big)}{\sum_{k \in B_{\mathrm{act}}} \exp\big(\ln \alpha_k + \delta \tanh(\theta_k)\big)},$$
   với $\alpha_{\mathrm{cart}} = \alpha_{\mathrm{buy}} = 1{,}02$, $\delta = 0{,}02$ và điều chuẩn $0{,}01\lVert \boldsymbol{\theta} \rVert_2^2 / |B|$, đảm bảo trọng số trung bình luôn bằng $1{,}0$ và nằm trong miền chặn giải tích $[\alpha_b e^{-\delta}, \alpha_b e^{\delta}]$.

---

## 3. Bảng Tổng Hợp Kết Quả Thực Nghiệm Chính Thức

Bảng dưới đây đối chiếu toàn diện kết quả giữa công bố gốc SIGIR 2024, bản tái lập mã nguồn gốc, Hướng tiếp cận 1 và Hướng tiếp cận 2 trên cả hai bộ dữ liệu (tương ứng trực tiếp với **Bảng 2** trong [`paper/main.tex`](../paper/main.tex); ký hiệu $^*$ và $^{**}$ biểu thị mức ý nghĩa thống kê $p < 0{,}01$ và $p < 0{,}001$ theo kiểm định *paired t-test*):

| Bộ dữ liệu | Phân nhóm Tiếp cận | Mô hình Thực nghiệm | Best Epoch | Valid HR@10 | Test HR@10 | Test NDCG@10 | Test HR@20 | Test NDCG@20 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tmall** | Tham chiếu công bố gốc | BIPN (SIGIR 2024) | — | — | 0,1414 | 0,0741 | 0,2033 | 0,0898 |
| **Tmall** | Tái lập chuẩn ($\lambda_u = 1/N_u$, pat=5) | BIPN (`official-bipn`) | Ep 33 | 0,2249 | 0,1398 | 0,0736 | 0,2000 | 0,0884 |
| **Tmall** | **Hướng tiếp cận 1** | `LightGCN-Base` (Đối chứng) | Ep 20 | 0,0236 | 0,0141 | 0,0070 | 0,0197 | 0,0083 |
| **Tmall** | *(Khung cơ sở, trọng số $N_{\mathrm{all}}/n_b$)* | `BIPN-InvLearned` (Softmax $N_{\mathrm{all}}/n_b$) | Ep 1 | 0,0154 | 0,0089 | 0,0045 | 0,0132 | 0,0055 |
| **Tmall** | | `BIPN-InvWeight` ($\alpha=2$) | Ep 4 | 0,0120 | 0,0071 | 0,0032 | 0,0127 | 0,0046 |
| **Tmall** | **Hướng tiếp cận 2** | `LightGCN-Aligned` (Đối chứng) | Ep 56 | 0,0595 | 0,0369 | 0,0199 | 0,0492 | 0,0229 |
| **Tmall** | *(Chuẩn hóa đồng bộ, $\lambda_u^{\mathrm{rank}}$ logarit* | `BIPN-Aligned` ($\lambda_u = 1/N_u$, pat=10) | Ep 47 | 0,2268 | 0,1410 | 0,0732 | 0,2032 | 0,0884 |
| **Tmall** | *& Trọng số bảo toàn thang đo)* | **`BIPN-Learned` (Đề xuất)** | **Ep 47** | **0,2391**$^{**}$ | **0,1460**$^{*}$ | **0,0772**$^{*}$ | **0,2075**$^{*}$ | **0,0923**$^{*}$ |
| **Tmall** | | **`BIPN-Weighted` (Đề xuất)** | **Ep 58** | **0,2397**$^{**}$ | **0,1472**$^{**}$ | **0,0771**$^{*}$ | **0,2108**$^{**}$ | **0,0928**$^{**}$ |
| **Retailrocket** | **Hướng tiếp cận 1** | `LightGCN-Base` (Đối chứng) | Ep 1 | 0,0011 | 0,0003 | 0,0001 | 0,0007 | 0,0002 |
| **Retailrocket** | *(Khung cơ sở, trọng số $N_{\mathrm{all}}/n_b$)* | `BIPN-Base` ($\lambda_u = 1/N_u$) | Ep 1 | 0,0814 | 0,1275 | 0,0901 | 0,1515 | 0,0962 |
| **Retailrocket** | | `BIPN-InvWeight` ($\alpha=2$) | Ep 1 | 0,0757 | 0,1229 | 0,0865 | 0,1482 | 0,0929 |
| **Retailrocket** | **Hướng tiếp cận 2** | `LightGCN-Aligned` (Đối chứng) | Ep 12 | 0,0226 | 0,0046 | 0,0035 | 0,0055 | 0,0037 |
| **Retailrocket** | *(Chuẩn hóa đồng bộ, $\lambda_u^{\mathrm{rank}}$ logarit* | `BIPN-Aligned` ($\lambda_u = 1/N_u$) | Ep 5 | 0,4983 | 0,8571 | 0,7451 | 0,8787 | 0,7506 |
| **Retailrocket** | *& Trọng số bảo toàn thang đo)* | **`BIPN-Weighted` (Đề xuất)** | **Ep 2** | **0,5198**$^{*}$ | **0,8590** | **0,7459** | **0,8805** | **0,7513** |
| **Retailrocket** | | **`BIPN-Learned` (Đề xuất)** | **Ep 4** | **0,5198**$^{*}$ | **0,8599**$^{*}$ | **0,7462** | **0,8805** | **0,7514** |

### Kiểm định ý nghĩa thống kê cặp (Paired t-test):
- **Trên Tmall (`29.285` người dùng tập kiểm tra):**
  - `BIPN-Weighted` so với `official-bipn` (Ep 33): $\text{HR@10}$ ($0{,}1472$ vs $0{,}1398$, $t = 3{,}39, p = 7{,}02 \times 10^{-4}$), $\text{NDCG@10}$ ($0{,}0771$ vs $0{,}0736$, $t = 2{,}74, p = 6{,}22 \times 10^{-3}$), $\text{HR@20}$ ($0{,}2108$ vs $0{,}2000$, $t = 4{,}37, p = 1{,}23 \times 10^{-5}$), $\text{NDCG@20}$ ($0{,}0928$ vs $0{,}0884$, $t = 3{,}56, p = 3{,}76 \times 10^{-4}$).
  - Bóc tách riêng $\lambda_u^{\mathrm{rank}}$ logarit so với $\lambda_u = 1/N_u$ trên cùng bộ trọng số `BIPN-Weighted`: $\text{Test HR@10}$ tăng từ $0{,}1447 \to 0{,}1472$ ($t = 3{,}74, p = 1{,}84 \times 10^{-4}$), $\text{Test NDCG@10}$ tăng từ $0{,}0754 \to 0{,}0771$ ($t = 6{,}38, p = 1{,}78 \times 10^{-10}$).
- **Trên Retailrocket (`885` người dùng tập xác thực & `11.719` người dùng tập kiểm tra):**
  - `BIPN-Learned` so với `BIPN-Aligned`: $\text{Valid HR@10}$ ($0{,}5198$ vs $0{,}4983$, $t = 2{,}98, p = 2{,}96 \times 10^{-3}$), $\text{Valid NDCG@10}$ ($0{,}3375$ vs $0{,}3233$, $t = 3{,}53, p = 4{,}34 \times 10^{-4}$), $\text{Test HR@10}$ ($0{,}8599$ vs $0{,}8571$, $t = 3{,}08, p = 2{,}09 \times 10^{-3}$).

---

## 4. Danh mục Tệp Minh chứng Thực nghiệm trong Kho lưu trữ

- **Bản thảo bài báo khoa học chuẩn Springer LNCS:** [`paper/main.tex`](../paper/main.tex) và danh mục tài liệu tham khảo [`paper/refs.bib`](../paper/refs.bib).
- **Checkpoints của Hướng tiếp cận 2 (Đang hoạt động):** `bipn/checkpoints/*.pt` và nhật ký huấn luyện `bipn/logs/*.log`.
- **Lưu trữ kết quả của Hướng tiếp cận 1 và các bản chạy đối chứng:**
  - [`backups/20261025_suboptimal_results_pre_improvement/`](../backups/20261025_suboptimal_results_pre_improvement/README.md)
  - [`backups/20261027_pre_architecture_alignment_and_lambda_improvement/`](../backups/20261027_pre_architecture_alignment_and_lambda_improvement/README.md)
- **Mã nguồn bài báo gốc phục vụ kiểm chứng tái lập:** [`official-bipn/`](../official-bipn/).
