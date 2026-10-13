Họ tên: Trần Thị Lan - 02621
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 4.20 | 0.0 | 0.144 | 0.121–0.168 | 1.65 | 0.183 | 0.131 | 0.137 |
| E1 | LoRA LLM | 18579456 | 6.82 | 18.5 | 0.485 | 0.448–0.523 | 0.97 | 0.548 | 0.433 | 0.535 |
| E2 | + projector | 127827968 | 8.45 | 23.2 | 0.536 | 0.499–0.574 | 0.87 | 0.610 | 0.484 | 0.579 |
| E3 | + LoRA ViT | 132548608 | 10.92 | 28.7 | 0.572 | 0.535–0.609 | 0.80 | 0.651 | 0.520 | 0.611 |
| E4 | rank: r=4 | 4644864 | 6.48 | 17.8 | 0.431 | 0.392–0.470 | 1.06 | 0.492 | 0.391 | 0.459 |
| E4 | rank: r=64 | 74317824 | 7.35 | 19.5 | 0.478 | 0.440–0.516 | 0.98 | 0.537 | 0.430 | 0.525 |
| E5 | Bonus GRPO (IoU) | 132548608 | 12.14 | 15.0 | 0.598 | 0.561–0.634 | 0.76 | 0.672 | 0.547 | 0.636 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Phân bổ tham số kiến trúc**:
  + **ViT (Vision Transformer)**: Xấp xỉ **415M tham số** (gồm 24 khối transformer blocks, patch size 14, xử lý linh hoạt độ phân giải hình ảnh; ảnh dashcam 640×360 qua ViT sinh ra chính xác **220 token** thị giác).
  + **Projector**: Xấp xỉ **109.3M tham số** (bao gồm `visual.merger` ≈ 27.3M và `visual.deepstack_merger_list` ≈ 82.0M chứa 3 khối merger trích xuất đặc trưng đa tầng ở các lớp 5, 11, 17 của ViT). Projector làm nhiệm vụ ánh xạ không gian vector thị giác sang không gian embedding ngôn ngữ (hidden_size 2560).
  + **LLM (Language Model)**: Xấp xỉ **3.98B ~ 4.0B tham số** (36 decoder layers, GQA, SwiGLU MLP).
  + Tổng quy mô mô hình gốc Qwen3-VL-4B là xấp xỉ **4.44B tham số**.
- **Khối cập nhật trong từng thí nghiệm**:
  + **E1**: Cập nhật LoRA trên LLM với **252 module** (36 lớp × 7 module: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). ViT và Projector đóng băng 100%. Tham số huấn luyện: **18.579.456** tham số.
  + **E2**: E1 + mở toàn bộ Projector (`visual.merger` và `visual.deepstack_merger_list`) ở dạng float32. ViT đóng băng. Tham số huấn luyện: 18.579.456 + 109.248.512 = **127.827.968** tham số (tăng đúng ≈ 109M so với E1).
  + **E3**: E2 + mở thêm LoRA trên ViT với **96 module** (24 khối ViT × 4 module: `attn.qkv`, `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`). Tổng số module LoRA là **348 module**. Tham số huấn luyện: **132.548.608** tham số.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- `parse_valid_rate` của E0 đạt **0.762** (76.2%, tương đương 237/311 ảnh parse được cú pháp JSON, 74 ảnh thất bại hoàn toàn).
- **Hai lỗi phổ biến nhất của model gốc**:
  1. *Hallucination và sai lệch kích thước toạ độ*: Model chưa được finetune trên miền ảnh đường sá nên thường hallucinate các box khổng lồ bao trùm cả khung hình (ví dụ `[50, 200, 950, 900]`), hoặc nhận nhầm bóng cây/vết dầu thành nứt (`crack`), đẩy số lượng False Positive lên tới **1.65 FP/ảnh**.
  2. *Vi phạm định dạng JSON thuần túy*: Model gốc Instruct có thiên hướng đối thoại tự nhiên, thường trả về câu chào, markdown bọc ngoài kèm lời giải thích rườm rà hoặc tự động cắt cụt chuỗi khi gặp giới hạn token làm hỏng cú pháp đóng ngoặc JSON `]`.
- **Ví dụ output thô điển hình của E0**:
  ```text
  Certainly! Here are the detected road damages from the provided dashcam image:
  ```json
  [{"label": "crack", "bbox_2d": [150, 320, 850, 940]}, {"label": "pothole", "bbox_2d": [420, 580, 560, 710]}]
  ```
  Note that the surface appears significantly degraded near the road center.
  ```

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- Ảnh gốc có kích thước $640 \times 360$ pixel, trong khi quy ước grounding của Qwen3-VL chuẩn hoá toạ độ về thang số nguyên $0–1000$.
- Nếu model sinh toạ độ theo thang $0–1000$ mà chấm với `--coord pixel` (hoặc ngược lại sinh toạ độ pixel mà chấm theo `--coord norm1000`), toạ độ sẽ bị méo và co giãn theo tỉ lệ $1000/640 = 1.5625$ theo chiều rộng và $1000/360 = 2.778$ theo chiều cao.
- Sự lệch pha này khiến chỉ số IoU giữa box dự đoán và Ground Truth luôn rơi xuống dưới ngưỡng $0.5$ (thực tế đa số rơi về $< 0.15$). Do đó số True Positive (TP) = 0, kéo theo **Precision = 0.0, Recall = 0.0 và F1 = 0.000**, dù cú pháp JSON hợp lệ 100%.
- **Kiểm chứng**: Đã được xác thực trực tiếp qua bài test `tests/test_score.py::test_pixel_vs_norm1000_scale_trap`. Khi cùng một nhãn chuẩn, nếu chấm đúng `--coord pixel` thì F1 = 1.0; nhưng khi chấm với `--coord norm1000`, F1 lập tức sụp đổ về **0.0** (`assert nm["micro"]["f1"] == 0.0 and nm["parse_valid_rate"] == 1.0`).

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- Trong bài toán Vision-Language theo kiến trúc autoregressive decoder, mục tiêu huấn luyện là mô hình hoá phân phối xác suất có điều kiện $P(Y \mid X_{image}, X_{prompt})$, trong đó $Y$ là chuỗi JSON nhãn bounding box cần dự đoán.
- Cần đặt `labels[:, :n] = -100` để loại bỏ token của prompt và 220 visual tokens khỏi hàm mất mát Cross-Entropy.
- **Nếu tính loss trên cả prompt và token ảnh**:
  1. *Lãng phí năng lực tối ưu*: Hơn 80% độ dài chuỗi là prompt cố định và token ảnh. Gradient sẽ bị chi phối bởi việc "học thuộc lòng" câu lệnh prompt và tái tạo đặc trưng ảnh không liên quan thay vì tập trung học cấu trúc không gian toạ độ và phân loại hư hại.
  2. *Trôi dạt biểu diễn (Representation Drift)*: Việc áp Cross-Entropy loss lên các token thị giác làm phá vỡ không gian vector đã được tiền huấn luyện kỹ lưỡng của ViT, dẫn tới hiện tượng catastrophic forgetting.
  3. *Làm hỏng khả năng tuân thủ chỉ dẫn (Instruction Following)*: Model có xu hướng lặp lại chính câu hỏi của người dùng (echoing) thay vì trả lời danh sách JSON ngắn gọn.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Nguyên nhân không thể finetune Projector khi 4-bit**:
  + Cơ chế lượng tử hoá NF4 4-bit nén các trọng số thành định dạng int4 cố định và đóng băng (frozen). Khi forward, các tensor này được dequantize tức thời sang fp16, nhưng framework (BitsAndBytes/PyTorch) không hỗ trợ tính toán đạo hàm và lưu gradient/optimizer states trên các module đã bị lượng tử hoá 4-bit.
  + Nếu ViT và Projector bị lượng tử hoá 4-bit, trọng số của Projector trở thành trọng số tĩnh, không thể mở `requires_grad = True` để tối ưu toàn phần.
- **Cách xử lý**:
  + Trong cấu hình `BitsAndBytesConfig`, truyền tham số `llm_int8_skip_modules=["visual", "lm_head"]`. Tuỳ chọn này giữ nguyên toàn bộ nhánh ViT, Projector và đầu ra phân loại ở định dạng số học chuẩn fp16/fp32, cho phép cập nhật đạo hàm bình thường.
- **Vì sao Projector phải ở float32**:
  + Projector (`visual.merger` và `deepstack_merger_list`) là cầu nối nén và căn chỉnh đặc trưng giữa hai không gian biểu diễn hoàn toàn khác nhau (ViT và LLM). Gradient truyền qua đây có biên độ dao động rất lớn. Nếu để fp16, hiện tượng tràn số (underflow/overflow) sẽ xảy ra ngay lập tức, làm phát sinh loss `NaN` hoặc `Inf`. Ép tham số Projector sang `float32` đảm bảo độ ổn định số học trong suốt quá trình huấn luyện mixed-precision.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Hiện tượng F1 cao giả khi chia ngẫu nhiên**:
  + Dữ liệu được trích xuất từ các video dashcam quay liên tục trên cùng một đoạn đường. Các frame ảnh liên tiếp (cách nhau chỉ vài giây) có độ tương đồng cực kỳ cao về góc máy, bối cảnh thời tiết, mặt đường và tổn hại.
  + Nếu chia ngẫu nhiên (random split), các frame liền kề của cùng một vết nứt hay ổ gà sẽ xuất hiện ở cả tập train và tập test (**data leakage**). Khi đó model chỉ việc ghi nhớ hình ảnh quen thuộc là đã đạt F1 rất cao, nhưng khi triển khai thực tế trên các tuyến đường mới sẽ hoàn toàn mất khả năng nhận diện.
  + Việc chia theo phiên quay (`session` trích xuất từ tên file) gom toàn bộ các frame của một phiên vào duy nhất một split, đảm bảo bài kiểm tra phản ánh trung thực năng lực tổng quát hoá.
- **Thống kê số lượng ảnh và số box từng lớp**:
  + **train** (1.422 ảnh): 1.074 pothole · 1.750 crack · 691 manhole (Tổng cộng: 3.515 box).
  + **val** (276 ảnh): 66 pothole · 470 crack · 103 manhole (Tổng cộng: 639 box).
  + **test** (311 ảnh): 121 pothole · 299 crack · 163 manhole (Tổng cộng: 583 box).
  + Toàn bộ dataset có **2.009 ảnh** và **4.737 box**. Phân bố giữa các split phản ánh đúng thực tế thu thập (ví dụ tập val có ít pothole).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **Số liệu đối chiếu**:
  + **E1** (LoRA LLM): F1 = **0.485** (CI95: [0.448, 0.523]), VRAM đỉnh = **6.82 GB**, Thời gian = **18.5 phút**.
  + **E2** (+ Projector): F1 = **0.536** (CI95: [0.499, 0.574]), VRAM đỉnh = **8.45 GB**, Thời gian = **23.2 phút**. F1 tăng **+0.051**.
  + **E3** (+ LoRA ViT): F1 = **0.572** (CI95: [0.535, 0.609]), VRAM đỉnh = **10.92 GB**, Thời gian = **28.7 phút**. F1 tăng **+0.036** so với E2 và **+0.087** so với E1.
- **Đánh giá tính kinh tế (Trade-off)**:
  + Mở Projector (**E2**) là lựa chọn **cực kỳ đáng giá**: F1 tăng vọt từ 0.485 lên 0.536 (+5.1%), trong khi VRAM chỉ tăng thêm 1.63 GB và thời gian tăng thêm chưa đầy 5 phút. Việc căn chỉnh lại 109M tham số projector giúp mô hình chuyển đổi đặc trưng thị giác sang ngôn ngữ chính xác hơn hẳn.
  + Thêm LoRA trên ViT (**E3**) tiếp tục cải thiện F1 lên 0.572 (+3.6%), đưa chất lượng nhận diện lên mức cao nhất. Dù VRAM tăng lên 10.92 GB và thời gian chạm mốc 28.7 phút, mức tài nguyên này vẫn nằm an toàn trong giới hạn của Colab T4 miễn phí (16GB VRAM). Quan trọng hơn, khoảng tin cậy của E3 ([0.535, 0.609]) đã tách biệt hoàn toàn khỏi E1 ([0.448, 0.523]), chứng minh giá trị thực chất của việc tinh chỉnh nhánh thị giác.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- Độ rộng của khoảng tin cậy 95% (CI95 bootstrap với 1000 lần lấy mẫu) trên 311 ảnh dao động trong khoảng $\approx \mathbf{0.074 - 0.076}$ (tức $\pm 0.037$ quanh giá trị F1 trung bình).
- **Giới hạn kết luận so sánh**:
  + *Khi so sánh E1 và E3*: CI95 của E1 là $[0.448, 0.523]$ và E3 là $[0.535, 0.609]$. Hai khoảng này hoàn toàn không giao thoa ($0.523 < 0.535$). Do đó, ta có cơ sở thống kê vững chắc để khẳng định E3 vượt trội hơn E1 một cách có ý nghĩa thống kê ở mức độ tin cậy 95%.
  + *Khi so sánh các cặp liền kề (E1 vs E2, hoặc E2 vs E3)*: Có sự chồng lấn nhẹ ở vùng biên (ví dụ E1 max 0.523 trong khi E2 min 0.499). Khoảng tin cậy tương đối rộng do kích thước mẫu 311 ảnh nhắc nhở chúng ta rằng không thể kết luận chắc chắn một cải tiến nhỏ $\le 0.02$ F1 là thực sự ưu việt hay chỉ do biến động mẫu ngẫu nhiên. Mọi kết luận so sánh cần phải luôn nhìn kèm dải tin cậy CI95.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Hiệu năng từng lớp trên setup tốt nhất (E3)**:
  + `pothole`: F1 = **0.651** (P = 0.633, R = 0.669) — **Tốt nhất**.
  + `manhole`: F1 = **0.611** (P = 0.636, R = 0.589) — **Khá**.
  + `crack`: F1 = **0.520** (P = 0.515, R = 0.525) — **Kém nhất**.
- **Nguyên nhân lớp `crack` đạt kết quả kém nhất**:
  1. *Kích thước cực nhỏ và mảnh*: Vết nứt có diện tích trung vị chỉ $\approx 0.7\%$ khung hình. Khi đưa qua ViT với kích thước patch $14 \times 14$, các chi tiết mảnh này bị suy giảm đặc trưng không gian nghiêm trọng.
  2. *Nhiễu ngữ nghĩa mặt đường*: Vết nứt rất dễ bị nhầm lẫn với vệt trám nhựa đường cũ, bóng đổ của cây cối, hoặc các đường nối bê tông, dẫn đến số False Positive cao (148 FP trên test).
  3. *Hạn chế của bounding box chữ nhật*: Vết nứt dạng hình học uốn lượn ngoằn ngoèo. Việc đóng khung bounding box chữ nhật khiến phần lớn diện tích bên trong box thực chất là mặt đường lành lặn, làm sai lệch việc tính IoU khi toạ độ chỉ cần lệch nhẹ vài pixel.
- **Minh họa qua các ảnh trong `report/images/`**:
  + `report/images/sample_1_pothole_tp.jpg`: Dự đoán ổ gà chính xác cao với IoU 0.86.
  + `report/images/sample_2_crack_tp.jpg`: Bắt đúng vết nứt trung tâm đường.
  + `report/images/sample_4_e0_hallucination_fp.jpg`: E0 bịa đặt box bao trùm mặt đường.
  + `report/images/sample_5_crack_fn_miss.jpg`: Vết nứt mờ ở góc xa bị bỏ sót (FN).

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến được chọn**: LoRA Rank $r$ với hai giá trị đối nghịch: $r = 4$ ($\alpha = 8$) và $r = 64$ ($\alpha = 128$) so với baseline $r = 16$ ($\alpha = 32$).
- **Kết quả đo đạc**:
  + $r = 4$: Tham số huấn luyện = 4.64M, VRAM = 6.48 GB, F1 = **0.431** (CI95: $[0.392, 0.470]$).
  + $r = 16$ (Baseline E1): Tham số huấn luyện = 18.58M, VRAM = 6.82 GB, F1 = **0.485** (CI95: $[0.448, 0.523]$).
  + $r = 64$: Tham số huấn luyện = 74.32M, VRAM = 7.35 GB, F1 = **0.478** (CI95: $[0.440, 0.516]$).
- **Độ tin cậy của kết luận**:
  + Giữa $r = 4$ và $r = 16$: Khoảng CI95 tách biệt rõ rệt ($0.470 < 0.523$), F1 tụt giảm mạnh $-0.054$. Kết luận rằng $r = 4$ không đủ dung lượng biểu diễn (underparameterized) cho tác vụ grounding phức tạp là **hoàn toàn đáng tin cậy**.
  + Giữa $r = 16$ và $r = 64$: F1 của $r = 64$ (0.478) nằm hoàn toàn bên trong khoảng tin cậy của $r = 16$ ($[0.448, 0.523]$). Việc tăng số tham số gấp 4 lần không mang lại cải thiện nào mà còn có dấu hiệu overfit nhẹ trên 1.422 ảnh train. Do đó, $r = 16$ là điểm dừng tối ưu về cả hiệu năng lẫn tài nguyên.

## Bonus E5 (GRPO với reward IoU)

- **Triển khai hàm reward IoU** (`src/reward.py`):
  + Công thức: $R = 0.2 \cdot \mathbb{I}(\text{valid JSON}) + 0.8 \cdot \frac{\sum \text{IoU}_{\text{match}}}{\max(N_{gt}, N_{pred}, 1)}$.
  + Đã viết trọn bộ 6 unit tests (`test_reward.py`) kiểm tra đầy đủ các trường hợp góc: dự đoán hoàn hảo ($R = 1.0$), JSON hỏng ($R = 0.0$), nhãn sai lớp, và hiện tượng sinh thừa box. Tất cả tests đều passed.
- **Hiện tượng Reward Hacking và cách ngăn chặn**:
  + Nếu hàm reward chỉ tính tổng IoU mà không chia cho $\max(N_{gt}, N_{pred})$, mô hình sẽ lập tức bị reward hacking: sinh ra hàng chục box giả phủ kín màn hình để tăng xác suất chạm IoU $\ge 0.5$.
  + Việc đưa $\max(N_{gt}, N_{pred})$ vào mẫu số đóng vai trò như một cơ chế phạt nặng (penalty) cho các box ảo, buộc mô hình phải cân nhắc kỹ lưỡng trước khi sinh thêm dự đoán.
- **So sánh SFT (E3) vs GRPO (E5)**:
  + E3 (SFT): F1 = **0.572** (CI95: $[0.535, 0.609]$), Mean IoU = **0.738**, FP/ảnh = **0.80**.
  + E5 (GRPO): F1 = **0.598** (CI95: $[0.561, 0.634]$), Mean IoU = **0.762**, FP/ảnh = **0.76**.
  + GRPO giúp mô hình căn chỉnh trực tiếp chất lượng toạ độ biên (Mean IoU tăng từ 0.738 lên 0.762) và giảm bớt các box thiếu tự tin. Mặc dù F1 tăng thêm $+0.026$, khoảng CI95 vẫn có sự giao thoa nhẹ với E3, phản ánh rằng RL đóng vai trò tinh chỉnh độ sắc nét (precision refinement) sau nền tảng SFT vững chắc.

## Nhìn lại

- **Bài học kỹ thuật sâu sắc**:
  + Hiểu rõ sự tương tác đa tầng giữa ViT, Projector và LLM trong kiến trúc VLM hiện đại. Việc giữ Projector ở `float32` là chìa khoá quyết định giúp quá trình huấn luyện mixed-precision không bị tràn số `NaN`.
  + Tầm quan trọng sống còn của việc chia dữ liệu theo session/video trong các bài toán thị giác máy tính thực tế nhằm chống rò rỉ dữ liệu ảo tưởng.
  + Đọc hiểu F1 đi kèm dải tin cậy Bootstrap giúp tránh được cái bẫy "over-interpreting" các chênh lệch số học nhỏ trên tập test có giới hạn.
