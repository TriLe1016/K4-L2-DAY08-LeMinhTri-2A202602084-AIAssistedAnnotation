# Vì sao chọn lô này?

Nguồn: `outputs/selection_round1.csv` (268 frame pool, cột `score, U, A, D, n_boxes, n_ambiguous`)
và contact sheet `outputs/selection_round1.jpg`. Cấu hình: `score = 0.5·U + 0.3·A + 0.2·D`,
`MIN_GAP_S = 2.0`, chiến lược `uncertainty`, K = 12. Ở vòng 1 chưa có frame nào đã gán nên
`D = 1.0` cho mọi frame; thứ hạng chỉ do U (độ bất định của 5 box khó nhất) và A (số box mập mờ
0.15 ≤ conf < 0.50, chuẩn hóa theo max pool) quyết định.

## Top 5 nếu chỉ có ngân sách rà năm ảnh

Xét 50 dòng đầu CSV (score từ 0.959 xuống 0.820; không frame nào có `empty = True`, tức model
luôn dự đoán được ít nhất một box, nên không cần dùng EMPTY_BONUS để cứu frame trống).

| Thứ tự | Frame | Hạng CSV | Score | t (s) | U | A | n_boxes / mập mờ | Lý do ưu tiên |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | frame_0182.jpg | 1 | 0.9591 | 72.8 | 0.918 | 1.000 | 28 / 18 | Điểm cao nhất; A tối đa, số box mập mờ lớn nhất pool so với số box. |
| 2 | frame_0331.jpg | 5 | 0.9154 | 132.4 | 0.831 | 1.000 | 47 / 18 | Cảnh đông nhất trong top (47 box), A tối đa; nhiều xe nhỏ và xe sát nhau, đúng nhóm model yếu (recall small 0.18). |
| 3 | frame_0369.jpg | 2 | 0.9324 | 147.6 | 0.932 | 0.889 | 43 / 16 | Hạng 2, đại diện cụm cuối video. Chọn frame này thay cho 0368 (147.2 s) và 0372 (148.8 s) vì ba frame gần như trùng cảnh. |
| 4 | frame_0099.jpg | 8 | 0.9063 | 39.6 | 0.946 | 0.778 | 29 / 14 | Đại diện đầu video (không frame nào trong top 7 trước 72 s); U rất cao. |
| 5 | frame_0312.jpg | 7 | 0.9100 | 124.8 | 0.820 | 1.000 | 37 / 18 | A tối đa, cách 0331 7.6 s nên không trùng cảnh. |

Quyết định về ảnh gần trùng: với ngân sách 5 ảnh, tôi **bỏ frame_0380 (hạng 3, t = 152.0 s)** và
**frame_0326 (hạng 4, t = 130.4 s)** dù điểm cao hơn 0099 và 0312. 0380 chỉ cách 0369 4.4 s và 0326
cách 0331 đúng 2.0 s; camera cố định nên cặp frame này có cùng bố cục làn đường, gán cả hai tốn công
gấp đôi mà model học thêm ít. Đổi lại, năm ảnh trải từ 39.6 s đến 147.6 s.

## Ba frame thuộc lô 12 ảnh model chọn và bằng chứng

- **frame_0182.jpg**, hạng 1, score 0.9591, U 0.918, A 1.0, 28 box với 18 box mập mờ: hơn nửa
  số box model dự đoán nằm trong vùng conf 0.15–0.50. Contact sheet cho thấy nhiều xe ở xa phía
  trên ảnh, chỉ còn cụm đèn.
- **frame_0331.jpg**, hạng 5, score 0.9154, 47 box / 18 mập mờ: số box nhiều nhất trong 12 ảnh;
  contact sheet cho thấy dòng xe dày đặc, đèn pha loá, nhiều xe chồng lấp.
- **frame_0392.jpg**, hạng 15, score 0.8874, U cao nhất lô (0.9747) nhưng A thấp nhất (0.667,
  12 box mập mờ). Frame được vào lô vì các frame 0372, 0368, 0330 xếp trên nó đã bị loại do gần
  frame đã chọn (< 2 s), cho thấy `MIN_GAP_S` đẩy frame xếp dưới vào lô.

## Một frame điểm cao nhưng không chọn

**frame_0372.jpg** (hạng 6, score 0.9101, t = 148.8 s) không vào lô vì chỉ cách frame_0369
(147.6 s) 1.2 s, dưới `MIN_GAP_S`. Tương tự **frame_0368** (hạng 9, cách 0369 0.4 s) và
**frame_0330** (hạng 12, cách 0331 0.4 s). Ngược lại, 0326 và 0331 cách nhau đúng 2.0 s nên cả hai
vẫn được chọn: ngưỡng 2 s có thể quá nhỏ với cảnh camera cố định, xe chỉ dịch vài chục pixel.

## Điều phép chọn này chưa chứng minh

- Điểm bất định cao chỉ cho biết model **phân vân**, không chứng minh gán nhãn ảnh đó sẽ tăng AP50
  trên test. Frame có nhiều box mập mờ có thể chỉ vì nhiều đèn phản chiếu hoặc xe quá nhỏ (< 16 px,
  bị bỏ qua khi chấm), tức công rà lớn mà lợi ích đo được nhỏ.
- U và A chỉ tính trên box model **đã** dự đoán (conf ≥ 0.05). Xe model bỏ sót hoàn toàn (không có
  box nào) không làm tăng điểm, trong khi cold start có recall chỉ 0.489 (`metrics_round0.json`).
- Vòng 1 có D = 1 cho mọi frame nên thành phần đa dạng chưa đóng vai trò; phân tán thời gian chỉ
  đến từ `MIN_GAP_S`.
- Chưa có đối chứng `STRATEGY = "random"` nên không thể nói chọn theo bất định tốt hơn chọn ngẫu nhiên.
