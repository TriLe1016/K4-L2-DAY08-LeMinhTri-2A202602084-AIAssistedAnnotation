# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Le Minh Tri

Công cụ gán nhãn đã dùng: CVAT chạy Docker local (v2.74.1), import/export định dạng Ultralytics YOLO Detection 1.0

Nguồn số liệu: `reports/rounds_table.md`, `outputs/metrics_round0.json`, `outputs/metrics_round1.json`,
`outputs/selection_round1.csv`, `outputs/selection_round2.csv`, `outputs/round1_diff.md`,
`outputs/compare_round0.jpg`, `outputs/compare_round1.jpg`. Nhãn test do mô hình tạo, chưa được người
rà, nên mọi số đo dưới đây là mức khớp với bộ tham chiếu này, không phải chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Video là một cảnh quay liên tục từ camera cố định (không có cắt cảnh), lấy mẫu 2.5 ảnh/giây. Hai ảnh
cách nhau 0.4 s gần như giống hệt nhau và mỗi chiếc xe ở trong khung hình vài giây. Nếu chia ngẫu
nhiên, cùng một chiếc xe ở cùng vị trí sẽ xuất hiện cả trong ảnh train và ảnh test: mô hình được
chấm trên chính những xe nó đã học, tức là rò rỉ dữ liệu (data leakage). Số đo test khi đó sẽ bị
**lệch lên (lạc quan)**, cao hơn khả năng thật trên cảnh chưa thấy.

Vì vậy tập test gồm 20 ảnh ở 4 đoạn quanh giây 20, 60, 100, 140; các ảnh trong ±4 s quanh mỗi đoạn
(112 ảnh) bị loại làm vùng đệm; 268 ảnh còn lại là pool. Ảnh pool gần test nhất vẫn cách 4.4 s
(`data/DATA.md`). Tôi không sửa `data/test/labels/` và không đưa ảnh test vào train.

## 2. Mô hình khởi đầu lạnh (cold start)

| vòng | model | ảnh train | box train | AP50 | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Tại conf 0.25, cold start có TP 197, FP 16, FN 206 trên 403 box tham chiếu (`metrics_round0.json`):
ít báo nhầm nhưng **bỏ sót hơn nửa số xe**. Trong `compare_round0.jpg`, box bỏ sót (vàng) tập trung ở:

- **xe bật đèn pha đi về phía camera**, kể cả xe lớn ở gần, ví dụ xe to góc dưới trái frame_0250 và
  xe dưới trái frame_0050: đèn pha loá làm mất đường viền thân xe;
- **xe nhỏ ở xa chỉ còn cụm đèn đỏ** phía trên ảnh: recall small chỉ 0.182 (66 box), so với 0.547
  (medium) và 0.561 (large).

Recall theo kích thước cho thấy mô hình COCO chưa quen với xe ban đêm chỉ nhìn thấy qua đèn; đây là
nhóm cần nhãn người nhất. Tuy vậy recall large cũng chỉ 0.561, nên vấn đề không chỉ là kích thước mà
còn là điều kiện ánh sáng.

Ca cần rà lại nhãn tham chiếu trước khi kết luận mô hình sai: ở frame_0250 và frame_0150, cold start
có box FP (đỏ) nằm **đè lên cụm xe đèn đỏ phía xa mà tham chiếu cũng có box**, chỉ khác kích thước.
FP ở đây có thể do box tham chiếu (cũng do mô hình tạo) ôm lệch nên IoU < 0.5, chứ không chắc mô hình
báo nhầm vật không phải xe. Cần người xem lại các box tham chiếu đó.

## 3. Chiến lược chọn mẫu

`score = W_U·U + W_A·A + W_D·D` với trọng số 0.5 / 0.3 / 0.2:

- **U** (độ bất định): trung bình 5 giá trị `1 − |2c − 1|` lớn nhất trong ảnh, bằng 1 khi mô hình
  phân vân nhất (conf ≈ 0.5). Ảnh có nhiều box khó được ưu tiên.
- **A** (mập mờ): số box có 0.15 ≤ conf < 0.50, chia cho max trong pool.
- **D** (đa dạng): khoảng cách thời gian tới frame đã gán gần nhất. Vòng 1 chưa có frame nào đã gán
  nên D = 1 với mọi ảnh.
- **`MIN_GAP_S = 2.0`**: khi chọn tham lam theo score, bỏ frame cách frame đã chọn dưới 2 s, vì
  camera cố định làm hai frame gần nhau gần như trùng; gán cả hai tốn công mà mô hình học thêm ít.

Theo `reports/SELECTION.md`: frame_0182 (hạng 1, score 0.9591, 18/28 box mập mờ), frame_0331
(47 box, A = 1.0) và frame_0392 (hạng 15, U cao nhất lô 0.9747) nằm trong lô. frame_0372 (hạng 6,
score 0.9101) **không được chọn** vì chỉ cách frame_0369 1.2 s; tương tự frame_0368 và frame_0330.
Nhờ đó frame_0392 hạng 15 mới vào lô. Với ngân sách 5 ảnh tôi còn bỏ thêm 0380 và 0326 vì gần 0369
và 0331, để trải lô từ 39.6 s đến 147.6 s.

Điểm bất định **không chứng minh** ảnh đó sẽ cải thiện mô hình: nó chỉ đo mô hình phân vân trên các
box nó đã dự đoán. Box mập mờ có thể là đèn phản chiếu hoặc xe < 16 px (bị bỏ qua khi chấm), và xe
bị bỏ sót hoàn toàn không đóng góp gì vào U hay A. Thực tế tôi phải **thêm 173 box** mà mô hình
không đề xuất, nhiều hơn số box được giữ lại. Chưa có đối chứng `STRATEGY = "random"` nên chưa thể
nói chọn theo bất định tốt hơn chọn ngẫu nhiên.

## 4. Các vòng học chủ động (active learning)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 330 | 0.510 | -0.261 | 1.000 | 0.032 | 0.062 | 0.000 | 0.020 | 0.171 |

**Mức sửa nhãn gợi ý (vòng 1, `outputs/round1_diff.md`):** mô hình đề xuất 169 box; sau khi sửa còn
330 box: accepted 155, edited 2, deleted 12, added 173 (accept rate 92%). Lỗi chính của pre-label là
**bỏ sót** (173 box thêm) chứ không phải báo nhầm (12 box xoá), khớp với precision cao / recall thấp ở
vòng 0.

**Thay đổi số đo:** AP50 giảm 0.261 (0.771 → 0.510) so với cold start (vòng 1 cũng là vòng trước).
Tại conf 0.25 mô hình chỉ còn TP 13, FP 0, FN 390: precision 1.0 nhưng recall 0.032. Recall giảm ở
mọi nhóm: small 0.182 → 0.000, medium 0.547 → 0.020, large 0.561 → 0.171. Không nhóm nào tốt lên
theo số đo tổng.

**Giải thích có bằng chứng:** AP50 vẫn 0.510 trong khi recall@0.25 gần 0, nghĩa là mô hình vẫn xếp
hạng box xe khá đúng nhưng **gán độ tin cậy rất thấp**: gần như mọi box nằm dưới ngưỡng 0.25. Bằng
chứng thứ hai: trong `selection_round2.csv`, mô hình mới chỉ dự đoán trung bình 6.1 box/ảnh (2–11) ở conf ≥ 0.05 trên
268 ảnh pool, so với trung bình 27.9 box/ảnh (14–53) của cold start trong `selection_round1.csv`. Nguyên nhân tôi cho là hợp lý nhất nằm ở
cấu hình train của notebook: fine-tune từ `yolov8n.pt` sang **1 lớp** nên đầu phân loại được khởi tạo
lại, trong khi 12 ảnh với `batch=16` chỉ cho **1 bước cập nhật mỗi epoch**, tức khoảng 50 bước cho
50 epoch. Đầu phân loại mới chưa đủ bước để nâng độ tin cậy. Đây là giả thuyết, chưa được kiểm bằng
thí nghiệm riêng.

**Một ca đổi sau fine-tune (`compare_round1.jpg`):** ở frame_0250, xe lớn bật đèn pha **góc dưới
trái** bị cold start bỏ sót (vàng) nhưng mô hình vòng 1 bắt được (xanh, TP 1). Đây đúng là kiểu xe
tôi thêm nhiều nhất khi sửa nhãn (xe gần camera, đèn pha loá, bị cắt ở mép dưới). Ngược lại, ở
frame_0050 cold start có TP 11 nhưng vòng 1 còn TP 0: các xe rõ ở giữa ảnh bị tụt dưới ngưỡng 0.25.

**Phân biệt ba nguồn bằng chứng:**

- *Quan sát độc lập* (`BLIND_SCAN.md`, khoá trước khi xem pre-label): frame_0099 tôi đếm 24 xe và dự
  đoán AI sẽ sót xe bị che và xe tối phía xa bên trái.
- *Lỗi pre-label đã sửa* (`round1_diff.md`, `REVIEW_LOG.csv`): frame_0099 mô hình đề xuất 13 box, sau
  khi sửa còn 24 box (thêm 11), **đúng bằng số xe tôi đếm độc lập**. Các box thêm tập trung ở cụm xe
  xa bên trái và xe gần camera. Ngoài ra tôi xoá/tách các box AI ôm chung 2 xe (frame_0326, 0312, 0107).
- *Kết quả mô hình sau train* (`metrics_round1.json`): nhãn tốt hơn **không** tự chuyển thành số đo
  tốt hơn trong vòng này, vì mô hình sau 50 bước cập nhật chưa đủ tự tin.

**Ca khó theo guideline:** box AI ôm chung hai xe đứng sát/xếp trước sau. GUIDELINE yêu cầu hai box
riêng. Tôi thống nhất cách xử lý: thu box AI về một xe và vẽ box mới cho xe còn lại, không xoá hẳn.
Với xe chỉ thấy cụm đèn (frame_0187), tôi vẽ box theo thân xe đoán được quanh đèn, không chỉ khoanh
hai chấm đèn, và không tính vệt đèn trên mặt đường. Xe < 16 px tôi không cố gán vì bị bỏ qua khi chấm.

## 5. Kết luận và giới hạn

**So với cold start:** vòng 1 kém hơn trên cùng 20 ảnh test (AP50 0.510 so với 0.771; recall@0.25
0.032 so với 0.489). Kết quả này **không chứng minh nhãn đã sửa sai**: nó cho thấy quy trình
fine-tune với 12 ảnh/50 bước chưa đủ để mô hình 1 lớp mới đạt độ tin cậy bằng mô hình COCO.

**Quyết định:** tôi **dừng, không gán thêm vòng 2 ngay**. Lô vòng 2 do chính mô hình yếu này chọn
(`selection_round2.csv`, điểm tối đa chỉ 0.68 và mỗi ảnh chỉ 2–11 box) nên độ bất định của nó không
đáng tin. Trước khi tốn thêm công gán, cần kiểm cấu hình train: tăng số bước cập nhật (batch nhỏ hơn
hoặc nhiều epoch hơn), hoặc train từ checkpoint giữ kiến thức COCO, rồi đo lại trên cùng tập test.

**Hai ca còn yếu/bất định cho vòng sau:**

1. **Xe nhỏ phía xa chỉ còn đèn đỏ**: recall small 0.182 ở vòng 0 và 0.000 ở vòng 1. Chi phí rà cao
   vì mỗi ảnh có nhiều xe nhỏ; với lô vòng 1 tôi phải thêm trung bình khoảng 14 box/ảnh (173/12).
2. **Cụm xe đông, sát nhau** như frame_0331 và frame_0392, nơi AI hay vẽ một box ôm hai xe.
   Nguy cơ ảnh gần trùng rõ ở `selection_round2.csv`: frame_0074 (29.6 s) đứng hạng 2 nhưng
   frame_0073, 0072, 0070 (28.0–29.2 s) xếp ngay sau; nếu `MIN_GAP_S` không loại, lô sẽ lặp một cảnh.

**Giới hạn của kết luận:**

- Test chỉ 20 ảnh từ 4 đoạn thời gian; chênh < ~0.01 AP50 chưa đủ kết luận. Mức giảm 0.261 lớn hơn
  nhiều nên xu hướng giảm là thật, nhưng độ lớn chính xác không chắc.
- 14 box tham chiếu cao < 16 px bị bỏ qua, nên số đo không phản ánh xe rất xa.
- Nhãn tham chiếu do mô hình tạo, chưa người rà: mô hình khớp tốt với tham chiếu có thể chỉ vì giống
  mô hình tạo tham chiếu; box nhãn người ôm khác kích thước có thể bị tính FP/FN.
- Train và đánh giá mỗi cấu hình chỉ một lần (một seed).

**Tự QC nhãn của tôi:** tôi đã chạy kiểm tra box chồng nhau (IoU > 0.4) trên 330 box. Sau khi sửa
lại, còn **frame_0331** (một box ôm xe trắng và xe đèn đỏ phía xa giữa ảnh) và **frame_0392** (một box
cao ôm hai xe xếp trước sau) chưa tách; ở frame_0392 tôi cũng đã xoá một box ở giữa ảnh phía xa mà khi
rà lại có thể là xe buýt/xe tải. Đây là lỗi nhãn đã biết, còn trong bộ train vòng 1.

**Nếu AP50 giảm, kiểm tra gì trước khi train thêm:** (1) mô hình có dự đoán box ở conf thấp không
(AP50 còn 0.51 cho thấy có) để phân biệt thiếu tự tin với học sai; (2) số bước cập nhật và cấu hình
fine-tune; (3) nhãn train có lỗi hệ thống không, như box gộp hai xe hoặc box ôm vệt đèn; (4) test có
bị lẫn vào train không (đã kiểm: không).
