# Annotation Guideline v2.0 — DRAFT

**Trạng thái: DRAFT, CHƯA FREEZE.** Nguồn dữ liệu chính thức là
[`guideline_v2.0_draft.yaml`](guideline_v2.0_draft.yaml) (14 rule GR-01
đến GR-14, machine-readable). File này là bản đọc-hiểu song song, không
thay thế file yaml.

**Provenance**: tổng hợp từ domain reviewer (session 2026-09-19/20), review
100% 2 workbook (`review_priority_singleton_anomalies.xlsx` 9 case,
`review_guideline_ambiguities.xlsx` 61 surface + 195 occurrence sâu cho 9
surface trọng điểm). KHÔNG dựa trên majority vote tự động.

**Trước khi freeze thành bản chính thức, cần**:
1. Người review ký tên vào field `author` trong file yaml.
2. Chốt GR-10 (đơn vị hành chính LOC vs ORG, hiện đang OPEN 50/50).
3. Review 524 candidate GR-11 (`dynasty_era_candidates.xlsx`).
4. Quyết định GR-09 (loại bỏ 會試/經筵 khỏi gold hay để dành schema v3 EVENT).

## Nguyên tắc chung

### Flat-NER priority rule (schema không nested)

Khi 1 chuỗi ký tự có thể đọc theo ≥2 tầng nghĩa lồng nhau (ví dụ 1 cụm vừa
là chức danh vừa gắn với 1 người cụ thể), vì schema **không hỗ trợ nested
entity**, phải chọn **1 tầng nghĩa duy nhất** theo thứ tự ưu tiên:

1. Nếu câu xác định được **1 cá nhân cụ thể** làm referent chính → ưu tiên
   **PER** (GR-07, GR-12).
2. Nếu không xác định được cá nhân cụ thể, nhưng có 1 **thực thể tổ chức/
   chính thể cụ thể** làm referent → **ORG** (GR-03, GR-04, GR-05).
3. Nếu chỉ là **cách gọi chung chung** không định danh ai/cái gì cụ thể →
   **TITLE** (GR-08).

Đây là priority rule mặc định khi 2 reviewer bất đồng về tầng nghĩa — không
áp dụng máy móc, ngữ cảnh cụ thể luôn override.

### PER vs TITLE (miếu hiệu / tôn hiệu vua) — xem GR-07, GR-08

- Miếu hiệu/thụy hiệu/niên hiệu+帝 định danh **một vua cụ thể**, dùng làm
  chủ ngữ/tân ngữ của hành động → **PER**.
- Kính xưng áp dụng được cho **bất kỳ ai giữ ngôi** (không định danh duy
  nhất) → **TITLE**.

### ORG vs TITLE (chức quan / cơ quan) — xem GR-01, GR-02

- Gắn với **1 cá nhân giữ chức** → TITLE.
- Chỉ **bộ máy/nha môn** (thường có 司/臺/院/部/衙門) → ORG.
- **Blocker kỹ thuật đã phát hiện (GR-02)**: 16/33 case `御史` thực ra là
  `御史臺` bị cắt do lỗi encode PUA — xem mục PUA normalization bên dưới,
  đây là lỗi **data/tokenization**, không phải lỗi judgment của annotator.

### PER / ORG / LOC cho dòng họ, triều đại, chính thể — xem GR-03, GR-04, GR-05, GR-06

Quy tắc chung: **dựa vào referent trong ngữ cảnh câu cụ thể, KHÔNG dựa vào
surface form cố định**. Cùng 1 chuỗi ký tự (`莫氏`, `吳`, `陳`, `哀牢`,
`林邑`) có thể đúng ở PER, ORG, hoặc LOC tùy câu — xem ví dụ đối lập trong
GR-04/GR-05.

### LOC — hậu tố hành chính

Hậu tố hành chính (`州 府 營 路 鎮 縣 坊 道 社 里`) phải được bao gồm trong
span LOC khi xuất hiện liền sau tên riêng, **trừ khi** ngữ cảnh dùng tỉnh
lược đã xác lập rõ từ câu trước (elliptical reference). 89 stem hiện có cả
2 dạng trong corpus (`artifacts/audit_v1/reports/boundary_admin_suffix_report.json`)
— cần review từng occurrence, chưa mass-correct.

### DTM boundary policy

- Từ chỉ ngày/tiết/mùa/can chi → DTM (GR-14).
- **KHÔNG dùng rule cứng gộp 2 tag DTM liền kề** (ví dụ mùa+ngày) — đã thử
  và xác nhận làm giảm F1 (xem memory dự án `dvsktt-ner-project`, thử
  2026-08-16, F1 DTM giảm 0.578→0.482) vì nhiều cặp DTM liền kề là 2 gold
  entity thật riêng biệt. Giữ quyết định gộp/tách theo từng câu cụ thể.
- Niên hiệu+năm là DTM khi đứng riêng (`洪德二十年`); khi có polity/triều
  đứng trước cần tách theo GR-11 (`明成化十八年` → `明`/ORG + `成化十八年`/DTM).

## Việc CHƯA xong (không tự chốt trong bản draft này)

| Vấn đề | Trạng thái |
|---|---|
| GR-10 (đơn vị hành chính LOC vs ORG) | OPEN, 50/50, cần domain expert chốt 1 chiều mặc định |
| GR-11 (524 candidate tách polity+era) | Chưa review, chỉ mới scan |
| GR-09 (EVENT: 會試/經筵) | Chưa quyết định remove hay giữ tạm |
| 107 codepoint PUA chưa verify (992 lần) | Không liên quan trực tiếp lỗi 御史臺, cần audit riêng nếu muốn xử lý |
| 52 surface trong `guideline_ambiguities` không có occurrence-level review | Quyết định hiện tại chỉ ở mức surface, KHÔNG auto-apply được (xem `apply_adjudication.py`) |
