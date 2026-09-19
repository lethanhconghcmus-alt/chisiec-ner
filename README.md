# Ancient Chinese NER

NER pipeline for Ancient Chinese text using GuwenBERT + CRF.

**Trained on:** AnChineseNERE + CHisIEC + BEEVETA + C-CLUE  
**Entity types:** PER · LOC · TITLE · DTM · ORG  
**Backbone:** `ethanyt/guwenbert-base`

---

## Repo Structure

```
├── src/
│   ├── models.py        # GuwenBertCRF, GuwenBertLinear
│   └── data_utils.py    # CoNLL reader, NERDataset
├── api/
│   ├── main.py          # FastAPI app
│   └── predictor.py     # Inference logic
├── outputs/
│   └── ancient/
│       └── guwenbert_crf/
│           ├── best.pt
│           └── label_map.json
├── Dockerfile
└── requirements.txt
```

---

## API

### Start server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker build -t ancient-ner .
docker run -p 8000:8000 ancient-ner
```

### Endpoints

**GET /health**
```json
{"status": "ok", "model_loaded": true}
```

**POST /predict**

Request:
```json
{
  "sentences": "以其徃關上接領北朝頒賞勑諭銀幣故也。\n命胡士楊、阮名實、阮廷正等徃關上候命。"
}
```

Response:
```json
{
  "data": [
    {
      "text": "以其徃關上接領北朝頒賞勑諭銀幣故也。",
      "entities": []
    },
    {
      "text": "命胡士楊、阮名實、阮廷正等徃關上候命。",
      "entities": [
        {"text": "胡士楊", "label": "PER", "start": 1, "end": 4},
        {"text": "阮名實", "label": "PER", "start": 5, "end": 8},
        {"text": "阮廷正", "label": "PER", "start": 9, "end": 12}
      ]
    }
  ]
}
```

### Environment Variables

| Variable    | Default                                          | Description            |
|-------------|--------------------------------------------------|------------------------|
| `BACKBONE`  | `ethanyt/guwenbert-base`                         | HuggingFace model name |
| `CKPT_PATH` | `outputs/ancient/guwenbert_crf/best.pt`          | Model checkpoint       |
| `LABEL_MAP` | `outputs/ancient/guwenbert_crf/label_map.json`   | Label map JSON         |
| `MAX_LEN`   | `128`                                            | Max token length       |

---

## Training

See `notebook45a88c3f07.ipynb` for full training pipeline.

---

## BIOES + boundary-head multi-task training (M0/M1/M2)

Nâng cấp thêm start/end boundary head phụ trợ (auxiliary), huấn luyện
multi-task cùng CRF, KHÔNG cần annotate mới (nhãn boundary suy tự động từ
BIO/BIOES sẵn có). Baseline gốc (`GuwenBertCRF`/`GuwenBertLinear`, scheme
BIO) hoàn toàn không đổi — mọi lệnh cũ chạy y hệt trước.

**Kiến trúc M2**: `encoder -> {ner_classifier -> CRF (BIOES), start_classifier, end_classifier}`.
Xem docstring `src/models.py:BertCRFBoundaryNER` và `src/bioes_utils.py` để
biết chi tiết cách xử lý mask/label -100/CRF decode không hợp lệ.

### 1. Preprocess BIO -> BIOES (chỉ cần xem trước, script train tự convert)

```python
from src.data_utils import read_conll
from src.bioes_utils import convert_dataset_bio_to_bioes

data = read_conll("data/ancient/train.txt")
bioes_data, stats = convert_dataset_bio_to_bioes(data, mode="strict")  # "repair" nếu muốn tự sửa I-X mồ côi
print(stats)  # num_sequences, num_sequences_repaired, num_tags_repaired
```

### 2. Train M0 / M1 / M2

```bash
# M0 — baseline gốc, BIO, không boundary head (KHÔNG đổi gì so với trước)
python scripts/train.py model.method=guwenbert_crf model.backbone=SIKU-BERT/sikubert \
    checkpoint.output_dir=outputs/ancient_m0/seed42 project.seed=42

# M1 — BIOES, vẫn CRF cũ, không boundary head (đo lợi ích của BIOES đơn thuần)
python scripts/train.py model.method=guwenbert_crf model.backbone=SIKU-BERT/sikubert \
    data.label_scheme=bioes data.bioes_mode=strict \
    checkpoint.output_dir=outputs/ancient_m1/seed42 project.seed=42

# M2 — BIOES + boundary heads (model đề xuất)
python scripts/train_boundary.py config=configs/config_boundary.yaml \
    model.backbone=SIKU-BERT/sikubert model.enable_boundary_auxiliary=true \
    model.boundary_weight=0.1 model.boundary_loss_type=weighted_bce \
    checkpoint.output_dir=outputs/ancient_boundary_m2/seed42 project.seed=42
```

`model.backbone` cũng nhận `hsc748NLP/GujiRoBERTa_jian` (GujiRoBERTa) hoặc
`ethanyt/guwenbert-base` (GuwenBERT). Chạy lại với `project.seed=43/44/45/46`
và `checkpoint.output_dir=.../seedNN` riêng cho từng seed để multi-seed.

### 3. Evaluate / error report

M0/M1 dùng `Evaluator` (đã có sẵn, `scripts/train.py` tự gọi
`full_report` + `error_analysis` cuối quá trình train). M2 dùng
`BoundaryEvaluator` (`src/evaluator_boundary.py`), cũng tự gọi cuối
`scripts/train_boundary.py`. Kết quả:
- `<output_dir>/<method>/test_report.json` — P/R/F1 strict entity-level toàn cục + từng loại.
- `<output_dir>/<method>/test_errors.jsonl` — lỗi từng câu (mục F.9: gold/pred BIOES, spans, start/end gold/pred, error_categories).
- `<output_dir>/<method>/test_error_summary.json` — đếm theo category + focus report (PER/LOC len>=2, LOC hậu tố hành chính 州/府/營/路/鎮/縣/坊, partial-span examples).

### 4. Aggregate multi-seed results (M0 vs M1 vs M2)

```bash
python scripts/aggregate_results.py \
    --exp M0="outputs/ancient_m0/*/results.json" \
    --exp M1="outputs/ancient_m1/*/results.json" \
    --exp M2="outputs/ancient_boundary_m2/*/results.json" \
    --out results/ablation_summary
```

Ghi ra `results/ablation_summary.csv` (1 dòng/seed/experiment, đủ cột theo
mục G), `.md` (bảng mean±std + delta M1-M0/M2-M1 theo từng seed), `.json`.

### 5. Unit tests

```bash
pip install pytest
pytest tests/ -q
```

`tests/test_bioes_utils.py` (BIO<->BIOES, boundary label derivation, CRF
decode repair), `tests/test_dataset_boundary.py` (alignment/padding/-100),
`tests/test_model_boundary.py` (loss masking, focal loss, overfit nhỏ),
`tests/test_evaluator_boundary.py` (span error categorization, JSONL
export), `tests/test_aggregate_results.py`. Test model dùng backbone giả
`hf-internal-testing/tiny-random-bert` (nhanh, không cần GPU) chỉ để xác
minh wiring đúng — KHÔNG phản ánh chất lượng thật trên SikuBERT/GujiRoBERTa.

### Config flags chính (`configs/config_boundary.yaml`)

| Key | Ý nghĩa |
|---|---|
| `data.label_scheme` (config.yaml, M0/M1) | `bio` (mặc định) hoặc `bioes` |
| `data.bioes_mode` | `strict` (mặc định, raise lỗi kèm sample_id+vị trí nếu BIO gốc lỗi) hoặc `repair` |
| `model.method` | `guwenbert_crf`/`guwenbert_linear` (M0/M1) hoặc `bert_crf_boundary` (M2) |
| `model.enable_boundary_auxiliary` | `false` = tắt boundary loss, chỉ còn `crf_loss` (vẫn BIOES) |
| `model.boundary_weight` | hệ số λ trong `crf_loss + λ*(start_loss+end_loss)`, mặc định 0.1 |
| `model.boundary_loss_type` | `bce` / `weighted_bce` (mặc định) / `focal` |
| `model.boundary_pos_weight_max` | clip pos_weight (tính từ TRAIN split) cho `weighted_bce` |
| `model.focal_alpha`, `model.focal_gamma` | tham số focal loss |
| `training.encoder_lr/ner_head_lr/boundary_head_lr/crf_lr` | differential LR — xem `src/trainer_boundary.py` |

---

## Annotation consistency audit (KHÔNG sửa nhãn tự động)

Sau khi M0/M1/M2/M3 hội tụ về cùng 1 dải F1 (~0.65-0.67, xem lịch sử ablation
ở trên), hướng cải thiện tiếp theo là data-centric: audit tính nhất quán
của gold label hiện có (ORG↔TITLE, PER↔TITLE, LOC↔ORG, DTM, boundary địa
danh hành chính) trước khi thu thập/annotate thêm data mới.

`scripts/annotation_audit.py` CHỈ ĐỌC train/dev/test hiện có, KHÔNG sửa bất
kỳ nhãn nào, KHÔNG ghi đè dataset gốc — chỉ sinh **candidate inconsistency**
(ứng viên cần người adjudicate), không khẳng định gold sai.

```bash
python scripts/annotation_audit.py \
    --train data/dvsk/dvsk_train.txt \
    --dev   data/dvsk/dvsk_dev.txt \
    --test  data/dvsk/dvsk_test.txt \
    --out-dir artifacts/audit_v1 \
    --top-n 300 \
    [--source-map path/to/record_source_map.json]  # optional, join document_id (chapter DVSKTT)
```

Output (`artifacts/audit_v1/`, KHÔNG track git — regenerable, xem `.gitignore`):
- `reports/ambiguity_report.{csv,jsonl}` — surface form gán >=2 nhãn khác nhau.
- `reports/boundary_admin_suffix_report.json` — LOC lúc có lúc không có hậu
  tố hành chính (州/府/營/路/鎮/縣/坊/道/社/里) trên cùng 1 "stem".
- `reports/merge_split_report.json` — surface vừa có bằng chứng bị TÁCH
  (đứng cạnh entity khác loại) vừa có bằng chứng bị GỘP (là tiền tố của 1
  entity dài hơn) ở nơi khác trong corpus — tổng quát hóa từ script cũ
  `find_office_inconsistency.py` cho các cặp ORG/LOC/PER × TITLE.
- `reports/confusable_pair_report.json` — cross-reference theo đúng 5 cặp
  dễ nhầm (ORG-TITLE, PER-TITLE, LOC-ORG, DTM-*).
- `review/review_workbook.{csv,xlsx}` — bảng review, mỗi dòng 1 occurrence
  cụ thể cần adjudicate, sắp theo `priority_score` (heuristic — xem
  `src/audit_utils.py:compute_priority_score`), cột `reviewer_decision`/
  `final_label`/`final_start`/`final_end` để TRỐNG cho reviewer điền, KHÔNG
  tự động điền.
- `summary.md` — thống kê tổng quan + top 50 ambiguous surfaces + top
  boundary inconsistencies.
- `manifest.json` — timestamp, config, checksum MD5 của từng file data đầu
  vào (để biết report ứng với đúng version data nào).

**Model disagreement (mục 6, optional)**: `build_review_workbook()` hỗ trợ
tham số `model_disagreements` nếu có checkpoint + prediction, nhưng lần
chạy audit_v1 KHÔNG bật (không có checkpoint M0 gốc cục bộ — chỉ có M1/M2/M3
kiến trúc khác, dùng nhầm sẽ gắn nhãn "M0 disagreement" sai sự thật). Muốn
bật: tự chạy inference, build dict `{(split, sample_id): {"confidence":...}}`
rồi truyền vào `build_review_workbook`.

### Quy trình tạo dataset v2 sau adjudication

```
dataset_v1   = data/dvsk/dvsk_{train,dev,test}.txt hiện tại (đã freeze, dùng cho M0-M3)
guideline_v2 = rule mới thống nhất sau khi reviewer quyết định các case trong review_workbook
dataset_v2   = dataset_v1 + labels đã adjudicate (CÓ changelog, KHÔNG sửa lặng lẽ)
```

1. Reviewer mở `review_workbook.xlsx`, điền `reviewer_decision`
   (OK/SỬA/BỎ), `final_label`/`final_start`/`final_end` nếu sửa,
   `guideline_rule_id` tham chiếu rule mới nếu có.
2. Freeze `guideline_v2` (văn bản rule, giống `docs/annotation_guideline.md`
   ở repo `ancient-chinese-ner`) TRƯỚC khi áp bất kỳ thay đổi nào — không
   sửa data trước rồi mới viết rule diễn giải ngược.
3. Viết script `apply_adjudication.py` (chưa có — làm ở bước sau) đọc
   `review_workbook.xlsx` đã điền, áp đúng các dòng có `reviewer_decision`,
   sinh `dvsk_{train,dev,test}_v2.txt` + report số dòng đã sửa (đối chiếu
   pattern đã dùng thành công trước đó: `scripts/apply_gold_review.py` ở
   repo `ancient-chinese-ner`, gold review v2.1).
4. Chạy lại **M0 (SikuBERT+CRF+BIO)** trên `dataset_v2` TRƯỚC khi thu thập
   data mới — tách bạch được "F1 tăng do label quality" hay "do thêm data"
   (thứ tự này quan trọng — làm ngược sẽ khiến active-learning pool thu
   thập sau kế thừa đúng ambiguity chưa xử lý).
5. Chỉ sau đó mới build raw pool + active learning (B0-B4 active-vs-random,
   xem memory `dvsktt-bioes-boundary-heads`).
