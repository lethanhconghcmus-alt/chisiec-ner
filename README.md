# CHisIEC NER — Full Pipeline

NER cho dataset văn bản lịch sử Trung Quốc cổ đại (CHisIEC) với 3 phương pháp:
1. **GuwenBERT + CRF** — pre-train trên cổ văn, ưu tiên thử trước
2. **RoBERTa + BiLSTM + CRF** — nắm bắt context dài tốt hơn
3. **RoBERTa + KAN + CRF** — học pattern phi tuyến với Kolmogorov-Arnold Network

---

## Cấu trúc project

```
chisiec_ner/
├── configs/
│   └── config.yaml              # Tất cả hyperparams — chỉnh ở đây
├── src/
│   ├── data_utils.py            # CoNLL reader, validation, Dataset
│   ├── models.py                # 3 NER models
│   ├── trainer.py               # Train loop: fp16, early stopping
│   ├── evaluator.py             # F1, error analysis, confusion matrix
│   └── utils.py                 # Seed, logging, checkpoint manager
├── scripts/
│   ├── run_eda.py               # EDA + data validation
│   ├── train.py                 # Entry point train
│   ├── evaluate_compare.py      # So sánh 3 methods
│   └── inference.py             # Predict câu mới
├── notebooks/
│   └── kaggle_pipeline.ipynb    # Notebook chạy trên Kaggle
├── data/                        # Đặt train.txt, dev.txt, test.txt vào đây
└── requirements.txt
```

---

## Setup (macOS local)

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/chisiec-ner.git
cd chisiec-ner

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Đặt data vào thư mục data/
cp /path/to/chisiec/train.txt data/
cp /path/to/chisiec/dev.txt   data/
cp /path/to/chisiec/test.txt  data/
```

---

## Workflow

### Bước 1 — EDA & Validate data
```bash
python scripts/run_eda.py
# Output: outputs/eda/*.png, outputs/eda/*_stats.json
```

### Bước 2 — Train
```bash
# Method 1 (khuyến nghị thử trước)
python scripts/train.py model.method=guwenbert_crf

# Method 2
python scripts/train.py model.method=roberta_bilstm_crf

# Method 3
python scripts/train.py model.method=roberta_kan_crf

# Override bất kỳ config nào qua CLI
python scripts/train.py model.method=guwenbert_crf training.epochs=15 training.batch_size=8
```

### Bước 3 — So sánh kết quả
```bash
python scripts/evaluate_compare.py
# Output: outputs/compare_*.png, outputs/comparison_summary.csv
```

### Bước 4 — Inference
```bash
# Predict 1 câu
python scripts/inference.py --method guwenbert_crf --text "唐太宗李世民貞觀元年"

# Predict cả file
python scripts/inference.py --method guwenbert_crf \
    --input_file data/test.txt --output_file outputs/predictions.txt
```

---

## Chạy trên Kaggle GPU

```bash
# Push code lên GitHub
git add . && git commit -m "init" && git push

# Trong Kaggle Notebook:
!git clone https://github.com/YOUR_USERNAME/chisiec-ner.git
%cd chisiec-ner
!pip install -q -r requirements.txt
!python scripts/train.py model.method=guwenbert_crf training.fp16=true
```

Xem `notebooks/kaggle_pipeline.ipynb` để chạy full pipeline.

---

## Config quan trọng (`configs/config.yaml`)

| Field | Default | Ghi chú |
|-------|---------|---------|
| `model.method` | `guwenbert_crf` | Chọn 1 trong 3 methods |
| `training.epochs` | `10` | Tăng lên 15 nếu cần |
| `training.batch_size` | `16` | Giảm xuống 8 nếu OOM |
| `training.fp16` | `true` | Bật khi chạy Kaggle GPU |
| `early_stopping.patience` | `3` | Dừng nếu không cải thiện sau 3 epoch |
| `logging.use_wandb` | `false` | Đổi `true` + điền `wandb_entity` để track |

---

## Output sau khi train

```
outputs/
├── guwenbert_crf/
│   ├── results.json                  # F1 scores + history
│   ├── label_map.json
│   ├── train.log
│   ├── guwenbert_crf_best.pt         # Best model weights
│   ├── test_report.json              # Per-entity P/R/F1
│   ├── test_confusion_matrix.png
│   ├── test_fp_fn.png
│   └── test_error_analysis.json
├── roberta_bilstm_crf/  ...
├── roberta_kan_crf/     ...
├── compare_test_f1.png
├── compare_training_curves.png
├── compare_entity_f1.png
└── comparison_summary.csv
```

---

## Pre-trained Models

| Method | Backbone |
|--------|---------|
| `guwenbert_crf` | `ethanyt/guwenbert-base` |
| `roberta_bilstm_crf` | `hfl/chinese-roberta-wwm-ext` |
| `roberta_kan_crf` | `hfl/chinese-roberta-wwm-ext` |
