"""
annotation_audit_deepdive.py — vòng review thận trọng thứ 2, theo đúng yêu
cầu: tách riêng (A) candidate lỗi gõ/nhãn cục bộ (singleton anomaly) khỏi
(B) ambiguity mang tính hệ thống cần rule guideline (KHÔNG coi minority là
lỗi), cộng 1 case file riêng cho 光紹帝 và deep-dive đầy đủ occurrence cho
9 surface được chỉ định. CHỈ ĐỌC dữ liệu, KHÔNG sửa nhãn, KHÔNG tự điền
reviewer_decision/final_label/semantic_role_candidate.

Usage:
  python scripts/annotation_audit_deepdive.py \
      --train data/dvsk/dvsk_train.txt --dev data/dvsk/dvsk_dev.txt \
      --test data/dvsk/dvsk_test.txt --out-dir artifacts/audit_v1 \
      [--source-map path/to/record_source_map.json]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from src.data_utils import read_conll
from src.audit_utils import (
    REVIEWER_DECISION_OPTIONS,
    SEMANTIC_ROLE_OPTIONS,
    build_ambiguity_report,
    build_case_file_rows,
    extract_all_spans,
    find_adjacent_split_variants,
    find_guideline_ambiguities,
    find_singleton_anomalies,
    load_source_map,
)

DEEPDIVE_SURFACES = ["日燏", "御史", "承司", "明成化", "莫氏", "吳", "陳", "哀牢", "林邑"]
LIGHT_SHAO_DI = "光紹帝"


# ── Excel formatting helper (frozen header, wrap text, dropdown) ───────────
def write_formatted_xlsx(path, sheets: dict, dropdown_columns: dict = None, wrap_columns: dict = None):
    """
    sheets: {sheet_name: DataFrame}
    dropdown_columns: {sheet_name: {col_name: [options]}} -- thêm data
      validation dropdown (không ép giá trị, chỉ gợi ý).
    wrap_columns: {sheet_name: [col_name, ...]} -- bật wrap_text + rộng cột
      cho các cột text dài (context/sentence).
    """
    dropdown_columns = dropdown_columns or {}
    wrap_columns = wrap_columns or {}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

        for name, df in sheets.items():
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "A2"
            n_rows = len(df) + 1
            wrap_cols = wrap_columns.get(name, [])
            for col_idx, col_name in enumerate(df.columns, start=1):
                letter = get_column_letter(col_idx)
                if col_name in wrap_cols:
                    ws.column_dimensions[letter].width = 60
                    for row in range(2, n_rows + 1):
                        ws.cell(row=row, column=col_idx).alignment = Alignment(
                            wrap_text=True, vertical="top"
                        )
                else:
                    ws.column_dimensions[letter].width = min(max(len(str(col_name)) + 2, 10), 25)

            for col_name, options in dropdown_columns.get(name, {}).items():
                if col_name not in df.columns:
                    continue
                col_idx = list(df.columns).index(col_name) + 1
                letter = get_column_letter(col_idx)
                formula = '"' + ",".join(options) + '"'
                dv = DataValidation(type="list", formula1=formula, allow_blank=True)
                ws.add_data_validation(dv)
                dv.add(f"{letter}2:{letter}{n_rows}")


def build_case_light_shao_di_markdown(rows: list, split_variants: list) -> str:
    lines = ["# Case file: 光紹帝 (toàn bộ occurrence)", ""]
    lines.append(
        "Mục tiêu: kiểm tra kỹ TRƯỚC khi coi occurrence duy nhất mang nhãn "
        "ORG là lỗi gõ. KHÔNG tự sửa label ở đây."
    )
    lines.append("")
    label_counts = {}
    for r in rows:
        label_counts[r["gold_label"]] = label_counts.get(r["gold_label"], 0) + 1
    lines.append(f"Tổng occurrence: **{len(rows)}**. Phân bố nhãn: {label_counts}")
    lines.append("")

    minority_label = min(label_counts, key=lambda l: label_counts[l])
    lines.append(f"Nhãn thiểu số: **{minority_label}** ({label_counts[minority_label]} lần)")
    lines.append("")

    lines.append("## Kiểm tra tách segmentation (光紹 + 帝 ở nơi khác?)")
    if split_variants:
        lines.append(f"PHÁT HIỆN {len(split_variants)} chỗ bị tách:")
        for v in split_variants:
            lines.append(
                f"- {v['split']}#{v['sample_id']}: `{v['part1_surface']}`/{v['part1_label']} + "
                f"`{v['part2_surface']}`/{v['part2_label']} — {v['context']}"
            )
    else:
        lines.append("KHÔNG tìm thấy chỗ nào 光紹帝 bị tách thành 2 entity liền kề "
                      "(光紹+帝 hoặc biến thể khác) — segmentation nhất quán.")
    lines.append("")

    lines.append("## Toàn bộ 20 occurrence")
    lines.append("")
    lines.append("| entity_id | split | sample_id | doc_id | label | span_check | neighbors | context ±30 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        marker = " **← NHÃN THIỂU SỐ**" if r["gold_label"] == minority_label else ""
        lines.append(
            f"| {r['entity_id']} | {r['split']} | {r['sample_id']} | {r['document_id']} | "
            f"{r['gold_label']}{marker} | {r['span_check_ok']} | {r['neighboring_entities']} | "
            f"{r['context_pm30']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--source-map", default=None)
    ap.add_argument("--out-dir", default="artifacts/audit_v1")
    args = ap.parse_args()

    splits_data = {
        "train": read_conll(args.train),
        "dev": read_conll(args.dev),
        "test": read_conll(args.test),
    }
    source_map = load_source_map(args.source_map)
    entities = extract_all_spans(splits_data, source_map, context_window=30)
    ambiguity_report = build_ambiguity_report(entities)

    review_dir = os.path.join(args.out_dir, "review")
    os.makedirs(review_dir, exist_ok=True)

    # ── 1. Case file 光紹帝 ──────────────────────────────────────────────
    shao_di_rows = build_case_file_rows(entities, LIGHT_SHAO_DI, context_window=30)
    split_variants = find_adjacent_split_variants(entities, LIGHT_SHAO_DI)
    case_md = build_case_light_shao_di_markdown(shao_di_rows, split_variants)
    with open(os.path.join(review_dir, "case_light_shao_di.md"), "w", encoding="utf-8") as f:
        f.write(case_md)

    shao_di_df = pd.DataFrame(shao_di_rows)
    write_formatted_xlsx(
        os.path.join(review_dir, "case_light_shao_di.xlsx"),
        sheets={"光紹帝_all_occurrences": shao_di_df},
        wrap_columns={"光紹帝_all_occurrences": ["full_sentence", "context_pm30", "neighboring_entities"]},
    )

    # ── 2A. Singleton anomalies (top 30) ───────────────────────────────
    singleton_anomalies = find_singleton_anomalies(ambiguity_report, entities, min_occurrences=10)
    singleton_rows = []
    for i, c in enumerate(singleton_anomalies[:30]):
        singleton_rows.append({
            "rank": i + 1,
            "surface": c["surface"],
            "minority_label": c["minority_label"],
            "majority_label": c["majority_label"],
            "majority_count": c["majority_count"],
            "total_occurrences": c["total_occurrences"],
            "label_counts": json.dumps(c["label_counts"], ensure_ascii=False),
            "record_ref": f"{c['minority_split']}:{c['minority_sample_id']}",
            "minority_document_id": c["minority_document_id"],
            "minority_context": c["minority_context"],
            "reviewer_decision": "",
            "reviewer_notes": "",
        })
    singleton_df = pd.DataFrame(singleton_rows)
    write_formatted_xlsx(
        os.path.join(review_dir, "review_priority_singleton_anomalies.xlsx"),
        sheets={"singleton_anomalies": singleton_df},
        dropdown_columns={"singleton_anomalies": {"reviewer_decision": REVIEWER_DECISION_OPTIONS}},
        wrap_columns={"singleton_anomalies": ["minority_context", "label_counts"]},
    )

    # ── 2B. Guideline ambiguities (surface-level, ranked by frequency) ─
    guideline_ambiguities = find_guideline_ambiguities(ambiguity_report, min_second_count=2)
    guideline_rows = []
    for r in guideline_ambiguities:
        guideline_rows.append({
            "surface": r["surface"],
            "total_occurrences": r["total_occurrences"],
            "num_unique_labels": r["num_unique_labels"],
            "label_counts": json.dumps(r["label_counts"], ensure_ascii=False),
            "splits_present": ",".join(r["splits_present"]),
            "reviewer_decision": "",
            "reviewer_notes": "",
            "guideline_rule_id": "",
        })
    guideline_df = pd.DataFrame(guideline_rows)

    # ── deep-dive full occurrence export cho 9 surface chỉ định ────────
    deepdive_rows = []
    for surface in DEEPDIVE_SURFACES:
        for row in build_case_file_rows(entities, surface, context_window=30):
            row = dict(row)
            row["surface"] = surface
            row["reviewer_decision"] = ""
            deepdive_rows.append(row)
    deepdive_df = pd.DataFrame(deepdive_rows)
    cols = ["surface"] + [c for c in deepdive_df.columns if c != "surface"]
    deepdive_df = deepdive_df[cols]

    write_formatted_xlsx(
        os.path.join(review_dir, "review_guideline_ambiguities.xlsx"),
        sheets={
            "guideline_ambiguities": guideline_df,
            "deepdive_occurrences": deepdive_df,
        },
        dropdown_columns={
            "guideline_ambiguities": {"reviewer_decision": REVIEWER_DECISION_OPTIONS},
            "deepdive_occurrences": {
                "reviewer_decision": REVIEWER_DECISION_OPTIONS,
                "semantic_role_candidate": SEMANTIC_ROLE_OPTIONS,
            },
        },
        wrap_columns={
            "guideline_ambiguities": ["label_counts"],
            "deepdive_occurrences": ["full_sentence", "context_pm30", "neighboring_entities"],
        },
    )

    # ── guideline_v2 draft (KHÔNG áp dụng, chỉ đề xuất) ─────────────────
    guideline_draft = build_guideline_v2_draft()
    with open(os.path.join(args.out_dir, "guideline_v2_draft.md"), "w", encoding="utf-8") as f:
        f.write(guideline_draft)

    # ── 8. Update summary với 3 con số tách biệt ────────────────────────
    n_singleton_surfaces = len({c["surface"] for c in singleton_anomalies})
    n_guideline_surfaces = len(guideline_ambiguities)
    n_unresolved = len(set(ambiguity_report and [r["surface"] for r in ambiguity_report]) -
                        {c["surface"] for c in singleton_anomalies} -
                        {r["surface"] for r in guideline_ambiguities})
    deepdive_summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "KHONG goi tat ca 251 ambiguous surfaces la 'label errors'. "
            "3 nhom tach biet:"
        ),
        "suspected_isolated_annotation_errors": {
            "num_surfaces_with_singleton_anomaly": n_singleton_surfaces,
            "num_singleton_occurrences_total": len(singleton_anomalies),
        },
        "context_dependent_legitimate_ambiguities": {
            "num_surfaces": n_guideline_surfaces,
        },
        "unresolved_requires_domain_expert": {
            "num_surfaces_not_yet_classified": n_unresolved,
            "note": "chua qualif y cho ca 2 nhom tren (vd < min_occurrences "
                     "hoac chi 1 nhan phu khong du 'canh tranh that su') "
                     "-- can NEEDS_DOMAIN_EXPERT thay vi tu dong xep loai.",
        },
    }
    with open(os.path.join(args.out_dir, "deepdive_summary.json"), "w", encoding="utf-8") as f:
        json.dump(deepdive_summary, f, ensure_ascii=False, indent=2)

    print(f"Singleton anomaly candidates: {len(singleton_anomalies)} occurrences "
          f"across {n_singleton_surfaces} surfaces (top 30 exported)")
    print(f"Guideline ambiguity surfaces: {n_guideline_surfaces}")
    print(f"Unresolved/unclassified surfaces: {n_unresolved}")
    print(f"Output -> {review_dir}, {args.out_dir}/guideline_v2_draft.md, "
          f"{args.out_dir}/deepdive_summary.json")

    print("\n=== TOP 10 SINGLETON ANOMALY CANDIDATES ===")
    for c in singleton_anomalies[:10]:
        print(f"'{c['surface']}': minority={c['minority_label']}(1) vs "
              f"majority={c['majority_label']}({c['majority_count']}), "
              f"total={c['total_occurrences']}, record={c['minority_split']}:{c['minority_sample_id']}")

    print("\n=== TOP 10 GUIDELINE AMBIGUITY CANDIDATES (by frequency) ===")
    for r in guideline_ambiguities[:10]:
        print(f"'{r['surface']}': {r['label_counts']} (total={r['total_occurrences']}, "
              f"splits={r['splits_present']})")

    print(f"\n=== 光紹帝: {len(shao_di_rows)} occurrences, split_variants_found={len(split_variants)} ===")


def build_guideline_v2_draft() -> str:
    return """# Guideline v2 — DRAFT (chưa áp dụng, chờ adjudication)

**KHÔNG áp dụng nhãn nào từ file này.** Đây là đề xuất rule dựa trên pattern
quan sát được từ annotation audit (`artifacts/audit_v1/`), cần reviewer/
domain expert xác nhận trước khi freeze thành guideline chính thức.

## PERSON / RULER

- Miếu hiệu/thụy hiệu/đế hiệu của 1 cá nhân cụ thể (vd `光紹帝`, `昭皇`) khi
  dùng để CHỈ ĐÍCH DANH người đó (chủ ngữ/tân ngữ hành động) → **PER**.
- Khi cùng chuỗi ký tự dùng như 1 DANH XƯNG CHUNG không gắn với 1 cá nhân cụ
  thể trong câu (vd liệt kê tước vị, câu tổng quát về ngôi vị) → **TITLE**.
- Schema hiện flat (không nested) → PRIORITY RULE khi mơ hồ: ưu tiên PER
  nếu có thể xác định referent cụ thể trong câu; chỉ dùng TITLE khi câu
  KHÔNG cho phép xác định 1 cá nhân cụ thể.
- Case cần xác nhận: `光紹帝` hiện 15 PER / 4 TITLE / 1 ORG — audit đề xuất
  ORG (1 lần) là lỗi gõ; PER vs TITLE (15 vs 4) cần rule trên để xử lý nhất
  quán các occurrence còn lại.

## DYNASTY / CLAN / POLITY

- Referent là TẬP THỂ CHÍNH TRỊ đang hành động như 1 chủ thể (khởi binh,
  cai trị, giao chiến) → **ORG**.
- Referent là CÁ NHÂN/NHÓM NGƯỜI cụ thể thuộc dòng họ đó (không phải bản
  thân thể chế) → **PER**.
- Referent là ĐỊA BÀN/lãnh thổ gắn với dòng họ/quốc gia đó → **LOC**.
- Rule dựa trên NGỮ CẢNH, KHÔNG dựa trên surface form cố định — cùng 1
  surface (`莫氏`, `吳`, `陳`, `哀牢`, `林邑`) có thể đúng ở cả 3 nhãn tùy câu.
- Case cần xác nhận: `莫氏` (ORG:12/PER:10/LOC:1), `吳` (PER:15/ORG:11/LOC:1),
  `陳` (ORG:6/PER:4/LOC:1), `哀牢` (ORG:20/LOC:16), `林邑` (ORG:5/PER:4/LOC:1)
  — LOC=1 ở 3/5 case có vẻ là lỗi gõ (audit riêng), phần ORG/PER còn lại
  CẦN rule ngữ cảnh trên, không nên mass-correct theo majority.

## OFFICE / INSTITUTION

- Chức vụ/tước vị GẮN VỚI 1 cá nhân cụ thể trong câu (ai giữ chức gì) →
  **TITLE**.
- Cơ quan/tập thể hành chính hoạt động NHƯ 1 tổ chức (ban hành lệnh, có
  thẩm quyền tập thể, không gắn 1 cá nhân) → **ORG**.
- Case mơ hồ 2 chiều (vd `禮部` vừa là bộ máy hành chính vừa gắn với 1 quan
  chức đứng đầu) → ưu tiên ORG nếu câu nói về CƠ QUAN, TITLE nếu câu nói về
  NGƯỜI giữ chức đó.
- Case cần xác nhận: `御史` (TITLE:29/ORG:4), `承司` (ORG:11/TITLE:6) — cần
  ví dụ cụ thể từng occurrence thiểu số để quyết định có phải lỗi gõ hay là
  2 cách dùng hợp lệ.

## TIME

- Niên hiệu (vd `明成化`) khi dùng ĐỂ ĐỊNH VỊ THỜI GIAN 1 sự kiện (thường đi
  kèm số năm, "X niên") → **DTM**.
- Khi niên hiệu dùng để CHỈ TRIỀU ĐẠI/CHÍNH THỂ ban hành niên hiệu đó (ít
  gặp hơn) → **ORG**.
- Case cần xác nhận: `明成化` hiện ORG:2/DTM:2/LOC:2 — chia đều, CẦN xem lại
  cả 6 occurrence cụ thể (không suy đoán từ tần suất).

## LOCATION (boundary hậu tố hành chính)

- Hậu tố hành chính (州/府/營/路/鎮/縣/坊/道/社/里) PHẢI được bao gồm trong
  span LOC khi xuất hiện liền sau tên riêng, TRỪ KHI ngữ cảnh rõ ràng dùng
  tỉnh lược (elliptical reference) đã xác lập từ câu trước.
- 89 stem hiện có cả 2 dạng (có/không hậu tố) trong corpus — audit KHÔNG
  đủ để phân biệt tỉnh lược hợp lệ vs annotation thiếu sót, cần review
  từng occurrence trong `reports/boundary_admin_suffix_report.json`.
"""


if __name__ == "__main__":
    main()
