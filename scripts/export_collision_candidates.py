"""
export_collision_candidates.py — xuất 8 collision case phát hiện bởi
apply_adjudication.py dry-run (boundary_split_fill_in.xlsx widen sang
đè lên entity đã tồn tại), PHÂN LOẠI RIÊNG từng case (KHÔNG coi mọi
overlap là merge candidate). Đọc-only, không sửa gì, không tự chọn
selected_action.

Usage:
  python scripts/export_collision_candidates.py \
      --train ... --dev ... --test ... \
      --out artifacts/audit_v1/review/collision_merge_candidates.xlsx
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

from src.data_utils import read_conll
from src.audit_utils import extract_all_spans, load_source_map

# ── 8 case đã biết (phát hiện từ apply_adjudication.py dry-run lần 2) ──────
# Phân loại theo đúng phân tích mục B của yêu cầu -- KHÔNG generic-hoá thành
# 1 rule "overlap -> merge". Mỗi case có overlap_type + action candidates
# RIÊNG, và ghi rõ lý do/điều cần kiểm tra trước khi reviewer chọn.
KNOWN_COLLISIONS = [
    {
        "collision_id": 1, "split": "train", "sample_id": 948,
        "proposed_span": (65, 70), "proposed_label": "DTM",
        "overlap_type": "guideline_conflict_GR11",
        "action_candidates": "KEEP_SEPARATE_AS_ORG_PLUS_DTM;MERGE_ENTITY_AS_DTM;CORRECT_BOUNDARY;NEEDS_DOMAIN_EXPERT",
        "validation_consequence": "Dry-run bao 'would_overlap_another_entity' vi 十七年/DTM da ton tai rieng.",
        "note": (
            "guideline_v2.0_draft GR-11 de xuat tach ORG(明)+DTM(成化N年) cho "
            "prefix polity rieng biet. O day gold hien tai la 1 entity "
            "明成化/DTM (KHONG phai 明/ORG rieng) dung canh 十七年/DTM co san "
            "-- CAN xac nhan: gold that su la [明成化][十七年] hay la mot "
            "loi khac (vd thieu entity 明/ORG). KHONG tu merge cho den khi "
            "GR-11 duoc chot."
        ),
    },
    {
        "collision_id": 2, "split": "train", "sample_id": 1050,
        "proposed_span": (126, 131), "proposed_label": "DTM",
        "overlap_type": "guideline_conflict_GR11",
        "action_candidates": "KEEP_SEPARATE_AS_ORG_PLUS_DTM;MERGE_ENTITY_AS_DTM;CORRECT_BOUNDARY;NEEDS_DOMAIN_EXPERT",
        "validation_consequence": "would_overlap_another_entity voi 十六年/DTM da ton tai rieng.",
        "note": "Cung mau voi case 1. Luu y gold entity 明成化 o day dang la LOC, khong phai ORG/DTM nhu cac case khac -- CAN kiem tra rieng truoc khi gop nhom.",
    },
    {
        "collision_id": 3, "split": "train", "sample_id": 1186,
        "proposed_span": (109, 114), "proposed_label": "DTM",
        "overlap_type": "guideline_conflict_GR11",
        "action_candidates": "KEEP_SEPARATE_AS_ORG_PLUS_DTM;MERGE_ENTITY_AS_DTM;CORRECT_BOUNDARY;NEEDS_DOMAIN_EXPERT",
        "validation_consequence": "would_overlap_another_entity voi 十八年/DTM da ton tai rieng.",
        "note": "Cung mau voi case 1.",
    },
    {
        "collision_id": 4, "split": "dev", "sample_id": 141,
        "proposed_span": (74, 79), "proposed_label": "DTM",
        "overlap_type": "guideline_conflict_GR11",
        "action_candidates": "KEEP_SEPARATE_AS_ORG_PLUS_DTM;MERGE_ENTITY_AS_DTM;CORRECT_BOUNDARY;NEEDS_DOMAIN_EXPERT",
        "validation_consequence": "would_overlap_another_entity voi 十五年/DTM da ton tai rieng.",
        "note": "Cung mau voi case 1-3. Gold entity 明成化 o day la LOC (giong case 2).",
    },
    {
        "collision_id": 5, "split": "train", "sample_id": 209,
        "proposed_span": (103, 105), "proposed_label": "PER",
        "overlap_type": "adjacent_merge_candidate",
        "action_candidates": "MERGE_ENTITY_AS_PER;KEEP_SEPARATE_PER_PLUS_TITLE;GUIDELINE_AMBIGUITY",
        "validation_consequence": "would_overlap_another_entity voi 太宗/PER da ton tai rieng.",
        "note": (
            "陳太宗 (Tran Thai Tong) la 1 nguoi cu the -- co the la referent "
            "duy nhat (PER gop) hoac 陳=dynasty/ORG + 太宗=ruler-title/TITLE "
            "(2 tang nghia long nhau, schema flat khong ho tro ca 2). "
            "Reviewer PHAI chon 1 trong 3: (a) PER(陳太宗) gop, "
            "(b) giu rieng PER(陳)+TITLE(太宗) [hien tai la ORG(陳)+PER(太宗), "
            "can sua ca nhan truoc], (c) policy khac theo guideline."
        ),
    },
    {
        "collision_id": 6, "split": "train", "sample_id": 107,
        "proposed_span": (33, 35), "proposed_label": "PER",
        "overlap_type": "boundary_conflict",
        "action_candidates": "CORRECT_BOUNDARY;KEEP_SEPARATE;NEEDS_DOMAIN_EXPERT",
        "validation_consequence": "would_overlap_another_entity voi 黎/PER da ton tai rieng.",
        "note": (
            "Cau goc '...黎輔陳之勇冠三軍...' -- CAN xac dinh day la 2 nhan "
            "danh rieng biet dung canh nhau (黎 + 陳, vd liet ke 2 tuong) hay "
            "1 nguoi ten '黎輔陳'/'輔陳' bi tach sai. KHONG mac dinh MERGE --"
            " co the offset reviewer dien (33,35) sai (nen la mo rong 陳 "
            "rieng, khong dinh lien 黎)."
        ),
    },
    {
        "collision_id": 7, "split": "train", "sample_id": 437,
        "proposed_span": (82, 84), "proposed_label": "TITLE",
        "overlap_type": "flat_schema_overlap",
        "action_candidates": "KEEP_GOLD;CORRECT_BOUNDARY;NEEDS_DOMAIN_EXPERT",
        "validation_consequence": "would_overlap_another_entity voi 故都/LOC (chu '都' dang thuoc LOC nay).",
        "note": (
            "'故都御史' -- 都 dang duoc gan cho 故都/LOC (kinh do cu). Muon mo "
            "rong 御史 thanh 都御史/TITLE thi PHAI bot '都' khoi 故都 truoc "
            "(2 sua dong thoi, khong phai 1 merge don gian). KHONG lien quan "
            "PUA normalization (khong co ky tu PUA o day). Can xac nhan: "
            "'都御史' co phai 1 chuc danh day du hay khong truoc khi sua."
        ),
    },
]

# Case 8 (test:139, "大行") ĐÃ LOẠI KHỎI danh sách collision thật -- phát
# hiện đây là FALSE POSITIVE do bug trong scripts/export_boundary_split_fillin.py
# (khớp entity nguồn chỉ theo label, không theo surface -- khi 1 câu có
# nhiều entity cùng nhãn TITLE, code lấy nhầm 經畧使[114-116] thay vì
# 大行[119-120]). Đã fix bug + patch lại đúng entity_id=22804 trong
# boundary_split_fill_in.xlsx (final_start=117/final_end=121 người review
# điền vẫn ĐÚNG, chỉ old span bị trỏ nhầm). Sau khi sửa, span mới
# (117,121) KHÔNG chồng lấn entity nào khác trong câu -- không còn là
# collision, xử lý như CORRECT_BOUNDARY bình thường qua
# process_boundary_split_fillin().


def _neighbors_str(ents):
    return "; ".join(f"{e['surface']}/{e['label']}[{e['start']}-{e['end']}](id={e['entity_id']})" for e in ents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--source-map", default=None)
    ap.add_argument("--out", default="artifacts/audit_v1/review/collision_merge_candidates.xlsx")
    ap.add_argument("--out-md", default="artifacts/audit_v1/review/collision_merge_candidates.md")
    args = ap.parse_args()

    splits_data = {
        "train": read_conll(args.train),
        "dev": read_conll(args.dev),
        "test": read_conll(args.test),
    }
    source_map = load_source_map(args.source_map)
    entities = extract_all_spans(splits_data, source_map, context_window=50)

    by_sentence = {}
    for e in entities:
        by_sentence.setdefault((e["split"], e["sample_id"]), []).append(e)

    rows = []
    md_sections = ["# Collision merge candidates — 8 case (phân loại riêng, KHÔNG MERGE mặc định)", ""]

    for meta in KNOWN_COLLISIONS:
        split, sample_id = meta["split"], meta["sample_id"]
        ents_here = sorted(by_sentence.get((split, sample_id), []), key=lambda e: e["start"])
        text = ents_here[0]["text"] if ents_here else ""
        ps, pe = meta["proposed_span"]
        ctx_s, ctx_e = max(0, ps - 50), min(len(text), pe + 1 + 50)
        marked = f"{text[ctx_s:ps]}〖{text[ps:pe+1]}〗{text[pe+1:ctx_e]}"

        original_entities_str = _neighbors_str(ents_here)

        rows.append({
            "collision_id": meta["collision_id"],
            "split": split,
            "sample_id": sample_id,
            "document_id": ents_here[0].get("document_id") if ents_here else None,
            "full_text": text,
            "marked_context_pm50": marked,
            "original_entities": original_entities_str,
            "proposed_new_span": f"{ps}-{pe}",
            "proposed_new_label": meta["proposed_label"],
            "overlap_type": meta["overlap_type"],
            "proposed_action_candidates": meta["action_candidates"],
            "likely_overextended_boundary": meta["collision_id"] == 8,
            "analysis_note": meta["note"],
            "selected_action": "",
            "merged_entity_ids": "",
            "expected_resulting_surface": "",
            "resulting_start": "",
            "resulting_end": "",
            "resulting_label": "",
            "guideline_rule_id": "",
            "reviewer_notes": "",
            "validation_consequence_if_unresolved": meta["validation_consequence"],
        })

        md_sections.append(f"## Collision {meta['collision_id']}: {split}:{sample_id}")
        md_sections.append(f"- overlap_type: **{meta['overlap_type']}**")
        md_sections.append(f"- proposed action candidates: {meta['action_candidates']}")
        md_sections.append(f"- proposed new span/label: {ps}-{pe} / {meta['proposed_label']}")
        md_sections.append(f"- original entities trong câu: {original_entities_str}")
        md_sections.append(f"- context: {marked}")
        md_sections.append(f"- ghi chú: {meta['note']}")
        md_sections.append(f"- hệ quả nếu không xử lý: {meta['validation_consequence']}")
        md_sections.append("")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="collisions", index=False)
        ws = writer.sheets["collisions"]
        ws.freeze_panes = "A2"
        wrap_cols = {"full_text", "marked_context_pm50", "original_entities", "analysis_note"}
        for col_idx, col_name in enumerate(df.columns, start=1):
            letter = get_column_letter(col_idx)
            if col_name in wrap_cols:
                ws.column_dimensions[letter].width = 60
                for row in range(2, len(df) + 2):
                    ws.cell(row=row, column=col_idx).alignment = Alignment(wrap_text=True, vertical="top")
            else:
                ws.column_dimensions[letter].width = max(len(col_name) + 2, 14)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_sections))

    print(f"Exported {len(rows)} collision case(s) -> {args.out}, {args.out_md}")
    for meta in KNOWN_COLLISIONS:
        print(f"  [{meta['collision_id']}] {meta['split']}:{meta['sample_id']} overlap_type={meta['overlap_type']}")


if __name__ == "__main__":
    main()
