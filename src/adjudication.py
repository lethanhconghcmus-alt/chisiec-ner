"""
adjudication.py — logic thuần cho scripts/apply_adjudication.py: đọc review
workbook đã điền, validate nghiêm ngặt, sinh ProposedChange. KHÔNG bao giờ
tự ghi dataset_v2 ở module này — chỉ trả về dữ liệu; script CLI quyết định
ghi ra đĩa hay không dựa theo --dry-run/--apply.

Nguyên tắc cứng (theo yêu cầu, KHÔNG thương lượng):
- Chỉ auto-apply khi occurrence được review TRỰC TIẾP (có entity_id/offset
  cụ thể trong workbook) — surface-level-only decision (sheet
  guideline_ambiguities) KHÔNG bao giờ tự apply vì suy ra sẽ là majority
  vote trá hình.
- Chỉ CORRECT_LABEL được tự derive nhãn mới (qua SEMANTIC_ROLE_TO_LABEL,
  không mơ hồ). CORRECT_BOUNDARY/SPLIT_ENTITY/MERGE_ENTITY cần offset mới
  TƯỜNG MINH trong workbook (final_start/final_end) — hiện các workbook đã
  export KHÔNG có cột này => luôn bị validate error/skip, KHÔNG suy đoán từ
  free-text reviewer_notes.
- NEEDS_DOMAIN_EXPERT/UNSURE/GUIDELINE_AMBIGUITY/KEEP_GOLD không bao giờ
  tạo ProposedChange.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.audit_utils import ENTITY_TYPES, _file_checksum

ALLOWED_APPLY_DECISIONS = {
    "CORRECT_LABEL", "CORRECT_BOUNDARY", "SPLIT_ENTITY", "MERGE_ENTITY",
    "REMOVE_OUT_OF_SCHEMA_ENTITY",
}
NEVER_APPLY_DECISIONS = {"GUIDELINE_AMBIGUITY", "NEEDS_DOMAIN_EXPERT", "UNSURE", "KEEP_GOLD"}

# Chỉ dùng khi reviewer_decision == CORRECT_LABEL. "OTHER"/blank KHÔNG map
# (thường là case cần SPLIT_ENTITY/xử lý riêng, không phải đổi nhãn đơn giản).
SEMANTIC_ROLE_TO_LABEL = {
    "PERSON": "PER",
    "RULER": "PER",
    "DYNASTY": "ORG",
    "CLAN": "ORG",
    "POLITY": "ORG",
    "PLACE": "LOC",
    "OFFICE": "TITLE",
    "INSTITUTION": "ORG",
    "TIME": "DTM",
}


@dataclass
class ProposedChange:
    change_id: int
    sample_id: int
    split: str
    document_id: Optional[str]
    rule_id: str
    reviewer_decision: str
    old_span: tuple
    old_label: str
    new_span: tuple
    new_label: str
    original_text: str
    normalized_text_if_any: Optional[str]
    reviewer_notes: str
    source_workbook_row: str
    validation_status: str  # "valid" | "error"
    validation_message: str = ""

    def to_row(self) -> dict:
        d = dict(
            change_id=self.change_id, sample_id=self.sample_id, split=self.split,
            document_id=self.document_id, rule_id=self.rule_id,
            reviewer_decision=self.reviewer_decision,
            old_span=list(self.old_span), old_label=self.old_label,
            new_span=list(self.new_span), new_label=self.new_label,
            original_text=self.original_text,
            normalized_text_if_any=self.normalized_text_if_any,
            reviewer_notes=self.reviewer_notes,
            source_workbook_row=self.source_workbook_row,
            validation_status=self.validation_status,
            validation_message=self.validation_message,
        )
        return d


@dataclass
class SkippedDecision:
    source_workbook_row: str
    split: Optional[str]
    sample_id: Optional[object]
    surface: Optional[str]
    reviewer_decision: str
    reason: str


def build_entity_index(entities: list) -> dict:
    """(split, sample_id) -> list[entity dict] (tham chiếu trực tiếp entities
    hiện có, dùng để validate 'source span matches gold surface' trước khi
    apply — nếu dataset thay đổi so với lúc export workbook, match sẽ fail
    và bị log validation error thay vì áp nhầm)."""
    by_sentence = defaultdict(list)
    for e in entities:
        by_sentence[(e["split"], e["sample_id"])].append(e)
    return by_sentence


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end < b_start or a_start > b_end)


def check_no_overlap_after_edit(entities_in_sentence: list, new_start: int, new_end: int,
                                 exclude_start: int, exclude_end: int) -> bool:
    """True nếu (new_start,new_end) không đè lên bất kỳ entity nào KHÁC
    trong câu (bỏ qua chính entity đang sửa, nhận diện qua old span)."""
    for e in entities_in_sentence:
        if e["start"] == exclude_start and e["end"] == exclude_end:
            continue
        if spans_overlap(new_start, new_end, e["start"], e["end"]):
            return False
    return True


def process_singleton_anomalies(df: pd.DataFrame) -> tuple:
    """
    review_priority_singleton_anomalies.xlsx KHÔNG có cột final_label/
    final_start/final_end (chỉ có reviewer_notes tự do) -- theo đúng
    nguyên tắc "record không có final_label/final offsets hợp lệ" =>
    KHÔNG auto-apply. Mọi dòng đều bị skip với lý do tường minh, để user
    biết cần bổ sung cột structured trước khi apply được.
    """
    skipped = []
    for i, row in df.iterrows():
        decision = str(row.get("reviewer_decision") or "").strip()
        ref = str(row.get("record_ref") or "")
        split, sample_id = (ref.split(":", 1) + [None])[:2] if ref else (None, None)
        if not decision:
            reason = "reviewer_decision_empty"
        elif decision == "KEEP_GOLD":
            reason = "no_change_needed_keep_gold"
        else:
            reason = "missing_structured_final_label_or_offset_fields"
        skipped.append(SkippedDecision(
            source_workbook_row=f"singleton_anomalies:row{i}",
            split=split, sample_id=sample_id, surface=row.get("surface"),
            reviewer_decision=decision, reason=reason,
        ))
    return [], skipped


def process_guideline_ambiguities_surface_level(df: pd.DataFrame) -> list:
    """Sheet guideline_ambiguities là surface-level (không phải occurrence-
    level) -- áp bất kỳ decision nào ở đây đồng nghĩa suy diễn theo majority
    vote, bị cấm tường minh. Luôn skip, kể cả CORRECT_LABEL/SPLIT_ENTITY."""
    skipped = []
    for i, row in df.iterrows():
        skipped.append(SkippedDecision(
            source_workbook_row=f"guideline_ambiguities:row{i}",
            split=None, sample_id=None, surface=row.get("surface"),
            reviewer_decision=str(row.get("reviewer_decision") or ""),
            reason="surface_level_only_not_occurrence_reviewed_majority_vote_forbidden",
        ))
    return skipped


def process_deepdive_occurrences(df: pd.DataFrame, entity_index: dict,
                                  change_id_start: int = 0) -> tuple:
    proposed, skipped = [], []
    change_id = change_id_start

    for i, row in df.iterrows():
        decision = str(row.get("reviewer_decision") or "").strip()
        split = row.get("split")
        sample_id = row.get("sample_id")
        surface = row.get("gold_surface")
        old_label = row.get("gold_label")
        wb_row_id = f"deepdive_occurrences:entity_id={row.get('entity_id')}"

        try:
            old_start = int(row.get("gold_start"))
            old_end = int(row.get("gold_end"))
            sample_id_int = int(sample_id)
        except (TypeError, ValueError):
            skipped.append(SkippedDecision(wb_row_id, split, sample_id, surface,
                                            decision, "invalid_or_missing_offset_fields"))
            continue

        if not decision or decision in NEVER_APPLY_DECISIONS:
            reason = "no_change_needed_or_empty" if not decision or decision == "KEEP_GOLD" \
                else f"decision_not_auto_applicable:{decision}"
            skipped.append(SkippedDecision(wb_row_id, split, sample_id_int, surface, decision, reason))
            continue

        if decision not in ALLOWED_APPLY_DECISIONS:
            skipped.append(SkippedDecision(wb_row_id, split, sample_id_int, surface, decision,
                                            f"unrecognized_decision_value:{decision}"))
            continue

        if decision != "CORRECT_LABEL":
            # CORRECT_BOUNDARY / SPLIT_ENTITY / MERGE_ENTITY / REMOVE_OUT_OF_SCHEMA_ENTITY
            # đều cần offset/spec mới TƯỜNG MINH -- workbook hiện tại không
            # có cột này (chỉ có ghi chú tự do), nên luôn skip có lý do rõ.
            skipped.append(SkippedDecision(wb_row_id, split, sample_id_int, surface, decision,
                                            f"missing_structured_offset_or_split_spec_for:{decision}"))
            continue

        role = row.get("semantic_role_candidate")
        new_label = SEMANTIC_ROLE_TO_LABEL.get(role)
        if new_label is None:
            skipped.append(SkippedDecision(wb_row_id, split, sample_id_int, surface, decision,
                                            f"semantic_role_candidate_not_mappable:{role!r}"))
            continue
        if new_label == old_label:
            skipped.append(SkippedDecision(wb_row_id, split, sample_id_int, surface, decision,
                                            "derived_label_equals_gold_no_change"))
            continue

        ents_here = entity_index.get((split, sample_id_int), [])
        live_match = next(
            (e for e in ents_here if e["start"] == old_start and e["end"] == old_end
             and e["label"] == old_label), None,
        )
        status, msg = "valid", ""
        if live_match is None:
            status = "error"
            msg = "source_span_or_label_mismatch_vs_live_data (dataset co the da doi tu luc export workbook)"
        elif live_match["surface"] != surface:
            status = "error"
            msg = f"surface_mismatch live={live_match['surface']!r} workbook={surface!r}"
        elif new_label not in ENTITY_TYPES:
            status = "error"
            msg = f"invalid_new_label:{new_label}"
        elif not check_no_overlap_after_edit(ents_here, old_start, old_end, old_start, old_end):
            status = "error"
            msg = "would_overlap_another_entity (khong nen xay ra vi span khong doi, chi doi nhan)"

        proposed.append(ProposedChange(
            change_id=change_id, sample_id=sample_id_int, split=split,
            document_id=row.get("document_id"), rule_id="derived-from-semantic-role-mapping",
            reviewer_decision=decision, old_span=(old_start, old_end), old_label=old_label,
            new_span=(old_start, old_end), new_label=new_label,
            original_text=row.get("full_sentence"), normalized_text_if_any=None,
            reviewer_notes="", source_workbook_row=wb_row_id,
            validation_status=status, validation_message=msg,
        ))
        change_id += 1

    return proposed, skipped


def process_boundary_split_fillin(df: pd.DataFrame, entity_index: dict,
                                   change_id_start: int = 0) -> tuple:
    """
    boundary_split_fill_in.xlsx (scripts/export_boundary_split_fillin.py):
    mỗi dòng có old span (gold_start/gold_end/gold_label) + final span mới
    (final_start/final_end/final_label) reviewer điền tường minh. row_type
    == "SPLIT_ENTITY" nhưng split2_start ĐỂ TRỐNG được hiểu là reviewer đã
    quyết định KHÔNG tách thật (gộp lại thành 1 span rộng hơn, 1 nhãn duy
    nhất) — đây là quyết định hợp lệ của reviewer, không phải lỗi thiếu dữ
    liệu, xử lý như CORRECT_BOUNDARY thông thường. Nếu split2_start CÓ giá
    trị, sinh 2 ProposedChange (part1 dùng final_*, part2 dùng split2_*),
    cùng rule_id để nhóm lại khi review changelog.
    """
    proposed, skipped = [], []
    change_id = change_id_start

    for i, row in df.iterrows():
        row_type = str(row.get("row_type") or "")
        wb_row_id = f"boundary_split_fillin:row{i}(entity_id={row.get('entity_id')})"
        split = row.get("split")
        surface = row.get("gold_surface")
        old_label = row.get("gold_label")

        try:
            sample_id = int(row.get("sample_id"))
            old_start = int(row.get("gold_start"))
            old_end = int(row.get("gold_end"))
        except (TypeError, ValueError):
            skipped.append(SkippedDecision(wb_row_id, split, row.get("sample_id"), surface,
                                            row_type, "invalid_or_missing_old_offset_fields"))
            continue

        final_label = row.get("final_label")
        final_start_raw = row.get("final_start")
        final_end_raw = row.get("final_end")
        if pd.isna(final_label) or pd.isna(final_start_raw) or pd.isna(final_end_raw):
            skipped.append(SkippedDecision(wb_row_id, split, sample_id, surface, row_type,
                                            "final_label_or_offset_still_missing"))
            continue
        try:
            final_start = int(final_start_raw)
            final_end = int(final_end_raw)
        except (TypeError, ValueError):
            skipped.append(SkippedDecision(wb_row_id, split, sample_id, surface, row_type,
                                            "final_offset_not_integer"))
            continue
        if final_label not in ENTITY_TYPES:
            skipped.append(SkippedDecision(wb_row_id, split, sample_id, surface, row_type,
                                            f"invalid_final_label:{final_label}"))
            continue
        if final_start > final_end or final_start < 0:
            skipped.append(SkippedDecision(wb_row_id, split, sample_id, surface, row_type,
                                            "invalid_offset_range"))
            continue

        ents_here = entity_index.get((split, sample_id), [])
        live_match = next(
            (e for e in ents_here if e["start"] == old_start and e["end"] == old_end
             and e["label"] == old_label), None,
        )
        notes = str(row.get("fill_notes") or row.get("existing_reviewer_notes") or "")

        if live_match is None:
            proposed.append(ProposedChange(
                change_id=change_id, sample_id=sample_id, split=split,
                document_id=row.get("document_id"), rule_id=f"boundary_split_fillin:{row_type}",
                reviewer_decision=row_type, old_span=(old_start, old_end), old_label=old_label,
                new_span=(final_start, final_end), new_label=final_label,
                original_text=row.get("full_sentence"), normalized_text_if_any=None,
                reviewer_notes=notes, source_workbook_row=wb_row_id,
                validation_status="error",
                validation_message="source_span_or_label_mismatch_vs_live_data",
            ))
            change_id += 1
            continue

        text_len = len(live_match["text"])
        status, msg = "valid", ""
        if live_match["surface"] != surface:
            status, msg = "error", f"surface_mismatch live={live_match['surface']!r} workbook={surface!r}"
        elif final_end >= text_len:
            status, msg = "error", f"final_end_out_of_bounds (text_len={text_len})"
        elif not check_no_overlap_after_edit(ents_here, final_start, final_end, old_start, old_end):
            status, msg = "error", "would_overlap_another_entity"

        split2_start_raw = row.get("split2_start")
        is_real_split = row_type == "SPLIT_ENTITY" and not pd.isna(split2_start_raw)

        if not is_real_split:
            proposed.append(ProposedChange(
                change_id=change_id, sample_id=sample_id, split=split,
                document_id=row.get("document_id"), rule_id=f"boundary_split_fillin:{row_type}",
                reviewer_decision=row_type, old_span=(old_start, old_end), old_label=old_label,
                new_span=(final_start, final_end), new_label=final_label,
                original_text=live_match["text"], normalized_text_if_any=None,
                reviewer_notes=notes, source_workbook_row=wb_row_id,
                validation_status=status, validation_message=msg,
            ))
            change_id += 1
        else:
            try:
                s2_start = int(row.get("split2_start"))
                s2_end = int(row.get("split2_end"))
                s2_label = row.get("split2_label")
            except (TypeError, ValueError):
                skipped.append(SkippedDecision(wb_row_id, split, sample_id, surface, row_type,
                                                "split2_offset_not_integer"))
                continue
            if s2_label not in ENTITY_TYPES or s2_start > s2_end or s2_start < 0 or s2_end >= text_len:
                skipped.append(SkippedDecision(wb_row_id, split, sample_id, surface, row_type,
                                                "invalid_split2_fields"))
                continue
            if spans_overlap(final_start, final_end, s2_start, s2_end):
                status, msg = "error", "split_parts_overlap_each_other"
            part1_status = status
            part2_status = status
            if status == "valid" and not check_no_overlap_after_edit(
                ents_here, s2_start, s2_end, old_start, old_end,
            ):
                part2_status, msg = "error", "split_part2_would_overlap_another_entity"

            proposed.append(ProposedChange(
                change_id=change_id, sample_id=sample_id, split=split,
                document_id=row.get("document_id"), rule_id=f"boundary_split_fillin:SPLIT_ENTITY:part1",
                reviewer_decision=row_type, old_span=(old_start, old_end), old_label=old_label,
                new_span=(final_start, final_end), new_label=final_label,
                original_text=live_match["text"], normalized_text_if_any=None,
                reviewer_notes=notes, source_workbook_row=wb_row_id,
                validation_status=part1_status, validation_message=msg if part1_status != "valid" else "",
            ))
            change_id += 1
            proposed.append(ProposedChange(
                change_id=change_id, sample_id=sample_id, split=split,
                document_id=row.get("document_id"), rule_id=f"boundary_split_fillin:SPLIT_ENTITY:part2",
                reviewer_decision=row_type, old_span=(old_start, old_end), old_label=old_label,
                new_span=(s2_start, s2_end), new_label=s2_label,
                original_text=live_match["text"], normalized_text_if_any=None,
                reviewer_notes=notes, source_workbook_row=wb_row_id,
                validation_status=part2_status, validation_message=msg if part2_status != "valid" else "",
            ))
            change_id += 1

    return proposed, skipped


SELECTED_ACTION_OPTIONS = [
    "MERGE_ENTITY", "KEEP_SEPARATE", "CORRECT_BOUNDARY", "REMOVE_ENTITY",
    "GUIDELINE_AMBIGUITY", "NEEDS_DOMAIN_EXPERT",
]
# CHỈ MERGE_ENTITY được implement để tự động validate+apply ở vòng này
# (theo đúng yêu cầu mục C — "Thêm support MERGE_ENTITY"). Các action khác
# (KEEP_SEPARATE/CORRECT_BOUNDARY/REMOVE_ENTITY/GUIDELINE_AMBIGUITY/
# NEEDS_DOMAIN_EXPERT) chỉ được LOG Ý ĐỊNH, không tự apply gì — cần bước
# implement riêng sau nếu cần.
IMPLEMENTED_SELECTED_ACTIONS = {"MERGE_ENTITY"}


@dataclass
class ProposedMerge:
    change_id: int
    sample_id: int
    split: str
    document_id: Optional[str]
    rule_id: str
    source_entity_ids: list
    source_spans: list  # [(start, end, label, surface), ...]
    resulting_span: tuple
    resulting_label: str
    reviewer_notes: str
    source_workbook_row: str
    validation_status: str
    validation_message: str = ""

    def to_row(self) -> dict:
        return dict(
            change_id=self.change_id, sample_id=self.sample_id, split=self.split,
            document_id=self.document_id, rule_id=self.rule_id,
            source_entity_ids=list(self.source_entity_ids),
            source_spans=[list(s) for s in self.source_spans],
            resulting_span=list(self.resulting_span), resulting_label=self.resulting_label,
            reviewer_notes=self.reviewer_notes, source_workbook_row=self.source_workbook_row,
            validation_status=self.validation_status, validation_message=self.validation_message,
        )


def build_entity_by_id(entities: list) -> dict:
    return {e["entity_id"]: e for e in entities}


def validate_and_build_merge(
    row_id: str, sample_id, split, document_id, merged_entity_ids: list,
    entity_by_id: dict, entity_index: dict, resulting_start, resulting_end,
    resulting_label: str, expected_resulting_surface: Optional[str],
    guideline_rule_id: Optional[str], reviewer_notes: Optional[str],
    change_id: int,
) -> "tuple[Optional[ProposedMerge], Optional[SkippedDecision]]":
    """
    Validate đầy đủ theo mục C. Trả (ProposedMerge, None) nếu tạo được
    change (dù valid hay error), hoặc (None, SkippedDecision) nếu thiếu
    field bắt buộc đến mức không tạo được change record nào (không rõ
    entity nguồn -> không có gì để log làm change).
    """
    if not merged_entity_ids or len(merged_entity_ids) < 2:
        return None, SkippedDecision(row_id, split, sample_id, None, "MERGE_ENTITY",
                                      "merged_entity_ids_missing_or_less_than_2")
    if not guideline_rule_id or (isinstance(guideline_rule_id, float)):
        return None, SkippedDecision(row_id, split, sample_id, None, "MERGE_ENTITY",
                                      "missing_guideline_rule_id")
    if not reviewer_notes or not str(reviewer_notes).strip():
        return None, SkippedDecision(row_id, split, sample_id, None, "MERGE_ENTITY",
                                      "missing_reviewer_notes")

    sources = []
    for eid in merged_entity_ids:
        e = entity_by_id.get(eid)
        if e is None:
            return None, SkippedDecision(row_id, split, sample_id, None, "MERGE_ENTITY",
                                          f"source_entity_id_not_found:{eid}")
        sources.append(e)

    error_msg = None
    if len({(e["split"], e["sample_id"]) for e in sources}) != 1:
        error_msg = "source_entities_not_all_same_sample"
    elif sources[0]["split"] != split or sources[0]["sample_id"] != sample_id:
        error_msg = "source_entities_do_not_match_declared_split_sample_id"

    # Khoảng cách giữa các entity nguồn: KHÔNG bắt buộc liền kề tuyệt đối --
    # 1 case thật (黎輔陳 = Lê Phụ Trần) có khoảng cách 1 ký tự (輔) giữa 2
    # entity 1-chữ vì tên bị cắt làm đôi lúc annotate; khoảng trống đó
    # CHÍNH LÀ bằng chứng cần merge, không phải lý do từ chối. An toàn thật
    # sự nằm ở 3 chỗ khác: (1) resulting_span phải khớp CHÍNH XÁC min/max
    # nguồn (không tự ý nuốt thêm ký tự ngoài phạm vi khai báo), (2)
    # expected_resulting_surface phải khớp đúng text thật tại offset đó,
    # (3) resulting_span không được đè lên entity thứ 3 nào khác (bao gồm
    # cả entity nằm lọt trong khoảng trống). Chỉ chặn cứng khi khoảng cách
    # quá lớn (>10 ký tự) — dấu hiệu nhập sai entity_id nhiều khả năng hơn
    # là 1 tên bị cắt.
    sources_sorted = sorted(sources, key=lambda e: e["start"])
    if error_msg is None:
        for a, b in zip(sources_sorted, sources_sorted[1:]):
            gap = b["start"] - a["end"] - 1
            if gap > 10 and not spans_overlap(a["start"], a["end"], b["start"], b["end"]):
                error_msg = f"source_entities_gap_too_large_suspicious ({gap} ky tu)"
                break

    if resulting_label not in ENTITY_TYPES:
        error_msg = error_msg or f"invalid_resulting_label:{resulting_label}"

    min_start = min(e["start"] for e in sources)
    max_end = max(e["end"] for e in sources)
    if error_msg is None and (resulting_start != min_start or resulting_end != max_end):
        error_msg = (
            f"resulting_span_does_not_exactly_cover_sources "
            f"(expected {(min_start, max_end)}, got {(resulting_start, resulting_end)})"
        )

    if error_msg is None:
        text = sources[0]["text"]
        actual_surface = text[resulting_start:resulting_end + 1]
        if expected_resulting_surface and actual_surface != expected_resulting_surface:
            error_msg = (
                f"expected_resulting_surface_mismatch "
                f"(text={actual_surface!r} vs expected={expected_resulting_surface!r})"
            )

    if error_msg is None:
        ents_here = entity_index.get((split, sample_id), [])
        source_ids = {e["entity_id"] for e in sources}
        others = [e for e in ents_here if e["entity_id"] not in source_ids]
        for o in others:
            if spans_overlap(resulting_start, resulting_end, o["start"], o["end"]):
                error_msg = f"merged_span_would_overlap_other_entity:{o['surface']}/{o['label']}"
                break

    status = "error" if error_msg else "valid"
    change = ProposedMerge(
        change_id=change_id, sample_id=sample_id, split=split, document_id=document_id,
        rule_id=str(guideline_rule_id), source_entity_ids=list(merged_entity_ids),
        source_spans=[(e["start"], e["end"], e["label"], e["surface"]) for e in sources],
        resulting_span=(resulting_start, resulting_end), resulting_label=resulting_label,
        reviewer_notes=str(reviewer_notes), source_workbook_row=row_id,
        validation_status=status, validation_message=error_msg or "",
    )
    return change, None


def process_collision_candidates(df: pd.DataFrame, entities: list, entity_index: dict,
                                  change_id_start: int = 0) -> tuple:
    """
    collision_merge_candidates.xlsx: mỗi dòng là 1 collision case đã được
    PHÂN LOẠI RIÊNG (overlap_type khác nhau, KHÔNG coi mọi overlap là
    merge candidate). CHỈ selected_action == "MERGE_ENTITY" được thực sự
    validate+build change; mọi action khác (kể cả đã điền) chỉ LOG Ý ĐỊNH,
    không tự apply — cần implement riêng nếu muốn tự động các nhánh đó.
    Dòng để trống selected_action -> skip, lý do "pending_review".
    """
    entity_by_id = build_entity_by_id(entities)
    proposed, skipped = [], []
    change_id = change_id_start

    for i, row in df.iterrows():
        row_id = f"collision_merge_candidates:row{i}(collision_id={row.get('collision_id')})"
        split = row.get("split")
        sample_id = row.get("sample_id")
        action = row.get("selected_action")
        action = str(action).strip() if not pd.isna(action) else ""

        if not action:
            skipped.append(SkippedDecision(row_id, split, sample_id, None, "", "pending_review"))
            continue
        if action not in SELECTED_ACTION_OPTIONS:
            skipped.append(SkippedDecision(row_id, split, sample_id, None, action,
                                            f"unrecognized_selected_action:{action}"))
            continue
        if action not in IMPLEMENTED_SELECTED_ACTIONS:
            skipped.append(SkippedDecision(row_id, split, sample_id, None, action,
                                            f"action_not_yet_implemented_for_auto_apply:{action}"))
            continue

        # action == MERGE_ENTITY
        raw_ids = row.get("merged_entity_ids")
        merged_entity_ids = []
        if not pd.isna(raw_ids):
            merged_entity_ids_raw = [s.strip() for s in str(raw_ids).split(";") if s.strip()]
            # entity_id thật trong entities list la int (xem
            # extract_all_spans); Excel luon doc merged_entity_ids ve dang
            # text "1567;1568" -- ep kieu ve int de khop dung, giu nguyen
            # string goc neu khong phai so (se bi bao loi "not_found" ro
            # rang thay vi so sanh sai kieu am tham).
            merged_entity_ids = []
            for s in merged_entity_ids_raw:
                try:
                    merged_entity_ids.append(int(s))
                except ValueError:
                    merged_entity_ids.append(s)

        def _to_int_or_none(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        resulting_start = _to_int_or_none(row.get("resulting_start"))
        resulting_end = _to_int_or_none(row.get("resulting_end"))
        resulting_label = row.get("resulting_label")
        resulting_label = str(resulting_label).strip() if not pd.isna(resulting_label) else None

        if resulting_start is None or resulting_end is None or not resulting_label:
            skipped.append(SkippedDecision(row_id, split, sample_id, None, action,
                                            "missing_resulting_span_or_label"))
            continue

        change, skip = validate_and_build_merge(
            row_id=row_id, sample_id=int(sample_id), split=split, document_id=row.get("document_id"),
            merged_entity_ids=merged_entity_ids, entity_by_id=entity_by_id, entity_index=entity_index,
            resulting_start=resulting_start, resulting_end=resulting_end,
            resulting_label=resulting_label,
            expected_resulting_surface=row.get("expected_resulting_surface"),
            guideline_rule_id=row.get("guideline_rule_id"), reviewer_notes=row.get("reviewer_notes"),
            change_id=change_id,
        )
        if skip is not None:
            skipped.append(skip)
            continue
        proposed.append(change)
        change_id += 1

    return proposed, skipped


def validate_dataset_checksums(prior_manifest: dict, current_checksums: dict) -> list:
    """So sánh checksum dataset hiện tại vs. checksum lúc audit_v1 được sinh
    (manifest.json gốc). Mismatch -> global validation error (không phải
    per-row) vì toàn bộ workbook có thể đã review trên data KHÁC bản hiện
    tại."""
    errors = []
    prior = prior_manifest.get("dataset_checksums", {}) if prior_manifest else {}
    for split, cur_sum in current_checksums.items():
        prior_sum = prior.get(split)
        if prior_sum and prior_sum != cur_sum:
            errors.append(
                f"Checksum mismatch cho split '{split}': audit_v1 manifest={prior_sum}, "
                f"hiện tại={cur_sum}. Dataset co the da thay doi sau khi workbook duoc "
                f"review -- KHONG an toan de apply cho den khi xac minh lai."
            )
    return errors
