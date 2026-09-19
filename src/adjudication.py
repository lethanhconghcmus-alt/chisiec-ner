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
