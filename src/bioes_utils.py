"""
bioes_utils.py — BIO <-> BIOES conversion, span decoding, boundary-label
derivation cho multi-task NER + CRF + start/end boundary heads.

Toàn bộ hàm ở đây thuần Python (không phụ thuộc torch) để dễ unit test và
tái sử dụng ở cả bước tiền xử lý dữ liệu lẫn bước decode/evaluate.

── LỰA CHỌN KIẾN TRÚC MASK (mục B trong yêu cầu) ──────────────────────────
Codebase hiện tại (src/models.py:crf_step) đã áp dụng PHƯƠNG ÁN 1: ép
mask[:, 0] = True (vị trí [CLS]) để thỏa mãn ràng buộc của thư viện
`torchcrf` (yêu cầu timestep đầu tiên luôn valid, không hỗ trợ mask có "lỗ"
ở giữa hai vị trí True). Các vị trí [CLS]/[SEP]/padding/subword-tiếp-theo
vẫn có label=-100 ở tầng dataset; khi build crf_labels cho CRF, các vị trí
-100 được thay bằng id của nhãn "O" (KHÔNG phải id 0 tùy tiện — xem
`o_label_id` trong models.py) để không nhiễu training. Khi decode/evaluate,
các vị trí có label gốc = -100 bị loại bỏ hoàn toàn khỏi span/metric (xem
evaluator.py). Module này giữ nguyên phương án đó cho nhất quán với pipeline
sẵn có; không đổi sang "pack liên tục" (phương án 2) vì phải viết lại toàn
bộ tầng attention/positional embedding — không cần thiết cho yêu cầu này.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from src.utils import get_logger

logger = get_logger(__name__)

ENTITY_TYPES = ["PER", "LOC", "ORG", "TITLE", "DTM"]

BIOES_LABELS = ["O"] + [
    f"{prefix}-{etype}"
    for etype in ENTITY_TYPES
    for prefix in ("B", "I", "E", "S")
]
# NOTE: thứ tự trên là per-type (B/I/E/S liền nhau cho từng loại), không phải
# per-prefix. Đây KHÔNG phải yêu cầu bắt buộc của đề bài (đề chỉ liệt kê đủ
# 21 nhãn), nhưng cố định thứ tự này (thay vì sorted() alphabetically như
# build_label_map cũ) để label2id/id2label ổn định, dễ đọc giữa các lần
# chạy — quan trọng cho reproducibility khi so sánh M0/M1/M2.


def build_bioes_label_map(entity_types: Optional[list] = None) -> tuple:
    """Label map cố định cho scheme BIOES (không tự suy ra từ data)."""
    types = entity_types or ENTITY_TYPES
    labels = ["O"] + [f"{p}-{t}" for t in types for p in ("B", "I", "E", "S")]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


# ── A. BIO VALIDATION + BIO -> BIOES ────────────────────────────────────────
class BIOFormatError(ValueError):
    pass


def find_bio_violations(labels: list, sample_id=None) -> list:
    """
    Kiểm tra 1 chuỗi nhãn BIO. Trả về list vi phạm dạng:
        {"position": i, "tag": label, "prev_type": ..., "reason": ...}
    KHÔNG raise — chỉ báo cáo. Dùng để validate trước khi convert (mục A).
    """
    violations = []
    prev_type = None
    for i, lab in enumerate(labels):
        if lab == "O":
            prev_type = None
            continue
        if "-" not in lab or lab.split("-", 1)[0] not in ("B", "I"):
            violations.append({
                "position": i, "tag": lab, "prev_type": prev_type,
                "reason": f"Unrecognized BIO tag '{lab}' (sample_id={sample_id})",
            })
            prev_type = None
            continue
        prefix, etype = lab.split("-", 1)
        if prefix == "I" and prev_type != etype:
            violations.append({
                "position": i, "tag": lab, "prev_type": prev_type,
                "reason": (
                    f"I-{etype} tại vị trí {i} không có B-{etype}/I-{etype} "
                    f"liền trước (prev_type={prev_type}, sample_id={sample_id})"
                ),
            })
        prev_type = etype
    return violations


def convert_bio_to_bioes(labels: list, sample_id=None, mode: str = "strict") -> tuple:
    """
    Chuyển 1 chuỗi nhãn BIO -> BIOES.

    mode:
      - "strict": raise BIOFormatError (kèm sample_id + vị trí) nếu gặp
        I-X không có B-X/I-X cùng loại liền trước.
      - "repair": coi I-X mồ côi đó là B-X rồi convert tiếp (không raise).

    Returns: (bioes_labels, num_repaired)
    """
    if mode not in ("strict", "repair"):
        raise ValueError(f"Unknown mode: {mode!r}, expect 'strict' or 'repair'")

    violations = find_bio_violations(labels, sample_id=sample_id)
    if violations and mode == "strict":
        v = violations[0]
        raise BIOFormatError(
            f"Invalid BIO sequence (sample_id={sample_id}, pos={v['position']}, "
            f"tag={v['tag']}): {v['reason']}. {len(violations)} violation(s) total."
        )

    # Bước 1: repair BIO (coi I-X mồ côi thành B-X) để mọi entity-run đều
    # bắt đầu bằng B- trước khi convert sang BIOES.
    fixed = []
    prev_type = None
    num_repaired = 0
    for lab in labels:
        if lab == "O":
            fixed.append("O")
            prev_type = None
            continue
        prefix, etype = lab.split("-", 1)
        if prefix == "B":
            fixed.append(lab)
        elif prefix == "I":
            if prev_type == etype:
                fixed.append(lab)
            else:
                fixed.append(f"B-{etype}")
                num_repaired += 1
        else:
            raise BIOFormatError(f"Unrecognized BIO tag '{lab}' (sample_id={sample_id})")
        prev_type = etype

    # Bước 2: quét theo entity-run liên tục cùng loại -> BIOES
    n = len(fixed)
    bioes = []
    i = 0
    while i < n:
        lab = fixed[i]
        if lab == "O":
            bioes.append("O")
            i += 1
            continue
        etype = lab.split("-", 1)[1]
        j = i + 1
        while j < n and fixed[j] == f"I-{etype}":
            j += 1
        run_len = j - i
        if run_len == 1:
            bioes.append(f"S-{etype}")
        else:
            bioes.append(f"B-{etype}")
            bioes.extend([f"I-{etype}"] * (run_len - 2))
            bioes.append(f"E-{etype}")
        i = j

    return bioes, num_repaired


def convert_dataset_bio_to_bioes(data: list, mode: str = "strict") -> tuple:
    """
    data: list[(tokens, bio_labels)] — dùng sample index làm sample_id.
    Returns: (new_data with bioes labels, stats dict)
    """
    out = []
    total_repaired = 0
    n_sequences_repaired = 0
    for sample_id, (tokens, labels) in enumerate(data):
        bioes, n_rep = convert_bio_to_bioes(labels, sample_id=sample_id, mode=mode)
        out.append((tokens, bioes))
        if n_rep:
            total_repaired += n_rep
            n_sequences_repaired += 1

    stats = {
        "mode": mode,
        "num_sequences": len(data),
        "num_sequences_repaired": n_sequences_repaired,
        "num_tags_repaired": total_repaired,
    }
    if mode == "repair" and total_repaired:
        logger.warning(
            f"[BIO->BIOES repair] {total_repaired} tag(s) sửa (coi I-X mồ côi "
            f"thành B-X) trên {n_sequences_repaired}/{len(data)} câu."
        )
    else:
        logger.info(
            f"[BIO->BIOES] {len(data)} câu convert xong (mode={mode}, "
            f"repaired={total_repaired})."
        )
    return out, stats


def derive_boundary_labels_from_bio(bio_labels: list) -> tuple:
    """
    Suy start/end nhị phân TRỰC TIẾP từ nhãn BIO (dùng cho M3: BIO + boundary
    heads, KHÔNG cần convert sang BIOES — tránh confound giữa "đổi label
    scheme" và "thêm boundary heads" khi so sánh với M0/M1/M2).
      start[i] = 1 nếu tag là B-X.
      end[i]   = 1 nếu tag != O VÀ (i là token cuối HOẶC token kế tiếp không
                 phải I-X CÙNG loại X) -- tức là entity "đóng" ở vị trí i,
                 bất kể đó là B-X (entity 1 token) hay I-X (entity nhiều
                 token, i là I cuối cùng của chuỗi).
    """
    n = len(bio_labels)
    start = [1 if t.startswith("B-") else 0 for t in bio_labels]
    end = [0] * n
    for i, tag in enumerate(bio_labels):
        if tag == "O":
            continue
        etype = tag.split("-", 1)[1]
        is_last = (i == n - 1) or (bio_labels[i + 1] != f"I-{etype}")
        end[i] = 1 if is_last else 0
    return start, end


# ── B. BOUNDARY LABEL DERIVATION ────────────────────────────────────────────
def derive_boundary_labels(bioes_labels: list) -> tuple:
    """
    start[i] = 1 nếu tag là B-* hoặc S-*, ngược lại 0.
    end[i]   = 1 nếu tag là E-* hoặc S-*, ngược lại 0.
    Phải gọi TRƯỚC khi pad/align vào subword — xem NERDataset trong
    data_utils.py (boundary label được suy từ nhãn word-level gốc, sau đó
    align theo word_ids() giống hệt ner label, KHÔNG suy lại từ nhãn đã pad).
    """
    start = [1 if (t.startswith("B-") or t.startswith("S-")) else 0 for t in bioes_labels]
    end = [1 if (t.startswith("E-") or t.startswith("S-")) else 0 for t in bioes_labels]
    return start, end


def compute_boundary_pos_weight(
    dataset: list, max_pos_weight: float = 10.0, scheme: str = "bioes",
) -> dict:
    """
    Tính pos_weight cho BCEWithLogitsLoss (boundary_loss_type=weighted_bce)
    TỪ TRAIN SPLIT (dataset: list[(tokens, labels)], labels ở đúng `scheme`
    — "bioes" hoặc "bio", xem derive_boundary_labels/
    derive_boundary_labels_from_bio). Clip theo max_pos_weight. Log số
    positive/negative/pos_weight cho start và end. KHÔNG được gọi với
    dev/test data (caller chịu trách nhiệm chỉ truyền train).
    """
    if scheme not in ("bio", "bioes"):
        raise ValueError(f"Unknown scheme: {scheme!r}")
    derive_fn = derive_boundary_labels_from_bio if scheme == "bio" else derive_boundary_labels

    n_pos_start = n_pos_end = n_valid = 0
    for _, labels in dataset:
        start, end = derive_fn(labels)
        n_valid += len(labels)
        n_pos_start += sum(start)
        n_pos_end += sum(end)

    n_neg_start = n_valid - n_pos_start
    n_neg_end = n_valid - n_pos_end

    def _pw(n_neg, n_pos):
        if n_pos == 0:
            return max_pos_weight
        return min(n_neg / n_pos, max_pos_weight)

    start_pw = _pw(n_neg_start, n_pos_start)
    end_pw = _pw(n_neg_end, n_pos_end)

    logger.info(
        f"[boundary pos_weight] start: pos={n_pos_start} neg={n_neg_start} "
        f"pos_weight={start_pw:.3f} | end: pos={n_pos_end} neg={n_neg_end} "
        f"pos_weight={end_pw:.3f} (clip max={max_pos_weight})"
    )
    return {
        "start_pos_weight": start_pw,
        "end_pos_weight": end_pw,
        "start_num_positive": n_pos_start,
        "start_num_negative": n_neg_start,
        "end_num_positive": n_pos_end,
        "end_num_negative": n_neg_end,
    }


# ── SPAN DECODING (BIOES -> entity spans) ───────────────────────────────────
def repair_bioes_sequence(tags: list) -> tuple:
    """
    torchcrf KHÔNG enforce hard transition constraints (không có cơ chế cấm
    ví dụ 'O -> I-PER' hay 'B-PER -> E-LOC' trong quá trình decode/viterbi
    của thư viện này). Vì vậy chuỗi BIOES decode ra từ CRF có thể invalid.
    Hàm này validate + sửa (lenient repair) TRƯỚC khi convert sang spans:
      - 'I-X'/'E-X' xuất hiện mà không có entity cùng loại đang mở -> coi là
        'B-X' mới (mở entity mới, tự sửa).
      - Entity đang mở gặp 'O', 'S-*', hoặc 'B-*' khác/entity mới mà chưa có
        E- đóng lại -> tự đóng bằng cách coi token liền trước là E-X.
      - Cuối chuỗi vẫn còn entity mở -> đóng tại token cuối.
    Trả về (repaired_tags, num_repairs). Dùng ở evaluator trước khi tính spans
    và log số lần repair (không im lặng sửa).
    """
    n = len(tags)
    out = list(tags)
    num_repairs = 0
    open_type = None
    open_start = None

    def _close_at(idx):
        nonlocal num_repairs
        if open_type is None:
            return
        if idx == open_start:
            out[idx] = f"S-{open_type}"
        else:
            out[idx] = f"E-{open_type}"
        num_repairs += 1

    for i, tag in enumerate(tags):
        if tag == "O":
            if open_type is not None:
                _close_at(i - 1)
                open_type = None
            out[i] = "O"
            continue

        prefix, etype = tag.split("-", 1) if "-" in tag else (None, None)
        if prefix not in ("B", "I", "E", "S"):
            raise BIOFormatError(f"Unrecognized BIOES tag '{tag}' at position {i}")

        if prefix == "S":
            if open_type is not None:
                _close_at(i - 1)
                open_type = None
            out[i] = tag
            continue

        if prefix == "B":
            if open_type is not None:
                _close_at(i - 1)
            open_type, open_start = etype, i
            out[i] = tag
            continue

        # prefix in ("I", "E")
        if open_type == etype:
            out[i] = tag
            if prefix == "E":
                open_type = None
        else:
            # Mồ côi: không có B-etype đang mở (hoặc loại khác đang mở)
            if open_type is not None:
                _close_at(i - 1)
            open_type, open_start = etype, i
            out[i] = f"B-{etype}"
            num_repairs += 1

    if open_type is not None:
        _close_at(n - 1)

    return out, num_repairs


def bioes_to_spans(tags: list) -> list:
    """
    Chuyển chuỗi BIOES (PHẢI đã hợp lệ — gọi repair_bioes_sequence() trước
    nếu tags đến từ CRF decode) thành list[(start, end_inclusive, type)].
    Raise nếu gặp cấu trúc invalid (an toàn hơn là âm thầm bỏ qua).
    """
    n = len(tags)
    spans = []
    i = 0
    while i < n:
        tag = tags[i]
        if tag == "O":
            i += 1
            continue
        prefix, etype = tag.split("-", 1)
        if prefix == "S":
            spans.append((i, i, etype))
            i += 1
        elif prefix == "B":
            j = i
            while j < n and tags[j] != f"E-{etype}":
                if j > i and not tags[j].startswith(f"I-{etype}"):
                    raise BIOFormatError(
                        f"Invalid BIOES sequence at position {j}: expected "
                        f"I-{etype}/E-{etype} while entity {etype} open from {i}, "
                        f"got '{tags[j]}'. Call repair_bioes_sequence() first."
                    )
                j += 1
            if j >= n:
                raise BIOFormatError(
                    f"Entity {etype} opened at {i} never closed with E-{etype}. "
                    f"Call repair_bioes_sequence() first."
                )
            spans.append((i, j, etype))
            i = j + 1
        else:
            raise BIOFormatError(
                f"Unexpected tag '{tag}' at position {i} (dangling I-/E- "
                f"without B-). Call repair_bioes_sequence() first."
            )
    return spans


def spans_to_bioes(spans: list, length: int) -> list:
    """Nghịch đảo của bioes_to_spans — hữu ích cho unit test round-trip."""
    tags = ["O"] * length
    for start, end, etype in spans:
        if start == end:
            tags[start] = f"S-{etype}"
        else:
            tags[start] = f"B-{etype}"
            for k in range(start + 1, end):
                tags[k] = f"I-{etype}"
            tags[end] = f"E-{etype}"
    return tags


@dataclass
class BoundaryErrorCounts:
    exact_match: int = 0
    correct_type_wrong_boundary: int = 0
    predicted_is_prefix: int = 0
    predicted_is_suffix: int = 0
    predicted_longer: int = 0
    wrong_type_exact_boundary: int = 0
    missed_entity: int = 0
    spurious_entity: int = 0


# ── HARD-CONSTRAINED CRF TRANSITIONS (BIOES state machine) ─────────────────
# Bối cảnh: torchcrf.CRF KHÔNG enforce hard transition constraint -- toàn bộ
# 21x21 cặp nhãn đều được phép về mặt thuật toán Viterbi, chỉ bị "khuyên can"
# gián tiếp qua giá trị học được của transitions[i,j]. Với train nhỏ (1,277
# câu) + 21 nhãn BIOES, nhiều cặp bất hợp lệ (O->I-X, B-X->O, B-X->E-Y khác
# loại, ...) không đủ dữ liệu phản ví dụ để model tự học tránh -> observed
# 402/2518 lần CRF decode ra chuỗi invalid trên test set M2 (repair() phải
# sửa). Cách sửa đúng: bake hard constraint trực tiếp vào transitions/
# start_transitions/end_transitions của torchcrf bằng cách gán 1 penalty rất
# âm cho MỌI cặp bất hợp lệ, làm lại sau MỖI optimizer.step() (vì AdamW vẫn
# có thể nhích các phần tử đó lên 1 chút do gradient len qua đường đi khác
# trong partition function) -- kỹ thuật "projected/masked constrained CRF"
# tiêu chuẩn khi không sửa lại thuật toán Viterbi của thư viện.
def _prefix_type(tag: str):
    if tag == "O":
        return "O", None
    prefix, etype = tag.split("-", 1)
    return prefix, etype


def bioes_legal_transition(prev_tag: str, next_tag: str) -> bool:
    """O/E-X/S-X (kết thúc 1 "cụm") chỉ được theo sau bởi O/B-Y/S-Y (bắt đầu
    cụm mới hoặc không có gì). B-X/I-X (đang mở cụm loại X) chỉ được theo
    sau bởi I-X/E-X (tiếp tục ĐÚNG loại X)."""
    p_prefix, p_type = _prefix_type(prev_tag)
    n_prefix, n_type = _prefix_type(next_tag)
    if p_prefix in ("O", "E", "S"):
        return n_prefix in ("O", "B", "S")
    if p_prefix in ("B", "I"):
        return n_prefix in ("I", "E") and n_type == p_type
    raise ValueError(f"Unrecognized tag prefix in '{prev_tag}'")


def bioes_legal_start(tag: str) -> bool:
    prefix, _ = _prefix_type(tag)
    return prefix in ("O", "B", "S")


def bioes_legal_end(tag: str) -> bool:
    prefix, _ = _prefix_type(tag)
    return prefix in ("O", "E", "S")


def bio_legal_transition(prev_tag: str, next_tag: str) -> bool:
    """State machine cho BIO (KHÔNG có E-/S-, chỉ 2 prefix B/I + O) — dùng
    cho M3 (BIO + boundary heads). Khác BIOES ở chỗ B-X/I-X không BẮT BUỘC
    phải đóng bằng E-X: được phép "buông" sang O hoặc B-Y bất kỳ lúc nào
    (entity 1-token hoặc n-token đều hợp lệ mà không cần marker kết thúc
    riêng). Chỉ 1 ràng buộc thật: I-X chỉ được theo sau B-X/I-X CÙNG loại X
    (đây chính là điều kiện `find_bio_violations()`/`convert_bio_to_bioes()`
    đã dùng để phát hiện "orphan I")."""
    p_prefix, p_type = _prefix_type(prev_tag)
    n_prefix, n_type = _prefix_type(next_tag)
    if n_prefix == "I":
        return p_prefix in ("B", "I") and p_type == n_type
    return n_prefix in ("O", "B")


def bio_legal_start(tag: str) -> bool:
    prefix, _ = _prefix_type(tag)
    return prefix in ("O", "B")  # I-* không được mở đầu chuỗi (orphan I)


def bio_legal_end(tag: str) -> bool:
    return True  # BIO không có marker đóng riêng -- mọi tag đều hợp lệ ở cuối


_SCHEME_RULES = {
    "bioes": (bioes_legal_transition, bioes_legal_start, bioes_legal_end),
    "bio": (bio_legal_transition, bio_legal_start, bio_legal_end),
}


def build_transition_masks(label2id: dict, scheme: str = "bioes") -> tuple:
    """
    Trả về (illegal_transition, illegal_start, illegal_end) — BoolTensor,
    True tại vị trí BỊ CẤM, theo state machine của `scheme` ("bio" hoặc
    "bioes"). Dùng với apply_hard_transition_constraints() để bake vào
    torchcrf.CRF. QUAN TRỌNG: dùng SAI scheme cho label space (vd áp luật
    BIOES lên 1 label map chỉ có B/I/O) sẽ ép sai — vd cấm nhầm B-X->O hợp
    lệ trong BIO — nên luôn truyền đúng scheme khớp với label2id thực tế.
    """
    if scheme not in _SCHEME_RULES:
        raise ValueError(f"Unknown scheme: {scheme!r}, expect 'bio' or 'bioes'")
    legal_transition, legal_start, legal_end = _SCHEME_RULES[scheme]

    n = len(label2id)
    id2label = {v: k for k, v in label2id.items()}
    illegal_transition = torch.ones(n, n, dtype=torch.bool)
    illegal_start = torch.ones(n, dtype=torch.bool)
    illegal_end = torch.ones(n, dtype=torch.bool)

    for i in range(n):
        ti = id2label[i]
        if legal_start(ti):
            illegal_start[i] = False
        if legal_end(ti):
            illegal_end[i] = False
        for j in range(n):
            tj = id2label[j]
            if legal_transition(ti, tj):
                illegal_transition[i, j] = False

    return illegal_transition, illegal_start, illegal_end


def build_bioes_transition_masks(label2id: dict) -> tuple:
    """Backward-compat wrapper (dùng ở models.py mặc định) — tương đương
    build_transition_masks(label2id, scheme="bioes")."""
    return build_transition_masks(label2id, scheme="bioes")


def apply_hard_transition_constraints(
    crf, illegal_transition: torch.Tensor, illegal_start: torch.Tensor,
    illegal_end: torch.Tensor, penalty: float = float("-inf"),
) -> None:
    """
    Ghi đè in-place transitions/start_transitions/end_transitions của
    torchcrf.CRF: mọi cặp/vị trí bị cấm -> `penalty`. PHẢI gọi lại hàm này
    sau MỖI optimizer.step() (xem src/trainer_boundary.py) để giữ ràng buộc
    trong suốt quá trình train, không chỉ lúc khởi tạo.

    **Vì sao mặc định `-inf` chứ không phải 1 số âm hữu hạn (vd -100000)**:
    một hằng số âm hữu hạn KHÔNG phải hard constraint thật — với emission
    score đủ lớn (kể cả không thực tế, ví dụ do lỗi khởi tạo hoặc
    adversarial input), tổng điểm của đường đi bất hợp lệ vẫn có thể vượt
    đường đi hợp lệ tốt nhất (unit test
    `test_viterbi_never_starts_with_I_or_E_even_with_adversarial_emissions`
    phát hiện chính xác lỗ hổng này với penalty=-100000 và emission=1e6).
    `-inf` loại bỏ hoàn toàn lỗ hổng: `torch.max` (dùng trong Viterbi decode
    của torchcrf) luôn bỏ qua nhánh `-inf` trừ khi MỌI nhánh đều `-inf`
    (không xảy ra vì luôn có ít nhất 1 đường hợp lệ). Không gây NaN ở decode
    (chỉ dùng max/argmax). Ở training (NLL loss dùng logsumexp qua
    `_compute_normalizer`), `exp(-inf) = 0` nên các đường bất hợp lệ đóng
    góp đúng 0 vào partition function — an toàn cả dưới fp16 autocast vì
    `-inf` là giá trị hợp lệ của IEEE754 (không bị tràn số như 1 hằng số âm
    hữu hạn có |giá trị| > 65504, giới hạn fp16).
    """
    with torch.no_grad():
        crf.transitions.masked_fill_(illegal_transition.to(crf.transitions.device), penalty)
        crf.start_transitions.masked_fill_(illegal_start.to(crf.start_transitions.device), penalty)
        crf.end_transitions.masked_fill_(illegal_end.to(crf.end_transitions.device), penalty)
