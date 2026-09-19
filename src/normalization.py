"""
normalization.py — PUA (Private Use Area) text normalization với alignment
tường minh raw<->normalized offset. KHÔNG sửa dataset gốc; chỉ cung cấp hàm
để audit tool / apply_adjudication dùng khi cần remap offset.

Bối cảnh (guideline_v2 GR-02): 2 codepoint PUA đã VERIFY thủ công đều là dị
thể của 臺 (đài) xuất hiện trong 御史臺 bị OCR/nhập liệu dùng font riêng:
  U+F0C51 -> 臺
  U+F025F -> 臺
CHỈ áp dụng mapping đã verify — codepoint PUA khác gặp trong corpus (nếu có)
PHẢI được liệt kê ở pua_inventory.csv với mapping_status="unmapped", KHÔNG
tự đoán.
"""

from __future__ import annotations

from typing import Optional

# ── PUA mapping đã verify (configs/normalization_v2.yaml là nguồn cấu hình
# chính thức cho pipeline thật; hằng số này là fallback/default khi không
# truyền config, và dùng trực tiếp trong unit test). ─────────────────────
VERIFIED_PUA_MAPPING = {
    0xF0C51: "臺",
    0xF025F: "臺",
}

# Các dải Unicode coi là "suspicious" khi audit (mục C.1 yêu cầu)
PUA_RANGES = [
    (0xE000, 0xF8FF),        # PUA chuẩn (BMP)
    (0xF0000, 0xFFFFD),      # Supplementary PUA-A
    (0x100000, 0x10FFFD),    # Supplementary PUA-B
]
VARIATION_SELECTOR_RANGES = [
    (0xFE00, 0xFE0F),
    (0xE0100, 0xE01EF),
]
REPLACEMENT_CHAR = 0xFFFD


def codepoint_category(cp: int) -> Optional[str]:
    """Trả 'pua' / 'variation_selector' / 'replacement_char' / None."""
    if cp == REPLACEMENT_CHAR:
        return "replacement_char"
    for lo, hi in PUA_RANGES:
        if lo <= cp <= hi:
            return "pua"
    for lo, hi in VARIATION_SELECTOR_RANGES:
        if lo <= cp <= hi:
            return "variation_selector"
    return None


def normalize_text_with_alignment(text: str, mapping: dict = None) -> tuple:
    """
    Áp `mapping` (codepoint -> chuỗi thay thế) lên `text`, trả về:
      normalized_text: str
      raw_to_normalized: list[int] -- raw_to_normalized[i] = vị trí BẮT ĐẦU
        của ký tự raw thứ i trong normalized_text (dùng để remap start
        offset của entity).
      normalized_to_raw: list[int] -- normalized_to_raw[j] = chỉ số ký tự
        raw đã sinh ra ký tự normalized thứ j (nhiều-1 nếu 1 raw char sinh
        ra nhiều normalized char).
      applied_rules: list[dict] -- log từng lần thay thế thật sự xảy ra
        (raw_index, codepoint, raw_char, replacement).
    KHÔNG BAO GIỜ xóa ký tự (replacement rỗng) -- mapping chỉ được thay thế
    1-hoặc-nhiều ký tự, không bao giờ 0 ký tự, đảm bảo "no silent character
    deletion". Deterministic: cùng input luôn ra cùng output.
    """
    mapping = VERIFIED_PUA_MAPPING if mapping is None else mapping
    normalized_chars = []
    raw_to_normalized = []
    normalized_to_raw = []
    applied_rules = []

    for i, ch in enumerate(text):
        cp = ord(ch)
        raw_to_normalized.append(len(normalized_chars))
        if cp in mapping:
            replacement = mapping[cp]
            if replacement == "":
                raise ValueError(
                    f"Mapping cho U+{cp:04X} là chuỗi rỗng -- vi phạm quy tắc "
                    f"'no silent character deletion', từ chối normalize."
                )
            applied_rules.append({
                "raw_index": i, "codepoint": f"U+{cp:04X}",
                "raw_char": ch, "replacement": replacement,
            })
        else:
            replacement = ch
        for rc in replacement:
            normalized_to_raw.append(i)
            normalized_chars.append(rc)

    normalized_text = "".join(normalized_chars)
    return normalized_text, raw_to_normalized, normalized_to_raw, applied_rules


def remap_span_inclusive(raw_to_normalized: list, raw_start: int, raw_end: int,
                          text_len: int, normalized_len: int) -> tuple:
    """
    Remap (raw_start, raw_end) [inclusive, convention thống nhất toàn repo]
    sang (normalized_start, normalized_end) [inclusive].
    """
    n_start = raw_to_normalized[raw_start]
    if raw_end + 1 < len(raw_to_normalized):
        end_char_len = raw_to_normalized[raw_end + 1] - raw_to_normalized[raw_end]
    else:
        end_char_len = normalized_len - raw_to_normalized[raw_end]
    n_end = raw_to_normalized[raw_end] + end_char_len - 1
    return n_start, n_end
