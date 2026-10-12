"""
Dựng từ điển chữ Hán → âm Hán Việt → web/static/hanviet.json

  python3 web/make_hanviet.py ~/Downloads/NER-2/NER_AnhThieu/*.xlsx

Nguồn 1 (ưu tiên): cột `sino` (phiên âm) ↔ `chinese` trong các file Excel DVSKTT.
  Mỗi câu được căn chữ ↔ âm tiết 1-1, chỉ lấy câu có số chữ = số âm tiết.
  Mỗi chữ giữ âm đọc xuất hiện nhiều nhất (đa âm thì liệt kê thêm).
Nguồn 2 (bổ sung chữ còn thiếu): trường kVietnamese của Unicode Unihan.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

OUT = Path(__file__).parent / "static" / "hanviet.json"
UNIHAN_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"
UNIHAN_CACHE = Path(__file__).parent / ".unihan_vietnamese.txt"

HAN_PUNCT = re.compile(r"[\s.,;:?!、，。；：？！“”‘’「」『』（）《》()\[\]\"'…—·-]")
SYL_SPLIT = re.compile(r"[\s.,;:?!、，。；：？！“”‘’「」『』（）《》()\[\]\"'…—·-]+")


def align_excel(paths: list[str]) -> tuple[dict, int, int]:
    counts: dict[str, Counter] = defaultdict(Counter)
    ok = total = 0
    for p in paths:
        if Path(p).name.startswith("~$"):  # file khoá tạm của Excel
            continue
        ws = openpyxl.load_workbook(os.path.expanduser(p), read_only=True, data_only=True).active
        for row in ws.iter_rows(min_row=2, values_only=True):
            sino, han = (row[2], row[3]) if len(row) >= 4 else (None, None)
            if not sino or not han:
                continue
            # câu trong cả hai cột cách nhau bằng xuống dòng
            for h_line, s_line in zip(str(han).splitlines(), str(sino).splitlines()):
                chars = HAN_PUNCT.sub("", h_line)
                syls = [s for s in SYL_SPLIT.split(s_line) if s]
                if not chars:
                    continue
                total += 1
                if len(chars) != len(syls):
                    continue
                ok += 1
                for ch, syl in zip(chars, syls):
                    counts[ch][syl.lower()] += 1
    return counts, ok, total


def load_unihan() -> dict[str, list[str]]:
    if not UNIHAN_CACHE.exists():
        print(f"Tải Unihan từ {UNIHAN_URL} …")
        data = urllib.request.urlopen(UNIHAN_URL, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            lines = z.read("Unihan_Readings.txt").decode("utf-8").splitlines()
        UNIHAN_CACHE.write_text("\n".join(l for l in lines if "\tkVietnamese\t" in l), encoding="utf-8")
    out = {}
    for line in UNIHAN_CACHE.read_text(encoding="utf-8").splitlines():
        cp, _, val = line.split("\t")
        out[chr(int(cp[2:], 16))] = val.split()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", nargs="+")
    args = ap.parse_args()

    counts, ok, total = align_excel(args.xlsx)
    print(f"Excel: căn được {ok}/{total} câu, {len(counts)} chữ")

    dic: dict[str, list[str]] = {}
    for ch, c in counts.items():
        top = c.most_common()
        # bỏ âm hiếm (thường do lệch căn) — giữ âm chiếm >= 10% số lần
        n = sum(c.values())
        dic[ch] = [s for s, k in top if k / n >= 0.1] or [top[0][0]]

    unihan = load_unihan()
    added = 0
    for ch, readings in unihan.items():
        if ch not in dic:
            dic[ch] = [r.lower() for r in readings]
            added += 1
    print(f"Unihan: thêm {added} chữ → tổng {len(dic)} chữ")

    OUT.write_text(json.dumps(dic, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"→ {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
