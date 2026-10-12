"""
Lấy N đoạn đã gán nhãn từ file Excel NER_AnhThieu → web/static/gold.json
để xem song song với kết quả model (trang /gold).

  python3 web/make_gold.py "~/Downloads/NER-2/NER_AnhThieu/NER 1288-1388.xlsx" -n 10

Nhãn trong Excel chỉ là danh sách thực thể (không có vị trí) nên span được suy ra
bằng cách tìm mọi lần xuất hiện trong đoạn, ưu tiên chuỗi dài hơn khi chồng nhau.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path

import openpyxl

KEY_MAP = {"TIME": "DTM"}  # file dùng lẫn TIME / DTM
OUT = Path(__file__).parent / "static" / "gold.json"


def locate(text: str, ents: dict) -> list:
    cands = sorted({(e, KEY_MAP.get(t, t)) for t, lst in ents.items() for e in lst if e},
                   key=lambda x: -len(x[0]))
    taken = [False] * len(text)
    spans = []
    for ent, label in cands:
        for m in re.finditer(re.escape(ent), text):
            s, e = m.start(), m.end()
            if not any(taken[s:e]):
                taken[s:e] = [True] * (e - s)
                spans.append({"label": label, "start": s, "end": e, "text": ent})
    return sorted(spans, key=lambda x: x["start"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()

    ws = openpyxl.load_workbook(os.path.expanduser(args.xlsx), data_only=True).active
    docs = []
    for i, (link, _, sino, han, ner) in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if not han or not ner:
            continue
        ents = ast.literal_eval(ner)
        # chuẩn hoá: mỗi câu gốc (kết thúc bằng '.') thành 1 dòng, bỏ khoảng trắng
        sents = [s.strip() for s in re.split(r"(?<=\.)", re.sub(r"\s", "", han)) if s.strip()]
        text = "".join(s.replace(".", "。") for s in sents)
        spans = locate(text, ents)
        missing = sorted({e for lst in ents.values() for e in lst if e and e not in text})
        n_types = len({s["label"] for s in spans})
        docs.append({"row": i, "text": text, "raw": re.sub(r"[。]", "", text),
                     "sino": sino, "entities": spans, "missing": missing, "_score": n_types})

    # ưu tiên đoạn có nhiều loại nhãn, giữ thứ tự gốc
    picked = sorted(sorted(docs, key=lambda d: -d["_score"])[:args.n], key=lambda d: d["row"])
    for k, d in enumerate(picked, 1):
        d["id"] = k
        del d["_score"]
    OUT.write_text(json.dumps({"source": Path(args.xlsx).name, "docs": picked},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    raw_txt = OUT.with_name("gold_raw.txt")
    raw_txt.write_text("\n\n".join(f"# Đoạn {d['id']} (dòng {d['row']})\n{d['raw']}" for d in picked) + "\n",
                       encoding="utf-8")
    print(f"{len(picked)} đoạn → {OUT}\n       văn bản thô → {raw_txt}")
    for d in picked:
        print(f"  #{d['id']} dòng {d['row']}: {len(d['raw'])} chữ, {len(d['entities'])} span,"
              f" không tìm thấy trong text: {d['missing'] or '-'}")


if __name__ == "__main__":
    main()
