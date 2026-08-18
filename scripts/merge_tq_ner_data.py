"""
Merge CHisIEC + C-CLUE + CMAG into unified CoNLL BIO (PER/LOC/OFI) for
cross-domain zero-shot training, matching chisiec-ner's read_conll() format
(token<TAB>label per line, blank line = sentence boundary).

Schema decisions (from conversation with user):
- BOOK (CHisIEC) dropped -> O
- OFI / JOB -> OFI (unified title label)
- ORG (C-CLUE only) -> LOC (align with TQ mainstream convention: political
  entities merged into LOC)
- CMAG has no BOOK/ORG, only PER/LOC/OFI already
- Dynasty-name-as-LOC convention kept as-is (TQ side), no remapping
"""
import pickle
import sys

TMP = "C:/Users/PC/AppData/Local/Temp/"


def write_conll(sentences, path):
    """sentences: list of (tokens, labels)"""
    with open(path, "w", encoding="utf-8") as f:
        for tokens, labels in sentences:
            for t, l in zip(tokens, labels):
                f.write(f"{t}\t{l}\n")
            f.write("\n")


# ── CHisIEC (BIOES, char\tlabel, blank-line separated) ──────────────────────
def parse_chisiec(path):
    sentences = []
    tokens, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    sentences.append((tokens, labels))
                tokens, labels = [], []
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            ch, tag = parts
            if tag == "O":
                bio = "O"
            else:
                pos, typ = tag.split("-")
                if typ == "BOOK":
                    bio = "O"  # drop BOOK
                else:
                    if typ == "OFI":
                        typ = "OFI"
                    # B/S -> B-, I/E -> I-
                    bio_pos = "B" if pos in ("B", "S") else "I"
                    bio = f"{bio_pos}-{typ}"
            tokens.append(ch)
            labels.append(bio)
        if tokens:
            sentences.append((tokens, labels))
    return sentences


# ── C-CLUE (space-separated char / space-separated tag, one sentence per line) ──
def parse_cclue(src_path, lbl_path):
    sentences = []
    with open(src_path, encoding="utf-8") as fs, open(lbl_path, encoding="utf-8") as fl:
        for s_line, l_line in zip(fs, fl):
            toks = s_line.split()
            tags = l_line.split()
            if len(toks) != len(tags) or not toks:
                continue
            new_tags = []
            for tag in tags:
                if tag == "O":
                    new_tags.append("O")
                    continue
                pos, typ = tag.split("-")
                if typ == "JOB":
                    typ = "OFI"
                elif typ == "ORG":
                    typ = "LOC"  # unify: ORG merged into LOC
                elif typ not in ("PER", "LOC"):
                    new_tags.append("O")  # drop WAR/BOO/other minor types
                    continue
                new_tags.append(f"{pos}-{typ}")
            sentences.append((toks, new_tags))
    return sentences


# ── CMAG (pickle: list of [tokens, tag_ids], BIES scheme) ───────────────────
CMAG_ID2TYPE = {1: "LOC", 2: "LOC", 3: "LOC", 4: "LOC",
                6: "OFI", 7: "OFI", 8: "OFI", 9: "OFI",
                10: "PER", 11: "PER", 12: "PER", 13: "PER"}
CMAG_ID2POS = {1: "B", 2: "E", 3: "I", 4: "S",
               6: "B", 7: "E", 8: "I", 9: "S",
               10: "B", 11: "E", 12: "I", 13: "S"}


def parse_cmag(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    sentences = []
    for toks, tag_ids in data:
        labels = []
        cur_type = None
        for tg in tag_ids:
            if tg == 5 or tg == 0:
                labels.append("O")
                cur_type = None
                continue
            typ = CMAG_ID2TYPE[tg]
            pos = CMAG_ID2POS[tg]
            bio_pos = "B" if pos in ("B", "S") else "I"
            labels.append(f"{bio_pos}-{typ}")
        if len(labels) == len(toks) and toks:
            sentences.append((toks, labels))
    return sentences


def stats(name, sentences):
    from collections import Counter
    c = Counter()
    for _, labels in sentences:
        for l in labels:
            if l != "O":
                c[l.split("-")[1]] += 1
    print(f"{name}: {len(sentences)} sentences, entity counts: {dict(c)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    chisiec_train = parse_chisiec(TMP + "chisiec_train.txt")
    chisiec_dev = parse_chisiec(TMP + "chisiec_dev.txt")
    stats("CHisIEC train", chisiec_train)
    stats("CHisIEC dev", chisiec_dev)

    cclue_train = parse_cclue(TMP + "cclue_train_src.txt", TMP + "cclue_train_lbl.txt")
    cclue_dev = parse_cclue(TMP + "cclue_dev_src.txt", TMP + "cclue_dev_lbl.txt")
    stats("C-CLUE train", cclue_train)
    stats("C-CLUE dev", cclue_dev)

    cmag_train = parse_cmag(TMP + "data_cmag_train.pkl")
    cmag_dev = parse_cmag(TMP + "cmag_dev.pkl")
    stats("CMAG train", cmag_train)
    stats("CMAG dev", cmag_dev)

    merged_train = chisiec_train + cclue_train + cmag_train
    merged_dev = chisiec_dev + cclue_dev + cmag_dev

    write_conll(merged_train, TMP + "tq_merge_train.txt")
    write_conll(merged_dev, TMP + "tq_merge_dev.txt")

    stats("MERGED train", merged_train)
    stats("MERGED dev", merged_dev)
    print("done")
