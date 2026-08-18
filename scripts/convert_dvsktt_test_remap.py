"""
Convert DVSKTT test-gold (jsonl, {entity|TYPE} markup in `output`) into
char-level CoNLL BIO, remapped to the unified TQ schema (PER/LOC/OFI/DTM):
  TITLE -> OFI
  ORG   -> LOC
  + add OFI tag for untagged occurrences of 陛下 (per guideline decision)
DTM kept as-is (TQ-trained model will never predict it; scored separately).
"""
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

LABEL_MAP = {"TITLE": "OFI", "ORG": "LOC", "PER": "PER", "LOC": "LOC", "DTM": "DTM"}

TAG_RE = re.compile(r"\{([^{}|]+)\|([A-Z]+)\}")


def parse_output_markup(output_text):
    """Strip {entity|TYPE} markup, return (plain_chars, bio_labels)."""
    chars = []
    labels = []
    pos = 0
    for m in TAG_RE.finditer(output_text):
        # plain text before this entity
        plain = output_text[pos:m.start()]
        for ch in plain:
            chars.append(ch)
            labels.append("O")
        entity_text, etype = m.group(1), m.group(2)
        mapped = LABEL_MAP.get(etype, etype)
        for i, ch in enumerate(entity_text):
            chars.append(ch)
            labels.append(f"{'B' if i == 0 else 'I'}-{mapped}")
        pos = m.end()
    # trailing plain text
    for ch in output_text[pos:]:
        chars.append(ch)
        labels.append("O")
    return chars, labels


def add_bixia_tag(chars, labels):
    """Tag untagged occurrences of 陛下 as OFI."""
    text = "".join(chars)
    for m in re.finditer("陛下", text):
        s, e = m.start(), m.end()
        if all(labels[i] == "O" for i in range(s, e)):
            labels[s] = "B-OFI"
            labels[s + 1] = "I-OFI"
    return labels


def convert(path_in, path_out):
    n_sent = 0
    with open(path_in, encoding="utf-8") as fin, open(path_out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            output_text = rec["output"]
            chars, labels = parse_output_markup(output_text)
            labels = add_bixia_tag(chars, labels)
            for ch, lb in zip(chars, labels):
                fout.write(f"{ch}\t{lb}\n")
            fout.write("\n")
            n_sent += 1
    print(f"{path_in} -> {path_out}: {n_sent} sentences")


if __name__ == "__main__":
    base = "D:/ancient-chinese-ner/data/processed/ner_clean/"
    out_dir = "D:/bio_source/tq_merge/"
    for split in ["test", "dev", "train"]:
        convert(base + f"{split}.jsonl", out_dir + f"dvsktt_{split}_remap.txt")
