import sys
sys.stdout.reconfigure(encoding="utf-8")

def strip_dtm(path_in, path_out):
    n = 0
    with open(path_in, encoding="utf-8") as fin, open(path_out, "w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.rstrip("\n")
            if not raw.strip():
                fout.write("\n")
                continue
            ch, tag = raw.split("\t")
            if tag.endswith("-DTM"):
                tag = "O"
                n += 1
            fout.write(f"{ch}\t{tag}\n")
    print(f"{path_in} -> {path_out}: stripped {n} DTM tag lines")

strip_dtm("D:/bio_source/tq_merge/dvsktt_test_remap.txt",
          "D:/bio_source/tq_merge/dvsktt_test_remap_no_dtm.txt")
