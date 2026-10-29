# Phase 4, Step 4.1: MinHash deduplication

from datasketch import MinHash, MinHashLSH
import jsonlines, re, os

def get_shingles(text, k=5):
    text = re.sub(r'\s+', ' ', text.lower())
    return set(text[i:i+k] for i in range(len(text)-k+1))

def build_minhash(text):
    m = MinHash(num_perm=128)
    for shingle in get_shingles(text):
        m.update(shingle.encode('utf-8'))
    return m

def dedup_jsonl(input_path, output_path, threshold=0.8):
    if not os.path.exists(input_path):
        print(f"  Skipping {input_path} (not found)")
        return 0, 0
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept = []

    with jsonlines.open(input_path) as reader:
        items = list(reader)

    for i, item in enumerate(items):
        text = item.get("question", "") + " " + item.get("answer", "")
        m = build_minhash(text)
        try:
            if not lsh.query(m):
                lsh.insert(str(i), m)
                kept.append(item)
        except Exception:
            kept.append(item)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(kept)

    print(f"{input_path}: {len(items)} -> {len(kept)} after dedup")
    return len(items), len(kept)

for split in ["forget", "retain", "boundary"]:
    for task in ["entity", "concept", "skill"]:
        dedup_jsonl(
            f"knot_data/knot_{task}/{split}_raw.jsonl",
            f"knot_data/knot_{task}/{split}_deduped.jsonl"
        )

print("Deduplication complete.")
