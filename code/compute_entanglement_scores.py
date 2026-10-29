# Phase 5, Step 5.1: Compute embedding-based entanglement scores
# Uses sentence-transformers (local, no API key needed)

import numpy as np
import jsonlines, os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Fix SSL cert issue
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

# Use HF mirror for faster download in China
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Downloads ~90MB on first run, cached afterwards
MODEL_NAME = "all-MiniLM-L6-v2"
print(f"Loading sentence-transformers model: {MODEL_NAME}")
print(f"Using HF endpoint: {os.environ['HF_ENDPOINT']}")
model = SentenceTransformer(MODEL_NAME)

def encode_qa(qa_items, batch_size=64):
    texts = [f"Q: {item['question']} A: {item['answer']}" for item in qa_items]
    return model.encode(texts, batch_size=batch_size,
                        show_progress_bar=True, normalize_embeddings=True)

def compute_max_similarity(forget_embs, retain_embs):
    scores = []
    for f_emb in tqdm(forget_embs, desc="Scoring"):
        sims = retain_embs @ f_emb
        scores.append(float(np.max(sims)))
    return scores

for task in ["entity", "concept", "skill"]:
    forget_path = f"knot_data/knot_{task}/forget_verified.jsonl"
    retain_path = f"knot_data/knot_{task}/retain_verified.jsonl"

    if not os.path.exists(forget_path) or not os.path.exists(retain_path):
        print(f"Skipping {task}: missing files (forget: {os.path.exists(forget_path)}, retain: {os.path.exists(retain_path)})")
        # Create empty output file if forget_verified doesn't exist
        output_path = f"knot_data/knot_{task}/forget_with_emb_score.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with jsonlines.open(output_path, "w") as w:
            pass  # Create empty file
        continue

    with jsonlines.open(forget_path) as r:
        forget_items = list(r)
    with jsonlines.open(retain_path) as r:
        retain_items = list(r)

    if not forget_items:
        print(f"Skipping {task}: forget set is empty ({len(forget_items)} items)")
        # Create empty output file
        output_path = f"knot_data/knot_{task}/forget_with_emb_score.jsonl"
        with jsonlines.open(output_path, "w") as w:
            pass
        continue

    if not retain_items:
        print(f"Warning: {task} retain set is empty ({len(retain_items)} items)")
        # Still process with empty retain set, all scores will be 0
        scores = [0.0] * len(forget_items)
        for item, score in zip(forget_items, scores):
            item["embedding_score"] = score
        output_path = f"knot_data/knot_{task}/forget_with_emb_score.jsonl"
        with jsonlines.open(output_path, "w") as w:
            w.write_all(forget_items)
        print(f"{task}: created output with zero scores (no retain data)")
        continue

    print(f"Encoding {task} forget ({len(forget_items)} items)...")
    forget_embs = encode_qa(forget_items)
    print(f"Encoding {task} retain ({len(retain_items)} items)...")
    retain_embs = encode_qa(retain_items)

    scores = compute_max_similarity(forget_embs, retain_embs)
    for item, score in zip(forget_items, scores):
        item["embedding_score"] = score

    output_path = f"knot_data/knot_{task}/forget_with_emb_score.jsonl"
    with jsonlines.open(output_path, "w") as w:
        w.write_all(forget_items)

    print(f"{task}: mean={np.mean(scores):.3f}, min={np.min(scores):.3f}, max={np.max(scores):.3f}")

print("Embedding scores complete.")
