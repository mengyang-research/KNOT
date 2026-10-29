import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

# Phase 5, Step 5.3: LLM Judge dependency score (Concept)

from openai import OpenAI
import jsonlines, json, re, random, time, os
from config import DEEPSEEK_API_KEY, MODEL

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

JUDGE_PROMPT = """Rate the knowledge dependency between the following forget question and retain question on a scale of 1-5.

Forget question: {forget_q}
Retain question: {retain_q}

Definition of dependency: Is understanding the core concept needed to answer the forget question also necessary to answer the retain question?

1 = No dependency (completely independent knowledge)
2 = Weak dependency (tangentially related)
3 = Moderate dependency (share some underlying concepts)
4 = Strong dependency (same core concept, different applications)
5 = Inseparable (cannot answer one without the other)

Respond with ONLY a JSON: {{"score": 1-5, "reason": "brief explanation"}}"""

def compute_llm_scores(forget_items, retain_items, sample_n=5):
    scores = []
    for i, fi in enumerate(forget_items):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(forget_items)}")
            sys.stdout.flush()
        sampled_retain = random.sample(retain_items, min(sample_n, len(retain_items)))
        item_scores = []
        for ri in sampled_retain:
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                        forget_q=fi["question"], retain_q=ri["question"]
                    )}],
                    max_tokens=150,
                    temperature=0
                )
                raw = resp.choices[0].message.content
                raw = re.sub(r'```(?:json)?\s*', '', raw).strip()
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    item_scores.append(result.get("score", 3) / 5.0)
                time.sleep(0.1)
            except Exception as ex:
                print(f"  Judge error: {ex}")
                sys.stdout.flush()
        scores.append(sum(item_scores) / len(item_scores) if item_scores else 0.5)
    return scores

# Run for concept task
forget_path = "knot_data/knot_concept/forget_with_emb_score.jsonl"
retain_path = "knot_data/knot_concept/retain_verified.jsonl"

if not os.path.exists(forget_path):
    print(f"Missing {forget_path}, skipping LLM judge scoring")
else:
    with jsonlines.open(forget_path) as r:
        forget_items = list(r)
    with jsonlines.open(retain_path) as r:
        retain_items = list(r)

    print(f"Computing LLM judge scores for {len(forget_items)} concept forget items...")
    sys.stdout.flush()
    scores = compute_llm_scores(forget_items, retain_items, sample_n=5)

    for item, score in zip(forget_items, scores):
        item["llm_judge_score"] = score

    with jsonlines.open(forget_path, "w") as w:
        w.write_all(forget_items)

    avg = sum(scores) / len(scores) if scores else 0
    print(f"Concept LLM judge scores done. Mean: {avg:.3f}")
    sys.stdout.flush()
