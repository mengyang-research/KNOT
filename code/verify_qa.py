# Phase 4, Step 4.2: LLM fact verification of QA pairs

from openai import OpenAI
import jsonlines, json, re, os, time
from config import DEEPSEEK_API_KEY, MODEL

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

VERIFY_PROMPT = """Given the following question and answer, determine if the answer is factually correct based on general knowledge.

Question: {question}
Answer: {answer}

Respond with ONLY a JSON object: {{"correct": true/false, "confidence": 0-1, "note": "brief explanation if incorrect"}}"""

def verify_batch(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"  Skipping {input_path} (not found)")
        return

    with jsonlines.open(input_path) as r:
        items = list(r)

    verified = []
    for item in items:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": VERIFY_PROMPT.format(
                    question=item["question"], answer=item["answer"]
                )}],
                max_tokens=200,
                temperature=0
            )
            raw = resp.choices[0].message.content
            raw = re.sub(r'```(?:json)?\s*', '', raw).strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if result.get("correct", False) and result.get("confidence", 0) >= 0.7:
                    item["verify_confidence"] = result["confidence"]
                    verified.append(item)
            time.sleep(0.1)
        except Exception as ex:
            print(f"  Verify error: {ex}")
            # Keep item on error (fail open)
            item["verify_confidence"] = 0.7
            verified.append(item)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, "w") as w:
        w.write_all(verified)
    print(f"{input_path}: {len(verified)}/{len(items)} kept after verification")

# Verify forget and boundary splits (retain is typically too large; skip for now)
for task in ["entity", "concept", "skill"]:
    for split in ["forget", "boundary"]:
        verify_batch(
            f"knot_data/knot_{task}/{split}_deduped.jsonl",
            f"knot_data/knot_{task}/{split}_verified.jsonl"
        )
    # For retain, just copy deduped -> verified (too many to LLM-verify cost-effectively)
    retain_src = f"knot_data/knot_{task}/retain_deduped.jsonl"
    retain_dst = f"knot_data/knot_{task}/retain_verified.jsonl"
    if os.path.exists(retain_src):
        with jsonlines.open(retain_src) as r:
            items = list(r)
        os.makedirs(os.path.dirname(retain_dst), exist_ok=True)
        with jsonlines.open(retain_dst, "w") as w:
            w.write_all(items)
        print(f"retain ({task}): {len(items)} items copied (not LLM-verified to save cost)")
    else:
        print(f"retain ({task}): source file {retain_src} not found, creating empty verified file")
        os.makedirs(os.path.dirname(retain_dst), exist_ok=True)
        with jsonlines.open(retain_dst, "w") as w:
            pass  # Create empty file

print("Verification complete.")
