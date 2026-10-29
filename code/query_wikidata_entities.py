# Phase 1, Step 1.1: Generate 300 entities with triples via DeepSeek (batched)

from openai import OpenAI
import json, os, time
from collections import defaultdict
from config import DEEPSEEK_API_KEY, MODEL

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

os.makedirs("knot_data/raw/entities", exist_ok=True)
os.makedirs("knot_data/raw/wikidata", exist_ok=True)

OCC_TYPES = ["scientist", "politician", "artist", "athlete", "business_leader", "other"]
TARGET_PER_OCC = 50
BATCH_SIZE = 10  # Generate in smaller batches to avoid token limit

GEN_ENTITY_PROMPT = """Generate exactly {n} real, well-known {occ_type} people in JSON array format.

For each person, provide:
- "label": full name
- "forget_triples": array of 2-3 personal facts with keys: subject, predicate, predicate_id, object
- "retain_triples": array of 2-3 public achievement facts with keys: subject, predicate, predicate_id, object
- "wikipedia_text": short biography (~100 words)

Return ONLY the JSON array, no markdown or explanations.
Ensure the JSON is complete and properly closed."""

all_entities = []
entity_id_counter = 1000

def parse_entities(raw_text):
    """Parse entities from raw API response with multiple fallback methods"""
    import re

    entities = []

    # Try direct JSON parse
    try:
        entities = json.loads(raw_text.strip())
        if isinstance(entities, list):
            return entities
        elif isinstance(entities, dict):
            if 'data' in entities and isinstance(entities['data'], list):
                return entities['data']
            return [entities]
    except:
        pass

    # Try to find JSON array with regex
    match = re.search(r'\[.*\]', raw_text, re.DOTALL)
    if match:
        try:
            entities = json.loads(match.group())
            if isinstance(entities, list):
                return entities
        except:
            pass

    # Try to find JSON array with bracket balancing
    start_idx = raw_text.find('[')
    if start_idx != -1:
        depth = 0
        for i in range(start_idx, len(raw_text)):
            if raw_text[i] == '[':
                depth += 1
            elif raw_text[i] == ']':
                depth -= 1
                if depth == 0:
                    try:
                        entities = json.loads(raw_text[start_idx:i+1])
                        if isinstance(entities, list):
                            return entities
                    except:
                        break

    # Extract individual objects as last resort
    matches = re.findall(r'\{[^{}]*"label"[^"}]*\}[^{}]*', raw_text)
    if matches:
        entities = [json.loads(m) for m in matches if json.loads(m).get('label')]

    return entities

for occ_type in OCC_TYPES:
    print(f"Generating {TARGET_PER_OCC} {occ_type} entities...")

    occ_entities = []

    # Generate in batches
    for batch_num in range(0, TARGET_PER_OCC, BATCH_SIZE):
        batch_size = min(BATCH_SIZE, TARGET_PER_OCC - batch_num)
        print(f"  Batch {batch_num//BATCH_SIZE + 1}: {batch_num + 1}-{batch_num + batch_size} entities")

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": GEN_ENTITY_PROMPT.format(n=batch_size, occ_type=occ_type)}],
                max_tokens=4000,
                temperature=0.3
            )
            raw = resp.choices[0].message.content

            # Save raw response for debugging
            debug_file = f"knot_data/raw/entities/{occ_type}_batch{batch_num//BATCH_SIZE + 1}_response.txt"
            with open(debug_file, "w", encoding='utf-8') as f:
                f.write(raw)

            entities = parse_entities(raw)

            if entities:
                for e in entities:
                    if isinstance(e, dict):
                        if 'forget_triples' not in e:
                            e['forget_triples'] = []
                        if 'retain_triples' not in e:
                            e['retain_triples'] = []
                        if 'wikipedia_text' not in e:
                            e['wikipedia_text'] = ''
                        if 'label' in e:
                            e['occupation_type'] = occ_type
                            e['entity_id'] = f"Q{entity_id_counter}"
                            entity_id_counter += 1
                            occ_entities.append(e)

                print(f"    Parsed {len(entities)} entities, total for {occ_type}: {len(occ_entities)}")
            else:
                print(f"    Failed to parse entities from response")

        except Exception as ex:
            print(f"    Error: {ex}")

        time.sleep(1)

    all_entities.extend(occ_entities)
    print(f"  Completed {occ_type}: {len(occ_entities)} entities")

print(f"Total entities collected: {len(all_entities)}")

# Trim to 300 with balanced distribution
by_occ = defaultdict(list)
for e in all_entities:
    if isinstance(e, dict) and e.get('occupation_type'):
        by_occ[e['occupation_type']].append(e)

selected, i = [], 0
while len(selected) < 300 and any(by_occ[o] for o in OCC_TYPES):
    occ = OCC_TYPES[i % len(OCC_TYPES)]
    if by_occ[occ]:
        selected.append(by_occ[occ].pop(0))
    i += 1

print(f"Selected {len(selected)} entities")

# Save in the expected format
entity_triples = {}
for e in selected:
    entity_triples[e["entity_id"]] = {
        "label": e["label"],
        "forget_triples": e.get("forget_triples", []),
        "retain_triples": e.get("retain_triples", []),
        "wikipedia_text": e.get("wikipedia_text", "")
    }

with open("knot_data/raw/wikidata/entity_triples.json", "w", encoding='utf-8') as f:
    json.dump(entity_triples, f, ensure_ascii=False, indent=2)

occ_dist = defaultdict(int)
for e in selected:
    occ_dist[e.get("occupation_type", "other")] += 1
print(f"Occupation distribution: {dict(occ_dist)}")
print("Saved to knot_data/raw/wikidata/entity_triples.json")