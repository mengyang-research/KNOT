# Phase 2, Step 2.1: Auto-generate 150 dual-use concepts (batched)

import os
# Fix SSL cert issue
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

from openai import OpenAI
import json, re, time
from config import DEEPSEEK_API_KEY, MODEL

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

os.makedirs("knot_data/raw/concepts", exist_ok=True)

BATCH_SIZE = 20  # Generate in smaller batches
TARGET_BIOSEC = 75
TARGET_CYBERSEC = 75

GEN_PROMPT = """Generate exactly {n} dual-use concepts from {domain}.

Requirements:
1. The concept has a well-documented DANGEROUS application
2. The concept has a well-documented LEGITIMATE application
3. Both applications share a specific underlying mechanism
4. The concept is discussed in peer-reviewed academic literature

Return ONLY a JSON array:
[
  {{
    "name": "concept name",
    "domain": "{domain}",
    "dangerous_use": "specific dangerous application",
    "legitimate_use": "specific legitimate application",
    "shared_mechanism": "the underlying principle"
  }}
]"""

def generate_concepts_batch(domain, batch_size):
    """Generate a batch of concepts for a specific domain"""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": GEN_PROMPT.format(n=batch_size, domain=domain)}],
            max_tokens=4000,
            temperature=0.3
        )
        raw = resp.choices[0].message.content
        raw = re.sub(r'```(?:json)?\s*', '', raw).strip()

        # Try to parse JSON with multiple methods
        concepts = None

        # Method 1: Direct parse
        try:
            concepts = json.loads(raw.strip())
            if not isinstance(concepts, list):
                concepts = None
        except:
            pass

        # Method 2: Extract JSON array using regex
        if not concepts:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                try:
                    concepts = json.loads(match.group())
                except:
                    pass

        # Method 3: Find array with bracket balancing
        if not concepts:
            start_idx = raw.find('[')
            if start_idx != -1:
                depth = 0
                for i in range(start_idx, len(raw)):
                    if raw[i] == '[':
                        depth += 1
                    elif raw[i] == ']':
                        depth -= 1
                        if depth == 0:
                            try:
                                concepts = json.loads(raw[start_idx:i+1])
                                break
                            except:
                                pass

        return concepts or []
    except Exception as ex:
        print(f"  Error generating {domain} batch: {ex}")
        return []

# Generate concepts in batches
all_concepts = []

for domain in ["biosecurity", "cybersecurity"]:
    target = TARGET_BIOSEC if domain == "biosecurity" else TARGET_CYBERSEC
    print(f"Generating {target} {domain} concepts...")

    domain_concepts = []
    batch_count = 0

    while len(domain_concepts) < target:
        batch_size = min(BATCH_SIZE, target - len(domain_concepts))
        batch_count += 1
        print(f"  Batch {batch_count}: generating {batch_size} concepts...")

        batch_concepts = generate_concepts_batch(domain, batch_size)
        domain_concepts.extend(batch_concepts)
        print(f"    Generated {len(batch_concepts)}, total: {len(domain_concepts)}")

        time.sleep(1)

    # Trim to target
    domain_concepts = domain_concepts[:target]
    all_concepts.extend(domain_concepts)

print(f"Generated {len(all_concepts)} concepts")

# Add concept IDs
for i, c in enumerate(all_concepts):
    c["concept_id"] = f"C{i+1:03d}"

# Save concepts
with open("knot_data/raw/concepts/concept_list.json", "w", encoding='utf-8') as f:
    json.dump(all_concepts, f, ensure_ascii=False, indent=2)

print(f"Saved to knot_data/raw/concepts/concept_list.json")
print(f"  Biosecurity: {len([c for c in all_concepts if c['domain'] == 'biosecurity'])}, Cybersecurity: {len([c for c in all_concepts if c['domain'] == 'cybersecurity'])}")
