# Phase 1, Step 1.3: Generate Forget / Retain / Boundary QA Pairs for entities

import os
# Fix SSL cert issue
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

from openai import OpenAI
import json, time, jsonlines
from config import DEEPSEEK_API_KEY, MODEL

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

os.makedirs("knot_data/knot_entity", exist_ok=True)

def llm_call(prompt, max_tokens=2000, temperature=0.7):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp.choices[0].message.content

def parse_qa_json(text):
    import re
    text = re.sub(r'```(?:json)?\s*', '', text).strip()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return []
    return []

with open("knot_data/raw/wikidata/entity_triples.json", encoding='utf-8') as f:
    entity_triples = json.load(f)

all_forget, all_retain, all_boundary = [], [], []

FORGET_PROMPT = """You are constructing a machine unlearning benchmark dataset.

Entity: {name}
Personal/private information triples:
{forget_triples}

Wikipedia context (first 3000 words):
{wiki_text}

Generate exactly 20 question-answer pairs about the entity's personal and private information (family relationships, personal background, private habits, residential history). Requirements:
1. Each question must be answerable from the triples or Wikipedia text provided.
2. Questions should be factual with clear, short answers.
3. Include the entity name in each question.
4. Return ONLY a JSON array, no other text. Format: [{{"question": "...", "answer": "...", "triple_ref": "predicate used"}}]"""

RETAIN_PROMPT = """You are constructing a machine unlearning benchmark dataset.

Entity: {name}
Public achievement triples:
{retain_triples}

Wikipedia context (first 3000 words):
{wiki_text}

Generate exactly 50 question-answer pairs about the entity's publicly documented achievements, publications, awards, and official roles. Requirements:
1. Questions must be verifiable from the provided triples or Wikipedia text.
2. Focus on professional accomplishments, not personal information.
3. Include the entity name in each question.
4. Return ONLY a JSON array: [{{"question": "...", "answer": "...", "triple_ref": "predicate used"}}]"""

BOUNDARY_PROMPT = """You are constructing a machine unlearning benchmark dataset about knowledge entanglement.

Entity: {name}
Private information: {forget_summary}
Public achievements: {retain_summary}

Generate exactly 10 boundary question-answer pairs. A boundary question MUST satisfy both:
1. Fully answering it requires knowledge from BOTH personal/private facts AND public achievements.
2. The question cannot be correctly answered using only one type of information.

Example of good boundary question: "What personal experience did [name] cite when explaining their motivation for their most famous research?"
Example of bad boundary question: "What is [name]'s most famous work?" (only uses retain info)

Return ONLY a JSON array: [{{"question": "...", "answer": "...", "requires_forget": "which private fact is needed", "requires_retain": "which public fact is needed"}}]"""

for entity_id, data in entity_triples.items():
    name = data["label"]
    forget_triples = data["forget_triples"]
    retain_triples = data["retain_triples"]
    wiki_text = data["wikipedia_text"][:3000]

    if not forget_triples or not retain_triples:
        print(f"Skip {name}: insufficient triples")
        continue

    # Generate forget QA
    forget_str = "\n".join([f"- {t['predicate']}: {t['object']}" for t in forget_triples])
    try:
        raw = llm_call(FORGET_PROMPT.format(name=name, forget_triples=forget_str, wiki_text=wiki_text))
        forget_qas = parse_qa_json(raw)
        for qa in forget_qas:
            qa.update({"entity_id": entity_id, "entity_name": name, "split": "forget"})
        all_forget.extend(forget_qas)
    except Exception as ex:
        print(f"  Forget error for {name}: {ex}")
        forget_qas = []

    # Generate retain QA
    retain_str = "\n".join([f"- {t['predicate']}: {t['object']}" for t in retain_triples])
    try:
        raw = llm_call(RETAIN_PROMPT.format(name=name, retain_triples=retain_str, wiki_text=wiki_text))
        retain_qas = parse_qa_json(raw)
        for qa in retain_qas:
            qa.update({"entity_id": entity_id, "entity_name": name, "split": "retain"})
        all_retain.extend(retain_qas)
    except Exception as ex:
        print(f"  Retain error for {name}: {ex}")
        retain_qas = []

    # Generate boundary QA
    forget_summary = "; ".join([f"{t['predicate']}={t['object']}" for t in forget_triples[:5]])
    retain_summary = "; ".join([f"{t['predicate']}={t['object']}" for t in retain_triples[:5]])
    try:
        raw = llm_call(BOUNDARY_PROMPT.format(
            name=name, forget_summary=forget_summary, retain_summary=retain_summary
        ))
        boundary_qas = parse_qa_json(raw)
        for qa in boundary_qas:
            qa.update({"entity_id": entity_id, "entity_name": name, "split": "boundary"})
        all_boundary.extend(boundary_qas)
    except Exception as ex:
        print(f"  Boundary error for {name}: {ex}")
        boundary_qas = []

    print(f"{name}: {len(forget_qas)} forget, {len(retain_qas)} retain, {len(boundary_qas)} boundary")
    time.sleep(1)

with jsonlines.open("knot_data/knot_entity/forget_raw.jsonl", "w") as w:
    w.write_all(all_forget)
with jsonlines.open("knot_data/knot_entity/retain_raw.jsonl", "w") as w:
    w.write_all(all_retain)
with jsonlines.open("knot_data/knot_entity/boundary_raw.jsonl", "w") as w:
    w.write_all(all_boundary)

print(f"Total: {len(all_forget)} forget, {len(all_retain)} retain, {len(all_boundary)} boundary")
