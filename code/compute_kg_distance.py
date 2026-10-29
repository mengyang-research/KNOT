# Phase 5, Step 5.2: KG distance computation (Entity only)

import networkx as nx
import json, os

with open("knot_data/raw/wikidata/entity_triples.json", encoding='utf-8') as f:
    entity_triples = json.load(f)

# Build knowledge graph: entity labels and their triple objects as nodes
G = nx.Graph()
for eid, data in entity_triples.items():
    label = data["label"]
    for t in data["forget_triples"] + data["retain_triples"]:
        subject = t.get("subject", label)
        obj = t.get("object", "")
        if subject and obj:
            G.add_edge(subject, obj, predicate=t.get("predicate", ""))

# Set of all entity label nodes (used to distinguish entity nodes from object nodes)
all_labels = {data["label"] for data in entity_triples.values()}

def kg_entanglement(entity_id, entity_data):
    """
    Improved KG entanglement score with multiple strategies:

    1. Direct object overlap: If forget and retain share exact objects -> high score
    2. 2-hop cross-entity connection: Third entity bridges forget and retain objects
    3. Token-level similarity: Jaccard similarity between tokenized object text

    Score is the maximum of all strategies (0-1 range).
    """
    forget_objects = [t["object"] for t in entity_data["forget_triples"] if t.get("object")]
    retain_objects = [t["object"] for t in entity_data["retain_triples"] if t.get("object")]

    if not forget_objects or not retain_objects:
        return 0.0

    # Strategy 1: Direct object overlap
    forget_set = set(forget_objects)
    retain_set = set(retain_objects)
    direct_overlap = len(forget_set & retain_set)
    if direct_overlap > 0:
        # High score for direct overlap: up to 1.0
        return min(direct_overlap / max(len(forget_set), len(retain_set)), 1.0)

    # Strategy 2: 2-hop cross-entity connection
    entity_label = entity_data["label"]
    connecting = 0
    total = len(forget_objects) * len(retain_objects)

    for f_obj in forget_objects:
        if f_obj not in G:
            continue
        # Other entity labels that reference this forget object
        f_entity_neighbors = {
            n for n in G.neighbors(f_obj)
            if n != entity_label and n in all_labels
        }

        for r_obj in retain_objects:
            if r_obj not in G:
                continue
            # Other entity labels that reference this retain object
            r_entity_neighbors = {
                n for n in G.neighbors(r_obj)
                if n != entity_label and n in all_labels
            }

            if f_entity_neighbors & r_entity_neighbors:
                connecting += 1

    if connecting > 0:
        return connecting / total

    # Strategy 3: Token-level Jaccard similarity (fallback)
    import re
    STOPWORDS = {
        'the', 'and', 'for', 'that', 'with', 'this', 'from', 'are', 'was',
        'has', 'had', 'not', 'but', 'have', 'who', 'one', 'all', 'can',
        'its', 'him', 'her', 'she', 'his', 'they', 'were', 'been', 'than',
        'also', 'born', 'died', 'date', 'place', 'time', 'year', 'name',
    }

    def tokenize(text):
        tokens = set()
        for w in re.findall(r'[a-z]{3,}', text.lower()):
            if w not in STOPWORDS:
                tokens.add(w)
        return tokens

    forget_tokens = set()
    for obj in forget_objects:
        forget_tokens |= tokenize(obj)

    retain_tokens = set()
    for obj in retain_objects:
        retain_tokens |= tokenize(obj)

    if forget_tokens and retain_tokens:
        intersection = len(forget_tokens & retain_tokens)
        union = len(forget_tokens | retain_tokens)
        return intersection / union

    return 0.0


entity_kg_scores = {}
for eid, data in entity_triples.items():
    entity_kg_scores[eid] = kg_entanglement(eid, data)

os.makedirs("knot_data/scores", exist_ok=True)
with open("knot_data/scores/entity_kg_scores.json", "w") as f:
    json.dump(entity_kg_scores, f, indent=2)

print(f"KG scores computed for {len(entity_kg_scores)} entities")
nonzero = sum(1 for s in entity_kg_scores.values() if s > 0)
avg = sum(entity_kg_scores.values()) / len(entity_kg_scores) if entity_kg_scores else 0
print(f"Non-zero KG scores: {nonzero}/{len(entity_kg_scores)} ({100*nonzero/max(len(entity_kg_scores),1):.1f}%)")
print(f"Average KG entanglement score: {avg:.3f}")

# Merge KG scores into the forget_with_emb_score file
import jsonlines

forget_path = "knot_data/knot_entity/forget_with_emb_score.jsonl"
if os.path.exists(forget_path):
    with jsonlines.open(forget_path) as r:
        items = list(r)
    for item in items:
        eid = item.get("entity_id", "")
        item["kg_score"] = entity_kg_scores.get(eid, 0.0)
    with jsonlines.open(forget_path, "w") as w:
        w.write_all(items)
    print(f"KG scores merged into {forget_path}")
