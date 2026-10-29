# Phase 6, Step 6.1: Generate dataset statistics

import jsonlines, json, os

stats = {}
for task in ["entity", "concept", "skill"]:
    stats[task] = {}
    for split in ["forget", "retain", "boundary"]:
        path = f"knot_data/knot_{task}/{split}_final.jsonl"
        try:
            with jsonlines.open(path) as r:
                items = list(r)
            stats[task][split] = len(items)
        except Exception:
            stats[task][split] = 0

print("\nDataset Statistics:")
print(f"{'Task':<15} {'Forget':>8} {'Retain':>8} {'Boundary':>10} {'Total':>8}")
grand_total = 0
for task, counts in stats.items():
    total = sum(counts.values())
    grand_total += total
    print(f"{task:<15} {counts['forget']:>8} {counts['retain']:>8} {counts['boundary']:>10} {total:>8}")
print(f"{'TOTAL':<15} {'':>8} {'':>8} {'':>10} {grand_total:>8}")

# Save stats
os.makedirs("knot_data", exist_ok=True)
with open("knot_data/dataset_stats.json", "w") as f:
    json.dump({"per_task": stats, "grand_total": grand_total}, f, indent=2)
print(f"\nStats saved to knot_data/dataset_stats.json")
