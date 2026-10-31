# KNOT Data Construction Pipeline

This directory contains the full pipeline for constructing the KNOT benchmark: entity/concept/skill QA generation, deduplication, verification, and entanglement scoring.

## Pipeline Overview

The pipeline runs in six phases, orchestrated by [`run_all.sh`](run_all.sh) with checkpoint-based resumption (completed steps are skipped on re-run):

| Phase | Step | Script | Description |
|-------|------|--------|-------------|
| 1 | 1.1 | `query_wikidata_entities.py` | Generate 300 candidate entity profiles (personal/professional triples in the Wikidata property schema + short biographies, LLM-batched) |
| 1 | 1.3 | `generate_entity_qa.py` | Generate forget / retain / boundary QA pairs for entities |
| 2 | 2.1 | `generate_concept_list.py` | Auto-generate 150 dual-use scientific concepts (batched) |
| 2 | 2.2 | `generate_concept_qa.py` | Generate Concept QA pairs |
| 3 | 3.1 | `generate_skill_qa.py` | Generate Skill QA pairs (Racket / Python / cross-paradigm) |
| 4 | 4.1 | `dedup.py` | MinHash deduplication |
| 4 | 4.2 | `verify_qa.py` | LLM fact verification of QA pairs |
| 5 | 5.1 | `compute_entanglement_scores.py` | Entanglement scores from `all-MiniLM-L6-v2` sentence-embedding cosine similarity (local, no API key) |
| 5 | 5.2 | `compute_kg_distance.py` | Per-entity knowledge-triple structural score (shared objects, bridging entities, content-word Jaccard fallback) (Entity) |
| 5 | 5.3 | `compute_llm_judge_score.py` | LLM-judge knowledge dependency scores (Concept) |
| 5 | 5.4 | `finalize_scores.py` | Merge scores and assign entanglement levels |
| 6 | 6.1 | `generate_stats.py` | Generate dataset statistics |

## Usage

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=your_key_here   # required by generation/verification steps
bash run_all.sh
```

- Configuration (API endpoints, model names, paths) is centralized in [`config.py`](config.py).
- Progress checkpoints are stored under `.checkpoints/`; delete a `.done` file to re-run a step.
- Step 5.1 uses `sentence-transformers` locally and requires no API key; QA generation and verification steps require access to an LLM API (see `config.py`).

## Output

The pipeline produces the JSONL files released in [`../dataset/`](../dataset/), each annotated with `entanglement_score` (continuous, [0, 1]) and `entanglement_level` (Low / Medium / High / Extreme).

## License

MIT. See the top-level [README](../README.md) for the dataset license and upstream references.
