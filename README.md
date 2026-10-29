# KNOT: A Knowledge Entanglement Benchmark for Robust Unlearning Evaluation

Official repository for the NeurIPS 2026 Datasets & Benchmarks Track paper:

> **KNOT: A Knowledge Entanglement Benchmark for Robust Unlearning Evaluation**
> Mengyang Li, Jingwen Wang, Yu Zhang, Shuang Liu, Zhong Zhang

Large language model (LLM) unlearning aims to selectively remove specific knowledge while preserving model utility. Existing benchmarks such as TOFU, WMDP, and MUSE evaluate unlearning on semantically disjoint forget and retain sets, making the task artificially easy. KNOT formalizes knowledge entanglement at three levels — **entity**, **concept**, and **skill** — and provides a benchmark of ~5,300 forget, ~9,800 retain, and ~5,000 boundary QA pairs, each annotated with a quantitative entanglement score. Evaluation of seven methods shows that (a) all methods degrade severely under high entanglement, (b) the coreset effect reported on standard benchmarks vanishes, and (c) existing metrics miss critical boundary behavior. We propose **Boundary Precision (BP)** and **Boundary Recall (BR)** to fill this gap.

## Repository Structure

```
KNOT/
├── code/       # Data construction and scoring pipeline
└── dataset/    # KNOT dataset (20,135 QA pairs) + Croissant metadata
```

- **[`code/`](code/)** — the full pipeline for constructing KNOT: entity selection, QA generation, deduplication, verification, and entanglement scoring. See [`code/README.md`](code/README.md).
- **[`dataset/`](dataset/)** — the released KNOT dataset in JSONL format with entanglement scores, plus the Croissant 1.0 metadata file with Responsible AI fields. See [`dataset/README.md`](dataset/README.md).

## Dataset at a Glance

| Category | Forget | Retain | Boundary | Total |
|----------|--------|--------|----------|-------|
| Entity   | 4,017  | 7,450  | 2,093    | 13,560 |
| Concept  | 1,199  | 2,237  | 2,885    | 6,321 |
| Skill    | 84     | 131    | 39       | 254 |
| **Total**| **5,300** | **9,818** | **5,017** | **20,135** |

## Quick Start

```bash
# Run the full data construction pipeline (requires API access, see code/README.md)
cd code
bash run_all.sh
```

## License

- **Dataset & code**: CC-BY 4.0
- **Upstream sources**: Wikipedia (CC-BY-SA 3.0), Wikidata (CC0), WMDP (MIT)

## Citation

If you use KNOT, please cite:

```bibtex
@inproceedings{knot2026,
  title     = {KNOT: A Knowledge Entanglement Benchmark for Robust Unlearning Evaluation},
  author    = {Li, Mengyang and Wang, Jingwen and Zhang, Yu and Liu, Shuang and Zhang, Zhong},
  booktitle = {Advances in Neural Information Processing Systems 39 (NeurIPS 2026) Datasets and Benchmarks Track},
  year      = {2026}
}
```

## Safety Notes

- KNOT-Concept contains dual-use scientific content (biosecurity, cybersecurity, chemistry). All hazardous content follows the WMDP threat model, with operational specifics removed.
- Public figure information in KNOT-Entity is sourced from publicly available Wikipedia/Wikidata.
- The dataset is released under CC-BY 4.0 with an Acceptable Use clause prohibiting re-identification of subjects, harassment, or capability uplift.
