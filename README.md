# Business Entity Resolution — Amazon ML Challenge 2026

Matches Source-1 business records to Source-2 / Source-3 records (US, India, France) under a macro F0.5 metric.
Everything is learned from the challenge data only — no external data, APIs or lookups.

## Portal results
| Submission | Portal F0.5 |
|---|---|
| v1 production pipeline (threshold 0.65) | 0.925 |
| v1 + calibrated decision layer (0.85 / empty-address 0.95 / native-script 0.70) | 0.929 |
| **v2 hybrid re-scoring, threshold 0.90** | **0.933** |

## Pipeline
1. **Normalisation** (`src/norm.py`, `src/exp3/norm3.py`): Unicode/accent cleanup, transliteration, legal-suffix stripping, address canonicalisation. `norm3` fixes a tokenizer bug that split Indic words at combining marks.
2. **Sharding**: test data routed by parsed state (US/India, learned last-component routing as fallback); records without a routable state are broadcast to every shard of their country; unseen countries (France) form one shard.
3. **Candidate generation** (`src/make_submission.py`): 8 retrieval channels (word & char n-gram TF-IDF on name/address, name 3-grams, number-normalised address, empty-address name quota, address keys, reverse retrieval) → learned LightGBM blocking ranker → top-10 per source (≤20 candidates per S1).
4. **Matching v2** (`src/rescore_v2.py`, `src/v2feats.py`): production candidates re-scored with norm3 text, 67 pairwise similarity features + 26 per-shard evidence features (candidate competition / margin, address-confirmed sibling agreement, name/address frequency, learned native→Latin token map) → LightGBM trained on the benchmark plus production-density replay data.
5. **Decision layer**: keep a pair if score ≥ threshold and it beats every competing S1 for the same record by ≥ 0.05 (each S2/S3 record has at most one owner).

## Validation lessons
- The original regional benchmark (~0.97) was far too optimistic: production shards contain country-wide broadcast pools and test pools are ~25% denser.
- A **production-faithful replay** (`experiments/final_optimization/00_production_replay/`) — the unmodified production code on held-out labelled training states — reproduced the portal score (~0.92) and became the main validation environment.
- Details: `experiments/` (reports per experiment: blocking, ownership, retrieval ablations, friend-model audit, replay, v2 ablations, test-vs-replay calibration shift).

## Layout
```
src/          all pipeline and experiment scripts (00_*..34_*, make_submission*.py, rescore_v2.py, v2feats.py, feats.py, blk.py, metric.py)
experiments/  final-day experiment scripts and reports
models/       trained LightGBM models + metadata (v1 production, v2 hybrid) and the learned transliteration map
```

## Reproduce
Place the challenge data under `dataset/{train,test}` and `utils/validate_submission.py`, copy `src/` to `pilot/`, then:
```
python pilot/make_submission.py all --team TEAM                                   # v1 pipeline (retrieval + scoring, ~7.5 h)
python pilot/rescore_v2.py submission_work <out_dir> <model_dir>                  # v2 re-scoring of cached candidates (~2.5 h)
python pilot/final_optimization/10_final_candidates/v2_hybrid/finalize.py 0.90 output_T090
```
Dependencies: `requirements.txt` (Python 3.14).
