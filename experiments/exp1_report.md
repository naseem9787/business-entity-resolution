# Exp1 report — density-faithful benchmark + ownership (all numbers cross-fitted over 5 state folds)
Benchmark: 124,965 S1 queries (US 65,639 / India 59,326; 15 states), pool S2 285,576 + S3 298,842 records. Validity checks: ALL PASSED
(state parse 100%/99.9997%; pool/S1 2.285 (S2) & 2.391 (S3) vs global 2.281 & 2.395; unowned share 26.6%/25.5% vs 26.6%/25.4%).
Candidates: top-15 per source (30 per entity), pair recall 0.9704, pairs 432,236 true / 419,443 in candidates.

| variant | macro F0.5 | gain vs B0 | 95% CI | micro P | micro R | singleton acc | false merges | missed |
|---|---|---|---|---|---|---|---|---|
| B0 pairwise LightGBM | 0.9638 | - | - | 0.9868 | 0.9276 | 0.9474 | 5349 | 31279 |
| A argmax+margin d=0 | 0.9668 | +0.0030 | [+0.0027,+0.0032] | 0.9892 | 0.9306 | 0.9540 | 4378 | 29994 |
| A d=0.05 | 0.9669 | +0.0030 | [+0.0027,+0.0033] | 0.9897 | 0.9299 | 0.9559 | 4185 | 30321 |
| A d=0.1 | 0.9668 | +0.0029 | [+0.0026,+0.0032] | 0.9899 | 0.9293 | 0.9562 | 4118 | 30569 |
| A d=0.2 | 0.9665 | +0.0027 | [+0.0024,+0.0030] | 0.9901 | 0.9284 | 0.9569 | 4028 | 30949 |
| D exclusive posterior | 0.9671 | +0.0033 | [+0.0030,+0.0036] | 0.9903 | 0.9292 | 0.9577 | 3952 | 30622 |
| B competitor-only stage 2 | 0.9668 | +0.0030 | [+0.0026,+0.0033] | 0.9894 | 0.9307 | 0.9516 | 4291 | 29941 |
| **E hybrid stage 2** | **0.9692** | **+0.0054** | **[+0.0049,+0.0058]** | 0.9897 | 0.9380 | 0.9718 | 4200 | 26782 |

Acceptance (E vs B0): gain>=0.002 yes; CI lower bound>0 yes; singleton not worse yes; micro recall change +1.04 pts (no drop) -> KEEP ownership. E beats best simple variant (D) by 0.0021 (>=0.0015) -> simplicity rule picks E.
Caveat: E does NOT transfer under leave-one-country-out (India->US: E 0.911 vs B0 0.927 at the source threshold) - see results.json LOCO.
