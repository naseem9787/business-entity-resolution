# Model v2 on the production-faithful replay (87,879 held-out S1; test-like density; owner-presence corrected; rule chosen on entity half A, reported on half B)
| model | B F0.5 | US | India | P | R | singleton | cand recall | false merges | rejected TP | rule (base/empty/native, margin .05) |
|---|---|---|---|---|---|---|---|---|---|---|
| A production model + rule .65 | .9168 | .9261 | .8994 | .9355 | .9093 | .807 | .9440 | 7574 | 4182 | .65/.65/.65 |
| B production model + rule .85/.95/.70 (portal ~.929) | .9365 | .9417 | .9268 | .9813 | .8721 | .932 | .9440 | 2008 | 8675 | .85/.95/.70 |
| B* production model, re-tuned rule | .9385 | .9440 | .9284 | .9791 | .8823 | .921 | .9440 | 2275 | 7451 | .80/.95/.60 |
| C0 norm3 retrieval+matcher, tokenizer only | .9332 | .9451 | .9111 | .9789 | .8709 | .921 | .9324 | 2266 | 7433 | |
| C1 + translit | .9337 | .9456 | .9115 | .9797 | .8714 | .931 | .9324 | 2176 | 7371 | |
| C2 + competition | .9385 | .9482 | .9205 | .9865 | .8684 | .952 | .9324 | 1431 | 7729 | |
| C3 full v2 (bench-trained) | .9394 | .9493 | .9208 | .9876 | .8723 | .956 | .9324 | 1323 | 7262 | |
| C5 full v2, bench+replay LOSO | .9443 | .9523 | .9292 | .9887 | .8796 | .953 | .9324 | 1214 | 6382 | |
| H0 HYBRID tokenizer only | .9370 | .9448 | .9225 | .9788 | .8787 | .919 | .9440 | 2297 | 7885 | |
| H1 HYBRID + translit | .9373 | .9453 | .9223 | .9797 | .8788 | .929 | .9440 | 2202 | 7877 | |
| H3 HYBRID full v2 (bench-trained) | .9443 | .9494 | .9349 | .9878 | .8822 | .958 | .9440 | 1321 | 7462 | .85/.99/.65 |
| **H5 HYBRID full v2, bench+replay LOSO** | **.9491** | **.9530** | **.9417** | .9884 | .8897 | .954 | .9440 | 1256 | 6558 | .70/.70/.50 |
Findings: (1) norm3 in RETRIEVAL lowers candidate recall (.944 -> .932; the production ranker was trained on the old representation) -> keep production retrieval.
(2) tokenizer fix alone and translit alone are ~neutral (+.000/+.001) on the replay; the gain comes from the competition/sibling/frequency evidence (+.006-.007)
and from training on production-density data (+.005). (3) Hybrid H5 = production candidates re-scored: +.0126 over the submitted rule B, both countries, all slices.
Caveat: replay over-predicted the last portal gain ~5x (.019 predicted vs ~.004 observed); France untested.

## Test files (hybrid H5 model on cached production candidates; candidate_pairs identical to baseline; all validated)
Label-free check: on the replay v2 predicts ~+1% more matches than the subA rule, on the test +8.3% in EVERY country -> calibration shift on test.
| T_BASE (=T_EMPTY, T_NATIVE=T-.2) | replay B F0.5 | US | India | replay P | test matches | vs subA |
|---|---|---|---|---|---|---|
| .70 (replay-optimal) | .9491 | .9530 | .9417 | .9884 | 5,702,765 | 1.073 |
| **.80 (recommended)** | **.9471** | .9513 | .9392 | .9930 | 5,545,450 | 1.044 |
| .85 | .9444 | .9492 | .9355 | .9948 | 5,445,396 | 1.025 |
| .90 | .9404 | .9462 | .9295 | .9963 | 5,320,155 | 1.001 |
Files: pilot/final_optimization/10_final_candidates/v2_hybrid/output_T070 | output_T080 | output_T090. Re-thresholding the cached v2 scores takes ~10 min (finalize.py <T> <dir>).
Full rescoring runtime: 2 h 24 min (no retrieval).
