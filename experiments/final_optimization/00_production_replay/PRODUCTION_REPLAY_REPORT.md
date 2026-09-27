# Production replay report (final day)
Replay = the UNMODIFIED production code (make_submission.predict, production models/threshold/margin, production routing incl. India broadcast)
run on 87,879 labelled TRAINING S1 from held-out states (US: ia, pa, ut = 57,428; India: andhra_pradesh, punjab = 30,451; none used to train the models),
against the FULL training S2/S3 pools (303,163 true pairs). Runtime 36 min.

## A. Base production replay (production rule thr .65 / margin .05)
| | F0.5 | P | R | singleton acc | empty-addr precision |
|---|---|---|---|---|---|
| raw (owners outside replay absent) | 0.9015 (US .910, India .885) | .912 | .909 | .76 | - |
| owner-presence corrected | 0.9180 (US .927, India .900) | .936 | .909 | .81 | .40 |
Portal: 0.9250. Benchmark (Exp2): 0.9711. -> The replay reproduces the portal degradation.
Candidate recall 0.943 (benchmark 0.993); only 299 true pairs are lost to routing; the rest are pushed out of the top-10 by broadcast records
(every US shard receives 111k unroutable/empty-address records, every India shard 479k).
Owner-presence correction: FP rate on records whose owner is another replay S1 is 1.8-1.9x lower than when the owner is outside the replay;
foreign-owned FP rows were blocked at that measured rate (the test has all owners present).
FP taxonomy (raw): 4,508 unowned, 411 owned by another replay S1, 21,763 owned by an S1 outside the replay. Unowned records are linked at 4.2% (US) / 7.2% (India) of their candidates.

## B. Density (withhold w of S1; density = 1/(1-w)); production rule, corrected
w=0 .9180 | w=.2 (test-like) .9170 | w=.5 .9145 -> modest monotone degradation, no collapse. Raw: .9015 / .8992 / .8940.

## C. India routing
broadcast (production): F .8853 P .898 R .899 cand-recall .937 | no broadcast: F .8521 P .982 R .682 cand-recall .723.
The broadcast pool is NECESSARY (27% of India true pairs live in unroutable records); it creates the false candidates that the decision layer must handle.

## D. Decision layer (selected on half A of entities, reported on held-out half B, corrected, w=0.2)
| rule | B F0.5 | US | India | singleton | matches/S1 |
|---|---|---|---|---|---|
| production .65/.05 | .9168 | .9261 | .8994 | .810 | 3.35 |
| global .85/.05 | ~.926 | | | | |
| **thr .85, empty-addr .95, native .70, margin .05** | **.9361** | **.9418** | **.9256** | .935 | 3.05 |
| same with margin .30 | .9366 | .9423 | .9260 | .938 | 3.05 |
Same ranking at w=0 and w=0.5 and in the uncorrected (pessimistic) replay (.8992 -> .9308). High-competition and top-2-close conditions add nothing and were dropped.
Country-specific thresholds added nothing (India/US offsets selected 0).

## E. Calibration (per-S1 candidates with s>=.65; share of pairs in the .2-.97 band)
benchmark US 3.35/.030, India 3.29/.045 | replay US 3.51/.069, India 3.60/.091 | TEST US 3.34/.053, India 3.60/.108, France 4.22/.119.
The replay matches the test's India distribution almost exactly; the benchmark does not. France is the most over-linked-looking country.

## Answers
Q1 Yes: labelled production replay gives .917 (corrected) vs portal .925 vs benchmark .971.
Q2 Combination, dominated by matcher calibration under production density: broadcast/unroutable pools (mostly empty-address records) put country-wide distractors into every shard;
    the .65 threshold was tuned at benchmark density; empty-address predictions are only ~40% precise; retrieval loses ~5% more pairs. Ownership works when the owner is present. France: unmeasurable, but its score distribution is the most shifted.
Q3 thr .85 / empty-address .95 / native-script .70 / margin .05 (country-agnostic): +0.019 corrected, +0.031 pessimistic, gains in both countries and all densities.
Q4 Risks: (a) France unlabelled - rule is country-agnostic and its effect on France is untested; (b) replay has fewer competing S1 than the test (only matters for margin, which is kept at the portal-proven .05);
    (c) replay states are 5 of ~60; (d) 7% more empty predictions (6.0% -> 6.9%).
Q5 Yes - one submission: pilot/final_optimization/10_final_candidates/subA_thr085_margin005/matching_results.tsv

## Root-cause diagnosis / best rule / evidence / submit / command
1. Root cause: benchmark density != production density (broadcast pools + 25% denser test pools); threshold/empty-address handling tuned on the wrong distribution.
2. Best validated decision layer: T_BASE .85, T_EMPTY .95, T_NATIVE .70, MARGIN .05.
3. Robust across held-out entities, US and India, three densities, corrected and pessimistic accounting.
4. Submit subA (validated, candidate_pairs identical to baseline).
5. python pilot/final_optimization/10_final_candidates/apply_decision_layer.py pilot/final_optimization/10_final_candidates/subA_thr085_margin005 0.85 0.05 0.95 0.70
