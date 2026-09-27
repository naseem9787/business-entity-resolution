# Test-vs-replay linkage shift of v2 (cached scores only; shift_analysis.py, additions_precision.py, band_precision.py)
v2(T=.80)/subA match ratio: replay US .990 India .985 | test US 1.054 India 1.041 France 1.028.
Two causes:
1. Fewer removals on test (explained, benign): on the replay v2 removes subA pairs that are ~55% precise, mostly empty-address records whose owner is outside the replay
   (replay artefact). On the test the owners are present, production's margin already blocks many of them -> v2 removes less (US .043 vs .126 per S1).
2. More additions on test (calibration risk): v2-only additions are concentrated in the types with low replay precision:
   production score <.2 (replay precision .78 owner-present) = 21-30% of test additions vs 5-11% on replay;
   production-contested (margin<.05, precision .58) = 11-24% of test additions vs 2-6% on replay.
Marginal-band precision (replay per-type precision re-weighted to the test composition; F0.5 break-even ~.75):
 band .70-.80: US ~.55, India ~.69, France ~.60 -> below break-even -> T=.70 harmful.
 band .80-.90: US ~.74, India ~.82, France ~.75 (count-weighted ~.78) -> at break-even: expected ~0, downside in US/France.
Comparison vs subA (5,313,551 matches, linked 93.11%, empty 119,391; US 3.06 / India 2.94 / France 3.46 per S1):
 T=.70 5,702,765 matches, linked 93.80%, empty 107,391, US 3.31 India 3.16 France 3.67
 T=.80 5,545,450 matches, linked 93.42%, empty 114,020, US 3.23 India 3.07 France 3.56
 T=.90 5,320,155 matches, linked 92.85%, empty ~123,950, US 3.12 India 2.93 France 3.40
Ownership: 0 within-file collisions in every file; subA vs T=.80 give 2,267 records to different S1 (US 771, India 1,164, France 332).
Recommendation: T=.90 (replay .9404 > subA rule .9365; same test volume as subA; drops only the at-break-even band). If it beats .929, T=.80 is the upside follow-up.
