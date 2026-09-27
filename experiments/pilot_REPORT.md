# Pilot experiment report (200k S1 entities; blocking benchmarked on 20k queries per country x source; matching on 40k queries, 12k validation)
Pilot pool = every S2/S3 match of the 200k sampled S1 + a proportional (9%) sample of all other S2/S3 records. => distractor density is ~9% of the real data.

## A/B. Dataset
Train S1 2.21M (US 60%, India 40%), S2 5.03M, S3 5.29M. Test S1 1.73M (US 663k, India 810k, France 259k), S2 4.89M, S3 5.08M.
Matches/S1: 0:5.6% 1:5.4% 2:17.0% 3:24.1% 4:21.9% 5:14.6% 6:7.5% 7:2.9% 8+:1.0% (mean 3.5). S2-only 6.5%, S3-only 7.5%.
Each S2/S3 record belongs to at most one S1. ~27% of S2 / 25% of S3 records have NO owner (pure distractors). 30% of S1 names are duplicated (chains).
Native-script names (India): S1 0%, S2 13%, S3 7%. Empty address: S1 0%, S2/S3 3.4%. PIN/ZIP essentially absent (0.2% India, 8% US).
## C/D. Blocking (recall of true S1->S2/S3 pairs; mean candidates/entity)
see pilot/summ_block.py output; best = fused union of word+char4 TF-IDF on name and address (address-weighted for India).
## Matching / thresholds / ablations: see pilot/log_match_K30.txt, log_analysis.txt, log_group.txt
