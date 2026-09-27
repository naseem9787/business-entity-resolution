# Friend TSV audit (label-free; script analyze_friend.py, metrics friend_audit.json)
- Friend: 3,432,263 pairs, 25.9% S1 empty, 1.98 matches/S1; S2 2.01M vs S3 1.42M (S3 pool is larger -> S3 under-recall).
- CERTAIN errors in friend file: 58,036 S2/S3 records are given to 2+ different S1 (216,615 pairs). Since a record has at most one owner, >=158,579 friend pairs (4.6%) are false merges -> friend precision <= 95.4%, contradicting the ">=97%" claim.
- Ours: 5,642,165 pairs, 5.96% empty, 3.26/S1, 0 exclusivity violations. Train prior: 3.46 true matches/S1, 5.6% singletons.
- Agreement: 2.98M common pairs (Jaccard 0.49). 568,700 S1 where friend is a strict subset of ours; 365,722 S1 friend-empty/ours-linked; only 19,571 ours-empty/friend-linked.
- Friend-only pairs (453,528): 54% are NOT in our final candidates (retrieval side), 46% are (median our score 0.29 = matcher rejected). They are over-represented in risky slices vs agreed pairs: house-number conflict 25% (vs 9%), empty address 13% (vs 1.3%), native script 16% (vs 1.2%), exact core name 37% (vs 63%). 25% of them go to a record that WE assign to a different S1.
- Ours-only pairs (2,663,430): our median score 0.996; friend gives the record to another S1 in only 0.9% of cases; 40% belong to S1 that friend leaves empty.
- 112,391 records are assigned to different S1 by the two systems (at least one wrong each).
Conclusion: the friend file is a low-recall system with measurable exclusivity violations, not a higher-precision oracle. Do not merge it; its "unique discoveries" are concentrated in the slices where both systems are least reliable.
