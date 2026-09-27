# Exp2 report - retrieval recovery (Exp1 outputs untouched; same 124,965-entity benchmark, same cross-fitting)
Absolute candidate recall @15/source: baseline A 0.9704 (reproduces Exp1) -> H learned ranker 0.9936 (India .9874, US .9992; empty-address pairs .563 -> .990).
Final F0.5 (stage-1 unchanged): Exp1 B0 0.9638 / E 0.9692.  H@15: B0 0.9660, A(d=.1) 0.9705, E 0.9707.  H@10: B0 0.9673, A(d=.05) 0.9711 (20 cands/entity).  H@5: B0 0.9663, A 0.9701.
Primary criterion (+0.002 over 0.9692): NOT met (best +0.0019, at 2/3 of the candidates). Recall gain does not convert: model-rejected pairs are now the bottleneck.
See results_*.json, ablation_summary.json, taxonomy.json, ranker_loco.json, kcurve.json.
