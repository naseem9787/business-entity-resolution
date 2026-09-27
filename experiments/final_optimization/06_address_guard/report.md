# Street-number guard - labeled evidence (benchmark, same first-number parser)
TRUE pairs with conflicting first house number: US 12.0%, India 9.3% (generator perturbs numbers: ranges, dropped digits, inserted "H.no").
In the .65-.95 score band precision is NOT lower for conflicts (US conflict .876 / equal .845; India .871 / .867).
Among predicted pairs: conflict precision .966 vs equal .996 (US). A hard veto would delete ~10% of true pairs at ~96.6% precision -> net loss.
=> The friend's strict street-number veto is NOT supported on US/India labels. Possible exception: France (61% of its ambiguous-band predictions conflict) - untestable without France labels.
