# Test vs benchmark distribution shift (label-free)
Pool records per S1 per source: train 2.28/2.40 (S2/S3) in both countries; test 2.86/2.97 India, 2.82/2.93 US, 2.71/2.82 France (+25%).
Candidate score distribution (share of candidate pairs per score bin [0,.05,.2,.5,.65,.8,.9,.97,1]):
 BENCH US    .801 .019 .010 .003 .004 .004 .009 .150 | >=.65 per S1 3.35 (truth 3.45)
 BENCH India .800 .020 .012 .004 .006 .008 .015 .136 | >=.65 per S1 3.29 (truth 3.41)
 TEST  US    .768 .036 .022 .007 .007 .007 .011 .143 | >=.65 per S1 3.34
 TEST  India .700 .062 .043 .015 .016 .014 .020 .130 | >=.65 per S1 3.60
 TEST  France.605 .110 .057 .017 .017 .013 .015 .165 | >=.65 per S1 4.22
Mass in the ambiguous band (.2-.97): US 1.8x, India 2.4x, France ~3x the benchmark. US test looks like the benchmark; India and France do not.
Our predicted pairs in the .65-.95 band: France 61% house-number conflict (US 35%, India 29%).
