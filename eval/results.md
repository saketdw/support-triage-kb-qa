# Evaluation results

## Retrieval — answerable buckets

| bucket | n | hit@1 | hit@3 | MRR | Wilson 95% CI on hit@1 |
|---|---|---|---|---|---|
| current | 10 | 10/10 | 10/10 | 1.00 | [72%, 100%] |
| historical | 4 | 4/4 | 4/4 | 1.00 | [51%, 100%] |
| active-notice | 2 | 1/2 | 2/2 | 0.75 | [9%, 91%] |
| evergreen | 10 | 8/10 | 9/10 | 0.85 | [49%, 94%] |
| **overall** | 26 | **23/26 (88%)** | | | [71%, 96%] |

## Expired notices (answerable as a negative)

4/4 answered as a negative citing the expired document as evidence.

## Unanswerable bucket (gold = NONE)

| behavior | count | qids | reading |
|---|---|---|---|
| abstained (correct) | 2 | q33, q34 | escalated to a human |
| hedged (safe) | 1 | q32 | named the gap; extraction cannot invent facts |
| answered (wrong) | 5 | q31, q35, q36, q37, q38 | quoted a related but non-answering document |

## Abstention threshold sweep

The fixed threshold (0.10) is one point on this curve; the sweep is the honest picture.

| threshold | abstained | of which truly unanswerable | answerable lost |
|---|---|---|---|
| 0.05 | 0 | 0 | 0 |
| 0.10 | 3 | 2 | 1 |
| 0.15 | 6 | 4 | 2 |
| 0.20 | 9 | 5 | 4 |
| 0.25 | 22 | 8 | 14 |

*Buckets are small; every number above carries the interval shown or wider. The historical bucket (n=4) at 4/4 is consistent with a true rate as low as 51% — the KB's own metadata could generate hundreds more dated questions to tighten it (see README, 'with more time').*
