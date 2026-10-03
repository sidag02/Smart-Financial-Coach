## Data

636,475 scored train outflows, 1058 planted (0.166%), 7,930 user-months; amount_outlier 369, duplicate 336, new_merchant_large 353

## Components alone (precision, recall of all planted)

| Component | Threshold | Flags | Precision | TP by kind |
|---|---|---|---|---|
| Duplicate (exact repeat within 90 min) | - | 337 | 0.973 | {'duplicate': 328} |
| Amount vs own history, own spread only | 5 | 1537 | 0.155 | {'amount_outlier': 237, 'duplicate': 1} |
| Amount vs own history, own spread only | 8 | 246 | 0.504 | {'amount_outlier': 124} |
| Amount vs own history, population spread | 4 | 733 | 0.325 | {'amount_outlier': 238} |
| Amount vs own history, population spread | 5 | 375 | 0.416 | {'amount_outlier': 156} |
| Amount vs own history, population spread | 6 | 208 | 0.500 | {'amount_outlier': 104} |
| New merchant: population price ratio | 3 | 639 | 0.485 | {'new_merchant_large': 288, 'amount_outlier': 22} |
| New merchant: population price ratio | 5 | 379 | 0.765 | {'new_merchant_large': 268, 'amount_outlier': 22} |
| New merchant: population price ratio | 8 | 258 | 0.899 | {'new_merchant_large': 213, 'amount_outlier': 19} |

## Combined rules: best recall at a precision target (grid on the same users)

| Target | z | Ratio | Min $ | Flags | Per user-month | Precision | Recall | Recall (clear) | Duplicate | Amount outlier | New merchant | Reason accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.70 | 5 | 5 | 0 | 1091 | 0.138 | 0.709 | 0.732 | 0.736 | 0.976 | 0.482 | 0.759 | 0.972 |
| 0.75 | 6 | 4 | 250 | 963 | 0.121 | 0.759 | 0.691 | 0.695 | 0.976 | 0.339 | 0.788 | 0.971 |
| 0.80 | 6 | 6 | 0 | 883 | 0.111 | 0.807 | 0.674 | 0.678 | 0.976 | 0.341 | 0.734 | 0.969 |

## At the 0.70 operating point

- precision of `duplicate` flags: 0.973 (337 flags)
- precision of `new_merchant` flags: 0.765 (379 flags)
- precision of `amount_unusual` flags: 0.416 (375 flags)
- false positives by process: {'discretionary': 174, 'one_off': 143}
- most frequent false-positive merchants: {'kaiser permanente': 27, 'monthly service fee': 13, 'overdraft item fee': 11, 'ticketmaster': 8, 'airbnb': 8, 'united airlines': 6}
- missed `new_merchant_large`: 85; 23 with no population price; median price ratio of the rest 1.69
- missed `amount_outlier`: 191; median z 3.81; tiers {'clear': 185, 'weak': 6}
- user bootstrap, 95%: precision 0.682-0.736, recall 0.705-0.759

## Baseline at the same flag volume

- Per-user z on amount (Technical Design baseline): 1091 flags, precision 0.044, recall 0.045
- Baseline plus the duplicate rule: 1091 flags, precision 0.342, recall 0.353
