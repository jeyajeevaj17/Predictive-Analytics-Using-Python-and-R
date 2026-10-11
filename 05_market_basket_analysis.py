# Experiment 5: Market Basket Analysis using Association Rules
# Uses only pandas; no extra association-rule package is required.
from itertools import combinations
from collections import Counter
import pandas as pd

transactions = [
    ["Bread", "Milk"], ["Bread", "Diapers", "Beer", "Eggs"],
    ["Milk", "Diapers", "Beer", "Cola"], ["Bread", "Milk", "Diapers", "Beer"],
    ["Bread", "Milk", "Diapers", "Cola"], ["Bread", "Eggs"],
    ["Milk", "Eggs"], ["Bread", "Milk", "Beer"]
]
transaction_sets = [set(t) for t in transactions]
n = len(transaction_sets)
all_items = sorted(set().union(*transaction_sets))
support_count = Counter()
for t in transaction_sets:
    for size in range(1, len(t) + 1):
        for combo in combinations(sorted(t), size):
            support_count[frozenset(combo)] += 1

support = {itemset: count / n for itemset, count in support_count.items()}
frequent = {s: v for s, v in support.items() if v >= 0.25}

print("MARKET BASKET ANALYSIS")
print("Frequent itemsets (minimum support = 0.25):")
rows = [{"itemset": ", ".join(sorted(s)), "support": round(v, 3)}
        for s, v in frequent.items()]
print(pd.DataFrame(rows).sort_values(["support", "itemset"], ascending=[False, True]).to_string(index=False))

rule_rows = []
for itemset, sup_union in frequent.items():
    if len(itemset) < 2:
        continue
    items = sorted(itemset)
    for size in range(1, len(items)):
        for lhs_tuple in combinations(items, size):
            lhs = frozenset(lhs_tuple)
            rhs = itemset - lhs
            if lhs not in support or rhs not in support:
                continue
            confidence = sup_union / support[lhs]
            lift = confidence / support[rhs]
            if confidence >= 0.5:
                rule_rows.append({
                    "antecedent": ", ".join(sorted(lhs)),
                    "consequent": ", ".join(sorted(rhs)),
                    "support": round(sup_union, 3),
                    "confidence": round(confidence, 3),
                    "lift": round(lift, 3)
                })
print("\\nAssociation rules (minimum confidence = 0.5):")
if rule_rows:
    print(pd.DataFrame(rule_rows).sort_values(
        ["confidence", "lift"], ascending=False
    ).to_string(index=False))
else:
    print("No rules met the selected threshold.")
