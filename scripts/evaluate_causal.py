"""Phase 3 evaluation: does the KG-constrained SCM attribute real failures to
the correct documented mechanism?

5-fold stratified cross-validation over all 10,000 rows: every one of the 339
machine failures is attributed by a model that never saw that row in
training. Ground truth is AI4I's per-row mechanism labels -- used here for
scoring only; the detection rules' formulas are never given to the model.
"""

import pathlib
import sys
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from vercausa.causal import NO_OBSERVABLE_CAUSE, CausalLayer, training_frame
from vercausa.kg import open_store
from vercausa.readings import DEFAULT_CSV

OBSERVABLE = ("TWF", "HDF", "PWF", "OSF")


def main() -> None:
    data = training_frame(DEFAULT_CSV)
    store = open_store()
    store.apply_schema()

    results = []
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in folds.split(data, data["machine_failure"]):
        layer = CausalLayer.fit_from_store(store, data.iloc[train_idx])
        test = data.iloc[test_idx]
        for _, row in test[test["machine_failure"] == 1].iterrows():
            results.append((row, layer.attribute(row)))

    observable = [(r, a) for r, a in results if any(r[m] for m in OBSERVABLE)]
    unobservable = [(r, a) for r, a in results if not any(r[m] for m in OBSERVABLE)]

    def truth(row):
        return {m for m in OBSERVABLE if row[m]}

    def recovered(attr):
        return {m for m, v in attr.responsibility.items() if v >= 0.5 and m != NO_OBSERVABLE_CAUSE}

    top1 = sum(a.top.mode in truth(r) for r, a in observable)
    exact = sum(recovered(a) == truth(r) for r, a in observable)
    wrongly_unexplained = sum(not a.explained for _, a in observable)
    flagged = sum(not a.explained for _, a in unobservable)

    print(f"Machine failures attributed out-of-fold: {len(results)}")
    print(f"\nFailures with a documented observable mechanism: {len(observable)}")
    print(f"  Top-1 mechanism correct:      {top1}/{len(observable)} ({100 * top1 / len(observable):.1f}%)")
    print(f"  Exact mechanism set correct:  {exact}/{len(observable)} ({100 * exact / len(observable):.1f}%)"
          "   <- includes the multi-mechanism failures")
    print(f"  Wrongly flagged unexplained:  {wrongly_unexplained}")

    print("\n  Recall per mechanism (responsibility >= 0.5):")
    for m in OBSERVABLE:
        rows = [(r, a) for r, a in observable if r[m]]
        hit = sum(a.responsibility[m] >= 0.5 for _, a in rows)
        print(f"    {m}: {hit}/{len(rows)}")

    print(f"\nFailures with NO observable mechanism (RNF-labelled or unlabelled): {len(unobservable)}")
    print(f"  Correctly flagged 'no observable cause': {flagged}/{len(unobservable)}")

    print("\nMisattributions:")
    misses = [(r, a) for r, a in observable if a.top.mode not in truth(r)]
    for r, a in misses or []:
        print(f"  row {int(r['source_row']):>5}  truth={sorted(truth(r))}  "
              f"model={a.top.mode} (p={a.top.probability:.3f})  "
              f"wear={r['tool_wear']:.0f}")
    if not misses:
        print("  none")

    print("\nEdge strengths (fit on all data, from the KG-permitted edge set only):")
    layer = CausalLayer.fit_from_store(store, data)
    for (src, dst), v in sorted(layer.edge_strengths(data).items(), key=lambda kv: (kv[0][1], -kv[1])):
        kind = "share of explanation" if dst in OBSERVABLE else "variance explained"
        print(f"  {src:<20} -> {dst:<20} {v:6.3f}  ({kind})")

    print("\nExample explanations:")
    shown = set()
    for r, a in results:
        m = a.top.mode
        if m in shown or m not in truth(r) or len(truth(r)) > 1:
            continue
        shown.add(m)
        drivers = ", ".join(
            f"{k} {v:+.2f}" for k, v in sorted(a.top.contributions.items(), key=lambda kv: -abs(kv[1]))
        )
        print(f"  row {int(r['source_row']):>5}  {m}  p={a.top.probability:.3f}  drivers: {drivers}")

    store.close()


if __name__ == "__main__":
    main()
