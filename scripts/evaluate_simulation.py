"""Phase 4 evaluation: do the SCM's "what if" answers match what the true
AI4I mechanisms say would have happened?

For every real HDF / PWF / OSF failure, intervene on each of that mode's
KG-permitted causes at 5 realistic levels (5th-95th percentile of observed
values; L/M/H for product variant), and compare the simulated outcome with
the outcome under the TRUE generating mechanism:
  - the documented failure rule, applied to
  - the true counterfactual state: in AI4I, process temperature is air
    temperature + 10K + noise, so do(air = a) shifts process temperature by
    the same amount; product variant sets tool wear *rate*, so changing it
    leaves the current tool wear unchanged.

The rules and generator are used only as the scoring oracle -- the model
never sees them. TWF is excluded: its true failure minute is random inside
the wear window, so there is no deterministic counterfactual to score.
5-fold cross-validation, same as Phase 3.

HDF via air temperature is reported separately as UNVERIFIABLE: the
documentation says process temperature follows air temperature 1:1, but the
data says ~0.65:1 (in levels and in row-to-row changes alike). The generator
code is not published, so there is no ground truth to score against.
"""

import collections
import pathlib
import sys
import warnings

import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from vercausa.causal import VARIANT_CODES, CausalLayer, training_frame
from vercausa.detection import check_hdf, check_osf, check_pwf
from vercausa.kg import open_store
from vercausa.readings import DEFAULT_CSV
from vercausa.simulation import SimulationLayer

MODES = ("HDF", "PWF", "OSF")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
CODE_TO_VARIANT = {v: k for k, v in VARIANT_CODES.items()}


def true_counterfactual(obs: dict, signal: str, value: float) -> dict:
    cf = dict(obs)
    cf[signal] = value
    if signal == "air_temperature":
        cf["process_temperature"] = obs["process_temperature"] + (value - obs["air_temperature"])
    return cf


def true_outcome(mode: str, s: dict) -> bool:
    if mode == "HDF":
        return check_hdf(s["air_temperature"], s["process_temperature"], s["rotational_speed"])
    if mode == "PWF":
        return check_pwf(s["torque"], s["rotational_speed"])
    return check_osf(s["tool_wear"], s["torque"], CODE_TO_VARIANT[int(s["product_variant"])])


def main() -> None:
    data = training_frame(DEFAULT_CSV)
    store = open_store()
    store.apply_schema()
    signals = [c for c in data.columns if c not in ("TWF", "HDF", "PWF", "OSF", "RNF",
                                                     "machine_failure", "source_row")]

    pairs = []
    factual_hits = collections.Counter()
    factual_totals = collections.Counter()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in folds.split(data, data["machine_failure"]):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        causal = CausalLayer.fit_from_store(store, train)
        sim = SimulationLayer(causal)
        for mode in MODES:
            events = test[(test[mode] == 1) & (test["machine_failure"] == 1)]
            for _, row in events.iterrows():
                obs = {s: float(row[s]) for s in signals}
                factual_totals[mode] += 1
                factual_hits[mode] += causal.failure_probability(mode, pd.DataFrame([obs]))[0] >= 0.5
                grid = []
                for signal in causal.explanatory_signals(mode):
                    levels = ([0.0, 1.0, 2.0] if signal == "product_variant"
                              else [float(train[signal].quantile(q)) for q in QUANTILES])
                    grid += [(signal, v) for v in levels]
                cf_frame = pd.DataFrame([sim.counterfactual_state(obs, {s: v}) for s, v in grid])
                probs = causal.failure_probability(mode, cf_frame)
                inside = causal.in_support(mode, cf_frame)
                for (signal, v), p, ins in zip(grid, probs, inside):
                    truth = true_outcome(mode, true_counterfactual(obs, signal, v))
                    pairs.append((mode, signal, bool(ins), truth, p >= 0.5))

    def report(label, subset):
        if not subset:
            return
        agree = sum(t == m for *_, t, m in subset)
        fixes = [m for *_, t, m in subset if not t]
        stays = [m for *_, t, m in subset if t]
        print(f"  {label:<30} agree {agree:>4}/{len(subset):<4} ({100 * agree / len(subset):5.1f}%)"
              f"   fix works: {sum(not m for m in fixes)}/{len(fixes)}"
              f"   fix fails: {sum(stays)}/{len(stays)}")

    unverifiable = [p for p in pairs if p[0] == "HDF" and p[1] == "air_temperature"]
    scored = [p for p in pairs if p not in unverifiable]

    print("Before any change -- does the model already see each real failure? (p >= 0.5)")
    for mode in MODES:
        print(f"  {mode}: {factual_hits[mode]}/{factual_totals[mode]}")

    print(f"\nWhat-if scenarios scored against the true mechanism: {len(scored)}")
    print("  'fix works' = truth says the change prevents the failure, and the model agrees")
    print("  'fix fails' = truth says it still fails, and the model agrees\n")
    report("ALL", scored)
    report("  inside observed data", [p for p in scored if p[2]])
    report("  OUTSIDE observed data", [p for p in scored if not p[2]])
    print()
    for mode in MODES:
        report(mode, [p for p in scored if p[0] == mode])
    print()
    for mode in MODES:
        for signal in sorted({p[1] for p in scored if p[0] == mode}):
            report(f"{mode} via {signal}", [p for p in scored if p[0] == mode and p[1] == signal])

    moved = sum(not m for *_, m in unverifiable)
    print(f"\nUNVERIFIABLE -- HDF via air_temperature ({len(unverifiable)} scenarios):")
    print(f"  model says the change prevents the failure in {moved}/{len(unverifiable)}; "
          "the documented 1:1 generator says never. The data's own 0.65:1 air->process")
    print("  relationship says cooling the air widens the temperature gap, so the model's answer")
    print("  is what this data supports. No ground truth exists to settle it.")

    store.close()


if __name__ == "__main__":
    main()
