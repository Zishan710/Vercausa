"""Phase 8 evaluation: causal (Vercausa) vs correlation-based predictive maintenance.

Three models, same 5-fold split as Phases 3-6, same targets (the per-mode
failure labels):

  Vercausa         the KG-constrained structural causal model (Phases 3-4)
  Correlational    one gradient-boosted classifier per failure mode on all
                   raw sensor signals -- standard predictive maintenance
  Corr. + physics  the same, plus the two physics features Vercausa uses
                   (power, temperature difference), to separate what the
                   features contribute from what the causal structure does

All three use the same classifier family (scikit-learn
HistGradientBoostingClassifier), so differences come from structure, not
from the learner. Compared on:

  1. Predicting failures (ROC AUC, average precision), out of fold.
  2. Root cause: is the signal the model blames one the documented rule
     actually reads? For the correlational models, per-failure Shapley
     values over the raw signals against a healthy baseline.
  3. What-if: as in evaluate_simulation.py -- change a TRUE cause to 5
     observed levels, compare with the true mechanism. Then the correlation
     trap: change a signal that is NOT a cause; the truth is that the failure
     still happens. A correlational model just re-scores the edited input;
     Vercausa runs the counterfactual through the causal graph.
  4. Actions: the same planning procedure (observed settings, singles then
     pairs, same ranking) driven by each model; the top action is scored
     against the true mechanisms. The correlational planner has no knowledge
     graph, so it may change any operator-adjustable signal.

HDF via air temperature is excluded from 3 for every model (unverifiable,
see evaluate_simulation.py).

    python scripts/evaluate_baseline.py     # about an hour (Shapley values for every failure)
"""

import itertools
import math
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from vercausa.agents import ACTIONABLE, AT_RISK, LOW_RESIDUAL, TOP_PER_SIGNAL, ActionAgents, operating_levels
from vercausa.causal import CausalLayer, training_frame
from vercausa.ground_truth import CODE_TO_VARIANT, TRUE_CAUSES, true_counterfactual, true_outcome
from vercausa.kg import open_store
from vercausa.readings import DEFAULT_CSV
from vercausa.schema import compute_derived
from vercausa.simulation import SimulationLayer

MODES = ("TWF", "HDF", "PWF", "OSF")
WHATIF_MODES = ("HDF", "PWF", "OSF")
RAW = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear", "product_variant"]
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
SHAPLEY_BASELINE = 100
NAMES = ("Vercausa", "Correlational", "Corr. + physics")


class Correlational:
    """Standard predictive maintenance: a classifier per mode on all signals."""

    def __init__(self, train: pd.DataFrame, physics: bool):
        self.physics = physics
        X = self.features(train)
        self.models = {m: HistGradientBoostingClassifier(random_state=0).fit(X, train[m].astype(int)) for m in MODES}

    def features(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[RAW].astype(float).copy()
        if self.physics:
            X["power"] = compute_derived("power", X)
            X["temperature_difference"] = compute_derived("temperature_difference", X)
        return X

    def prob(self, mode: str, frame: pd.DataFrame) -> np.ndarray:
        return self.models[mode].predict_proba(self.features(frame))[:, 1]

    def whatif(self, obs: dict, changes: dict) -> pd.DataFrame:
        # No causal graph: an intervention just edits the input.
        return pd.DataFrame([dict(obs, **changes)])

    def top_driver(self, mode: str, obs: dict, baseline: pd.DataFrame) -> str:
        """Exact interventional Shapley values over the raw signals."""
        n = len(RAW)
        subsets = list(itertools.product([0, 1], repeat=n))
        rows = []
        for mask in subsets:
            b = baseline[RAW].copy()
            for keep, s in zip(mask, RAW):
                if keep:
                    b[s] = obs[s]
            rows.append(b)
        p = self.prob(mode, pd.concat(rows, ignore_index=True)).reshape(len(subsets), len(baseline)).mean(axis=1)
        value = dict(zip(subsets, p))
        phi = {}
        for i, s in enumerate(RAW):
            total = 0.0
            for mask in subsets:
                if mask[i]:
                    continue
                k = sum(mask)
                with_i = mask[:i] + (1,) + mask[i + 1:]
                total += math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) * (value[with_i] - value[mask])
            phi[s] = total
        return max(phi, key=phi.get)


class Causal:
    """Vercausa's model behind the same interface."""

    def __init__(self, causal: CausalLayer):
        self.causal, self.sim = causal, SimulationLayer(causal)

    def prob(self, mode: str, frame: pd.DataFrame) -> np.ndarray:
        return self.causal.failure_probability(mode, frame)

    def whatif(self, obs: dict, changes: dict) -> pd.DataFrame:
        return pd.DataFrame([self.sim.counterfactual_state(obs, changes)])

    def top_driver(self, mode: str, obs: dict, baseline=None) -> str:
        c = self.causal.contributions(mode, pd.Series(obs))
        return max(c, key=c.get)


def safe(state: dict) -> bool:
    return not any(true_outcome(m, state) for m in MODES)


def correlational_plan(model: Correlational, obs: dict, levels: dict) -> dict:
    """Same planning procedure as the Action Layer, without the knowledge graph."""

    def score(changes):
        frame = model.whatif(obs, changes)
        risks = {m: float(model.prob(m, frame)[0]) for m in MODES}
        any_after = 1 - math.prod(1 - p for p in risks.values())
        size = sum(abs(v - obs[x]) / levels[x]["span"] for x, v in changes.items())
        return {"changes": changes, "risks": risks, "resolves": all(p < AT_RISK for p in risks.values()),
                "any_after": any_after, "effort": max(ACTIONABLE[x][0] for x in changes), "size": size}

    def key(a):
        return (not a["resolves"], a["any_after"] >= LOW_RESIDUAL, a["effort"], len(a["changes"]), a["size"],
                a["any_after"])

    singles = [score({s: v}) for s in ACTIONABLE for v in levels[s]["levels"]
               if abs(v - obs[s]) > 0.01 * levels[s]["span"]]
    results = singles
    if not any(a["resolves"] and a["effort"] < 3 for a in singles):
        top = {}
        for a in singles:
            top.setdefault(next(iter(a["changes"])), []).append(a)
        top = {s: sorted(acts, key=lambda a: (max(a["risks"].values()), a["any_after"]))[:TOP_PER_SIGNAL]
               for s, acts in top.items()}
        results = results + [score(dict(**a["changes"], **b["changes"]))
                             for x, y in itertools.combinations(sorted(top), 2) for a in top[x] for b in top[y]]
    return min(results, key=key)


def main() -> None:
    data = training_frame(DEFAULT_CSV)
    store = open_store()
    store.apply_schema()
    started = time.time()

    oof = {name: {m: np.zeros(len(data)) for m in MODES} for name in NAMES}
    root, whatif_true, whatif_trap, actions = ({n: [] for n in NAMES} for _ in range(4))

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in folds.split(data, data["machine_failure"]):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        causal = CausalLayer.fit_from_store(store, train)
        models = {NAMES[0]: Causal(causal), NAMES[1]: Correlational(train, physics=False),
                  NAMES[2]: Correlational(train, physics=True)}
        levels = operating_levels(train)
        typical = {s: float(train[s].median()) for s in RAW if s != "product_variant"}
        agents = ActionAgents(causal, models[NAMES[0]].sim, typical, levels)
        baseline = train[train["machine_failure"] == 0].sample(n=SHAPLEY_BASELINE, random_state=0)

        for name, model in models.items():
            for m in MODES:
                oof[name][m][test_idx] = model.prob(m, test)

        failures = test[(test["machine_failure"] == 1) & test[list(MODES)].any(axis=1)]
        for _, row in failures.iterrows():
            obs = {s: float(row[s]) for s in RAW}
            modes = [m for m in MODES if row[m] == 1]

            # 2. root cause (single-mechanism failures)
            if len(modes) == 1:
                for name, model in models.items():
                    root[name].append(model.top_driver(modes[0], obs, baseline) in TRUE_CAUSES[modes[0]])

            # 3. what-if
            for m in [x for x in modes if x in WHATIF_MODES]:
                for s in RAW:
                    if m == "HDF" and s == "air_temperature":
                        continue
                    grid = [0.0, 1.0, 2.0] if s == "product_variant" else [float(train[s].quantile(q)) for q in QUANTILES]
                    for v in grid:
                        truth = true_outcome(m, true_counterfactual(obs, {s: v}))
                        for name, model in models.items():
                            says = model.prob(m, model.whatif(obs, {s: v}))[0] >= 0.5
                            (whatif_true if s in TRUE_CAUSES[m] else whatif_trap)[name].append((truth, says))

            # 4. actions
            values = dict(obs, product_variant=CODE_TO_VARIANT[int(obs["product_variant"])])
            plan = agents.plan(values, modes)
            top = plan["recommendations"][0]["interventions"] if plan["recommendations"] else {}
            actions[NAMES[0]].append(bool(top) and safe(true_counterfactual(obs, top)))
            for name in NAMES[1:]:
                best = correlational_plan(models[name], obs, levels)
                actions[name].append(safe(true_counterfactual(obs, best["changes"])))

    def pct(xs):
        return f"{100 * sum(xs) / len(xs):5.1f}%" if xs else "   n/a"

    print(f"Out-of-fold over {len(data):,} readings; {time.time() - started:.0f}s\n")
    print(f"{'':34}" + "".join(f"{n:>17}" for n in NAMES))

    print("\n1. Predicting failures (out of fold)")
    for m in MODES:
        y = data[m].astype(int)
        print(f"  {m} ROC AUC / avg precision".ljust(34) + "".join(
            f"{roc_auc_score(y, oof[n][m]):>9.3f} / {average_precision_score(y, oof[n][m]):.3f}" for n in NAMES))

    print("\n2. Root cause: blamed signal is one the true rule reads")
    print(f"  single-mechanism failures ({len(root[NAMES[0]])})".ljust(34)
          + "".join(f"{pct(root[n]):>17}" for n in NAMES))

    print("\n3. What-if answers vs the true mechanism (failure predicted at p >= 0.5)")
    print(f"  change a true cause ({len(whatif_true[NAMES[0]])})".ljust(34)
          + "".join(f"{pct([t == s for t, s in whatif_true[n]]):>17}" for n in NAMES) + "   agreement")
    print(f"  change a NON-cause ({len(whatif_trap[NAMES[0]])})".ljust(34)
          + "".join(f"{pct([t == s for t, s in whatif_trap[n]]):>17}" for n in NAMES) + "   agreement")
    print("  -> wrongly says a non-cause change".ljust(34)
          + "".join(f"{pct([t and not s for t, s in whatif_trap[n]]):>17}" for n in NAMES) + "   (lower is better)")
    print("     prevents the failure")

    print("\n4. Top recommended action truly prevents every failure")
    print(f"  real failures ({len(actions[NAMES[0]])})".ljust(34) + "".join(f"{pct(actions[n]):>17}" for n in NAMES))
    store.close()


if __name__ == "__main__":
    main()
