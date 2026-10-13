"""Phase 6 evaluation: do the AI agents' recommendations actually prevent
real failures?

For every real machine failure with an observable cause (TWF, HDF, PWF,
OSF), the agents plan from the recorded sensor state, using a model and
operating levels fitted on the other folds (5-fold cross-validation, same
split as Phases 3-4). The top recommendation is then applied to the TRUE
generating mechanism (vercausa/ground_truth.py): it counts as a success only
if no documented failure rule fires afterwards -- not the one that happened,
and no new one either. The agents never see these rules.

Baseline: "replace the tool", the standard response of a maintenance
system that predicts a failure but cannot say why.

Recommendations that cool the air to prevent HDF are reported separately as
UNVERIFIABLE, for the reason given in evaluate_simulation.py: the documented
generator and the data disagree on how process temperature follows air
temperature, so there is no ground truth for them.

    python scripts/evaluate_actions.py
"""

import collections
import pathlib
import sys
import time
import warnings

from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from vercausa.agents import ActionAgents, operating_levels
from vercausa.causal import CausalLayer, training_frame
from vercausa.ground_truth import CODE_TO_VARIANT, true_counterfactual, true_outcome
from vercausa.kg import open_store
from vercausa.readings import DEFAULT_CSV
from vercausa.schema import SENSOR_SIGNALS
from vercausa.simulation import SimulationLayer

MODES = ("TWF", "HDF", "PWF", "OSF")
SIGNALS = [s.name for s in SENSOR_SIGNALS]


def safe(state: dict) -> bool:
    return not any(true_outcome(m, state) for m in MODES)


def main() -> None:
    data = training_frame(DEFAULT_CSV)
    store = open_store()
    store.apply_schema()

    cases = []
    started = time.time()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in folds.split(data, data["machine_failure"]):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        causal = CausalLayer.fit_from_store(store, train)
        typical = {s: float(train[s].median()) for s in SIGNALS}
        agents = ActionAgents(causal, SimulationLayer(causal), typical, operating_levels(train))
        failures = test[(test["machine_failure"] == 1) & test[list(MODES)].any(axis=1)]
        for _, row in failures.iterrows():
            obs = {s: float(row[s]) for s in SIGNALS + ["product_variant"]}
            values = dict(obs, product_variant=CODE_TO_VARIANT[int(row["product_variant"])])
            modes = [m for m in MODES if row[m] == 1]
            plan = agents.plan(values, modes)
            top = plan["recommendations"][0] if plan["recommendations"] else None
            cases.append({
                "modes": modes,
                "top": top,
                "works": top is not None and safe(true_counterfactual(obs, top["interventions"])),
                "any_shown_works": any(safe(true_counterfactual(obs, a["interventions"]))
                                       for a in plan["recommendations"]),
                "baseline_works": safe(true_counterfactual(obs, {"tool_wear": 0.0})),
                "unverifiable": top is not None and "HDF" in modes and "air_temperature" in top["interventions"],
                "combined": plan["combined"],
            })

    scored = [c for c in cases if not c["unverifiable"]]

    def line(label, subset):
        if not subset:
            return
        n = len(subset)
        planned = sum(c["top"] is not None for c in subset)
        works = sum(c["works"] for c in subset)
        any_ok = sum(c["any_shown_works"] for c in subset)
        base = sum(c["baseline_works"] for c in subset)
        print(f"  {label:<26} {n:>4}   plan found {planned:>4}   top action works {works:>4} ({100 * works / n:5.1f}%)"
              f"   one of top 3 works {any_ok:>4} ({100 * any_ok / n:5.1f}%)   replace-tool baseline {base:>4} ({100 * base / n:5.1f}%)")

    print(f"Real failures with an observable cause: {len(cases)}  (planned in {time.time() - started:.0f}s incl. training)")
    print("A recommendation 'works' if, under the TRUE documented mechanisms, no failure rule fires after it.\n")
    line("ALL (scored)", scored)
    for m in MODES:
        line(f"  {m} failures", [c for c in scored if m in c["modes"]])
    line("  needed the re-planner", [c for c in scored if c["combined"]])

    effort = collections.Counter(c["top"]["effort_label"] for c in scored if c["top"])
    print("\nWhat the top recommendation asks for:")
    for label, n in effort.most_common():
        print(f"  {label:<32} {n}")

    unver = [c for c in cases if c["unverifiable"]]
    print(f"\nUNVERIFIABLE -- top action cools the air to prevent HDF: {len(unver)} cases (not scored; "
          "see evaluate_simulation.py).")
    store.close()


if __name__ == "__main__":
    main()
