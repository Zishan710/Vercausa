"""Phase 8 evaluation: how often does Vercausa raise a false alarm?

Two alarms exist:

1. Detection Layer rules (open a failure event). HDF / PWF / OSF match the
   dataset's labels exactly (validate_detection.py), so they raise no false
   alarms there. The TWF rule fires on every reading inside the 200-240 min
   window, while the dataset fails the tool at one random minute of it, so
   its "false alarms" are readings where the tool is due but has not yet
   failed. Counted below.

2. The Action Layer's early warning: a failure mode is "at risk" when the
   causal model gives it >= 1% per reading (vercausa/agents.py). Scored out
   of fold (5-fold, same split as the other phases) against the labels:
   how many real failures were warned of, and how many healthy readings
   were flagged.

    python scripts/evaluate_alarms.py
"""

import pathlib
import sys
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from vercausa.agents import AT_RISK
from vercausa.causal import CausalLayer, training_frame
from vercausa.detection import TWF_WINDOW_MIN, check_twf
from vercausa.kg import open_store
from vercausa.readings import DEFAULT_CSV

MODES = ("TWF", "HDF", "PWF", "OSF")


def main() -> None:
    data = training_frame(DEFAULT_CSV)
    store = open_store()
    store.apply_schema()
    healthy = (data["machine_failure"] == 0).to_numpy()

    print("1. Detection rule for TWF (the only rule that does not match the labels exactly)")
    fires = data["tool_wear"].map(check_twf).to_numpy()
    twf = (data["TWF"] == 1).to_numpy()
    print(f"  readings in the {TWF_WINDOW_MIN[0]}-{TWF_WINDOW_MIN[1]} min window: {fires.sum()}; "
          f"real tool wear failures among them: {(fires & twf).sum()} of {twf.sum()}")
    print(f"  healthy readings flagged: {(fires & healthy).sum()} of {healthy.sum()} "
          f"({100 * (fires & healthy).mean() / healthy.mean():.2f}%) -- tools due for replacement")

    oof = {m: np.zeros(len(data)) for m in MODES}
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in folds.split(data, data["machine_failure"]):
        causal = CausalLayer.fit_from_store(store, data.iloc[train_idx])
        test = data.iloc[test_idx]
        for m in MODES:
            oof[m][test_idx] = causal.failure_probability(m, test)

    print(f"\n2. Early warning: mode at risk when p >= {AT_RISK:.0%} per reading (out of fold)")
    print(f"  {'':6}{'failures warned':>18}{'healthy flagged':>22}{'warnings that were real':>26}")
    for m in MODES:
        warn = oof[m] >= AT_RISK
        real = (data[m] == 1).to_numpy()
        print(f"  {m:<6}{f'{(warn & real).sum()}/{real.sum()}':>18}"
              f"{f'{(warn & healthy).sum()} ({100 * (warn & healthy).sum() / healthy.sum():.2f}%)':>22}"
              f"{f'{100 * (warn & real).sum() / max(warn.sum(), 1):.1f}%':>26}")

    any_warn = np.any([oof[m] >= AT_RISK for m in MODES], axis=0)
    observable = data[list(MODES)].any(axis=1).to_numpy()
    print(f"\n  Any mode at risk vs a failure with an observable cause:")
    print(f"    failures warned:  {(any_warn & observable).sum()}/{observable.sum()}")
    print(f"    healthy flagged:  {(any_warn & healthy).sum()}/{healthy.sum()} "
          f"({100 * (any_warn & healthy).sum() / healthy.sum():.2f}%)")
    worn = (data["tool_wear"] >= TWF_WINDOW_MIN[0]).to_numpy()
    print(f"      of which the tool is past {TWF_WINDOW_MIN[0]} min (due for replacement): "
          f"{(any_warn & healthy & worn).sum()}")
    print(f"      other healthy readings flagged: {(any_warn & healthy & ~worn).sum()} "
          f"({100 * (any_warn & healthy & ~worn).sum() / healthy.sum():.2f}%)")
    store.close()


if __name__ == "__main__":
    main()
