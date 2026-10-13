"""Simulation Layer: counterfactual "what if" over the Causal Layer's SCM.

The Digital Twin supplies the starting state; this layer does the
simulating (CLAUDE.md, architecture decision #2 -- twin = context,
SCM = simulation). Nothing here writes back to the twin.

Counterfactuals follow Pearl's three steps over the KG-permitted DAG:
  1. Abduction  -- for each non-root signal, recover the noise term that
     made its observed value differ from what its parents predict.
  2. Action     -- overwrite the intervened signals: do(X = x).
  3. Prediction -- recompute descendants in topological order, keeping the
     recovered noise, then re-score every failure mode's probability.

Signals use additive-noise mechanisms, so step 1 is exact. Derived
quantities (power, temperature difference) are recomputed from the
counterfactual signals by their physics formulas. Failure modes are binary
and their noise cannot be recovered from a single 0/1 observation, so they
are reported as probabilities under the counterfactual signal values.

Every result says whether it can be trusted: if a counterfactual lands where
the training data has no comparable examples, the learnt mechanism is
extrapolating and the mode is listed in `outside_observed`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Union

import networkx as nx
import numpy as np
import pandas as pd

from vercausa.causal import MODE_CODES, VARIANT_CODES, CausalLayer
from vercausa.kg import GraphStore

Intervention = Union[float, Callable[[float], float]]


def _any_failure(risks: dict) -> float:
    return 1 - math.prod(1 - p for p in risks.values())


@dataclass
class SimulationResult:
    interventions: dict
    factual: dict
    counterfactual: dict
    derived_before: dict
    derived_after: dict
    risk_before: dict
    risk_after: dict
    outside_observed: list

    @property
    def trustworthy(self) -> bool:
        """False if any mode's counterfactual inputs are outside observed data,
        i.e. the model had nothing comparable to learn from there."""
        return not self.outside_observed

    @property
    def any_before(self) -> float:
        return _any_failure(self.risk_before)

    @property
    def any_after(self) -> float:
        return _any_failure(self.risk_after)

    def changed_signals(self) -> dict:
        return {
            k: (self.factual[k], v)
            for k, v in self.counterfactual.items()
            if not math.isclose(v, self.factual[k], abs_tol=1e-9)
        }


class SimulationLayer:
    def __init__(self, causal: CausalLayer) -> None:
        self.causal = causal
        dag = causal.dag
        self._signals = [n for n in nx.topological_sort(dag) if dag.nodes[n]["kind"] == "signal"]

    def _normalise(self, state: dict) -> dict:
        out = {}
        for s in self._signals:
            if s not in state:
                raise ValueError(f"state is missing signal '{s}'")
            v = state[s]
            out[s] = float(VARIANT_CODES[v]) if s == "product_variant" and isinstance(v, str) else float(v)
        return out

    def _predict_signal(self, node: str, values: dict) -> float:
        parents = self.causal.parents(node)
        mech = self.causal.model.causal_mechanism(node)
        x = np.array([[values[p] for p in parents]], dtype=float)
        return float(np.asarray(mech.prediction_model.predict(x)).reshape(-1)[0])

    def counterfactual_state(self, observed: dict, interventions: dict) -> dict:
        unknown = set(interventions) - set(self._signals)
        if unknown:
            raise ValueError(
                f"can only intervene on signals {self._signals}; got {sorted(unknown)}"
            )
        obs = self._normalise(observed)
        cf = {}
        for node in self._signals:
            if node in interventions:
                v = interventions[node]
                v = v(obs[node]) if callable(v) else v
                cf[node] = float(VARIANT_CODES[v]) if isinstance(v, str) else float(v)
            elif not self.causal.parents(node):
                cf[node] = obs[node]
            else:
                noise = obs[node] - self._predict_signal(node, obs)
                cf[node] = self._predict_signal(node, cf) + noise
        return cf

    def risks(self, state: dict) -> dict:
        frame = pd.DataFrame([state])
        return {m: float(self.causal.failure_probability(m, frame)[0]) for m in MODE_CODES}

    def _derived(self, state: dict) -> dict:
        row = self.causal.with_derived(pd.DataFrame([state])).iloc[0]
        return {d: float(row[d]) for d in self.causal._derived}

    def simulate(self, observed: dict, interventions: dict) -> SimulationResult:
        factual = self._normalise(observed)
        cf = self.counterfactual_state(observed, interventions)
        frame = pd.DataFrame([cf])
        outside = [m for m in MODE_CODES if not self.causal.in_support(m, frame)[0]]
        return SimulationResult(
            interventions=dict(interventions),
            factual=factual,
            counterfactual=cf,
            derived_before=self._derived(factual),
            derived_after=self._derived(cf),
            risk_before=self.risks(factual),
            risk_after=self.risks(cf),
            outside_observed=outside,
        )

    def simulate_twin(self, store: GraphStore, machine_id: str, interventions: dict) -> SimulationResult:
        """What if we changed X on this machine, starting from its live twin state."""
        state = store.machine_state(machine_id)
        observed = {sig: s["value"] for sig, s in state["sensors"].items()}
        observed["product_variant"] = state["product_variant"]
        return self.simulate(observed, interventions)
