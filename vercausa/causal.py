"""Causal Layer: an SCM whose structure comes from the Knowledge Graph and
whose mechanisms are learnt from data.

Structure is read from the KG's candidate graph -- never searched for
(CLAUDE.md, architecture decision #1). Three kinds of mechanism:
  - Derived quantities (temperature difference, power): exact physics
    formulas taken from the KG, no noise, nothing learnt.
  - Non-root signals (e.g. process temperature <- air temperature): linear
    additive-noise models learnt from data.
  - Failure modes: classifiers that see ONLY their KG-permitted parents and
    learn thresholds and shape from data. The detection rules' thresholds
    are never given to them, so agreement with the AI4I ground-truth labels
    means the model recovered the documented mechanisms, not that it copied
    them.

Known modelling limitation: rotational_speed and torque are strongly
(negatively) correlated because both come from the machine's power setting,
which is not measured. The KG correctly draws no arrow between them, so they
enter as independent roots; interventions on one hold the other fixed, which
is exactly what do() means, but combinations far from the observed data get
less reliable estimates -- `in_support` flags those.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from dowhy import gcm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from vercausa.kg import GraphStore
from vercausa.schema import FAILURE_MODES, SIGNALS, compute_derived

VARIANT_CODES = {"L": 0, "M": 1, "H": 2}
MODE_CODES = tuple(m.code for m in FAILURE_MODES)
NO_OBSERVABLE_CAUSE = "RNF"
SUPPORT_QUANTILE = 0.99
# Candidate minimum leaf sizes for failure-mode classifiers. Chosen per mode
# at fit time by cross-validated log-loss on the training data: sharp,
# well-populated mechanisms (PWF) want small leaves; sparse, noisy ones (TWF,
# failure minute random within the wear window) need heavy smoothing or the
# classifier memorises individual minutes.
LEAF_GRID = (20, 50, 100, 200, 400)


def _select_min_samples_leaf(X: np.ndarray, y: np.ndarray) -> int:
    if y.sum() < 3 * 2:
        return LEAF_GRID[-1]
    losses = {}
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    for leaf in LEAF_GRID:
        oof = np.zeros(len(y))
        for tr, te in folds.split(X, y):
            clf = HistGradientBoostingClassifier(min_samples_leaf=leaf).fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        losses[leaf] = log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))
    return min(losses, key=losses.get)

gcm.config.disable_progress_bars()


def training_frame(csv_path) -> pd.DataFrame:
    """AI4I rows as SCM training data: signals + mechanism labels + Machine failure."""
    raw = pd.read_csv(csv_path)
    df = pd.DataFrame({s.name: raw[s.csv_column] for s in SIGNALS})
    df["product_variant"] = df["product_variant"].map(VARIANT_CODES)
    for code in MODE_CODES:
        df[code] = raw[code].astype(int)
    df["machine_failure"] = raw["Machine failure"].astype(int)
    df["source_row"] = raw["UDI"]
    return df


class FormulaModel(gcm.ml.PredictionModel):
    """A derived quantity's physics formula, wrapped as a DoWhy mechanism."""

    def __init__(self, name: str, parents: list) -> None:
        self.name = name
        self.parents = parents

    def fit(self, X, Y) -> None:
        pass

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        cols = {p: X[:, i] for i, p in enumerate(self.parents)}
        return np.asarray(compute_derived(self.name, cols), dtype=float).reshape(-1, 1)

    def clone(self) -> "FormulaModel":
        return FormulaModel(self.name, list(self.parents))


@dataclass
class MechanismScore:
    mode: str
    probability: float
    contributions: dict = field(default_factory=dict)


@dataclass
class Attribution:
    ranking: list
    responsibility: dict

    @property
    def top(self) -> MechanismScore:
        return self.ranking[0]

    @property
    def explained(self) -> bool:
        return self.top.mode != NO_OBSERVABLE_CAUSE


class CausalLayer:
    def __init__(self, dag: nx.DiGraph, baseline: pd.DataFrame, rnf_rate: float) -> None:
        self.dag = dag
        self.model = gcm.ProbabilisticCausalModel(dag)
        self._baseline = baseline
        self._rnf_rate = rnf_rate
        self._derived = [n for n in nx.topological_sort(dag) if dag.nodes[n]["kind"] == "derived"]
        self._support: dict = {}
        self.min_samples_leaf: dict = {}

    @classmethod
    def fit_from_store(
        cls, store: GraphStore, data: pd.DataFrame, baseline_size: int = 300, seed: int = 0
    ) -> "CausalLayer":
        dag = store.candidate_graph()
        healthy = data[data["machine_failure"] == 0]
        baseline = healthy.sample(n=min(baseline_size, len(healthy)), random_state=seed)
        layer = cls(dag, baseline, rnf_rate=float(data["RNF"].mean()))
        fit_data = layer.with_derived(data)[list(dag.nodes)]
        for node in dag.nodes:
            if dag.nodes[node]["kind"] == "failure_mode" and layer.parents(node):
                layer.min_samples_leaf[node] = _select_min_samples_leaf(
                    fit_data[layer.parents(node)].to_numpy(dtype=float),
                    fit_data[node].to_numpy(dtype=int),
                )
        layer._assign_mechanisms()
        for node in layer.min_samples_leaf:
            fit_data[node] = fit_data[node].astype(str)
        gcm.fit(layer.model, fit_data)
        layer._fit_support(fit_data)
        return layer

    def save(self, path) -> None:
        """Save the fitted model. Loading needs the same library versions
        (pinned in requirements.txt), since fitted scikit-learn models are
        version-specific."""
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "CausalLayer":
        return joblib.load(path)

    def _assign_mechanisms(self) -> None:
        for node in self.dag.nodes:
            parents = self.parents(node)
            kind = self.dag.nodes[node]["kind"]
            if not parents:
                mech = gcm.EmpiricalDistribution()
            elif kind == "derived":
                mech = gcm.AdditiveNoiseModel(FormulaModel(node, parents))
            elif kind == "failure_mode":
                mech = gcm.ClassifierFCM(gcm.ml.SklearnClassificationModel(
                    HistGradientBoostingClassifier(min_samples_leaf=self.min_samples_leaf[node])
                ))
            else:
                mech = gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor())
            self.model.set_causal_mechanism(node, mech)

    def _fit_support(self, data: pd.DataFrame) -> None:
        for mode in MODE_CODES:
            parents = self.parents(mode)
            if not parents:
                continue
            X = data[parents].to_numpy(dtype=float)
            scale = X.std(axis=0)
            scale[scale == 0] = 1.0
            # Spacing is measured between DISTINCT operating points: signals
            # logged at fixed resolution (tool wear in whole minutes) repeat
            # exactly, which would otherwise make the typical spacing zero and
            # flag every in-between value as unseen.
            points = np.unique(X / scale, axis=0)
            nn = NearestNeighbors(n_neighbors=2).fit(points)
            if len(points) < 2:
                threshold = float("inf")
            else:
                dist, _ = nn.kneighbors(points)
                threshold = float(np.quantile(dist[:, 1], SUPPORT_QUANTILE))
            self._support[mode] = (parents, scale, nn, threshold)

    def parents(self, node: str) -> list:
        return sorted(self.dag.predecessors(node))

    def explanatory_signals(self, mode: str) -> list:
        """Measured signals a mode depends on: direct signal parents plus the
        inputs of any derived-quantity parents. Explanations are given in
        these terms -- "torque was high", not "power was high"."""
        out = set()
        for p in self.parents(mode):
            if self.dag.nodes[p]["kind"] == "derived":
                out.update(self.parents(p))
            else:
                out.add(p)
        return sorted(out)

    def with_derived(self, rows: pd.DataFrame, only: list | None = None) -> pd.DataFrame:
        rows = rows.copy()
        for name in self._derived:
            if only is None or name in only:
                rows[name] = compute_derived(name, rows)
        return rows

    def _inputs(self, mode: str, rows: pd.DataFrame) -> np.ndarray:
        parents = self.parents(mode)
        return self.with_derived(rows, only=parents)[parents].to_numpy(dtype=float)

    def failure_probability(self, mode: str, rows: pd.DataFrame) -> np.ndarray:
        """P(mode = 1 | its parents) for each row of signal values."""
        parents = self.parents(mode)
        if not parents:
            return np.full(len(rows), self._rnf_rate if mode == NO_OBSERVABLE_CAUSE else 0.0)
        mech = self.model.causal_mechanism(mode)
        probs = mech.estimate_probabilities(self._inputs(mode, rows))
        classes = list(mech.get_class_names(np.arange(probs.shape[1])))
        return probs[:, classes.index("1")] if "1" in classes else np.zeros(len(rows))

    def in_support(self, mode: str, rows: pd.DataFrame) -> np.ndarray:
        """True where this mode's inputs lie inside the observed training data.

        Outside it, the learnt mechanism is extrapolating: there were no
        comparable examples to learn from, so its answer is a guess.
        """
        if mode not in self._support:
            return np.ones(len(rows), dtype=bool)
        _, scale, nn, threshold = self._support[mode]
        X = self._inputs(mode, rows) / scale
        dist, _ = nn.kneighbors(X, n_neighbors=1)
        return dist[:, 0] <= threshold

    def contributions(self, mode: str, row: pd.Series) -> dict:
        """Exact Shapley values of each explanatory signal's actual value vs.
        healthy operation.

        v(S) = mean over healthy baseline rows of P(mode | signals in S set to
        this reading's values, the rest taken from the baseline row) -- an
        interventional (do-style) replacement; derived quantities are
        recomputed from the mixed signals. Values sum to
        P(mode | this reading) - P(mode | typical healthy operation).
        """
        signals = self.explanatory_signals(mode)
        if not signals:
            return {}
        base = self._baseline.reset_index(drop=True)

        def value(subset: tuple) -> float:
            mixed = base.copy()
            for s in subset:
                mixed[s] = row[s]
            return float(self.failure_probability(mode, mixed).mean())

        k = len(signals)
        cache = {s: value(s) for r in range(k + 1) for s in itertools.combinations(signals, r)}
        shapley = {}
        for s in signals:
            others = [q for q in signals if q != s]
            total = 0.0
            for r in range(k):
                for subset in itertools.combinations(others, r):
                    with_s = tuple(sorted(subset + (s,), key=signals.index))
                    weight = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
                    total += weight * (cache[with_s] - cache[subset])
            shapley[s] = total
        return shapley

    def attribute(self, row: pd.Series) -> Attribution:
        """Which mechanism most likely produced this failure, and which signals drove it.

        Given that a failure occurred, mechanism m is responsible with
        probability P(m | x) / P(any failure | x), treating mechanisms as
        independent causes of the machine stopping (the AI4I OR-structure).
        """
        frame = row.to_frame().T.astype(float)
        probs = {m: float(self.failure_probability(m, frame)[0]) for m in MODE_CODES}
        p_any = 1 - math.prod(1 - p for p in probs.values())
        responsibility = {m: (p / p_any if p_any > 0 else 0.0) for m, p in probs.items()}
        ranking = [
            MechanismScore(m, probs[m])
            for m in sorted(MODE_CODES, key=lambda m: probs[m], reverse=True)
        ]
        for score in ranking:
            if score is ranking[0] or responsibility[score.mode] >= 0.5:
                score.contributions = self.contributions(score.mode, row)
        return Attribution(ranking, responsibility)

    def edge_strengths(self, data: pd.DataFrame) -> dict:
        """How strongly each cause drives what it points at, estimated from data.

        (signal, failure_mode): across the rows where that mode actually
        occurred, the share of the explanation each explanatory signal
        carries (mean |Shapley contribution|, normalised to sum to 1 per
        mode). Event-level on purpose: DoWhy's population-level
        arrow_strength is dominated by the ~99% healthy rows and is unstable
        run to run for failure modes this rare. Keyed by the measured signal
        even when it reaches the mode through a derived quantity.

        (signal, signal): fraction of the child's variance its parents
        explain (R^2 of the fitted mechanism; joint if there are several).

        Derived quantities are exact formulas and have no strength to learn.
        """
        strengths = {}
        for node in self.dag.nodes:
            parents = self.parents(node)
            kind = self.dag.nodes[node]["kind"]
            if not parents or kind == "derived":
                continue
            if kind == "failure_mode":
                signals = self.explanatory_signals(node)
                totals = dict.fromkeys(signals, 0.0)
                for _, row in data[data[node] == 1].iterrows():
                    for s, v in self.contributions(node, row).items():
                        totals[s] += abs(v)
                norm = sum(totals.values()) or 1.0
                for s in signals:
                    strengths[(s, node)] = totals[s] / norm
            else:
                mech = self.model.causal_mechanism(node)
                predicted = mech.prediction_model.predict(data[parents].to_numpy(dtype=float))
                residual = data[node].to_numpy(dtype=float) - np.asarray(predicted).reshape(-1)
                r2 = 1 - residual.var() / data[node].var(ddof=0)
                for p in parents:
                    strengths[(p, node)] = max(0.0, float(r2))
        return strengths
