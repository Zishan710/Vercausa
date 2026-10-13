"""Action Layer: AI agents that turn causal findings into ranked maintenance actions.

Five agents share one state and are wired together as a LangGraph workflow:

    diagnose → propose → simulate → rank ──(an easy action works)──→ done
                            ↑          └──(not yet)→ combine
                            └──────────────────────────────────────┘

1. Diagnose -- asks the Causal Layer which failure modes are at risk (for an
   event: its mode plus any other mode at risk) and what drove them.
2. Propose  -- builds candidate actions only on signals the knowledge graph
   permits as causes of those modes and that an operator can change, using
   settings the machine has actually run at (quantiles of observed data).
3. Simulate -- tests every candidate with the Simulation Layer: a
   counterfactual do() intervention on the state the twin recorded.
4. Rank     -- an action is recommended only if, in simulation, no failure
   mode is still at risk afterwards. Recommended actions are ordered: trusted
   results first, then low remaining risk, least disruption, fewest changes,
   smallest change. Only the best version of each distinct action is shown.
5. Combine  -- when no single action is enough (or the only ones that work
   are low-confidence or a facility change), pairs the most promising single
   actions on different signals and sends them back to Simulate.

Every step is deterministic and written to a trace, so a recommendation can
be explained and reproduced. No language model decides anything here; the
Reasoning Layer may put the result into words.
"""

from __future__ import annotations

import itertools
import warnings
from typing import TypedDict

import pandas as pd

with warnings.catch_warnings():
    # LangGraph's import emits a notice about a future LangChain default that
    # does not affect this module. LangChain re-enables its warnings when it is
    # imported, so it is muted after that and before LangGraph loads.
    import langchain_core  # noqa: F401
    warnings.simplefilter("ignore")
    from langgraph.graph import END, StateGraph

from vercausa.causal import CausalLayer
from vercausa.explain import MODE_NAMES, OBSERVABLE, explain_failure, model_row, whatif
from vercausa.simulation import SimulationLayer

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# A mode is "at risk" from 1% per reading. Tool wear failure builds up over
# many readings: its per-reading risk is ~0% below 190 min of wear and only
# 4-7% from 200 min on, so a 50% cut-off would never act on a worn tool. At
# 1%, every real failure in AI4I is flagged, while for the other modes at
# most 0.34% of healthy readings are (and for tool wear, the healthy readings
# flagged are tools already past 200 min).
AT_RISK = 0.01
LOW_RESIDUAL = 0.1
FACILITY = 3
TOP_PER_SIGNAL = 2
SHOWN = 3

# Signals an operator can change, how disruptive that is (1 = least), and how
# it is done. Process temperature is not here: it follows from the others.
ACTIONABLE = {
    "rotational_speed": (1, "setting change, no stop"),
    "torque": (1, "setting change, no stop"),
    "tool_wear": (2, "short stop to change the tool"),
    "air_temperature": (FACILITY, "facility change (cooling)"),
}
LABEL = {"rotational_speed": "spindle speed", "torque": "torque", "tool_wear": "tool wear",
         "air_temperature": "air temperature"}


def operating_levels(data: pd.DataFrame) -> dict:
    """Settings the machine has actually run at, per actionable signal.

    Candidate actions are drawn from these so the model is never asked about
    a setting it has no data for. For tool wear the only action is a new tool.
    """
    out = {}
    for s in ACTIONABLE:
        q = [float(data[s].quantile(p)) for p in LEVELS]
        out[s] = {"levels": [0.0] if s == "tool_wear" else q, "span": (q[-1] - q[0]) or 1.0}
    return out


def describe(signal: str, before: float, after: float) -> str:
    if signal == "tool_wear":
        return "Replace the cutting tool"
    up = after > before
    if signal == "rotational_speed":
        return f"{'Raise' if up else 'Lower'} the spindle speed to {after:,.0f} rpm"
    if signal == "torque":
        return (f"{'Increase' if up else 'Reduce'} the cutting load (feed rate or depth of cut) "
                f"so torque is about {after:.1f} Nm")
    return f"{'Warm' if up else 'Cool'} the air around the machine to {after - 273.15:.1f} °C"


def _pct(p: float) -> str:
    return "100%" if p >= 0.995 else "0%" if p < 0.001 else f"{p * 100:.1f}%" if p < 0.01 else f"{round(p * 100)}%"


def _names(modes) -> str:
    return " and ".join(MODE_NAMES[m].lower() for m in modes)


class PlanState(TypedDict, total=False):
    values: dict
    targets: list
    diagnosis: list
    candidates: list
    results: list
    combined: bool
    trace: list


class ActionAgents:
    def __init__(self, causal: CausalLayer, sim: SimulationLayer, typical: dict, levels: dict):
        self.causal, self.sim, self.typical, self.levels = causal, sim, typical, levels
        g = StateGraph(PlanState)
        g.add_node("diagnose", self.diagnose)
        g.add_node("propose", self.propose)
        g.add_node("simulate", self.simulate)
        g.add_node("rank", self.rank)
        g.add_node("combine", self.combine)
        g.set_entry_point("diagnose")
        g.add_conditional_edges("diagnose", lambda s: "propose" if s["targets"] else END)
        g.add_edge("propose", "simulate")
        g.add_edge("simulate", "rank")
        g.add_conditional_edges("rank", self._after_rank)
        g.add_edge("combine", "simulate")
        self.graph = g.compile()

    # ---------------------------------------------------------------- agents

    def _risks(self, values: dict) -> dict:
        frame = model_row(values).to_frame().T.astype(float)
        return {m: float(self.causal.failure_probability(m, frame)[0]) for m in OBSERVABLE}

    def diagnose(self, s: PlanState) -> dict:
        risks = self._risks(s["values"])
        at_risk = [m for m in OBSERVABLE if risks[m] >= AT_RISK]
        targets = list(dict.fromkeys((s.get("targets") or []) + at_risk))
        if not targets:
            return {"targets": [], "diagnosis": [], "trace": s["trace"] + [
                f"Diagnostician: no failure mode is at risk right now (every one below {_pct(AT_RISK)} per reading); "
            f"no action needed."]}
        diagnosis = []
        for m in targets:
            exp = explain_failure(self.causal, m, s["values"], self.typical)
            diagnosis.append({"mode": m, "name": MODE_NAMES[m], "risk": risks[m], "summary": exp["summary"],
                              "causes": [c["signal"] for c in exp["causes"]]})
        detected = s.get("targets") or []
        others = [m for m in targets if m not in detected]
        found = []
        if detected:
            found.append(f"{_names(detected)} detected")
        if others:
            found.append(f"{_names(others)} {'also ' if detected else ''}at risk "
                         f"({', '.join(f'{m} risk {_pct(risks[m])}' for m in others)})")
        return {"targets": targets, "diagnosis": diagnosis, "trace": s["trace"] + [
            f"Diagnostician: {'; '.join(found)}. {diagnosis[0]['summary']}"]}

    def propose(self, s: PlanState) -> dict:
        signals = [sig for m in s["targets"] for sig in self.causal.explanatory_signals(m) if sig in ACTIONABLE]
        signals = list(dict.fromkeys(signals))
        candidates = [{sig: level} for sig in signals for level in self._settings(sig, s["values"])]
        return {"candidates": candidates, "results": [], "combined": False, "trace": s["trace"] + [
            f"Planner: {len(candidates)} candidate actions on {', '.join(LABEL[x] for x in signals) or 'nothing'} "
            f"-- the knowledge graph's permitted causes of {_names(s['targets'])} that an operator can change, "
            f"at settings this machine has run at."]}

    def _settings(self, signal: str, values: dict) -> list:
        lv = self.levels[signal]
        return [x for x in lv["levels"] if abs(x - values[signal]) > 0.01 * lv["span"]]

    def simulate(self, s: PlanState) -> dict:
        values, out = s["values"], []
        for changes in s["candidates"]:
            r = whatif(self.sim, values, changes, s["targets"][0])
            still = [m for m in OBSERVABLE if r["risk_after"][m] >= AT_RISK]
            effort = max(ACTIONABLE[x][0] for x in changes)
            out.append({
                "title": " + ".join(describe(x, values[x], v) for x, v in changes.items()),
                "interventions": dict(changes),
                "changes": {x: {"before": float(values[x]), "after": float(v)} for x, v in changes.items()},
                "effort": effort,
                "effort_label": next(ACTIONABLE[x][1] for x in changes if ACTIONABLE[x][0] == effort),
                "size": sum(abs(v - values[x]) / self.levels[x]["span"] for x, v in changes.items()),
                "resolves": not still,
                "still_failing": still,
                "trustworthy": r["trustworthy"],
                "risk_before": {m: r["risk_before"][m] for m in s["targets"]},
                "risk_after": {m: r["risk_after"][m] for m in s["targets"]},
                "any_before": r["any_before"],
                "any_after": r["any_after"],
                "power_before": r["derived_before"]["power"],
                "power_after": r["derived_after"]["power"],
            })
        return {"results": s["results"] + out, "candidates": [], "trace": s["trace"] + [
            f"Simulator: tested {len(out)} {'combined ' if s.get('combined') else ''}actions as counterfactual "
            f"do() interventions on the recorded machine state."]}

    def rank(self, s: PlanState) -> dict:
        ranked = sorted(s["results"], key=lambda a: (
            not a["resolves"], not a["trustworthy"], a["any_after"] >= LOW_RESIDUAL,
            a["effort"], len(a["interventions"]), a["size"], a["any_after"]))
        for i, a in enumerate(ranked, 1):
            a["rank"] = i
            a["reasons"] = self._reasons(a, s["targets"])
        works = sum(a["resolves"] for a in ranked)
        trusted = sum(a["resolves"] and a["trustworthy"] for a in ranked)
        note = (f"{works} of {len(ranked)} clear every failure risk ({trusted} with full confidence)"
                if works else f"none of {len(ranked)} clears every failure risk")
        return {"results": ranked, "trace": s["trace"] + [f"Ranker: {note}."]}

    def combine(self, s: PlanState) -> dict:
        targets, values = s["targets"], s["values"]
        singles = [a for a in s["results"] if len(a["interventions"]) == 1]
        pairs, notes = [], []

        # An action that clears the targets but causes a new failure: look up
        # the new failure's causes in the knowledge graph and adjust one of them too.
        for a in singles:
            if all(a["risk_after"][m] < AT_RISK for m in targets) and a["still_failing"]:
                (sig, v), = a["interventions"].items()
                fixes = [x for m in a["still_failing"] for x in self.causal.explanatory_signals(m)
                         if x in ACTIONABLE and x != sig]
                fixes = list(dict.fromkeys(fixes))
                pairs += [{sig: v, x: level} for x in fixes for level in self._settings(x, values)]
                if fixes:
                    notes.append(f"{describe(sig, values[sig], v).lower()} clears {_names(targets)} but would cause "
                                 f"{_names(a['still_failing'])}, so pair it with a change to "
                                 f"{', '.join(LABEL[x] for x in fixes)}")

        # Otherwise: the most promising single actions on different signals, together.
        by_signal = {}
        for a in singles:
            by_signal.setdefault(next(iter(a["interventions"])), []).append(a)
        top = {sig: sorted(acts, key=lambda a: (max(a["risk_after"].values()), a["any_after"]))[:TOP_PER_SIGNAL]
               for sig, acts in by_signal.items()}
        pairs += [dict(**a["interventions"], **b["interventions"])
                  for x, y in itertools.combinations(sorted(top), 2) for a in top[x] for b in top[y]]

        unique = list({tuple(sorted(p.items())): p for p in pairs}.values())
        why = "; ".join(notes[:2]) if notes else "trying the most promising single actions together"
        return {"candidates": unique, "combined": True, "trace": s["trace"] + [
            f"Re-planner: no single setting change or tool change was enough with full confidence. "
            f"{why[0].upper() + why[1:]}. Testing {len(unique)} combined actions."]}

    def _after_rank(self, s: PlanState):
        easy = any(a["resolves"] and a["trustworthy"] and a["effort"] < FACILITY for a in s["results"])
        return END if easy or s.get("combined") else "combine"

    @staticmethod
    def _reasons(a: dict, targets: list) -> list:
        out = []
        for m in targets:
            out.append(f"{MODE_NAMES[m]} risk {_pct(a['risk_before'][m])} → {_pct(a['risk_after'][m])}")
        out.append(f"Any failure {_pct(a['any_before'])} → {_pct(a['any_after'])}")
        if not a["resolves"]:
            out.append(f"Rejected: {_names(a['still_failing'])} would still be at risk")
        if not a["trustworthy"]:
            out.append("Low confidence: outside the data the model learnt from")
        drop = 1 - a["power_after"] / a["power_before"] if a["power_before"] else 0.0
        if drop > 0.1:
            out.append(f"Cutting power falls by {round(drop * 100)}%, so material is removed more slowly")
        out.append(f"Effort: {a['effort_label']}")
        return out

    # ---------------------------------------------------------------- entry points

    def plan(self, values: dict, targets: list | None = None) -> dict:
        """Ranked actions for a machine state. `targets` are failure modes to
        clear even if the model scores them below the risk threshold (the mode
        of a detected failure event)."""
        targets = [m for m in (targets or []) if m in OBSERVABLE]
        s = self.graph.invoke({"values": dict(values), "targets": targets, "trace": []})
        ranked = s.get("results", [])
        good = [a for a in ranked if a["resolves"]]
        distinct = {}
        for a in good:
            distinct.setdefault(frozenset(a["interventions"]), a)
        shown = [dict(a, rank=i) for i, a in enumerate(list(distinct.values())[:SHOWN], 1)]
        return {
            "targets": s["targets"],
            "diagnosis": s["diagnosis"],
            "recommendations": shown,
            "more_that_work": len(good) - len(shown),
            "rejected": [a for a in ranked if not a["resolves"]],
            "considered": len(ranked),
            "combined": bool(s.get("combined")),
            "trace": s["trace"],
        }

    def for_event(self, event: dict) -> dict:
        return self.plan(event["values"], [event["mechanism"]])
