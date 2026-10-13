"""Explanations shared by the dashboard and the GraphRAG reasoning layer.

Everything here is derived from the Causal and Simulation layers; nothing is
generated freely, so any text produced from these structures can be traced
back to the model and the knowledge graph.
"""

from __future__ import annotations

import pandas as pd

from vercausa.causal import VARIANT_CODES, CausalLayer
from vercausa.schema import FAILURE_MODES, SENSOR_SIGNALS, compute_derived
from vercausa.simulation import SimulationLayer

MODE_NAMES = {m.code: m.name for m in FAILURE_MODES}
OBSERVABLE = ("TWF", "HDF", "PWF", "OSF")
SECONDARY_SHARE = 0.3

# How a signal sounds to a non-engineer when it is above / below its usual level.
PLAIN = {
    "rotational_speed": ("the spindle speed became too fast ({v:,.0f} rpm)",
                         "the spindle speed became too slow ({v:,.0f} rpm)"),
    "torque": ("the cutting force (torque) was too high ({v:.1f} Nm)",
               "the cutting force (torque) dropped too low ({v:.1f} Nm)"),
    "tool_wear": ("the tool was worn out ({v:.0f} min of use)",
                  "the tool was still fresh ({v:.0f} min of use)"),
    "air_temperature": ("the air around the machine was too warm ({c:.1f} °C)",
                        "the air around the machine was unusually cool ({c:.1f} °C)"),
    "process_temperature": ("the machining process ran too hot ({c:.1f} °C)",
                            "the machining process ran unusually cool ({c:.1f} °C)"),
}

# Which side of its usual level makes each signal harmful for each failure.
# Power failure is two-sided: an extreme speed with normal torque drives power
# too HIGH, so it must not be named as the reason power fell too LOW -- even
# though the causal model rightly credits it with raising power-failure risk.
HARMFUL = {
    "PWF_low": {"torque": "low", "rotational_speed": "low"},
    "PWF_high": {"torque": "high", "rotational_speed": "high"},
    "HDF": {"rotational_speed": "low", "air_temperature": "high", "process_temperature": "low"},
    "OSF": {"torque": "high", "tool_wear": "high"},
    "TWF": {"tool_wear": "high"},
}


def derived_values(values: dict) -> dict:
    return {
        "power": float(compute_derived("power", values)),
        "temperature_difference": float(compute_derived("temperature_difference", values)),
    }


def typical_values(readings) -> dict:
    """Median of each sensor signal -- what "usual" means for this machine."""
    return {
        s.name: float(pd.Series([r.values[s.name] for r in readings]).median())
        for s in SENSOR_SIGNALS
    }


def _consequence(mechanism: str, derived: dict) -> str:
    if mechanism == "PWF":
        return ("so the power went above the safe limit" if derived["power"] > 9000
                else "so the power fell below the safe minimum")
    return {
        "HDF": "so the machine couldn't get rid of its heat",
        "OSF": "so the tool was overstrained",
        "TWF": "so the tool reached the end of its safe life",
    }[mechanism]


def _phrase(signal: str, values: dict, typical: dict) -> str:
    if signal == "product_variant":
        return f"the product variant ({values['product_variant']}) has a lower strain limit"
    v = values[signal]
    high, low = PLAIN[signal]
    return (high if v >= typical[signal] else low).format(v=v, c=v - 273.15)


def plain_summary(mechanism: str, causes: list, values: dict, derived: dict, typical: dict) -> str:
    """One-sentence, non-technical root cause.

    Names the causes the causal model ranks highest, keeping only those
    deviating in the direction that produces this failure (see HARMFUL).
    """
    key = mechanism if mechanism != "PWF" else ("PWF_high" if derived["power"] > 9000 else "PWF_low")
    harmful = HARMFUL[key]

    def pushes(s):
        side = "high" if values[s] >= typical[s] else "low"
        return harmful.get(s) == side

    ranked = [c for c in causes if c["signal"] in harmful and pushes(c["signal"])]
    chosen = ranked[:1] + [c for c in ranked[1:2] if c["share"] >= SECONDARY_SHARE]
    if not chosen:
        chosen = causes[:1]
    text = " and ".join(_phrase(c["signal"], values, typical) for c in chosen)
    text += ", " + _consequence(mechanism, derived) + "."
    return text[0].upper() + text[1:]


def model_row(values: dict) -> pd.Series:
    """Signal values (product variant as L/M/H) in the form the causal model expects."""
    return pd.Series(dict({k: v for k, v in values.items() if k != "product_variant"},
                          product_variant=float(VARIANT_CODES[values["product_variant"]])))


def cause_path(causal: CausalLayer, mode: str) -> list:
    """Cause chain for a mode, read from the causal DAG: [{signal, via}]."""
    out = []
    for p in causal.parents(mode):
        if causal.dag.nodes[p]["kind"] == "derived":
            out += [{"signal": s, "via": p} for s in causal.parents(p)]
        else:
            out.append({"signal": p, "via": None})
    return out


def explain_failure(causal: CausalLayer, mechanism: str, values: dict, typical: dict) -> dict:
    """Root-cause explanation of one failure from the causal model."""
    row = model_row(values)
    contributions = causal.contributions(mechanism, row)
    total = sum(abs(v) for v in contributions.values()) or 1.0
    attribution = causal.attribute(row)
    causes = sorted(
        ({"signal": s, "value": float(row[s]), "share": abs(v) / total,
          "direction": "raised" if v >= 0 else "lowered"} for s, v in contributions.items()),
        key=lambda c: -c["share"],
    )
    derived = derived_values(values)
    return {
        "derived": derived,
        "probability": float(next(s.probability for s in attribution.ranking if s.mode == mechanism)),
        "causes": causes,
        "summary": plain_summary(mechanism, causes, values, derived, typical),
        "path": cause_path(causal, mechanism),
    }


def whatif(sim: SimulationLayer, values: dict, interventions: dict, focus: str) -> dict:
    """Counterfactual outcome of changing some signals, judged for one failure mode."""
    clean = {k: float(v) for k, v in interventions.items()}
    r = sim.simulate(values, clean)
    return {
        "mechanism": focus,
        "risk_before": {m: r.risk_before[m] for m in OBSERVABLE},
        "risk_after": {m: r.risk_after[m] for m in OBSERVABLE},
        "any_before": r.any_before,
        "any_after": r.any_after,
        "derived_before": r.derived_before,
        "derived_after": r.derived_after,
        "counterfactual": {k: v for k, v in r.counterfactual.items() if k != "product_variant"},
        "prevents_this": r.risk_after[focus] < 0.5 <= r.risk_before[focus],
        "still_failing": [m for m in OBSERVABLE if r.risk_after[m] >= 0.5],
        "other_failures_before": [m for m in OBSERVABLE if m != focus and r.risk_before[m] >= 0.5],
        "trustworthy": r.trustworthy,
        "outside_observed": [m for m in r.outside_observed if m in OBSERVABLE],
    }
