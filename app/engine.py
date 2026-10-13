"""Live replay engine behind the demo dashboard.

Feeds AI4I 2020 rows one at a time into the Digital Twin as if they were
arriving from the machine, and enriches every detected failure with the
Causal Layer's root-cause explanation. What-if requests go to the
Simulation Layer, starting from the state the twin recorded at that moment
(twin = context, SCM = simulation).
"""

from __future__ import annotations

import math
from collections import deque

import pandas as pd

from vercausa.causal import MODE_CODES, VARIANT_CODES, CausalLayer
from vercausa.kg import GraphStore
from vercausa.readings import DEFAULT_CSV, DEFAULT_MACHINE, SensorReading, load_ai4i
from vercausa.schema import FAILURE_MODES, SENSOR_SIGNALS, compute_derived
from vercausa.simulation import SimulationLayer
from vercausa.twin import DigitalTwin

MODE_NAMES = {m.code: m.name for m in FAILURE_MODES}
OBSERVABLE = ("TWF", "HDF", "PWF", "OSF")
HISTORY = 90
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


def _consequence(mechanism: str, derived: dict) -> str:
    if mechanism == "PWF":
        return ("so the power went above the safe limit" if derived["power"] > 9000
                else "so the power fell below the safe minimum")
    return {
        "HDF": "so the machine couldn't get rid of its heat",
        "OSF": "so the tool was overstrained",
        "TWF": "so the tool reached the end of its safe life",
    }[mechanism]


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


def _derived(values: dict) -> dict:
    return {
        "power": float(compute_derived("power", values)),
        "temperature_difference": float(compute_derived("temperature_difference", values)),
    }


class ReplayEngine:
    def __init__(self, store_factory, causal: CausalLayer, csv=DEFAULT_CSV, machine_id: str = DEFAULT_MACHINE):
        self.store_factory = store_factory
        self.causal = causal
        self.sim = SimulationLayer(causal)
        self.readings = list(load_ai4i(csv, machine_id=machine_id))
        self.typical = {
            s.name: float(pd.Series([r.values[s.name] for r in self.readings]).median())
            for s in SENSOR_SIGNALS
        }
        self.machine_id = machine_id
        self.version = 0
        self.speed = 2.0
        self.pause_on_failure = True
        self.reset()

    # ---------- replay control ----------

    def reset(self) -> None:
        self.store = self.store_factory()
        self.store.apply_schema()
        self.twin = DigitalTwin(self.store)
        self.twin.register(self.machine_id)
        self.position = 0
        self.playing = False
        self.events: list = []
        self.history: deque = deque(maxlen=HISTORY)
        self.last_reading: SensorReading | None = None
        self.last_new_events: list = []
        self.version += 1

    @property
    def finished(self) -> bool:
        return self.position >= len(self.readings)

    def step(self) -> list:
        """Ingest the next reading. Returns the failure events it opened."""
        if self.finished:
            self.playing = False
            return []
        reading = self.readings[self.position]
        self.position += 1
        result = self.twin.ingest(reading)
        self.last_reading = reading
        d = _derived(reading.values)
        self.history.append({"t": reading.timestamp, "power": d["power"],
                             "temperature_difference": d["temperature_difference"]})
        new = [self._explain(e, reading) for e in result.new_events]
        self.events.extend(new)
        self.last_new_events = new
        self.version += 1
        return new

    def next_failure(self) -> list:
        """Fast-forward to the next reading that opens a failure event."""
        while not self.finished:
            new = self.step()
            if new:
                return new
        return []

    # ---------- causal explanations ----------

    def _row(self, values: dict, variant: str) -> pd.Series:
        return pd.Series(dict(values, product_variant=float(VARIANT_CODES[variant])))

    def risks(self, values: dict, variant: str) -> dict:
        frame = self._row(values, variant).to_frame().T.astype(float)
        return {m: float(self.causal.failure_probability(m, frame)[0]) for m in OBSERVABLE}

    def _explain(self, event, reading: SensorReading) -> dict:
        row = self._row(reading.values, reading.product_variant)
        contributions = self.causal.contributions(event.mechanism, row)
        total = sum(abs(v) for v in contributions.values()) or 1.0
        attribution = self.causal.attribute(row)
        values = dict(reading.values, product_variant=reading.product_variant)
        causes = sorted(
            ({"signal": s, "value": float(row[s]), "share": abs(v) / total,
              "direction": "raised" if v >= 0 else "lowered"} for s, v in contributions.items()),
            key=lambda c: -c["share"],
        )
        derived = _derived(reading.values)
        return {
            "id": event.event_id,
            "mechanism": event.mechanism,
            "name": MODE_NAMES[event.mechanism],
            "timestamp": event.timestamp,
            "row": event.source_row,
            "reason": event.reasons,
            "values": values,
            "derived": derived,
            "probability": float(next(s.probability for s in attribution.ranking if s.mode == event.mechanism)),
            "causes": causes,
            "summary": plain_summary(event.mechanism, causes, values, derived, self.typical),
            "path": self.path(event.mechanism),
        }

    def path(self, mode: str) -> list:
        """Cause chain for a mode, read from the causal DAG: [(cause, via or None)]."""
        out = []
        for p in self.causal.parents(mode):
            if self.causal.dag.nodes[p]["kind"] == "derived":
                out += [{"signal": s, "via": p} for s in self.causal.parents(p)]
            else:
                out.append({"signal": p, "via": None})
        return out

    # ---------- what-if ----------

    def simulate(self, event_id: str, interventions: dict) -> dict:
        event = next(e for e in self.events if e["id"] == event_id)
        clean = {k: float(v) for k, v in interventions.items()}
        r = self.sim.simulate(event["values"], clean)
        focus = event["mechanism"]
        others_active = [m for m in OBSERVABLE if m != focus and r.risk_before[m] >= 0.5]
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
            "other_failures_before": others_active,
            "trustworthy": r.trustworthy,
            "outside_observed": [m for m in r.outside_observed if m in OBSERVABLE],
        }

    # ---------- snapshot for the UI ----------

    def state(self) -> dict:
        machine = self.store.machine_state(self.machine_id)
        r = self.last_reading
        live = None
        if r is not None:
            live = {
                "timestamp": r.timestamp,
                "row": r.source_row,
                "variant": r.product_variant,
                "values": r.values,
                "derived": _derived(r.values),
                "risk": self.risks(r.values, r.product_variant),
            }
        return {
            "version": self.version,
            "playing": self.playing,
            "speed": self.speed,
            "pause_on_failure": self.pause_on_failure,
            "position": self.position,
            "total": len(self.readings),
            "status": machine["status"],
            "active_alerts": machine["active_alerts"],
            "live": live,
            "history": list(self.history),
            "events": list(reversed(self.events[-40:])),
            "event_count": len(self.events),
            "new_event_ids": [e["id"] for e in self.last_new_events],
        }

    def meta(self) -> dict:
        ranges = {}
        for s in SENSOR_SIGNALS:
            vals = [rd.values[s.name] for rd in self.readings]
            ranges[s.name] = {"min": min(vals), "max": max(vals), "unit": s.unit}
        return {
            "machine": self.machine_id,
            "modes": {m: MODE_NAMES[m] for m in OBSERVABLE},
            "ranges": ranges,
            "thresholds": {"power_min": 3500, "power_max": 9000, "temperature_difference_min": 8.6},
        }


def json_safe(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj
