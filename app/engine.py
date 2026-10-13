"""Live replay engine behind the demo dashboard.

Feeds AI4I 2020 rows one at a time into the Digital Twin as if they were
arriving from the machine, and enriches every detected failure with the
Causal Layer's root-cause explanation. What-if requests go to the
Simulation Layer, starting from the state the twin recorded at that moment
(twin = context, SCM = simulation).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import replace

from vercausa.causal import CausalLayer
from vercausa.explain import (
    MODE_NAMES, OBSERVABLE, derived_values, explain_failure, model_row, typical_values, whatif,
)
from vercausa.readings import DEFAULT_CSV, DEFAULT_MACHINE, SensorReading, load_ai4i
from vercausa.schema import SENSOR_SIGNALS
from vercausa.simulation import SimulationLayer
from vercausa.twin import DigitalTwin

HISTORY = 90


class ReplayEngine:
    """Feeds readings into the twin and explains failures.

    mode="replay": the engine reads AI4I itself, one row per tick.
    mode="mqtt":   readings arrive from the network via ingest_external();
                   the machine publisher controls the pace, and the engine
                   reports the machine offline when messages stop.
    """

    def __init__(self, store_factory, causal: CausalLayer, csv=DEFAULT_CSV, machine_id: str = DEFAULT_MACHINE,
                 mode: str = "replay", clock=time.monotonic, offline_after: float = 5.0):
        self.store_factory = store_factory
        self.causal = causal
        self.sim = SimulationLayer(causal)
        self.readings = list(load_ai4i(csv, machine_id=machine_id))
        self.typical = typical_values(self.readings)
        self.machine_id = machine_id
        self.mode = mode
        self.clock = clock
        self.offline_after = offline_after
        self.broker_connected = False
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
        self.received = 0
        self.rejected = 0
        self.last_rejection: str | None = None
        self.last_message_at: float | None = None
        self._seq = 0
        self.version += 1

    @property
    def finished(self) -> bool:
        return self.mode == "replay" and self.position >= len(self.readings)

    def step(self) -> list:
        """Ingest the next replayed reading. Returns the failure events it opened."""
        if self.finished:
            self.playing = False
            return []
        reading = self.readings[self.position]
        self.position += 1
        return self._ingest(reading)

    # ---------- live (MQTT) input ----------

    def ingest_external(self, reading: SensorReading) -> list:
        """Ingest a reading that arrived over the network."""
        if reading.machine_id != self.machine_id:
            self.reject(f"unknown machine '{reading.machine_id}' (monitoring {self.machine_id})")
            return []
        self._seq += 1
        if reading.source_row <= 0:
            reading = replace(reading, source_row=self._seq)
        self.received += 1
        self.position += 1
        self.last_message_at = self.clock()
        return self._ingest(reading)

    def reject(self, reason: str, topic: str = "") -> None:
        self.rejected += 1
        self.last_rejection = reason
        self.version += 1

    def set_broker_connected(self, connected: bool) -> None:
        self.broker_connected = connected
        self.version += 1

    def link_state(self) -> str | None:
        if self.mode != "mqtt":
            return None
        if not self.broker_connected:
            return "broker-down"
        if self.last_message_at is None:
            return "waiting"
        return "offline" if self.clock() - self.last_message_at > self.offline_after else "live"

    def _ingest(self, reading: SensorReading) -> list:
        result = self.twin.ingest(reading)
        self.last_reading = reading
        d = derived_values(reading.values)
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

    def risks(self, values: dict, variant: str) -> dict:
        frame = model_row(dict(values, product_variant=variant)).to_frame().T.astype(float)
        return {m: float(self.causal.failure_probability(m, frame)[0]) for m in OBSERVABLE}

    def _explain(self, event, reading: SensorReading) -> dict:
        values = dict(reading.values, product_variant=reading.product_variant)
        return dict(
            {"id": event.event_id, "mechanism": event.mechanism, "name": MODE_NAMES[event.mechanism],
             "timestamp": event.timestamp, "row": event.source_row, "reason": event.reasons, "values": values},
            **explain_failure(self.causal, event.mechanism, values, self.typical),
        )

    # ---------- what-if ----------

    def simulate(self, event_id: str, interventions: dict) -> dict:
        event = next(e for e in self.events if e["id"] == event_id)
        return whatif(self.sim, event["values"], interventions, event["mechanism"])

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
                "derived": derived_values(r.values),
                "risk": self.risks(r.values, r.product_variant),
            }
        link = self.link_state()
        since = None if self.last_message_at is None else round(self.clock() - self.last_message_at, 1)
        return {
            "version": self.version,
            "playing": self.playing,
            "speed": self.speed,
            "pause_on_failure": self.pause_on_failure,
            "position": self.position,
            "total": len(self.readings) if self.mode == "replay" else None,
            "source": {
                "mode": self.mode,
                "link": link,
                "received": self.received,
                "rejected": self.rejected,
                "last_rejection": self.last_rejection,
                "seconds_since_last": since,
            },
            "status": "offline" if link in ("offline", "broker-down") and r is not None else machine["status"],
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
