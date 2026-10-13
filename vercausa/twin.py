"""Digital Twin: live state mirrored onto Knowledge Graph nodes.

This is a state mirror, not a simulator (CLAUDE.md, architecture decision #2).
It records what is true right now and what tripped; "what if" questions are
the Simulation Layer's job, run against the state this keeps current.

Events are opened on the rising edge only: a mechanism that stays tripped
across consecutive readings (e.g. TWF for the ~13 readings a tool spends in
its 200-240 min window) produces one FailureEvent, not one per reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vercausa.detection import DetectionResult, evaluate_reading
from vercausa.kg import FailureEventRecord, GraphStore
from vercausa.readings import SensorReading


@dataclass
class IngestResult:
    reading: SensorReading
    detection: DetectionResult
    new_events: list = field(default_factory=list)
    cleared: list = field(default_factory=list)


class DigitalTwin:
    def __init__(self, store: GraphStore) -> None:
        self.store = store
        self._active: dict = {}

    def register(self, machine_id: str) -> None:
        self.store.register_machine(machine_id)
        self._active[machine_id] = set(self.store.machine_state(machine_id)["active_alerts"])

    def ingest(self, reading: SensorReading) -> IngestResult:
        v = reading.values
        detection = evaluate_reading(
            product_variant=reading.product_variant,
            air_temp_k=v["air_temperature"],
            process_temp_k=v["process_temperature"],
            rpm=v["rotational_speed"],
            torque_nm=v["torque"],
            tool_wear_min=v["tool_wear"],
        )
        active = set(detection.tripped)
        previous = self._active.get(reading.machine_id, set())

        snapshot = dict(v, product_variant=reading.product_variant)
        new_events = [
            FailureEventRecord(
                event_id=f"{reading.machine_id}-{mech}-{reading.source_row}",
                machine_id=reading.machine_id,
                mechanism=mech,
                timestamp=reading.timestamp,
                source_row=reading.source_row,
                reasons=detection.reasons[mech],
                snapshot=snapshot,
            )
            for mech in sorted(active - previous)
        ]

        self.store.update_twin(
            machine_id=reading.machine_id,
            timestamp=reading.timestamp,
            product_variant=reading.product_variant,
            values=v,
            status="alert" if active else "running",
            active_alerts=sorted(active),
        )
        for event in new_events:
            self.store.add_failure_event(event)
        self._active[reading.machine_id] = active

        return IngestResult(reading, detection, new_events, sorted(previous - active))
