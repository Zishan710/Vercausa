"""Knowledge Graph schema and expert seed.

Every later layer reads this: the Detection Layer's trips become FailureEvent
nodes, the Causal Layer may only estimate edges that exist here
(CLAUDE.md, architecture decision #1), and the Digital Twin is just the live
properties on Machine/Sensor nodes (decision #2) -- it has no node type of
its own.

Two levels, kept separate on purpose:
- Type level (Signal, FailureMode): the causal structure. "Torque can cause
  power failure" is true of torque in general, not of one reading.
- Instance level (Machine, Sensor, FailureEvent): the twin's live state and
  the history of what actually tripped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeType:
    label: str
    properties: tuple
    description: str


@dataclass(frozen=True)
class EdgeType:
    type: str
    source: str
    target: str
    description: str


NODE_TYPES = (
    NodeType("Process", ("process_id", "name"), "A manufacturing step machines take part in."),
    NodeType(
        "Machine",
        ("machine_id", "status", "current_product_variant", "active_alerts", "last_updated"),
        "A physical unit. status/active_alerts/current_product_variant are live twin state.",
    ),
    NodeType(
        "Sensor",
        ("sensor_id", "signal", "unit", "current_value", "last_updated"),
        "One instrumented channel on one machine. current_value is live twin state.",
    ),
    NodeType("Signal", ("name", "unit", "kind"), "A measurable quantity, independent of any machine."),
    NodeType(
        "DerivedQuantity",
        ("name", "unit", "formula", "law"),
        "A physical quantity computed from signals by a textbook law (e.g. power = torque x speed).",
    ),
    NodeType("FailureMode", ("code", "name", "rule"), "A documented AI4I 2020 failure mechanism."),
    NodeType(
        "FailureEvent",
        ("event_id", "mechanism", "timestamp", "reasons", "snapshot"),
        "Created when the Detection Layer trips a rule (on the rising edge only).",
    ),
)

EDGE_TYPES = (
    EdgeType("PART_OF", "Machine", "Process", "Machine takes part in this process."),
    EdgeType("UPSTREAM_OF", "Machine", "Machine", "Physical line order between machines."),
    EdgeType("MEASURES", "Sensor", "Machine", "Sensor reads state from this machine."),
    EdgeType("READS", "Sensor", "Signal", "Which quantity this sensor instruments."),
    EdgeType("UPSTREAM_OF", "Signal", "Signal", "Physical influence between quantities."),
    EdgeType("FEEDS", "Signal", "DerivedQuantity", "Signal is an input to this derived quantity's formula."),
    EdgeType(
        "CANDIDATE_CAUSE_OF",
        "Signal | DerivedQuantity",
        "FailureMode",
        "The Causal Layer's permitted edge set. SCM estimation runs only over these.",
    ),
    EdgeType("INSTANCE_OF", "FailureEvent", "FailureMode", "Which mechanism tripped."),
    EdgeType("OBSERVED_ON", "FailureEvent", "Machine", "Where it tripped."),
)


@dataclass(frozen=True)
class Signal:
    name: str
    unit: str
    kind: str
    csv_column: str | None


SIGNALS = (
    Signal("air_temperature", "K", "continuous", "Air temperature [K]"),
    Signal("process_temperature", "K", "continuous", "Process temperature [K]"),
    Signal("rotational_speed", "rpm", "continuous", "Rotational speed [rpm]"),
    Signal("torque", "Nm", "continuous", "Torque [Nm]"),
    Signal("tool_wear", "min", "continuous", "Tool wear [min]"),
    Signal("product_variant", "", "categorical", "Type"),
)

SENSOR_SIGNALS = tuple(s for s in SIGNALS if s.kind == "continuous")


@dataclass(frozen=True)
class FailureMode:
    code: str
    name: str
    rule: str


FAILURE_MODES = (
    FailureMode("TWF", "Tool Wear Failure", "tool wear inside the 200-240 min window"),
    FailureMode("HDF", "Heat Dissipation Failure", "(process - air temp) < 8.6K AND rpm < 1380"),
    FailureMode("PWF", "Power Failure", "torque x speed outside [3500, 9000] W"),
    FailureMode("OSF", "Overstrain Failure", "wear x torque > 11000/12000/13000 minNm for L/M/H"),
    FailureMode("RNF", "Random Failure", "0.1% chance, independent of process state"),
)

# Physical influence between signals, from how the AI4I 2020 process works:
# process temperature is driven by ambient air temperature, and the product
# variant sets how fast the tool wears (H/M/L add 5/3/2 min per process).
# Rotational speed and torque are deliberately NOT linked: both come from the
# same power model and neither physically precedes the other.
SIGNAL_UPSTREAM_EDGES = (
    ("air_temperature", "process_temperature"),
    ("product_variant", "tool_wear"),
)

@dataclass(frozen=True)
class DerivedQuantity:
    name: str
    unit: str
    inputs: tuple
    formula: str
    law: str


# Only quantities given by general physics laws belong here -- nothing
# specific to the AI4I dataset. Thresholds on them are still learnt from data.
# Without these, a classifier on raw signals learns correlational shortcuts
# (e.g. "warm air => heat failure") that give wrong what-if answers.
DERIVED_QUANTITIES = (
    DerivedQuantity(
        "temperature_difference", "K", ("process_temperature", "air_temperature"),
        "process_temperature - air_temperature",
        "Newton's law of cooling: heat dissipation is driven by the temperature difference",
    ),
    DerivedQuantity(
        "power", "W", ("torque", "rotational_speed"),
        "torque * rotational_speed * 2*pi/60",
        "Mechanical power: P = torque x angular velocity",
    ),
)

_DERIVED_FUNCS = {
    "temperature_difference": lambda d: d["process_temperature"] - d["air_temperature"],
    "power": lambda d: d["torque"] * d["rotational_speed"] * 2 * math.pi / 60,
}


FEEDS_EDGES = tuple((i, d.name) for d in DERIVED_QUANTITIES for i in d.inputs)


def compute_derived(name: str, data):
    """Evaluate a derived quantity on a dict or DataFrame of signal values."""
    return _DERIVED_FUNCS[name](data)


# Expert seed: what each documented mechanism depends on. RNF has none --
# nothing observable causes it.
CANDIDATE_CAUSE_EDGES = (
    ("tool_wear", "TWF"),
    ("temperature_difference", "HDF"),
    ("rotational_speed", "HDF"),
    ("power", "PWF"),
    ("tool_wear", "OSF"),
    ("torque", "OSF"),
    ("product_variant", "OSF"),
)

DEFAULT_PROCESS = "milling"

CONSTRAINTS_CYPHER = (
    "CREATE CONSTRAINT process_id IF NOT EXISTS FOR (p:Process) REQUIRE p.process_id IS UNIQUE",
    "CREATE CONSTRAINT machine_id IF NOT EXISTS FOR (m:Machine) REQUIRE m.machine_id IS UNIQUE",
    "CREATE CONSTRAINT sensor_id IF NOT EXISTS FOR (s:Sensor) REQUIRE s.sensor_id IS UNIQUE",
    "CREATE CONSTRAINT signal_name IF NOT EXISTS FOR (s:Signal) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT derived_name IF NOT EXISTS FOR (d:DerivedQuantity) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT failure_mode IF NOT EXISTS FOR (f:FailureMode) REQUIRE f.code IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:FailureEvent) REQUIRE e.event_id IS UNIQUE",
)


def sensor_id(machine_id: str, signal: str) -> str:
    return f"{machine_id}:{signal}"
