"""Knowledge Graph storage.

Two interchangeable backends behind one interface:
- InMemoryGraphStore (networkx): the default, needs no infrastructure, so the
  pipeline runs on any laptop.
- Neo4jGraphStore (vercausa.kg_neo4j): used when NEO4J_URI is set.

Both expose the same queries, so nothing above this layer knows which one it
is talking to.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import networkx as nx

from vercausa.schema import (
    CANDIDATE_CAUSE_EDGES,
    DEFAULT_PROCESS,
    DERIVED_QUANTITIES,
    FAILURE_MODES,
    FEEDS_EDGES,
    SENSOR_SIGNALS,
    SIGNAL_UPSTREAM_EDGES,
    SIGNALS,
    sensor_id,
)

DERIVED_NAMES = {d.name for d in DERIVED_QUANTITIES}
CAUSE_LABELS = ("Signal", "DerivedQuantity")


def cause_label(name: str) -> str:
    return "DerivedQuantity" if name in DERIVED_NAMES else "Signal"


@dataclass(frozen=True)
class FailureEventRecord:
    event_id: str
    machine_id: str
    mechanism: str
    timestamp: str
    source_row: int
    reasons: str
    snapshot: dict = field(default_factory=dict)


class GraphStore(Protocol):
    def apply_schema(self) -> None: ...
    def register_machine(self, machine_id: str, process: str = DEFAULT_PROCESS) -> None: ...
    def update_twin(
        self,
        machine_id: str,
        timestamp: str,
        product_variant: str,
        values: dict,
        status: str,
        active_alerts: list,
    ) -> None: ...
    def add_failure_event(self, event: FailureEventRecord) -> None: ...
    def machine_state(self, machine_id: str) -> dict: ...
    def failure_events(
        self, machine_id: str, mechanism: str | None = None, limit: int | None = None
    ) -> list: ...
    def candidate_causes(self, mode: str, include_upstream: bool = True) -> list: ...
    def candidate_graph(self) -> nx.DiGraph: ...
    def close(self) -> None: ...


def _key(label: str, ident: str) -> tuple:
    return (label, ident)


class InMemoryGraphStore:
    def __init__(self) -> None:
        self.g = nx.DiGraph()
        self._events: list = []

    def apply_schema(self) -> None:
        g = self.g
        for s in SIGNALS:
            g.add_node(_key("Signal", s.name), label="Signal", name=s.name, unit=s.unit, kind=s.kind)
        for m in FAILURE_MODES:
            g.add_node(_key("FailureMode", m.code), label="FailureMode", code=m.code, name=m.name, rule=m.rule)
        for d in DERIVED_QUANTITIES:
            g.add_node(_key("DerivedQuantity", d.name), label="DerivedQuantity", name=d.name,
                       unit=d.unit, formula=d.formula, law=d.law)
        for a, b in SIGNAL_UPSTREAM_EDGES:
            g.add_edge(_key("Signal", a), _key("Signal", b), type="UPSTREAM_OF")
        for sig, derived in FEEDS_EDGES:
            g.add_edge(_key("Signal", sig), _key("DerivedQuantity", derived), type="FEEDS")
        for cause, mode in CANDIDATE_CAUSE_EDGES:
            g.add_edge(_key(cause_label(cause), cause), _key("FailureMode", mode), type="CANDIDATE_CAUSE_OF")

    def register_machine(self, machine_id: str, process: str = DEFAULT_PROCESS) -> None:
        g = self.g
        m = _key("Machine", machine_id)
        if m not in g:
            g.add_node(m, label="Machine", machine_id=machine_id, status="unknown",
                       current_product_variant=None, active_alerts=[], last_updated=None)
        p = _key("Process", process)
        g.add_node(p, label="Process", process_id=process, name=process)
        g.add_edge(m, p, type="PART_OF")
        for s in SENSOR_SIGNALS:
            sk = _key("Sensor", sensor_id(machine_id, s.name))
            if sk not in g:
                g.add_node(sk, label="Sensor", sensor_id=sk[1], signal=s.name, unit=s.unit,
                           current_value=None, last_updated=None)
            g.add_edge(sk, m, type="MEASURES")
            g.add_edge(sk, _key("Signal", s.name), type="READS")

    def update_twin(self, machine_id, timestamp, product_variant, values, status, active_alerts) -> None:
        m = self.g.nodes[_key("Machine", machine_id)]
        m.update(status=status, current_product_variant=product_variant,
                 active_alerts=list(active_alerts), last_updated=timestamp)
        for signal, value in values.items():
            s = self.g.nodes[_key("Sensor", sensor_id(machine_id, signal))]
            s.update(current_value=value, last_updated=timestamp)

    def add_failure_event(self, event: FailureEventRecord) -> None:
        ek = _key("FailureEvent", event.event_id)
        if ek in self.g:
            return
        self.g.add_node(ek, label="FailureEvent", **event.__dict__)
        self.g.add_edge(ek, _key("FailureMode", event.mechanism), type="INSTANCE_OF")
        self.g.add_edge(ek, _key("Machine", event.machine_id), type="OBSERVED_ON")
        self._events.append(event)

    def machine_state(self, machine_id: str) -> dict:
        mk = _key("Machine", machine_id)
        m = self.g.nodes[mk]
        sensors = {}
        for sk in self.g.predecessors(mk):
            node = self.g.nodes[sk]
            if node["label"] == "Sensor":
                sensors[node["signal"]] = {
                    "value": node["current_value"],
                    "unit": node["unit"],
                    "last_updated": node["last_updated"],
                }
        return {
            "machine_id": machine_id,
            "status": m["status"],
            "product_variant": m["current_product_variant"],
            "active_alerts": list(m["active_alerts"]),
            "last_updated": m["last_updated"],
            "sensors": sensors,
        }

    def failure_events(self, machine_id, mechanism=None, limit=None) -> list:
        matches = [
            e.__dict__ for e in reversed(self._events)
            if e.machine_id == machine_id and (mechanism is None or e.mechanism == mechanism)
        ]
        return matches if limit is None else matches[:limit]

    def candidate_causes(self, mode: str, include_upstream: bool = True) -> list:
        direct = {
            self.g.nodes[u]["name"]
            for u, _, d in self.g.in_edges(_key("FailureMode", mode), data=True)
            if d["type"] == "CANDIDATE_CAUSE_OF"
        }
        if not include_upstream:
            return sorted(direct)
        upstream = nx.DiGraph(
            (u, v) for u, v, d in self.g.edges(data=True) if d["type"] in ("UPSTREAM_OF", "FEEDS")
        )
        found = set(direct)
        for name in direct:
            k = _key(cause_label(name), name)
            if k in upstream:
                found |= {n[1] for n in nx.ancestors(upstream, k)}
        return sorted(found)

    def candidate_graph(self) -> nx.DiGraph:
        kinds = {"Signal": "signal", "DerivedQuantity": "derived", "FailureMode": "failure_mode"}
        dag = nx.DiGraph()
        for _, d in self.g.nodes(data=True):
            if d["label"] in kinds:
                dag.add_node(d.get("code") or d["name"], kind=kinds[d["label"]])
        for u, v, d in self.g.edges(data=True):
            if d["type"] in ("UPSTREAM_OF", "FEEDS", "CANDIDATE_CAUSE_OF") and u[0] in CAUSE_LABELS:
                dag.add_edge(u[1], v[1], type=d["type"])
        return dag

    def close(self) -> None:
        pass


def open_store() -> GraphStore:
    """Neo4j if NEO4J_URI is set, otherwise in-memory."""
    uri = os.environ.get("NEO4J_URI")
    if not uri:
        return InMemoryGraphStore()
    from vercausa.kg_neo4j import Neo4jGraphStore

    return Neo4jGraphStore(
        uri,
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ["NEO4J_PASSWORD"],
    )
