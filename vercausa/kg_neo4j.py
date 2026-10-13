"""Neo4j backend for the Knowledge Graph. Same interface as InMemoryGraphStore."""

from __future__ import annotations

import json

import networkx as nx
from neo4j import GraphDatabase

from vercausa.kg import FailureEventRecord
from vercausa.schema import (
    CANDIDATE_CAUSE_EDGES,
    CONSTRAINTS_CYPHER,
    DEFAULT_PROCESS,
    DERIVED_QUANTITIES,
    FAILURE_MODES,
    FEEDS_EDGES,
    SENSOR_SIGNALS,
    SIGNAL_UPSTREAM_EDGES,
    SIGNALS,
    sensor_id,
)


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str | None = None) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        self._db = database

    def _run(self, query: str, **params):
        records, _, _ = self._driver.execute_query(query, params, database_=self._db)
        return records

    def apply_schema(self) -> None:
        for stmt in CONSTRAINTS_CYPHER:
            self._run(stmt)
        self._run(
            "UNWIND $rows AS r MERGE (s:Signal {name: r.name}) SET s.unit = r.unit, s.kind = r.kind",
            rows=[{"name": s.name, "unit": s.unit, "kind": s.kind} for s in SIGNALS],
        )
        self._run(
            "UNWIND $rows AS r MERGE (f:FailureMode {code: r.code}) SET f.name = r.name, f.rule = r.rule",
            rows=[{"code": m.code, "name": m.name, "rule": m.rule} for m in FAILURE_MODES],
        )
        self._run(
            "UNWIND $rows AS r MERGE (d:DerivedQuantity {name: r.name}) "
            "SET d.unit = r.unit, d.formula = r.formula, d.law = r.law",
            rows=[
                {"name": d.name, "unit": d.unit, "formula": d.formula, "law": d.law}
                for d in DERIVED_QUANTITIES
            ],
        )
        self._run(
            "UNWIND $rows AS r MATCH (a:Signal {name: r[0]}), (b:Signal {name: r[1]}) "
            "MERGE (a)-[:UPSTREAM_OF]->(b)",
            rows=[list(e) for e in SIGNAL_UPSTREAM_EDGES],
        )
        self._run(
            "UNWIND $rows AS r MATCH (s:Signal {name: r[0]}), (d:DerivedQuantity {name: r[1]}) "
            "MERGE (s)-[:FEEDS]->(d)",
            rows=[list(e) for e in FEEDS_EDGES],
        )
        self._run(
            "UNWIND $rows AS r MATCH (c {name: r[0]}), (f:FailureMode {code: r[1]}) "
            "WHERE c:Signal OR c:DerivedQuantity "
            "MERGE (c)-[:CANDIDATE_CAUSE_OF]->(f)",
            rows=[list(e) for e in CANDIDATE_CAUSE_EDGES],
        )

    def register_machine(self, machine_id: str, process: str = DEFAULT_PROCESS) -> None:
        self._run(
            """
            MERGE (m:Machine {machine_id: $machine_id})
              ON CREATE SET m.status = 'unknown', m.active_alerts = []
            MERGE (p:Process {process_id: $process}) SET p.name = $process
            MERGE (m)-[:PART_OF]->(p)
            WITH m
            UNWIND $sensors AS s
            MATCH (sig:Signal {name: s.signal})
            MERGE (sn:Sensor {sensor_id: s.sensor_id})
              SET sn.signal = s.signal, sn.unit = s.unit
            MERGE (sn)-[:MEASURES]->(m)
            MERGE (sn)-[:READS]->(sig)
            """,
            machine_id=machine_id,
            process=process,
            sensors=[
                {"sensor_id": sensor_id(machine_id, s.name), "signal": s.name, "unit": s.unit}
                for s in SENSOR_SIGNALS
            ],
        )

    def update_twin(self, machine_id, timestamp, product_variant, values, status, active_alerts) -> None:
        self._run(
            """
            MATCH (m:Machine {machine_id: $machine_id})
            SET m.status = $status, m.current_product_variant = $variant,
                m.active_alerts = $alerts, m.last_updated = $ts
            WITH m
            UNWIND $values AS v
            MATCH (s:Sensor {sensor_id: v.sensor_id})
            SET s.current_value = v.value, s.last_updated = $ts
            """,
            machine_id=machine_id,
            status=status,
            variant=product_variant,
            alerts=list(active_alerts),
            ts=timestamp,
            values=[
                {"sensor_id": sensor_id(machine_id, sig), "value": val}
                for sig, val in values.items()
            ],
        )

    def add_failure_event(self, event: FailureEventRecord) -> None:
        self._run(
            """
            MATCH (m:Machine {machine_id: $machine_id}), (f:FailureMode {code: $mechanism})
            MERGE (e:FailureEvent {event_id: $event_id})
              ON CREATE SET e.mechanism = $mechanism, e.timestamp = $timestamp,
                            e.source_row = $source_row, e.reasons = $reasons,
                            e.snapshot = $snapshot
            MERGE (e)-[:INSTANCE_OF]->(f)
            MERGE (e)-[:OBSERVED_ON]->(m)
            """,
            event_id=event.event_id,
            machine_id=event.machine_id,
            mechanism=event.mechanism,
            timestamp=event.timestamp,
            source_row=event.source_row,
            reasons=event.reasons,
            snapshot=json.dumps(event.snapshot),
        )

    def machine_state(self, machine_id: str) -> dict:
        records = self._run(
            """
            MATCH (m:Machine {machine_id: $machine_id})
            OPTIONAL MATCH (s:Sensor)-[:MEASURES]->(m)
            RETURN m, collect(s) AS sensors
            """,
            machine_id=machine_id,
        )
        m, sensors = records[0]["m"], records[0]["sensors"]
        return {
            "machine_id": machine_id,
            "status": m.get("status"),
            "product_variant": m.get("current_product_variant"),
            "active_alerts": list(m.get("active_alerts") or []),
            "last_updated": m.get("last_updated"),
            "sensors": {
                s["signal"]: {
                    "value": s.get("current_value"),
                    "unit": s.get("unit"),
                    "last_updated": s.get("last_updated"),
                }
                for s in sensors
            },
        }

    def failure_events(self, machine_id, mechanism=None, limit=None) -> list:
        query = (
            "MATCH (e:FailureEvent)-[:OBSERVED_ON]->(:Machine {machine_id: $machine_id}) "
            "WHERE $mechanism IS NULL OR e.mechanism = $mechanism "
            "RETURN e ORDER BY e.timestamp DESC"
        )
        if limit is not None:
            query += " LIMIT $limit"
        records = self._run(query, machine_id=machine_id, mechanism=mechanism, limit=limit)
        return [
            {
                "event_id": r["e"]["event_id"],
                "machine_id": machine_id,
                "mechanism": r["e"]["mechanism"],
                "timestamp": r["e"]["timestamp"],
                "source_row": r["e"]["source_row"],
                "reasons": r["e"]["reasons"],
                "snapshot": json.loads(r["e"]["snapshot"]),
            }
            for r in records
        ]

    def candidate_causes(self, mode: str, include_upstream: bool = True) -> list:
        hops = "*0.." if include_upstream else "*0"
        records = self._run(
            f"MATCH (u)-[:UPSTREAM_OF|FEEDS{hops}]->(c)-[:CANDIDATE_CAUSE_OF]->"
            "(:FailureMode {code: $mode}) "
            "WHERE (u:Signal OR u:DerivedQuantity) AND (c:Signal OR c:DerivedQuantity) "
            "RETURN DISTINCT u.name AS name",
            mode=mode,
        )
        return sorted(r["name"] for r in records)

    def candidate_graph(self) -> nx.DiGraph:
        dag = nx.DiGraph()
        for r in self._run("MATCH (s:Signal) RETURN s.name AS n"):
            dag.add_node(r["n"], kind="signal")
        for r in self._run("MATCH (d:DerivedQuantity) RETURN d.name AS n"):
            dag.add_node(r["n"], kind="derived")
        for r in self._run("MATCH (f:FailureMode) RETURN f.code AS n"):
            dag.add_node(r["n"], kind="failure_mode")
        for r in self._run(
            "MATCH (a)-[r:UPSTREAM_OF|FEEDS|CANDIDATE_CAUSE_OF]->(b) "
            "WHERE a:Signal OR a:DerivedQuantity "
            "RETURN a.name AS a, CASE WHEN b:FailureMode THEN b.code ELSE b.name END AS b, "
            "type(r) AS t"
        ):
            dag.add_edge(r["a"], r["b"], type=r["t"])
        return dag

    def close(self) -> None:
        self._driver.close()
