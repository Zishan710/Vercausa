import os

import networkx as nx
import pytest

from vercausa.kg import InMemoryGraphStore
from vercausa.readings import SensorReading, load_ai4i
from vercausa.twin import DigitalTwin

MACHINE = "TEST-01"

HEALTHY = dict(air_temperature=298.1, process_temperature=308.6,
               rotational_speed=1551, torque=42.8, tool_wear=0)
HEAT_FAULT = dict(air_temperature=300.0, process_temperature=305.0,
                  rotational_speed=1300, torque=40.0, tool_wear=10)


def _reading(values, row, variant="M"):
    return SensorReading(MACHINE, f"2026-01-01T00:{row:02d}:00", variant, dict(values), row)


def _neo4j_store():
    from vercausa.kg_neo4j import Neo4jGraphStore

    store = Neo4jGraphStore(
        os.environ["NEO4J_TEST_URI"],
        os.environ.get("NEO4J_TEST_USER", "neo4j"),
        os.environ["NEO4J_TEST_PASSWORD"],
    )
    store._run("MATCH (n) DETACH DELETE n")
    return store


BACKENDS = [pytest.param(InMemoryGraphStore, id="memory")]
if os.environ.get("NEO4J_TEST_URI"):
    BACKENDS.append(pytest.param(_neo4j_store, id="neo4j"))


@pytest.fixture(params=BACKENDS)
def store(request):
    s = request.param()
    s.apply_schema()
    yield s
    s.close()


@pytest.fixture
def twin(store):
    t = DigitalTwin(store)
    t.register(MACHINE)
    return t


def test_healthy_reading_updates_state_without_events(twin, store):
    result = twin.ingest(_reading(HEALTHY, 1))
    assert result.new_events == []
    state = store.machine_state(MACHINE)
    assert state["status"] == "running"
    assert state["product_variant"] == "M"
    assert state["sensors"]["torque"]["value"] == 42.8
    assert state["sensors"]["torque"]["last_updated"] == "2026-01-01T00:01:00"
    assert store.failure_events(MACHINE) == []


def test_fault_opens_one_event_while_it_persists(twin, store):
    first = twin.ingest(_reading(HEAT_FAULT, 1))
    second = twin.ingest(_reading(HEAT_FAULT, 2))
    assert [e.mechanism for e in first.new_events] == ["HDF"]
    assert second.new_events == []
    assert store.machine_state(MACHINE)["active_alerts"] == ["HDF"]
    assert store.machine_state(MACHINE)["status"] == "alert"
    events = store.failure_events(MACHINE)
    assert len(events) == 1
    assert events[0]["source_row"] == 1
    assert events[0]["snapshot"]["rotational_speed"] == 1300


def test_clearing_then_retripping_opens_a_new_event(twin, store):
    twin.ingest(_reading(HEAT_FAULT, 1))
    cleared = twin.ingest(_reading(HEALTHY, 2))
    assert cleared.cleared == ["HDF"]
    assert store.machine_state(MACHINE)["status"] == "running"
    twin.ingest(_reading(HEAT_FAULT, 3))
    assert [e["source_row"] for e in store.failure_events(MACHINE)] == [3, 1]


def test_failure_events_filter_and_limit(twin, store):
    twin.ingest(_reading(HEAT_FAULT, 1))
    twin.ingest(_reading(HEALTHY, 2))
    twin.ingest(_reading(HEAT_FAULT, 3))
    assert len(store.failure_events(MACHINE, mechanism="HDF")) == 2
    assert store.failure_events(MACHINE, mechanism="PWF") == []
    assert len(store.failure_events(MACHINE, limit=1)) == 1


def test_restarted_twin_does_not_reopen_active_alert(store):
    first = DigitalTwin(store)
    first.register(MACHINE)
    first.ingest(_reading(HEAT_FAULT, 1))
    restarted = DigitalTwin(store)
    restarted.register(MACHINE)
    assert restarted.ingest(_reading(HEAT_FAULT, 2)).new_events == []


def test_candidate_causes_follow_upstream_and_feeds_edges(store):
    assert store.candidate_causes("HDF", include_upstream=False) == [
        "rotational_speed", "temperature_difference"
    ]
    assert store.candidate_causes("HDF") == [
        "air_temperature", "process_temperature", "rotational_speed", "temperature_difference"
    ]
    assert store.candidate_causes("PWF") == ["power", "rotational_speed", "torque"]
    assert store.candidate_causes("TWF", include_upstream=False) == ["tool_wear"]
    assert store.candidate_causes("TWF") == ["product_variant", "tool_wear"]
    assert store.candidate_causes("RNF") == []


def test_candidate_graph_is_a_dag_ending_in_failure_modes(store):
    dag = store.candidate_graph()
    assert nx.is_directed_acyclic_graph(dag)
    assert dag.has_edge("torque", "power")
    assert dag.has_edge("power", "PWF")
    assert dag.nodes["power"]["kind"] == "derived"
    assert dag.has_edge("air_temperature", "process_temperature")
    assert not dag.has_edge("torque", "PWF")
    assert not dag.has_edge("torque", "HDF")
    for node, kind in dag.nodes(data="kind"):
        if kind == "failure_mode":
            assert dag.out_degree(node) == 0


def test_replaying_real_data_opens_events_matching_detection(twin, store):
    readings = list(load_ai4i(machine_id=MACHINE, limit=300))
    opened = [e for r in readings for e in twin.ingest(r).new_events]
    assert opened
    assert len(store.failure_events(MACHINE)) == len(opened)
    assert store.machine_state(MACHINE)["last_updated"] == readings[-1].timestamp
