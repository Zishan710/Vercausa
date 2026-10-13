import inspect
import math

import networkx as nx
import pytest

from vercausa import detection
from vercausa.schema import (
    CANDIDATE_CAUSE_EDGES,
    DERIVED_QUANTITIES,
    FAILURE_MODES,
    FEEDS_EDGES,
    SIGNAL_UPSTREAM_EDGES,
    SIGNALS,
    compute_derived,
)

SIGNAL_NAMES = {s.name for s in SIGNALS}
DERIVED = {d.name: d for d in DERIVED_QUANTITIES}
MODE_CODES = {m.code for m in FAILURE_MODES}


def _seed_graph():
    return nx.DiGraph(list(SIGNAL_UPSTREAM_EDGES) + list(FEEDS_EDGES) + list(CANDIDATE_CAUSE_EDGES))


def test_candidate_edges_run_from_a_signal_or_derived_quantity_to_a_failure_mode():
    for cause, mode in CANDIDATE_CAUSE_EDGES:
        assert cause in SIGNAL_NAMES | set(DERIVED)
        assert mode in MODE_CODES


def test_derived_quantities_are_fed_only_by_signals():
    for sig, derived in FEEDS_EDGES:
        assert sig in SIGNAL_NAMES
        assert derived in DERIVED


def test_causal_seed_is_acyclic():
    assert nx.is_directed_acyclic_graph(_seed_graph())


def test_failure_modes_are_sinks():
    g = _seed_graph()
    for mode in MODE_CODES & set(g.nodes):
        assert g.out_degree(mode) == 0


def test_rnf_has_no_candidate_causes():
    assert not [e for e in CANDIDATE_CAUSE_EDGES if e[1] == "RNF"]


def test_power_formula_matches_detection_layer():
    values = {"torque": 40.0, "rotational_speed": 1500.0}
    assert compute_derived("power", values) == pytest.approx(40.0 * detection.rpm_to_rad_per_s(1500.0))
    assert compute_derived("power", {"torque": 10.0, "rotational_speed": 60.0}) == pytest.approx(20 * math.pi)


def test_temperature_difference_formula():
    values = {"process_temperature": 310.0, "air_temperature": 301.5}
    assert compute_derived("temperature_difference", values) == pytest.approx(8.5)


def _signals_behind(mode):
    out = set()
    for cause, m in CANDIDATE_CAUSE_EDGES:
        if m == mode:
            out |= set(DERIVED[cause].inputs) if cause in DERIVED else {cause}
    return out


def test_seed_matches_what_each_detection_rule_actually_reads():
    arg_to_signal = {
        "air_temp_k": "air_temperature",
        "process_temp_k": "process_temperature",
        "rpm": "rotational_speed",
        "torque_nm": "torque",
        "tool_wear_min": "tool_wear",
        "product_variant": "product_variant",
    }
    rules = {
        "TWF": detection.check_twf,
        "HDF": detection.check_hdf,
        "PWF": detection.check_pwf,
        "OSF": detection.check_osf,
    }
    for mode, fn in rules.items():
        reads = {arg_to_signal[p] for p in inspect.signature(fn).parameters}
        assert _signals_behind(mode) == reads, mode
