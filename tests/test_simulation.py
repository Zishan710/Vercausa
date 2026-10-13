import warnings

import pytest

from vercausa.causal import CausalLayer, training_frame
from vercausa.kg import InMemoryGraphStore
from vercausa.readings import DEFAULT_CSV, SensorReading
from vercausa.simulation import SimulationLayer
from vercausa.twin import DigitalTwin

warnings.filterwarnings("ignore")

SIGNALS = ("air_temperature", "process_temperature", "rotational_speed",
           "torque", "tool_wear", "product_variant")


@pytest.fixture(scope="module")
def data():
    return training_frame(DEFAULT_CSV)


@pytest.fixture(scope="module")
def store():
    s = InMemoryGraphStore()
    s.apply_schema()
    return s


@pytest.fixture(scope="module")
def sim(store, data):
    return SimulationLayer(CausalLayer.fit_from_store(store, data))


def _obs(data, source_row):
    row = data[data["source_row"] == source_row].iloc[0]
    return {s: float(row[s]) for s in SIGNALS}


def test_no_intervention_reproduces_the_factual_world(sim, data):
    obs = _obs(data, 9975)
    result = sim.simulate(obs, {})
    assert result.counterfactual == pytest.approx(result.factual)
    assert result.risk_after == pytest.approx(result.risk_before)
    assert result.changed_signals() == {}


def test_intervening_on_air_moves_process_temperature_by_the_learnt_slope(sim, data):
    obs = _obs(data, 9975)
    slope = (sim._predict_signal("process_temperature", {"air_temperature": obs["air_temperature"] + 1})
             - sim._predict_signal("process_temperature", {"air_temperature": obs["air_temperature"]}))
    cf = sim.counterfactual_state(obs, {"air_temperature": obs["air_temperature"] - 3})
    assert cf["process_temperature"] - obs["process_temperature"] == pytest.approx(-3 * slope)
    assert 0.5 < slope < 0.8


def test_intervention_leaves_non_descendants_untouched(sim, data):
    obs = _obs(data, 9975)
    cf = sim.counterfactual_state(obs, {"torque": 30.0})
    for s in ("air_temperature", "process_temperature", "rotational_speed", "tool_wear"):
        assert cf[s] == pytest.approx(obs[s])


def test_raising_speed_prevents_a_real_heat_failure(sim, data):
    row = data[(data["HDF"] == 1) & (data["machine_failure"] == 1)].iloc[0]
    obs = {s: float(row[s]) for s in SIGNALS}
    result = sim.simulate(obs, {"rotational_speed": 1600.0})
    assert result.risk_before["HDF"] >= 0.5
    assert result.risk_after["HDF"] < 0.5


def test_reducing_torque_prevents_the_power_failure_and_reports_power(sim, data):
    obs = _obs(data, 9975)
    result = sim.simulate(obs, {"torque": lambda t: t * 0.7})
    assert result.derived_before["power"] > 9000
    assert result.derived_after["power"] < 9000
    assert result.risk_before["PWF"] >= 0.5
    assert result.risk_after["PWF"] < 0.5
    assert result.any_after < result.any_before


def test_product_variant_can_be_set_by_letter(sim, data):
    obs = _obs(data, 9975)
    assert sim.counterfactual_state(obs, {"product_variant": "H"})["product_variant"] == 2.0


def test_cannot_intervene_on_failure_modes_or_unknown_nodes(sim, data):
    obs = _obs(data, 9975)
    with pytest.raises(ValueError):
        sim.simulate(obs, {"PWF": 0})
    with pytest.raises(ValueError):
        sim.simulate(obs, {"power": 5000})
    with pytest.raises(ValueError):
        sim.simulate(obs, {"humidity": 50})


def test_extreme_interventions_are_flagged_as_untrustworthy(sim, data):
    result = sim.simulate(_obs(data, 9975), {"torque": 400.0})
    assert "PWF" in result.outside_observed
    assert not result.trustworthy


def test_simulation_starts_from_live_twin_state(sim, store):
    twin = DigitalTwin(store)
    twin.register("SIM-01")
    values = dict(air_temperature=300.0, process_temperature=305.0,
                  rotational_speed=1300.0, torque=40.0, tool_wear=10.0)
    twin.ingest(SensorReading("SIM-01", "2026-01-01T00:00:00", "L", values, 1))
    result = sim.simulate_twin(store, "SIM-01", {})
    assert result.factual == pytest.approx(dict(values, product_variant=0.0))
    assert result.risk_before["HDF"] >= 0.5
