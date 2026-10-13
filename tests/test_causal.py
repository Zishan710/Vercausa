import warnings

import pytest

from vercausa.causal import CausalLayer, training_frame
from vercausa.kg import InMemoryGraphStore, _key
from vercausa.readings import DEFAULT_CSV

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def data():
    return training_frame(DEFAULT_CSV)


@pytest.fixture(scope="module")
def store():
    s = InMemoryGraphStore()
    s.apply_schema()
    return s


@pytest.fixture(scope="module")
def layer(store, data):
    return CausalLayer.fit_from_store(store, data)


def _row(data, source_row):
    return data[data["source_row"] == source_row].iloc[0]


def test_each_mechanism_sees_only_its_kg_permitted_parents(layer, store):
    for mode in ("TWF", "HDF", "PWF", "OSF", "RNF"):
        assert layer.parents(mode) == store.candidate_causes(mode, include_upstream=False)


def test_removing_an_edge_from_the_kg_removes_it_from_the_model(data):
    s = InMemoryGraphStore()
    s.apply_schema()
    s.g.remove_edge(_key("Signal", "torque"), _key("FailureMode", "OSF"))
    fenced = CausalLayer.fit_from_store(s, data)
    assert fenced.parents("OSF") == ["product_variant", "tool_wear"]


def test_explanations_are_in_terms_of_measured_signals(layer):
    assert layer.parents("PWF") == ["power"]
    assert layer.explanatory_signals("PWF") == ["rotational_speed", "torque"]
    assert layer.explanatory_signals("HDF") == [
        "air_temperature", "process_temperature", "rotational_speed"
    ]


def test_derived_quantity_mechanism_is_the_exact_formula(layer, data):
    row = _row(data, 9975)
    frame = layer.with_derived(row.to_frame().T.astype(float))
    mech = layer.model.causal_mechanism("power")
    x = frame[layer.parents("power")].to_numpy(dtype=float)
    assert float(mech.prediction_model.predict(x)[0, 0]) == pytest.approx(float(frame["power"].iloc[0]))
    assert float(frame["power"].iloc[0]) > 9000


def test_support_flags_inputs_never_seen_in_training(layer, data):
    real = data[data["machine_failure"] == 0].head(50)
    assert layer.in_support("PWF", real).mean() > 0.9
    absurd = real.head(1).copy()
    absurd["torque"] = 500.0
    assert not layer.in_support("PWF", absurd)[0]


def test_values_between_whole_minute_readings_are_not_flagged_as_unseen(layer):
    import pandas as pd
    wear = pd.DataFrame({"tool_wear": [190.9, 100.5, 12.3]})
    assert layer.in_support("TWF", wear).all()
    assert not layer.in_support("TWF", pd.DataFrame({"tool_wear": [900.0]}))[0]


def test_power_and_overstrain_failure_is_attributed_to_both_with_torque_driving_power(layer, data):
    attribution = layer.attribute(_row(data, 9975))
    assert {s.mode for s in attribution.ranking[:2]} == {"PWF", "OSF"}
    assert attribution.responsibility["PWF"] >= 0.5
    assert attribution.responsibility["OSF"] >= 0.5
    pwf = next(s for s in attribution.ranking if s.mode == "PWF")
    assert max(pwf.contributions, key=lambda k: abs(pwf.contributions[k])) == "torque"


def test_heat_failure_is_attributed_to_hdf(layer, data):
    row = data[(data["HDF"] == 1) & (data["machine_failure"] == 1)].iloc[0]
    assert layer.attribute(row).top.mode == "HDF"


def test_contributions_sum_to_change_from_healthy_baseline(layer, data):
    row = _row(data, 9975)
    contributions = layer.contributions("PWF", row)
    p_row = layer.failure_probability("PWF", row.to_frame().T.astype(float))[0]
    p_base = layer.failure_probability("PWF", layer._baseline).mean()
    assert sum(contributions.values()) == pytest.approx(p_row - p_base, abs=1e-9)


def test_rnf_has_no_parents_so_it_means_no_observable_cause(layer):
    assert layer.parents("RNF") == []


def test_saved_model_reloads_with_identical_answers(layer, data, tmp_path):
    path = tmp_path / "causal_layer.joblib"
    layer.save(path)
    reloaded = CausalLayer.load(path)
    rows = data.head(500)
    for mode in ("TWF", "HDF", "PWF", "OSF", "RNF"):
        assert (reloaded.failure_probability(mode, rows) == layer.failure_probability(mode, rows)).all()
    original, again = layer.attribute(_row(data, 9975)), reloaded.attribute(_row(data, 9975))
    assert again.top.mode == original.top.mode
    assert again.top.contributions == original.top.contributions


def test_edge_strength_shares_sum_to_one_per_failure_mode(layer, data):
    strengths = layer.edge_strengths(data)
    for mode in ("TWF", "HDF", "PWF", "OSF"):
        total = sum(v for (src, dst), v in strengths.items() if dst == mode)
        assert total == pytest.approx(1.0)
        assert {src for src, dst in strengths if dst == mode} == set(layer.explanatory_signals(mode))
    assert strengths[("air_temperature", "process_temperature")] > 0.5
    assert not any(dst in ("power", "temperature_difference") for _, dst in strengths)
