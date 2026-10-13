import warnings

import pandas as pd
import pytest

from vercausa.agents import ACTIONABLE, ActionAgents, operating_levels
from vercausa.causal import CausalLayer, training_frame
from vercausa.explain import typical_values
from vercausa.ground_truth import true_counterfactual, true_outcome
from vercausa.kg import InMemoryGraphStore
from vercausa.readings import DEFAULT_CSV, load_ai4i
from vercausa.simulation import SimulationLayer

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def agents():
    store = InMemoryGraphStore()
    store.apply_schema()
    causal = CausalLayer.fit_from_store(store, training_frame(DEFAULT_CSV))
    readings = list(load_ai4i())
    levels = operating_levels(pd.DataFrame([r.values for r in readings]))
    return ActionAgents(causal, SimulationLayer(causal), typical_values(readings), levels)


@pytest.fixture(scope="module")
def rows():
    return {r.source_row: dict(r.values, product_variant=r.product_variant) for r in load_ai4i()}


def _truly_safe(values, interventions):
    obs = dict(values, product_variant={"L": 0.0, "M": 1.0, "H": 2.0}[values["product_variant"]])
    state = true_counterfactual(obs, interventions)
    return not any(true_outcome(m, state) for m in ("TWF", "HDF", "PWF", "OSF"))


def test_low_power_failure_gets_more_cutting_load(agents, rows):
    plan = agents.plan(rows[51], ["PWF"])
    top = plan["recommendations"][0]
    assert top["interventions"].keys() == {"torque"}
    assert top["changes"]["torque"]["after"] > top["changes"]["torque"]["before"]
    assert top["resolves"] and top["trustworthy"] and top["rank"] == 1
    assert _truly_safe(rows[51], top["interventions"])
    assert [t.split(":")[0] for t in plan["trace"]] == ["Diagnostician", "Planner", "Simulator", "Ranker"]


def test_replanner_fixes_the_side_effect_of_raising_speed(agents, rows):
    # Raising speed alone clears this heat failure but pushes power over 9,000 W.
    plan = agents.plan(rows[3237], ["HDF"])
    assert plan["combined"]
    assert any(t.startswith("Re-planner") and "power failure" in t for t in plan["trace"])
    top = plan["recommendations"][0]
    assert top["interventions"].keys() == {"rotational_speed", "torque"}
    assert top["effort"] == 1
    assert _truly_safe(rows[3237], top["interventions"])


def test_tool_wear_failure_only_considers_a_new_tool(agents, rows):
    plan = agents.plan(rows[78], ["TWF"])
    assert plan["considered"] == 1
    assert plan["recommendations"][0]["title"] == "Replace the cutting tool"


def test_actions_stay_on_permitted_causes_and_observed_settings(agents, rows):
    plan = agents.plan(rows[70], ["OSF"])
    permitted = {s for m in plan["targets"] for s in agents.causal.explanatory_signals(m)}
    for a in plan["recommendations"] + plan["rejected"]:
        for sig, v in a["interventions"].items():
            assert sig in ACTIONABLE and sig in permitted
            assert v in agents.levels[sig]["levels"]


def test_shown_recommendations_are_distinct_and_ranked(agents, rows):
    plan = agents.plan(rows[3237], ["HDF"])
    keys = [frozenset(a["interventions"]) for a in plan["recommendations"]]
    assert len(keys) == len(set(keys))
    assert [a["rank"] for a in plan["recommendations"]] == list(range(1, len(keys) + 1))
    assert all(a["resolves"] for a in plan["recommendations"])
    assert not any(a["resolves"] for a in plan["rejected"])


def test_healthy_machine_needs_no_action(agents, rows):
    plan = agents.plan(rows[1])
    assert plan["targets"] == [] and plan["recommendations"] == []
    assert "no action needed" in plan["trace"][0]
