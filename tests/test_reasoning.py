import json
import warnings

import pandas as pd
import pytest

from vercausa.agents import ActionAgents, operating_levels
from vercausa.causal import CausalLayer, training_frame
from vercausa.explain import typical_values
from vercausa.kg import InMemoryGraphStore
from vercausa.readings import DEFAULT_CSV, load_ai4i
from vercausa.reasoning import (
    HELP, GraphRAG, GraphRetriever, OllamaLLM, rule_intent, understand, verify,
)
from vercausa.simulation import SimulationLayer
from vercausa.twin import DigitalTwin

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def retriever():
    store = InMemoryGraphStore()
    store.apply_schema()
    twin = DigitalTwin(store)
    twin.register("MILL-01")
    readings = list(load_ai4i())
    for r in readings:
        twin.ingest(r)
    causal = CausalLayer.fit_from_store(store, training_frame(DEFAULT_CSV))
    sim, typical = SimulationLayer(causal), typical_values(readings)
    agents = ActionAgents(causal, sim, typical, operating_levels(pd.DataFrame([r.values for r in readings])))
    return GraphRetriever(store, causal, sim, typical, "MILL-01", agents)


class ScriptedLLM:
    """Returns canned replies in order and records what it was asked."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, system, user, schema=None):
        self.calls.append((system, user, schema))
        return self.replies.pop(0)


@pytest.mark.parametrize("question, expected", [
    ("Why did the machine fail at reading 51?", {"intent": "explain_event", "row": 51}),
    ("What if torque was reduced by 20% in the last power failure?",
     {"intent": "what_if", "signal": "torque", "change_type": "percent", "change_value": -20.0, "mechanism": "PWF"}),
    ("What if we replace the tool at reading 9975?",
     {"intent": "what_if", "signal": "tool_wear", "change_type": "absolute", "change_value": 0.0, "row": 9975}),
    ("What would happen if speed went to 1500 rpm?", {"intent": "what_if", "signal": "rotational_speed",
                                                       "change_type": "absolute", "change_value": 1500.0}),
    ("How is the machine doing right now?", {"intent": "current_state"}),
    ("Which failures happened so far?", {"intent": "list_events"}),
])
def test_rules_read_explicit_details(question, expected):
    got = rule_intent(question)
    for k, v in expected.items():
        assert got.get(k) == v, (k, got)


def test_understand_falls_back_to_rules_without_llm():
    assert understand("Tell me a joke", None)["intent"] == "unsupported"
    assert understand("Why did reading 51 fail?", None) == {
        "intent": "explain_event", "row": 51, "mechanism": None, "signal": None,
        "change_type": None, "change_value": None}


def test_rules_override_llm_on_explicit_numbers():
    llm = ScriptedLLM(json.dumps({"intent": "explain_event", "row": 15, "mechanism": "PWF",
                                  "signal": None, "change_type": None, "change_value": None}))
    assert understand("Why did the power failure at reading 51 happen?", llm)["row"] == 51


def test_explain_event_facts_come_from_the_graph(retriever):
    facts = retriever.retrieve({"intent": "explain_event", "row": 51, "mechanism": None})
    text = " ".join(f.text for f in facts.items)
    assert "reading 51" in text and "Power Failure (PWF)" in text
    assert "power 1,378 W" in text
    assert "torque → power" in text
    assert "The cutting force (torque) dropped too low (4.6 Nm)" in text
    assert facts.items[0].source.startswith("(:FailureEvent")


def test_what_if_facts_run_the_simulation(retriever):
    facts = retriever.retrieve({"intent": "what_if", "row": 9975, "mechanism": "PWF", "signal": "torque",
                                "change_type": "percent", "change_value": -20.0})
    text = " ".join(f.text for f in facts.items)
    assert "torque 68.2 → 54.6 Nm (-20%)" in text
    assert "power 9,720 → 7,776 W" in text
    assert "100% before, 0% after" in text
    assert "Simulated risk of any failure: 100% before, 0.2% after; failures still likely after the change: none" in text


def test_current_state_and_list_events(retriever):
    state = " ".join(f.text for f in retriever.retrieve({"intent": "current_state"}).items)
    assert "status is running" in state and "Failure risk from the causal model right now" in state
    events = retriever.retrieve({"intent": "list_events", "mechanism": None}).items
    assert events[0].text.startswith("402 failure events are recorded")


def test_verify_accepts_grounded_and_rejects_invented_answers(retriever):
    facts = retriever.retrieve({"intent": "explain_event", "row": 51, "mechanism": None}).items
    assert verify("Power fell to 1378 W because torque dropped to 4.6 Nm [F3][F6].", facts) == []
    assert any("not in the facts" in i for i in verify("Torque was 12.5 Nm [F3].", facts))
    assert verify("Power was too low.", facts) == ["the answer cites no facts"]
    assert any("do not exist" in i for i in verify("It failed [F42].", facts))


def test_graphrag_uses_a_grounded_llm_answer(retriever):
    llm = ScriptedLLM(json.dumps({"intent": "explain_event", "row": 51, "mechanism": None, "signal": None,
                                  "change_type": None, "change_value": None}),
                      "The power fell to 1,378 W, below the safe minimum, because the cutting force dropped to 4.6 Nm [F3][F6].")
    ans = GraphRAG(retriever, llm).answer("Why did the machine fail at reading 51?")
    assert ans.source == "llm" and ans.issues == []
    assert "[F" in llm.calls[1][1] and "Question: Why did the machine fail" in llm.calls[1][1]


def test_graphrag_replaces_a_hallucinated_answer_with_facts(retriever):
    llm = ScriptedLLM(json.dumps({"intent": "explain_event", "row": 51, "mechanism": None, "signal": None,
                                  "change_type": None, "change_value": None}),
                      "The motor overheated to 95 °C and the bearings failed [F3].")
    ans = GraphRAG(retriever, llm).answer("Why did the machine fail at reading 51?")
    assert ans.source == "template"
    assert any("95" in i for i in ans.issues)
    assert "dropped too low (4.6 Nm)" in ans.text and "[F" in ans.text
    assert verify(ans.text, ans.facts) == []


def test_unsupported_questions_get_help_without_generation(retriever):
    llm = ScriptedLLM(json.dumps({"intent": "unsupported", "row": None, "mechanism": None, "signal": None,
                                  "change_type": None, "change_value": None}))
    ans = GraphRAG(retriever, llm).answer("What's the weather tomorrow?")
    assert ans.source == "help" and ans.text == HELP
    assert len(llm.calls) == 1


@pytest.mark.skipif(not OllamaLLM().available(), reason="local Ollama model not available")
def test_real_local_llm_answers_are_grounded(retriever):
    rag = GraphRAG(retriever, OllamaLLM())
    for q in ("Why did the machine fail at reading 51?",
              "What if torque was reduced by 20% at reading 9975 in the power failure?"):
        ans = rag.answer(q)
        assert ans.source in ("llm", "template")
        assert verify(ans.text, ans.facts) == []


@pytest.mark.parametrize("question, expected", [
    ("What if the cutting force was increased by 20%?", {"signal": "torque", "change_value": 20.0}),
    ("What if the cutting force was cut by 20%?", {"signal": "torque", "change_value": -20.0}),
    ("What caused the overstrain failure at reading 9975 when power was high?", {"mechanism": "OSF", "row": 9975}),
    ("Why did the power failure at reading 9975 happen, was it the heat?", {"mechanism": "PWF"}),
    ("Explain the HDF at reading 4852", {"mechanism": "HDF"}),
])
def test_rules_regressions(question, expected):
    got = rule_intent(question)
    for k, v in expected.items():
        assert got.get(k) == v, (k, got)


def test_llm_fields_are_type_checked():
    llm = ScriptedLLM(json.dumps({"intent": "what_if", "row": "9975", "mechanism": "XYZ", "signal": "humidity",
                                  "change_type": "lots", "change_value": "-20"}))
    got = understand("hmm", llm)
    assert got["row"] == 9975 and got["change_value"] == -20.0
    assert got["mechanism"] is None and got["signal"] is None and got["change_type"] is None


@pytest.mark.parametrize("question, expected", [
    ("What should I do about reading 9975?", {"intent": "recommend", "row": 9975}),
    ("How do we fix the power failure?", {"intent": "recommend", "mechanism": "PWF"}),
    ("What do you recommend right now?", {"intent": "recommend"}),
    ("What if we replace the tool at reading 9975?", {"intent": "what_if"}),
])
def test_rules_read_recommend_questions(question, expected):
    got = rule_intent(question)
    for k, v in expected.items():
        assert got.get(k) == v, (k, got)


def test_recommend_facts_come_from_the_action_layer(retriever):
    facts = retriever.retrieve({"intent": "recommend", "row": 51, "mechanism": None}).items
    plan = [f for f in facts if f.source.startswith("Action Layer")]
    assert plan[0].text.startswith("The Action Layer's agents tested")
    assert plan[1].text.startswith("Recommended action 1: Increase the cutting load")
    ans = GraphRAG(retriever, None).generate("What should I do about reading 51?",
                                             {"intent": "recommend"}, facts)
    assert "Recommended action 1" in ans.text and verify(ans.text, facts) == []


def test_recommend_for_live_state_reports_no_action_when_healthy(retriever):
    facts = retriever.retrieve({"intent": "recommend", "row": None, "mechanism": None}).items
    assert facts[0].text.startswith("Machine MILL-01 status is")
    assert any("no action is needed" in f.text or f.text.startswith("Recommended action") for f in facts)


def test_asking_what_to_do_overrides_the_llm():
    llm = ScriptedLLM(json.dumps({"intent": "current_state", "row": None, "mechanism": None, "signal": None,
                                  "change_type": None, "change_value": None}))
    assert understand("What should I do about reading 51?", llm)["intent"] == "recommend"


def test_an_explicit_reading_wins_over_a_wrongly_inferred_failure_type(retriever):
    # Reading 51 is a power failure; the LLM guessed tool wear.
    facts = retriever.retrieve({"intent": "explain_event", "row": 51, "mechanism": "TWF"}).items
    assert "reading 51" in facts[0].text and "Power Failure (PWF)" in facts[0].text
