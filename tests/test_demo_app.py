import warnings

import pytest
from fastapi.testclient import TestClient

from app.engine import ReplayEngine
from app.server import create_app
from vercausa.causal import CausalLayer, training_frame
from vercausa.kg import InMemoryGraphStore
from vercausa.readings import DEFAULT_CSV

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def causal():
    store = InMemoryGraphStore()
    store.apply_schema()
    return CausalLayer.fit_from_store(store, training_frame(DEFAULT_CSV))


@pytest.fixture
def engine(causal):
    return ReplayEngine(InMemoryGraphStore, causal)


def test_starts_empty_and_steps_one_reading(engine):
    assert engine.state()["live"] is None
    engine.step()
    s = engine.state()
    assert s["position"] == 1
    assert s["status"] == "running"
    assert set(s["live"]["risk"]) == {"TWF", "HDF", "PWF", "OSF"}


def test_next_failure_stops_on_a_reading_that_opens_an_event(engine):
    new = engine.next_failure()
    assert new
    s = engine.state()
    assert s["status"] == "alert"
    assert s["new_event_ids"] == [e["id"] for e in new]
    event = new[0]
    assert event["causes"]
    assert sum(c["share"] for c in event["causes"]) == pytest.approx(1.0)
    assert event["path"]


def test_power_failure_explanation_blames_torque_and_torque_cut_fixes_it(engine):
    while True:
        new = engine.next_failure()
        pwf = [e for e in new if e["mechanism"] == "PWF" and e["derived"]["power"] > 9000]
        if pwf:
            break
    e = pwf[0]
    assert e["causes"][0]["signal"] == "torque"
    assert {p["via"] for p in e["path"]} == {"power"}
    r = engine.simulate(e["id"], {"torque": e["values"]["torque"] * 0.8})
    assert r["prevents_this"]
    assert r["derived_after"]["power"] < 9000


def test_plain_summary_blames_low_torque_not_high_speed_for_low_power(engine):
    e = engine.next_failure()[0]
    assert e["row"] == 51 and e["derived"]["power"] < 3500
    assert e["summary"].startswith("The cutting force (torque) dropped too low")
    assert "speed" not in e["summary"]
    assert e["summary"].endswith("so the power fell below the safe minimum.")


def test_plain_summary_wording_per_failure_type(engine):
    by_mode = {}
    while len(by_mode) < 4 and not engine.finished:
        for e in engine.next_failure():
            by_mode.setdefault(e["mechanism"], e)
    assert "too high" in next(e for e in engine.events
                              if e["mechanism"] == "PWF" and e["derived"]["power"] > 9000)["summary"]
    assert "couldn't get rid of its heat" in by_mode["HDF"]["summary"]
    assert "°C" in by_mode["HDF"]["summary"] or "too slow" in by_mode["HDF"]["summary"]
    assert "overstrained" in by_mode["OSF"]["summary"]
    assert by_mode["TWF"]["summary"].startswith("The tool was worn out")


def test_reset_clears_the_replay(engine):
    engine.next_failure()
    engine.reset()
    s = engine.state()
    assert s["position"] == 0 and s["events"] == [] and s["live"] is None


def test_api_controls_and_simulate(engine):
    client = TestClient(create_app(engine))
    assert client.get("/api/meta").json()["machine"] == "MILL-01"
    assert client.post("/api/control", json={"action": "next_failure"}).status_code == 200
    state = client.get("/api/state").json()
    event = state["events"][0]
    signal = event["causes"][0]["signal"]
    r = client.post("/api/simulate", json={"event_id": event["id"], "interventions": {signal: event["values"][signal]}})
    assert r.status_code == 200
    assert r.json()["risk_after"] == pytest.approx(r.json()["risk_before"])
    assert client.post("/api/simulate", json={"event_id": "nope", "interventions": {}}).status_code == 404
    assert client.post("/api/control", json={"action": "fly"}).status_code == 400
    assert client.get("/").status_code == 200
