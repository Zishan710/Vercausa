import json
import os
import shutil
import socket
import subprocess
import time
import warnings

import pytest
from fastapi.testclient import TestClient

from app.engine import ReplayEngine
from app.server import create_app
from vercausa.causal import CausalLayer, training_frame
from vercausa.kg import InMemoryGraphStore
from vercausa.readings import DEFAULT_CSV, load_ai4i
from vercausa.streaming import (
    MessageError, Publisher, Subscriber, broker_running, decode_reading, encode_reading, start_broker_thread,
)

warnings.filterwarnings("ignore")

READINGS = list(load_ai4i(limit=60))


def _payload(**changes):
    topic, raw = encode_reading(READINGS[0])
    msg = json.loads(raw)
    for k, v in changes.items():
        if k in msg["values"]:
            msg["values"][k] = v
        elif v is None:
            msg.pop(k)
        else:
            msg[k] = v
    return topic, json.dumps(msg).encode()


def test_encode_decode_round_trip():
    for r in READINGS[:10]:
        assert decode_reading(*encode_reading(r)) == r


@pytest.mark.parametrize("payload, reason", [
    (b"not json", "not valid JSON"),
    (b"[1, 2]", "JSON object"),
    (_payload(machine_id=None)[1], "missing field 'machine_id'"),
    (_payload(timestamp="yesterday")[1], "not ISO 8601"),
    (_payload(product_variant="X")[1], "product_variant"),
    (_payload(torque=9999.0)[1], "outside the possible range"),
    (_payload(torque="high")[1], "must be a number"),
    (_payload(source_row=-3)[1], "non-negative"),
])
def test_bad_messages_are_rejected_with_a_reason(payload, reason):
    with pytest.raises(MessageError, match=reason):
        decode_reading("vercausa/MILL-01/readings", payload)


def test_machine_id_must_match_topic():
    topic, payload = encode_reading(READINGS[0])
    with pytest.raises(MessageError, match="does not match topic"):
        decode_reading("vercausa/MILL-02/readings", payload)


@pytest.fixture(scope="module")
def causal():
    store = InMemoryGraphStore()
    store.apply_schema()
    return CausalLayer.fit_from_store(store, training_frame(DEFAULT_CSV))


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_engine_tracks_link_state_and_goes_offline(causal):
    clock = FakeClock()
    eng = ReplayEngine(InMemoryGraphStore, causal, mode="mqtt", clock=clock, offline_after=5)
    assert eng.link_state() == "broker-down"
    eng.set_broker_connected(True)
    assert eng.link_state() == "waiting"
    eng.ingest_external(READINGS[0])
    assert eng.link_state() == "live"
    assert eng.state()["status"] == "running"
    clock.t += 6
    assert eng.link_state() == "offline"
    assert eng.state()["status"] == "offline"
    eng.ingest_external(READINGS[1])
    assert eng.link_state() == "live"
    assert eng.state()["source"]["received"] == 2


def test_engine_rejects_other_machines_and_counts_rejections(causal):
    eng = ReplayEngine(InMemoryGraphStore, causal, mode="mqtt")
    other = READINGS[0].__class__("MILL-99", READINGS[0].timestamp, "L", READINGS[0].values, 1)
    assert eng.ingest_external(other) == []
    eng.reject("'torque' = 9999 is outside the possible range")
    src = eng.state()["source"]
    assert src["received"] == 0 and src["rejected"] == 2
    assert "outside the possible range" in src["last_rejection"]


def test_external_readings_open_failure_events_like_replay(causal):
    eng = ReplayEngine(InMemoryGraphStore, causal, mode="mqtt")
    opened = [e for r in READINGS for e in eng.ingest_external(r)]
    assert [e["row"] for e in opened][:1] == [51]
    assert opened[0]["summary"].startswith("The cutting force (torque) dropped too low")


def test_mqtt_mode_only_allows_reset(causal):
    eng = ReplayEngine(InMemoryGraphStore, causal, mode="mqtt")
    with TestClient(create_app(eng, run_broker=False, broker=("127.0.0.1", _free_port()))) as client:
        assert client.post("/api/control", json={"action": "play"}).status_code == 409
        assert client.post("/api/control", json={"action": "next_failure"}).status_code == 409
        assert client.post("/api/control", json={"action": "reset"}).status_code == 200
        assert client.get("/api/state").json()["source"]["mode"] == "mqtt"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_readings_flow_end_to_end_through_a_real_broker():
    port = _free_port()
    start_broker_thread("127.0.0.1", port)
    time.sleep(0.4)
    got, rejected, link = [], [], []
    sub = Subscriber("test-sub", on_reading=got.append, on_reject=lambda reason, topic: rejected.append(reason),
                     on_link=link.append, host="127.0.0.1", port=port)
    assert _wait(lambda: link and link[-1] is True)
    pub = Publisher("test-machine", "127.0.0.1", port)
    time.sleep(0.3)
    for r in READINGS[:30]:
        pub.publish(r)
    topic, bad = _payload(torque=9999.0)
    pub.publish_raw(topic, bad)
    assert _wait(lambda: len(got) == 30 and rejected)
    assert got == READINGS[:30]
    assert "outside the possible range" in rejected[0]
    pub.close()
    sub.close()


MOSQUITTO = shutil.which("mosquitto") or next(
    (p for p in ("/opt/homebrew/opt/mosquitto/sbin/mosquitto", "/usr/local/opt/mosquitto/sbin/mosquitto")
     if os.path.exists(p)), None)


def test_broker_detection():
    port = _free_port()
    assert not broker_running("127.0.0.1", port)
    start_broker_thread("127.0.0.1", port)
    assert _wait(lambda: broker_running("127.0.0.1", port))


@pytest.mark.skipif(MOSQUITTO is None, reason="Mosquitto is not installed")
def test_readings_flow_through_mosquitto_at_qos1():
    port = _free_port()
    proc = subprocess.Popen([MOSQUITTO, "-p", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _wait(lambda: broker_running("127.0.0.1", port))
        got, link = [], []
        sub = Subscriber("test-sub-q1", on_reading=got.append, on_reject=lambda *a: None,
                         on_link=link.append, host="127.0.0.1", port=port, qos=1)
        assert _wait(lambda: link and link[-1] is True)
        pub = Publisher("test-machine-q1", "127.0.0.1", port, qos=1)
        time.sleep(0.3)
        for r in READINGS[:30]:
            pub.publish(r)
        assert _wait(lambda: len(got) == 30)
        assert got == READINGS[:30]
        pub.close()
        sub.close()
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_utc_z_timestamps_are_accepted():
    topic, raw = encode_reading(READINGS[0])
    msg = json.loads(raw)
    msg["timestamp"] = "2026-10-13T10:14:03Z"
    assert decode_reading(topic, json.dumps(msg).encode()).timestamp == "2026-10-13T10:14:03+00:00"


def test_replaying_the_same_rows_again_opens_new_distinct_events(causal):
    from dataclasses import replace
    eng = ReplayEngine(InMemoryGraphStore, causal, mode="mqtt")
    run1 = [e for r in READINGS for e in eng.ingest_external(replace(r, timestamp="2026-10-13T10:00:00+05:30"))]
    run2 = [e for r in READINGS for e in eng.ingest_external(replace(r, timestamp="2026-10-13T10:05:00+05:30"))]
    assert run1 and run2 and run1[0]["id"] != run2[0]["id"]
    assert len(eng.store.failure_events("MILL-01")) == len(eng.events) == 2
