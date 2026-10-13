"""Live sensor streaming over MQTT.

The message contract is the only thing a data source must honour: the CSV
replay, a hardware gateway or an OPC-UA bridge all publish the same JSON to
the same topic, and nothing downstream changes.

    topic:   vercausa/<machine_id>/readings
    payload: {"machine_id": "MILL-01", "timestamp": "2026-10-13T10:14:03+00:00",
              "product_variant": "L", "source_row": 51,
              "values": {"air_temperature": 298.1, "process_temperature": 308.6,
                         "rotational_speed": 1551, "torque": 42.8, "tool_wear": 0}}

Readings are published at QoS 0 by default: a new reading supersedes the last
one every second, so a lost message costs less than redelivery latency. The
bundled broker (mqttools) supports QoS 0 only; with Mosquitto running, the
dashboard uses it automatically and QoS 1 becomes available.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from datetime import datetime
from typing import Callable

import mqttools
import paho.mqtt.client as mqtt

from vercausa.readings import SensorReading
from vercausa.schema import SENSOR_SIGNALS

TOPIC = "vercausa/{machine_id}/readings"
SUBSCRIBE_TOPIC = "vercausa/+/readings"
DEFAULT_HOST, DEFAULT_PORT = "127.0.0.1", 1883
VARIANTS = ("L", "M", "H")

# Physically possible ranges for a milling machine. Anything outside is a
# sensor fault or a corrupted message, not a reading to act on.
PLAUSIBLE = {
    "air_temperature": (250.0, 400.0),
    "process_temperature": (250.0, 450.0),
    "rotational_speed": (0.0, 10_000.0),
    "torque": (0.0, 500.0),
    "tool_wear": (0.0, 1_000.0),
}


class MessageError(ValueError):
    """A message that does not meet the contract. str() says what is wrong."""


def encode_reading(reading: SensorReading) -> tuple:
    payload = {
        "machine_id": reading.machine_id,
        "timestamp": reading.timestamp,
        "product_variant": reading.product_variant,
        "source_row": reading.source_row,
        "values": {s.name: reading.values[s.name] for s in SENSOR_SIGNALS},
    }
    return TOPIC.format(machine_id=reading.machine_id), json.dumps(payload).encode()


def decode_reading(topic: str, payload: bytes) -> SensorReading:
    try:
        msg = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise MessageError("payload is not valid JSON")
    if not isinstance(msg, dict):
        raise MessageError("payload must be a JSON object")

    for field in ("machine_id", "timestamp", "product_variant", "values"):
        if field not in msg:
            raise MessageError(f"missing field '{field}'")
    machine_id = msg["machine_id"]
    if topic != TOPIC.format(machine_id=machine_id):
        raise MessageError(f"machine_id '{machine_id}' does not match topic '{topic}'")
    timestamp = msg["timestamp"]
    if isinstance(timestamp, str) and timestamp.endswith(("Z", "z")):
        timestamp = timestamp[:-1] + "+00:00"
    try:
        datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        raise MessageError(f"timestamp '{msg['timestamp']}' is not ISO 8601")
    if msg["product_variant"] not in VARIANTS:
        raise MessageError(f"product_variant must be one of {', '.join(VARIANTS)}")

    raw = msg["values"]
    if not isinstance(raw, dict):
        raise MessageError("'values' must be an object")
    values = {}
    for s in SENSOR_SIGNALS:
        if s.name not in raw:
            raise MessageError(f"missing value '{s.name}'")
        v = raw[s.name]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise MessageError(f"'{s.name}' must be a number")
        lo, hi = PLAUSIBLE[s.name]
        if not lo <= v <= hi:
            raise MessageError(f"'{s.name}' = {v} is outside the possible range {lo:g}–{hi:g} {s.unit}")
        values[s.name] = float(v)

    row = msg.get("source_row", 0)
    if isinstance(row, bool) or not isinstance(row, int) or row < 0:
        raise MessageError("'source_row' must be a non-negative integer")
    return SensorReading(machine_id, timestamp, msg["product_variant"], values, row)


def _client(client_id: str) -> mqtt.Client:
    return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, protocol=mqtt.MQTTv5)


class Publisher:
    def __init__(self, client_id: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, qos: int = 0):
        self.qos = qos
        self.client = _client(client_id)
        self.client.connect(host, port)
        self.client.loop_start()

    def publish(self, reading: SensorReading) -> None:
        topic, payload = encode_reading(reading)
        self.client.publish(topic, payload, qos=self.qos)

    def publish_raw(self, topic: str, payload: bytes) -> None:
        self.client.publish(topic, payload, qos=self.qos)

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()


class Subscriber:
    """Receives readings on a background thread and hands each one to a callback.

    on_reading(reading) gets every valid message; on_reject(reason, topic)
    gets every message that breaks the contract; on_link(connected) reports
    broker connection changes.
    """

    def __init__(self, client_id: str, on_reading: Callable, on_reject: Callable,
                 on_link: Callable = lambda connected: None,
                 host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, qos: int = 0):
        self.client = _client(client_id)

        def on_connect(client, userdata, flags, reason, props):
            ok = not reason.is_failure
            on_link(ok)
            if ok:
                client.subscribe(SUBSCRIBE_TOPIC, qos=qos)

        def on_disconnect(client, userdata, flags, reason, props):
            on_link(False)

        def on_message(client, userdata, msg):
            try:
                reading = decode_reading(msg.topic, msg.payload)
            except MessageError as e:
                on_reject(str(e), msg.topic)
                return
            on_reading(reading)

        self.client.on_connect = on_connect
        self.client.on_disconnect = on_disconnect
        self.client.on_message = on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=5)
        self.client.connect_async(host, port)
        self.client.loop_start()

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()


def broker_running(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 0.5) -> bool:
    """True if something (e.g. Mosquitto) is already accepting connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def serve_broker(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Run the bundled MQTT broker until cancelled. Must run inside an event loop."""
    await mqttools.Broker((host, port)).serve_forever()


def start_broker_thread(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> threading.Thread:
    """Run the bundled broker on its own thread and event loop."""
    thread = threading.Thread(target=lambda: asyncio.run(serve_broker(host, port)), daemon=True)
    thread.start()
    return thread
