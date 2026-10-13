"""Simulated machine: publishes AI4I 2020 readings over MQTT as a live feed.

Behaves like a sensor gateway on the shop floor -- one message per reading,
stamped with the current time, on topic vercausa/<machine>/readings. Stop it
with Ctrl+C and the dashboard marks the machine offline.

    python scripts/publish_machine.py                    # 1 reading/s from row 1
    python scripts/publish_machine.py --rate 5 --start 40
    python scripts/publish_machine.py --glitch-every 25  # send a corrupt reading now and then
"""

import argparse
import json
import pathlib
import sys
import time
from dataclasses import replace
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vercausa.readings import DEFAULT_MACHINE, load_ai4i
from vercausa.schema import compute_derived
from vercausa.streaming import DEFAULT_HOST, DEFAULT_PORT, Publisher, encode_reading


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish AI4I 2020 readings over MQTT")
    parser.add_argument("--broker", default=f"{DEFAULT_HOST}:{DEFAULT_PORT}", help="MQTT broker host:port")
    parser.add_argument("--machine", default=DEFAULT_MACHINE)
    parser.add_argument("--rate", type=float, default=1.0, help="readings per second")
    parser.add_argument("--start", type=int, default=1, help="first dataset row (UDI) to send")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many readings")
    parser.add_argument("--qos", type=int, choices=(0, 1), default=0, help="1 needs a full broker such as Mosquitto")
    parser.add_argument("--glitch-every", type=int, default=0,
                        help="every N readings, also send a corrupted one (impossible torque)")
    args = parser.parse_args()
    host, _, port = args.broker.rpartition(":")

    try:
        pub = Publisher(f"machine-{args.machine}", host, int(port), qos=args.qos)
    except OSError as e:
        sys.exit(f"Cannot reach the MQTT broker at {host}:{port} ({e}). "
                 "Start the dashboard first:  python -m app.server --source mqtt")

    print(f"Machine {args.machine} publishing to mqtt://{host}:{port} at {args.rate:g} reading(s)/s  (Ctrl+C to stop)")
    sent = 0
    try:
        for reading in load_ai4i(machine_id=args.machine):
            if reading.source_row < args.start:
                continue
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            reading = replace(reading, timestamp=now)
            pub.publish(reading)
            sent += 1
            v = reading.values
            power = compute_derived("power", v)
            print(f"→ row {reading.source_row:>5}  {v['rotational_speed']:>6.0f} rpm  {v['torque']:5.1f} Nm  "
                  f"{power:6.0f} W  wear {v['tool_wear']:3.0f} min  [{reading.product_variant}]")

            if args.glitch_every and sent % args.glitch_every == 0:
                topic, payload = encode_reading(reading)
                bad = json.loads(payload)
                bad["values"]["torque"] = 9999.0
                pub.publish_raw(topic, json.dumps(bad).encode())
                print("✗ sent a corrupted reading (torque 9999 Nm) — the dashboard should reject it")

            if args.limit and sent >= args.limit:
                break
            time.sleep(1 / args.rate)
    except KeyboardInterrupt:
        print("\nMachine stopped.")
    finally:
        time.sleep(0.2)
        pub.close()
    print(f"Sent {sent} readings.")


if __name__ == "__main__":
    main()
