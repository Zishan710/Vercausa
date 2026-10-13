"""Phase 2 demo: replay AI4I 2020 into the Knowledge Graph / Digital Twin.

Uses Neo4j if NEO4J_URI (+ NEO4J_PASSWORD) is set, otherwise an in-memory
graph. Phase 7 replaces this loop with a live MQTT/Kafka stream; the twin
code it calls stays the same.

    python scripts/build_twin.py --limit 2000
"""

import argparse
import collections
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vercausa.kg import open_store
from vercausa.readings import DEFAULT_MACHINE, load_ai4i
from vercausa.schema import FAILURE_MODES
from vercausa.twin import DigitalTwin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="rows to replay (default: all 10,000)")
    args = parser.parse_args()

    store = open_store()
    print(f"Backend: {type(store).__name__}")
    store.apply_schema()
    twin = DigitalTwin(store)
    twin.register(DEFAULT_MACHINE)

    opened = collections.Counter()
    n = 0
    start = time.perf_counter()
    for reading in load_ai4i(limit=args.limit):
        for event in twin.ingest(reading).new_events:
            opened[event.mechanism] += 1
        n += 1
    elapsed = time.perf_counter() - start

    print(f"Replayed {n} readings in {elapsed:.1f}s ({n / elapsed:.0f}/s)\n")

    print("Failure events opened (rising edge only):")
    for mode in FAILURE_MODES:
        print(f"  {mode.code}  {opened.get(mode.code, 0):>4}   {mode.name}")

    state = store.machine_state(DEFAULT_MACHINE)
    print(f"\nTwin state for {DEFAULT_MACHINE} @ {state['last_updated']}")
    print(f"  status={state['status']}  variant={state['product_variant']}  "
          f"active_alerts={state['active_alerts'] or 'none'}")
    for signal, s in sorted(state["sensors"].items()):
        print(f"  {signal:<20} {s['value']:>8} {s['unit']}")

    print("\nCandidate causes per failure mode (direct + upstream, from the graph):")
    for mode in FAILURE_MODES:
        causes = store.candidate_causes(mode.code)
        print(f"  {mode.code}: {', '.join(causes) if causes else '(none -- not observable)'}")

    print("\nMost recent events:")
    for e in store.failure_events(DEFAULT_MACHINE, limit=5):
        print(f"  row {e['source_row']:>5}  {e['mechanism']}  {e['reasons']}")

    store.close()


if __name__ == "__main__":
    main()
