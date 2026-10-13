"""Phase 4 demo: "what if we had changed X?" on real failure events.

Replays AI4I 2020 into the Digital Twin, then for the most recent event of
each failure type runs the fix a maintenance engineer would actually
consider, starting from the state the twin recorded at that moment. Ends
with a what-if on the machine's live state.

    python scripts/whatif_demo.py
"""

import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from vercausa.causal import CausalLayer, training_frame
from vercausa.kg import open_store
from vercausa.readings import DEFAULT_CSV, DEFAULT_MACHINE, load_ai4i
from vercausa.simulation import SimulationLayer
from vercausa.twin import DigitalTwin

UNITS = {"air_temperature": "K", "process_temperature": "K", "rotational_speed": "rpm",
         "torque": "Nm", "tool_wear": "min", "product_variant": "",
         "power": "W", "temperature_difference": "K"}


def scenario_for(event: dict):
    snap = event["snapshot"]
    mech = event["mechanism"]
    if mech in ("TWF", "OSF"):
        return "Replace the tool now", {"tool_wear": 0.0}
    if mech == "HDF":
        return "Raise spindle speed to 1500 rpm", {"rotational_speed": 1500.0}
    power = snap["torque"] * snap["rotational_speed"] * 2 * 3.141592653589793 / 60
    if power > 9000:
        return "Reduce torque by 20%", {"torque": lambda t: t * 0.8}
    return "Increase torque by 20%", {"torque": lambda t: t * 1.2}


def pct(p: float) -> str:
    return f"{p:.1%}" if p >= 0.01 or p == 0 else f"{p:.2%}"


def show(title: str, mode: str, result) -> None:
    print(f"  What if: {title}")
    for sig, (before, after) in result.changed_signals().items():
        print(f"    {sig:<22} {before:9.1f} -> {after:9.1f} {UNITS[sig]}")
    for d in result.derived_before:
        before, after = result.derived_before[d], result.derived_after[d]
        if abs(after - before) > 1e-6:
            print(f"    {d:<22} {before:9.1f} -> {after:9.1f} {UNITS[d]}   (physics)")
    print(f"    {mode} risk             {pct(result.risk_before[mode]):>9} -> {pct(result.risk_after[mode]):>9}")
    print(f"    any-failure risk       {pct(result.any_before):>9} -> {pct(result.any_after):>9}")
    if not result.trustworthy:
        print(f"    CAUTION: outside observed data for {', '.join(result.outside_observed)}"
              " -- the model is extrapolating here")


def main() -> None:
    data = training_frame(DEFAULT_CSV)
    store = open_store()
    store.apply_schema()
    sim = SimulationLayer(CausalLayer.fit_from_store(store, data))

    twin = DigitalTwin(store)
    twin.register(DEFAULT_MACHINE)
    for reading in load_ai4i():
        twin.ingest(reading)

    for mech in ("HDF", "PWF", "OSF", "TWF"):
        event = store.failure_events(DEFAULT_MACHINE, mechanism=mech, limit=1)[0]
        title, change = scenario_for(event)
        print(f"\n{mech} event at row {event['source_row']} ({event['timestamp']})")
        print(f"  Detected because: {event['reasons']}")
        result = sim.simulate(event["snapshot"], change)
        show(title, mech, result)

    state = store.machine_state(DEFAULT_MACHINE)
    print(f"\nLIVE twin state for {DEFAULT_MACHINE} @ {state['last_updated']} "
          f"(tool wear {state['sensors']['tool_wear']['value']:.0f} min)")
    show("Replace the tool now", "OSF", sim.simulate_twin(store, DEFAULT_MACHINE, {"tool_wear": 0.0}))

    store.close()


if __name__ == "__main__":
    main()
