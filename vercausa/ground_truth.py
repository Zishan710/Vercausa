"""AI4I 2020's documented failure mechanisms, used as a scoring oracle.

Evaluation scripts use this to check the model's answers against what the
dataset's generator says would really happen. The Causal, Simulation and
Action layers never see it.
"""

from __future__ import annotations

from vercausa.causal import VARIANT_CODES
from vercausa.detection import TWF_WINDOW_MIN, check_hdf, check_osf, check_pwf

CODE_TO_VARIANT = {v: k for k, v in VARIANT_CODES.items()}


def true_counterfactual(obs: dict, changes: dict) -> dict:
    """The true state after do(changes). In AI4I, process temperature is air
    temperature + 10K + noise, so changing the air shifts process temperature
    by the same amount; product variant sets tool wear *rate*, so changing it
    leaves the current tool wear unchanged."""
    cf = dict(obs)
    for signal, value in changes.items():
        cf[signal] = value
        if signal == "air_temperature" and "process_temperature" not in changes:
            cf["process_temperature"] = obs["process_temperature"] + (value - obs["air_temperature"])
    return cf


def true_outcome(mode: str, s: dict) -> bool:
    """Whether the documented rule for this mode fires in state s
    (product variant as its numeric code). For TWF: the tool has reached the
    200 min point after which the generator makes it fail."""
    if mode == "HDF":
        return check_hdf(s["air_temperature"], s["process_temperature"], s["rotational_speed"])
    if mode == "PWF":
        return check_pwf(s["torque"], s["rotational_speed"])
    if mode == "OSF":
        return check_osf(s["tool_wear"], s["torque"], CODE_TO_VARIANT[int(s["product_variant"])])
    if mode == "TWF":
        return s["tool_wear"] >= TWF_WINDOW_MIN[0]
    raise ValueError(f"no documented rule for {mode}")


# The signals each documented rule actually reads.
TRUE_CAUSES = {
    "TWF": {"tool_wear"},
    "HDF": {"air_temperature", "process_temperature", "rotational_speed"},
    "PWF": {"torque", "rotational_speed"},
    "OSF": {"tool_wear", "torque", "product_variant"},
}
