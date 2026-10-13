"""Detection Layer.

Evaluates a single sensor reading against the AI4I 2020 documented failure
mechanisms (Matzka, 2020). These are plain threshold checks, not a trained
model, by design (see CLAUDE.md, "Architecture decisions" #3): a rule tripping
is what triggers the Knowledge Layer + Causal Layer downstream, and the whole
platform's explainability claim depends on this first step being auditable.

Two of the five mechanisms are NOT fully recoverable from a single row's
process parameters, and the rule engine says so rather than pretending
otherwise:

- TWF (tool wear failure): the dataset's ground truth fires at one randomly
  chosen minute inside the 200-240 min window per tool run, not at every
  minute inside it. `check_twf` can only flag "this tool is inside its
  failure-eligible window" (a necessary, not sufficient, condition) -- that
  is what an at-risk warning should actually mean, so it is left as-is
  rather than reverse-engineered to match the label exactly.
- RNF (random failure): 0.1% chance, independent of every process parameter.
  No function of sensor state can detect it before it happens, so
  `check_rnf` always returns False. This is documented, not hidden, because
  a Detection Layer that silently ignored a failure mode would misrepresent
  what the system can and cannot see.

HDF, PWF, and OSF ARE deterministic functions of the row alone and are
expected to match the dataset's ground truth columns exactly -- see
scripts/validate_detection.py.
"""

from dataclasses import dataclass, field
import math

OSF_THRESHOLD_MINNM = {"L": 11000, "M": 12000, "H": 13000}
HDF_TEMP_DELTA_K = 8.6
HDF_MAX_RPM = 1380
PWF_MIN_WATTS = 3500
PWF_MAX_WATTS = 9000
TWF_WINDOW_MIN = (200, 240)
RNF_PROBABILITY = 0.001


def rpm_to_rad_per_s(rpm: float) -> float:
    return rpm * 2 * math.pi / 60


def check_twf(tool_wear_min: float) -> bool:
    lo, hi = TWF_WINDOW_MIN
    return lo <= tool_wear_min <= hi


def check_hdf(air_temp_k: float, process_temp_k: float, rpm: float) -> bool:
    return (process_temp_k - air_temp_k) < HDF_TEMP_DELTA_K and rpm < HDF_MAX_RPM


def check_pwf(torque_nm: float, rpm: float) -> bool:
    power_w = torque_nm * rpm_to_rad_per_s(rpm)
    return power_w < PWF_MIN_WATTS or power_w > PWF_MAX_WATTS


def check_osf(tool_wear_min: float, torque_nm: float, product_variant: str) -> bool:
    threshold = OSF_THRESHOLD_MINNM[product_variant]
    return tool_wear_min * torque_nm > threshold


def check_rnf() -> bool:
    return False


@dataclass
class DetectionResult:
    twf_candidate: bool
    hdf: bool
    pwf: bool
    osf: bool
    rnf_observable: bool
    triggered: bool
    reasons: dict = field(default_factory=dict)

    @property
    def tripped(self) -> list:
        return sorted(self.reasons)


def evaluate_reading(
    *,
    product_variant: str,
    air_temp_k: float,
    process_temp_k: float,
    rpm: float,
    torque_nm: float,
    tool_wear_min: float,
) -> DetectionResult:
    twf = check_twf(tool_wear_min)
    hdf = check_hdf(air_temp_k, process_temp_k, rpm)
    pwf = check_pwf(torque_nm, rpm)
    osf = check_osf(tool_wear_min, torque_nm, product_variant)
    rnf = check_rnf()

    reasons = {}
    if twf:
        reasons["TWF"] = (
            f"tool wear {tool_wear_min:.0f} min is inside the "
            f"{TWF_WINDOW_MIN[0]}-{TWF_WINDOW_MIN[1]} min replacement window"
        )
    if hdf:
        delta = process_temp_k - air_temp_k
        reasons["HDF"] = (
            f"temp delta {delta:.2f}K < {HDF_TEMP_DELTA_K}K and "
            f"{rpm:.0f} rpm < {HDF_MAX_RPM} rpm"
        )
    if pwf:
        power_w = torque_nm * rpm_to_rad_per_s(rpm)
        reasons["PWF"] = f"power {power_w:.0f}W outside [{PWF_MIN_WATTS}, {PWF_MAX_WATTS}]W"
    if osf:
        threshold = OSF_THRESHOLD_MINNM[product_variant]
        reasons["OSF"] = (
            f"wear x torque {tool_wear_min * torque_nm:.0f} minNm > "
            f"{threshold} minNm for variant {product_variant}"
        )

    return DetectionResult(
        twf_candidate=twf,
        hdf=hdf,
        pwf=pwf,
        osf=osf,
        rnf_observable=rnf,
        triggered=twf or hdf or pwf or osf,
        reasons=reasons,
    )
