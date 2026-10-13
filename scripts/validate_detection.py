"""Validate the Detection Layer's rules against the AI4I 2020 ground-truth
labels (data/ai4i2020.csv), row by row.

HDF, PWF, and OSF are deterministic functions of a single row and are
expected to match the dataset's labels exactly. TWF and RNF are not fully
recoverable from a single row (see vercausa/detection.py docstring) --
this script reports what CAN be checked for them instead of pretending
they should hit 100%.
"""

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vercausa.detection import check_hdf, check_osf, check_pwf, check_twf

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv"


def main() -> None:
    rows = list(csv.DictReader(DATA.open(encoding="utf-8-sig")))
    n = len(rows)

    hdf_correct = pwf_correct = osf_correct = 0
    twf_true_positives = twf_actual = 0

    for row in rows:
        air = float(row["Air temperature [K]"])
        process = float(row["Process temperature [K]"])
        rpm = float(row["Rotational speed [rpm]"])
        torque = float(row["Torque [Nm]"])
        wear = float(row["Tool wear [min]"])
        variant = row["Type"]

        hdf_pred = check_hdf(air, process, rpm)
        pwf_pred = check_pwf(torque, rpm)
        osf_pred = check_osf(wear, torque, variant)
        twf_pred = check_twf(wear)

        hdf_correct += hdf_pred == bool(int(row["HDF"]))
        pwf_correct += pwf_pred == bool(int(row["PWF"]))
        osf_correct += osf_pred == bool(int(row["OSF"]))

        if bool(int(row["TWF"])):
            twf_actual += 1
            twf_true_positives += twf_pred

    print(f"Rows: {n}")
    print(f"HDF exact match: {hdf_correct}/{n} ({100*hdf_correct/n:.2f}%)")
    print(f"PWF exact match: {pwf_correct}/{n} ({100*pwf_correct/n:.2f}%)")
    print(f"OSF exact match: {osf_correct}/{n} ({100*osf_correct/n:.2f}%)")
    print(
        f"TWF recall (actual failures whose wear was inside the "
        f"200-240 min window): {twf_true_positives}/{twf_actual}"
    )
    print(
        "RNF: 0 detectable from process state by definition -- "
        "it is an independent 0.1% chance, not a function of sensor readings."
    )


if __name__ == "__main__":
    main()
