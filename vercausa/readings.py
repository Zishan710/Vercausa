"""Sensor readings and the AI4I 2020 CSV loader.

Ground-truth failure labels are intentionally NOT carried on SensorReading:
the twin and Detection Layer must never see them. Evaluation code reads the
labels from the CSV separately.
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator

from vercausa.schema import SENSOR_SIGNALS

DEFAULT_CSV = pathlib.Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv"
DEFAULT_MACHINE = "MILL-01"
REPLAY_START = datetime(2026, 1, 1)


@dataclass(frozen=True)
class SensorReading:
    machine_id: str
    timestamp: str
    product_variant: str
    values: dict
    source_row: int


def load_ai4i(
    path: pathlib.Path = DEFAULT_CSV,
    machine_id: str = DEFAULT_MACHINE,
    interval_s: int = 60,
    limit: int | None = None,
) -> Iterator[SensorReading]:
    """Yield rows in UDI order as one machine's time series.

    Rows are genuinely sequential for a single machine -- tool wear climbs
    row by row and resets at each of the ~120 tool changes -- so replaying in
    order gives the twin a coherent history. Timestamps are synthetic.
    """
    with path.open(encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if limit is not None and i >= limit:
                return
            yield SensorReading(
                machine_id=machine_id,
                timestamp=(REPLAY_START + timedelta(seconds=i * interval_s)).isoformat(),
                product_variant=row["Type"],
                values={s.name: float(row[s.csv_column]) for s in SENSOR_SIGNALS},
                source_row=int(row["UDI"]),
            )
