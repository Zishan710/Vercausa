"""Download the AI4I 2020 Predictive Maintenance Dataset (UCI id 601) into data/."""

import pathlib
import urllib.request
import zipfile
import io

URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
DEST = pathlib.Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv"


def main() -> None:
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=30) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    csv_name = next(n for n in archive.namelist() if n.endswith(".csv"))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_bytes(archive.read(csv_name))
    print(f"Wrote {DEST}")


if __name__ == "__main__":
    main()
