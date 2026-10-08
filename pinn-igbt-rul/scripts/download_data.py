"""Download the NASA IGBT aging data and extract the four square-gate-signal devices used here.

Source: NASA Prognostics Center of Excellence, "IGBT Accelerated Aging".
Landing page: https://data.nasa.gov/dataset/insulated-gate-bipolar-transistor-igbt-accelerated-aging
Cite as: J. Celaya, Phil Wysocki, and K. Goebel (2009) "IGBT Accelerated Aging Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
UNTESTED here (no network when written): if the archive layout differs, see data/README.md.
Check the dataset's terms of use before redistributing any of the files.

Example:
    python scripts/download_data.py --out data/raw
"""
from __future__ import annotations

import argparse
import re
import urllib.request
import zipfile
from pathlib import Path

URL = "https://data.nasa.gov/docs/legacy/IGBTAgingData_04022009.zip"
# e.g. ".../Square Signal at gate and SMU data/Aging Data/Device 2/Device2  1.mat"
MEMBER = re.compile(r"Square Signal at gate and SMU data/Aging Data/Device \d+/(Device\d+\s+1\.mat)$")
DEVICES = ("Device2", "Device3", "Device4", "Device5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    archive = out / "IGBTAgingData_04022009.zip"
    if not archive.exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, archive)

    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            match = MEMBER.search(member)
            if match and match.group(1).startswith(DEVICES):
                target = out / Path(match.group(1)).name
                target.write_bytes(zf.read(member))
                print(f"Extracted {target}")


if __name__ == "__main__":
    main()
