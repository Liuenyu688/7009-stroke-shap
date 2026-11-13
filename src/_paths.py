"""Single source of truth for every path used by the scripts.

All paths are resolved relative to the repository root, so the project runs
from a fresh clone on any machine without editing a single line.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "healthcare-dataset-stroke-data.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
REPORT = ROOT / "report"

for _d in (OUT, FIG, REPORT):
    _d.mkdir(parents=True, exist_ok=True)
