"""
Benchmark datasets.

The frames are loaded through `parse_csv_bytes` — the exact code path the
Streamlit upload uses — rather than by calling the generators directly. That
matters: dtype inference happens at read time, so a task's ground truth is
only trustworthy if it is computed against the frame the agent actually sees.

The CSVs are seeded and committed, so a benchmark number published in the
README can be reproduced byte-for-byte.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd

from src.session_manager import parse_csv_bytes

DATA_DIR = Path(__file__).resolve().parents[1] / "demo_data"

DATASETS = {
    "retail_orders": "retail_orders.csv",
    "customer_churn": "customer_churn.csv",
}


@cache
def load(name: str) -> pd.DataFrame:
    """Load a benchmark dataset by short name."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(DATASETS)}")

    path = DATA_DIR / DATASETS[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Regenerate the demo data with:\n"
            "    python -m demo_data.generate"
        )
    return parse_csv_bytes(path.read_bytes(), path.name)
