"""
Regenerate the demo datasets.

    python -m demo_data.generate

The committed CSVs are the output of this script. They are generated rather
than downloaded so the repo stays self-contained and license-clean, and
seeded so the numbers in the README and the eval benchmark stay stable.

The point of these files is that the relationships are *real*: discount
genuinely erodes profit, tenure genuinely predicts churn. A demo built on
uniform noise makes every model look broken and every chart look flat.
Missing values are injected deliberately so the data-quality panels and
the NaN-alignment prompt rules have something to act on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = Path(__file__).parent


def _blank(rng: np.random.Generator, series: pd.Series, fraction: float) -> pd.Series:
    """Knock out a fraction of values to simulate real-world gaps."""
    mask = rng.random(len(series)) < fraction
    # `mask` keeps the float dtype, so the column still reads back as numeric
    # with genuine NaNs rather than becoming an object column of strings.
    return series.astype(float).mask(mask)


def retail_orders(n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    regions = ["North", "South", "East", "West", "Central"]
    # Deliberately uneven so "which region performs best" has a real answer.
    region_weights = [0.26, 0.18, 0.22, 0.20, 0.14]
    region_uplift = {"North": 1.18, "South": 0.88, "East": 1.05, "West": 0.96, "Central": 0.80}

    categories = {
        "Electronics": (180.0, 0.55),
        "Furniture": (240.0, 0.70),
        "Office Supplies": (35.0, 0.40),
        "Clothing": (60.0, 0.50),
    }
    segments = ["Consumer", "Corporate", "Home Office"]

    region = rng.choice(regions, size=n, p=region_weights)
    category = rng.choice(list(categories), size=n, p=[0.28, 0.17, 0.35, 0.20])
    segment = rng.choice(segments, size=n, p=[0.52, 0.30, 0.18])

    base_price = np.array([categories[c][0] for c in category])
    spread = np.array([categories[c][1] for c in category])
    uplift = np.array([region_uplift[r] for r in region])

    unit_price = np.round(base_price * uplift * rng.lognormal(0, spread, n), 2).clip(4.99, 4000)
    quantity = rng.integers(1, 9, size=n)

    # Corporate buys in bulk and negotiates harder.
    discount = np.where(
        segment == "Corporate",
        rng.choice([0.0, 0.1, 0.15, 0.2, 0.3], n, p=[0.20, 0.25, 0.25, 0.20, 0.10]),
        rng.choice([0.0, 0.05, 0.1, 0.15, 0.2], n, p=[0.42, 0.24, 0.18, 0.11, 0.05]),
    )

    sales = np.round(unit_price * quantity * (1 - discount), 2)
    # Margin shrinks as discount grows — this is the signal the correlation
    # question is supposed to find.
    margin = 0.32 - 0.85 * discount + rng.normal(0, 0.05, n)
    profit = np.round(sales * margin, 2)

    shipping_days = np.clip(rng.poisson(4, n) + (region == "Central") * 2, 1, None)

    # Returns are driven by discount, slow shipping and category.
    logit = (
        -3.0
        + 3.4 * discount
        + 0.11 * shipping_days
        + (category == "Clothing") * 1.1
        + (category == "Electronics") * 0.35
    )
    returned = rng.random(n) < 1 / (1 + np.exp(-logit))

    order_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 730, n), unit="D"
    )

    df = pd.DataFrame(
        {
            "OrderID": [f"ORD-{100000 + i}" for i in range(n)],
            "OrderDate": order_date.strftime("%Y-%m-%d"),
            "Region": region,
            "Category": category,
            "CustomerSegment": segment,
            "Quantity": quantity,
            "UnitPrice": unit_price,
            "Discount": discount,
            "Sales": sales,
            "Profit": profit,
            "ShippingDays": shipping_days,
            "Returned": np.where(returned, "Yes", "No"),
        }
    ).sort_values("OrderDate", ignore_index=True)

    df["ShippingDays"] = _blank(rng, df["ShippingDays"], 0.04)
    df["Discount"] = _blank(rng, df["Discount"], 0.02)
    return df


def customer_churn(n: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)

    age = np.clip(rng.normal(41, 14, n).round().astype(int), 18, 88)
    tenure = np.clip(rng.gamma(2.0, 9.0, n).round().astype(int), 0, 72)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]
    )
    # "No service" rather than "None": pandas treats the literal string
    # "None" as a missing value on read_csv, which would silently turn a
    # real category into 18% NaN.
    internet = rng.choice(["Fiber optic", "DSL", "No service"], n, p=[0.44, 0.38, 0.18])

    monthly = np.round(
        20
        + (internet == "Fiber optic") * 45
        + (internet == "DSL") * 22
        + rng.normal(0, 8, n),
        2,
    ).clip(18.0, 130.0)
    total = np.round(monthly * tenure * rng.uniform(0.94, 1.06, n), 2)

    support_calls = rng.poisson(1.4 + (internet == "Fiber optic") * 0.9, n)

    # Short tenure, month-to-month contracts and support friction drive churn.
    logit = (
        -0.35
        - 0.055 * tenure
        + 1.45 * (contract == "Month-to-month")
        - 0.85 * (contract == "Two year")
        + 0.34 * support_calls
        + 0.012 * (monthly - 65)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "CustomerID": [f"CUST-{70000 + i}" for i in range(n)],
            "Age": age,
            "Gender": rng.choice(["Female", "Male"], n),
            "TenureMonths": tenure,
            "Contract": contract,
            "InternetService": internet,
            "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "SupportCalls": support_calls,
            "SatisfactionScore": np.clip(
                (9.2 - 0.62 * support_calls + rng.normal(0, 1.4, n)).round(1), 1.0, 10.0
            ),
            "Churn": np.where(churn, "Yes", "No"),
        }
    )

    df["TotalCharges"] = _blank(rng, df["TotalCharges"], 0.03)
    df["SatisfactionScore"] = _blank(rng, df["SatisfactionScore"], 0.06)
    return df


DATASETS = {
    "retail_orders.csv": retail_orders,
    "customer_churn.csv": customer_churn,
}


def main() -> None:
    for filename, builder in DATASETS.items():
        frame = builder()
        path = OUT_DIR / filename
        frame.to_csv(path, index=False)
        print(f"{filename}: {len(frame):,} rows x {len(frame.columns)} cols -> {path}")


if __name__ == "__main__":
    main()
