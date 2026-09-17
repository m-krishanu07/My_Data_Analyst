"""
Benchmark tasks with executable ground truth.

The previous `evaluate_models.py` graded by substring-matching generated
source against a hand-written snippet, on an *empty* DataFrame. That scores
style, not correctness: `df.groupby("Department")` passes whether or not the
code runs, and a correct answer written differently fails.

Here each task carries:

  `check`      a predicate over the *text the sandbox actually returned* and
               the dataframe, so ground truth is recomputed from the data
               rather than hard-coded. Regenerating the datasets cannot
               silently invalidate the benchmark.
  `reference`  a known-good `def run(df)` producing the right answer.
               `run_benchmark.py --verify` executes every reference through
               the real sandbox and asserts its own `check` passes, which is
               what stops a broken predicate from quietly failing every model.

Numeric answers are matched by scanning the output for a number within
tolerance. That accepts any reasonable phrasing ("Total sales: 3,412,880.55",
a bare float, a markdown table) without accepting a wrong value.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

# ── Answer matching ───────────────────────────────────────────

# Requires a digit, so markdown table rules (`|---|`) and stray hyphens
# are not read as numbers.
_NUMBER = re.compile(r"-?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")


def numbers_in(text: str | None) -> list[float]:
    """Every number appearing in the result, thousands separators removed."""
    if not text:
        return []
    values = []
    for match in _NUMBER.finditer(text.replace(",", "")):
        try:
            values.append(float(match.group()))
        except ValueError:  # pragma: no cover - regex guarantees a float
            continue
    return values


def close(text: str | None, expected: float, rel_tol: float = 0.005, abs_tol: float = 0.0) -> bool:
    """True if any number in the output matches `expected` within tolerance."""
    tolerance = max(abs_tol, abs(expected) * rel_tol)
    return any(abs(value - expected) <= tolerance for value in numbers_in(text))


def close_rate(text: str | None, fraction: float) -> bool:
    """Accept a rate expressed either as a fraction (0.138) or a percent (13.8)."""
    return close(text, fraction, rel_tol=0.02) or close(text, fraction * 100, rel_tol=0.02)


def exact(text: str | None, expected: int) -> bool:
    """For counts, where 'close' is not good enough."""
    return close(text, expected, rel_tol=0.0, abs_tol=0.5)


def mentions(text: str | None, *tokens: str) -> bool:
    """True if every token appears in the output (case-insensitive)."""
    lowered = (text or "").lower()
    return all(token.lower() in lowered for token in tokens)


@dataclass(frozen=True)
class Task:
    id: str
    dataset: str
    category: str
    question: str
    reference: str
    check: Callable[[str | None, pd.DataFrame], bool]
    expects_figure: bool = False


# ── retail_orders ─────────────────────────────────────────────


def _r1(text, df):
    return close(text, float(df["Sales"].sum()))


def _r2(text, df):
    avg = df.groupby("Region")["Sales"].mean()
    return mentions(text, str(avg.idxmax())) and close(text, float(avg.max()))


def _r3(text, df):
    return close(text, float(df["Discount"].corr(df["Profit"])), rel_tol=0.05, abs_tol=0.01)


def _r4(text, df):
    return close_rate(text, float((df["Returned"] == "Yes").mean()))


def _r5(text, df):
    return exact(text, int(((df["Quantity"] >= 5) & (df["Discount"] > 0.10)).sum()))


def _r6(text, df):
    return close(text, float(df["ShippingDays"].median()), rel_tol=0.0, abs_tol=0.01)


def _r7(text, df):
    top = df.groupby("Category")["Profit"].sum().nlargest(3).index
    return mentions(text, *[str(name) for name in top])


def _always(text, df):
    """Chart tasks are scored on whether a figure came back, not on prose."""
    return True


def _r10(text, df):
    return exact(text, int(df["ShippingDays"].isna().sum()))


# ── customer_churn ────────────────────────────────────────────


def _c1(text, df):
    return close_rate(text, float((df["Churn"] == "Yes").mean()))


def _c2(text, df):
    rates = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean())
    return mentions(text, str(rates.idxmax())) and close_rate(text, float(rates.max()))


def _c3(text, df):
    avg = df.groupby("Churn")["TenureMonths"].mean()
    return close(text, float(avg["Yes"]), rel_tol=0.02) and close(
        text, float(avg["No"]), rel_tol=0.02
    )


def _c4(text, df):
    return close(
        text,
        float(df["SupportCalls"].corr(df["SatisfactionScore"])),
        rel_tol=0.05,
        abs_tol=0.01,
    )


def _c5(text, df):
    return exact(text, int(df["TotalCharges"].isna().sum()))


def _c7(text, df):
    avg = df.groupby("InternetService")["MonthlyCharges"].mean()
    return mentions(text, str(avg.idxmax())) and close(text, float(avg.max()))


def _c8(text, df):
    churned = df[df["Churn"] == "Yes"]
    return close(text, float(churned["SatisfactionScore"].median()), rel_tol=0.0, abs_tol=0.06)


TASKS: list[Task] = [
    Task(
        id="R1",
        dataset="retail_orders",
        category="aggregation",
        question="What is the total sales revenue across all orders?",
        reference=(
            "def run(df):\n"
            "    return (f\"Total sales: {df['Sales'].sum():,.2f}\", None)\n"
        ),
        check=_r1,
    ),
    Task(
        id="R2",
        dataset="retail_orders",
        category="grouping",
        question="Which region has the highest average sales per order, and what is that average?",
        reference=(
            "def run(df):\n"
            "    avg = df.groupby('Region')['Sales'].mean().sort_values(ascending=False)\n"
            "    return (avg, None)\n"
        ),
        check=_r2,
    ),
    Task(
        id="R3",
        dataset="retail_orders",
        category="correlation",
        question="What is the correlation between Discount and Profit?",
        reference=(
            "def run(df):\n"
            "    return (float(df['Discount'].corr(df['Profit'])), None)\n"
        ),
        check=_r3,
    ),
    Task(
        id="R4",
        dataset="retail_orders",
        category="aggregation",
        question="What percentage of orders were returned?",
        reference=(
            "def run(df):\n"
            "    rate = (df['Returned'] == 'Yes').mean() * 100\n"
            "    return (f'{rate:.2f}% of orders were returned', None)\n"
        ),
        check=_r4,
    ),
    Task(
        id="R5",
        dataset="retail_orders",
        category="filtering",
        question=(
            "How many orders had a quantity of 5 or more and a discount greater than 10%?"
        ),
        reference=(
            "def run(df):\n"
            "    mask = (df['Quantity'] >= 5) & (df['Discount'] > 0.10)\n"
            "    return (int(mask.sum()), None)\n"
        ),
        check=_r5,
    ),
    Task(
        id="R6",
        dataset="retail_orders",
        category="data quality",
        question="What is the median number of shipping days? Ignore missing values.",
        reference=(
            "def run(df):\n"
            "    return (float(df['ShippingDays'].median()), None)\n"
        ),
        check=_r6,
    ),
    Task(
        id="R7",
        dataset="retail_orders",
        category="ranking",
        question="Which three product categories generate the most total profit?",
        reference=(
            "def run(df):\n"
            "    return (df.groupby('Category')['Profit'].sum().nlargest(3), None)\n"
        ),
        check=_r7,
    ),
    Task(
        id="R8",
        dataset="retail_orders",
        category="visualisation",
        question="Create a bar chart of total sales by category.",
        reference=(
            "def run(df):\n"
            "    totals = df.groupby('Category')['Sales'].sum().reset_index()\n"
            "    fig = px.bar(totals, x='Category', y='Sales', "
            "title='Total sales by category')\n"
            "    return (totals, fig)\n"
        ),
        check=_always,
        expects_figure=True,
    ),
    Task(
        id="R9",
        dataset="retail_orders",
        category="visualisation",
        question="Plot total sales by month over time.",
        reference=(
            "def run(df):\n"
            "    out = df.copy()\n"
            "    out['Month'] = pd.to_datetime(out['OrderDate']).dt.to_period('M').astype(str)\n"
            "    trend = out.groupby('Month')['Sales'].sum().reset_index()\n"
            "    fig = px.line(trend, x='Month', y='Sales', title='Monthly sales')\n"
            "    return (trend, fig)\n"
        ),
        check=_always,
        expects_figure=True,
    ),
    Task(
        id="R10",
        dataset="retail_orders",
        category="data quality",
        question="How many missing values are there in the ShippingDays column?",
        reference=(
            "def run(df):\n"
            "    return (int(df['ShippingDays'].isna().sum()), None)\n"
        ),
        check=_r10,
    ),
    Task(
        id="C1",
        dataset="customer_churn",
        category="aggregation",
        question="What is the overall churn rate?",
        reference=(
            "def run(df):\n"
            "    rate = (df['Churn'] == 'Yes').mean() * 100\n"
            "    return (f'Churn rate: {rate:.2f}%', None)\n"
        ),
        check=_c1,
    ),
    Task(
        id="C2",
        dataset="customer_churn",
        category="grouping",
        question="Which contract type has the highest churn rate, and what is that rate?",
        reference=(
            "def run(df):\n"
            "    rates = df.groupby('Contract')['Churn'].apply("
            "lambda s: (s == 'Yes').mean() * 100)\n"
            "    return (rates.sort_values(ascending=False), None)\n"
        ),
        check=_c2,
    ),
    Task(
        id="C3",
        dataset="customer_churn",
        category="grouping",
        question="Compare the average tenure in months of churned versus retained customers.",
        reference=(
            "def run(df):\n"
            "    return (df.groupby('Churn')['TenureMonths'].mean(), None)\n"
        ),
        check=_c3,
    ),
    Task(
        id="C4",
        dataset="customer_churn",
        category="correlation",
        question="What is the correlation between SupportCalls and SatisfactionScore?",
        reference=(
            "def run(df):\n"
            "    return (float(df['SupportCalls'].corr(df['SatisfactionScore'])), None)\n"
        ),
        check=_c4,
    ),
    Task(
        id="C5",
        dataset="customer_churn",
        category="data quality",
        question="How many customers have a missing TotalCharges value?",
        reference=(
            "def run(df):\n"
            "    return (int(df['TotalCharges'].isna().sum()), None)\n"
        ),
        check=_c5,
    ),
    Task(
        id="C6",
        dataset="customer_churn",
        category="visualisation",
        question="Show the churn rate by internet service type as a bar chart.",
        reference=(
            "def run(df):\n"
            "    rates = df.groupby('InternetService')['Churn'].apply("
            "lambda s: (s == 'Yes').mean() * 100).reset_index(name='ChurnRate')\n"
            "    fig = px.bar(rates, x='InternetService', y='ChurnRate', "
            "title='Churn rate by internet service')\n"
            "    return (rates, fig)\n"
        ),
        check=_always,
        expects_figure=True,
    ),
    Task(
        id="C7",
        dataset="customer_churn",
        category="grouping",
        question=(
            "Which internet service type has the highest average monthly charges, "
            "and what is that average?"
        ),
        reference=(
            "def run(df):\n"
            "    avg = df.groupby('InternetService')['MonthlyCharges'].mean()\n"
            "    return (avg.sort_values(ascending=False), None)\n"
        ),
        check=_c7,
    ),
    Task(
        id="C8",
        dataset="customer_churn",
        category="filtering",
        question="What is the median satisfaction score among customers who churned?",
        reference=(
            "def run(df):\n"
            "    churned = df[df['Churn'] == 'Yes']\n"
            "    return (float(churned['SatisfactionScore'].median()), None)\n"
        ),
        check=_c8,
    ),
]

TASKS_BY_ID = {task.id: task for task in TASKS}
