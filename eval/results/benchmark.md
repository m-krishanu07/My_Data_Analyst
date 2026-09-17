# Benchmark

_Generated 2026-09-16 16:08 UTC by `python -m eval.run_benchmark`._

18 analysis tasks over two seeded datasets (retail_orders, customer_churn). Each task runs the full pipeline — prompt, LLM, AST validation, sandboxed execution, self-correction — and is scored on the **value the code returned**, checked against ground truth recomputed from the dataframe. Generated source is never string-matched.

## Results

| Model | pass@1 | Correct | Self-corrected | Median latency | Cost / query |
|---|---|---|---|---|---|
| `groq:openai/gpt-oss-120b` | 100% | 18/18 | 0 | 2.1s | $0.00054 |
| `groq:openai/gpt-oss-20b` | 100% | 18/18 | 1 | 5.1s | $0.00035 |

## By category

| Category | groq:openai/gpt-oss-120b | groq:openai/gpt-oss-20b |
|---|---|---|
| aggregation | 3/3 | 3/3 |
| correlation | 2/2 | 2/2 |
| data quality | 3/3 | 3/3 |
| filtering | 2/2 | 2/2 |
| grouping | 4/4 | 4/4 |
| ranking | 1/1 | 1/1 |
| visualisation | 3/3 | 3/3 |

## Failure taxonomy

- `groq:openai/gpt-oss-120b`: no failures.
- `groq:openai/gpt-oss-20b`: no failures.

## Reading this honestly

- `pass@1` is a single greedy sample per task at `temperature=0`; it is not an average over repeated runs.
- **Self-corrected** counts answers that were wrong on the first attempt and only became correct after the repair loop fed the execution error back to the model. It is the value the loop adds, isolated.
- Numeric answers match if a number within tolerance appears in the output, so a correct value buried in a table counts. Chart tasks are scored on whether a renderable figure crossed the sandbox boundary.
- `python -m eval.run_benchmark --verify` runs each task's reference implementation through the same scoring path and must be 18/18. That is what makes a low score attributable to the model rather than to a broken check.
- **Latency is wall-clock, and on a free tier it is mostly queueing.** The timer wraps the whole agent call, so provider-side rate-limit backoff (the client sleeps and retries on HTTP 429) lands inside the measurement. These numbers describe what a user on a shared free key experiences; they are not a measure of model speed and should not be read as a speed comparison between models.
- 18 tasks over two datasets is a smoke test, not a benchmark suite. If models tie at the top, the honest reading is that the suite is too easy to separate them — not that they are equally capable.
