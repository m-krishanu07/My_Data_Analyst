# My Data Analyst

Upload a CSV, ask a question in English, get a real answer — computed by Python
that an LLM wrote and a sandbox executed.

[![CI](https://github.com/m-krishanu07/My_Data_Analyst/actions/workflows/ci.yml/badge.svg)](https://github.com/m-krishanu07/My_Data_Analyst/actions/workflows/ci.yml)

```
You:  Which region has the highest average profit margin?
App:  [runs generated pandas in an isolated process]
      West — 18.4% average margin across 612 orders.
      ┌ bar chart ┐
      openai/gpt-oss-120b · 1,204 tokens · 1.9s · $0.0005
```

---

## What this is, and what it is not

**There is no training or fine-tuning anywhere in this project.** That is worth
stating plainly, because "an AI that analyses your data" usually implies a model
was trained on it. Nothing was.

The mechanism is simpler and more general: on every question the app sends the
model your file's *schema* — column names, dtypes, null counts, a few sample
values — and asks it to write a `run(df)` function against that schema. The
model has never seen your data and does not need to. This is why an arbitrary
CSV works on the first try, and why the two datasets in `demo_data/` are only
benchmark ground truth and demo fixtures, not a training set.

The model writes code. The code is what produces the number.

---

## Architecture

```
                 ┌──────────────────────────────────────────┐
  CSV upload ───▶│ session_manager.parse_csv_bytes          │
                 │   size / encoding / delimiter guards     │
                 └───────────────┬──────────────────────────┘
                                 │ DataFrame
                                 ▼
  question ─────▶┌──────────────────────────────────────────┐
                 │ agent.generate_and_execute               │
                 │   schema summary + chat history → prompt │
                 └───────────────┬──────────────────────────┘
                                 ▼
                 ┌──────────────────────────────────────────┐
                 │ LLM provider    groq │ bedrock │ replay   │
                 │   retry w/ backoff, token + cost accounting│
                 └───────────────┬──────────────────────────┘
                                 │ ```python def run(df): ...```
                                 ▼
                 ┌──────────────────────────────────────────┐   layer 1
                 │ sandbox.validator      AST policy check   │◀── static
                 └───────────────┬──────────────────────────┘
                                 │ pickle over a pipe
                                 ▼
                 ┌──────────────────────────────────────────┐   layer 2
                 │ sandbox.worker    separate OS process     │◀── isolation
                 │   rlimits, timeout, process-tree kill     │
                 └───────────────┬──────────────────────────┘
                                 │ JSON back (never pickle)
                                 ▼
                 ┌──────────────────────────────────────────┐
                 │ error?  ──▶ feed back to model, retry ×2  │
                 │ ok?     ──▶ render text + figure          │
                 └──────────────────────────────────────────┘
```

| Module | Responsibility |
|---|---|
| `src/sandbox/validator.py` | AST policy: imports, builtins, attributes, dunders |
| `src/sandbox/worker.py` | Long-lived child process that executes generated code |
| `src/sandbox/executor.py` | Worker lifecycle, timeouts, process-tree teardown |
| `src/sandbox/protocol.py` | Length-prefixed wire format, asymmetric encoding |
| `src/agent/agent.py` | Prompt assembly, self-correction loop, cost accounting |
| `src/llm/` | Provider abstraction: Groq, Bedrock, offline replay |
| `src/session_manager.py` | CSV parsing guards, cached schema summary, chat state |
| `src/ui/` | Streamlit components and CSS, separated from `app.py` |
| `eval/` | Execution-based benchmark with self-verifying ground truth |

---

## Security model

Generated code is untrusted input. It is written by a model that can be steered
by anything in the CSV — a column header is attacker-controlled text.

### Layer 1 — AST validation (`src/sandbox/validator.py`)

The obvious implementation is a substring blacklist, and it does not work.
Substring matching operates on *text*; escapes operate on *syntax*. Each of
these defeats a naive check and is now a permanent regression test in
`tests/test_validator.py`:

| Escape | Why a blacklist misses it |
|---|---|
| `eval ( '2+2' )` | a check for `"eval("` never sees the space |
| `__import__("os")` | a check for `"import os"` never sees this spelling |
| `().__class__.__bases__[0].__subclasses__()` | no forbidden word appears at all |
| `getattr(df, "__cl" + "ass__")` | the attribute name does not exist until runtime |
| `pd.io.common.os.system("...")` | `os` is never imported — pandas imported it for us |

So the validator parses the code and walks the AST, which is immune to
whitespace and spelling. It enforces:

- **Import allowlist** — pandas/numpy/plotting/stats/stdlib-analysis only.
- **Builtin denylist** — `eval`, `exec`, `compile`, `open`, `__import__`,
  `globals`, `getattr`/`setattr`/`delattr`. Dynamic attribute access is blocked
  outright, because `getattr` makes every other static rule decorative.
- **No dunder access** — one rule closes the entire `__class__ → __bases__ →
  __subclasses__` escape family.
- **No module hop** — reaching `os`, `sys`, `subprocess`, `ctypes` and friends
  *through attribute access* is blocked. `pd.io.common.os` really is the `os`
  module, and nothing about that chain is a dunder or an import, so this needed
  its own rule.
- **I/O denylist** — `read_csv`, `to_csv`, `savefig`, `to_pickle`, … The
  pandas object is already in the namespace, so without this the filesystem is
  wide open.
- **No class definitions, no async, no relative imports.**

The blocklists are deliberately *narrow*. A name is only blocked if its
dangerous binding is actually reachable; `rename` and `remove` were removed from
the list once the module hop was closed, because they were rejecting
`df.rename(columns=…)` and `list.remove(…)` — and that false positive was
costing a real repair round-trip on a third of the benchmark.

### Layer 2 — process isolation (`src/sandbox/worker.py`)

Validation alone is never sufficient. Static analysis cannot decide halting, so
`while True: pass` passes every rule above. Execution therefore happens in a
**separate OS process**:

- **Timeout that actually stops work.** A thread cannot be killed in Python, so
  the previous design's timeout was advisory. The worker is a real process:
  `taskkill /F /T` on Windows, `killpg` on POSIX, so orphaned grandchildren die
  too.
- **Memory ceiling** via `RLIMIT_AS`.
- **Asymmetric wire format.** The parent pickles *to* the worker; the worker
  replies in **JSON only**. Unpickling output from the process running untrusted
  code would hand it arbitrary code execution in the parent — the exact boundary
  the sandbox exists to defend.
- **Warm worker pool**, because importing pandas + sklearn + plotly per query
  would dominate latency.

### Known limitations — stated, not hidden

- **`RLIMIT_AS` is POSIX-only.** On Windows the memory cap is a no-op; the
  timeout and process kill still work. Deploy on Linux if the memory ceiling
  matters to you.
- **This is not a container or a VM.** The worker runs as the same user with the
  same filesystem access. It is a strong barrier against generated code, not
  against a determined attacker with a Python 0-day. For genuinely hostile
  input, run the whole app in the provided Docker image, and put the worker in
  its own network-less container.
- **Charts are executed code too.** A figure is serialized and rebuilt in the
  parent; matplotlib output crosses as PNG bytes, Plotly as JSON.

---

## Benchmark

`eval/` runs the **full pipeline** — prompt, LLM, validation, sandbox,
self-correction — over 18 analysis tasks on two seeded datasets, and scores the
**value the code returned** against ground truth recomputed from the dataframe.
Generated source is never string-matched.

| Model | pass@1 | Correct | Self-corrected | Median latency | Cost / query |
|---|---|---|---|---|---|
| `groq:openai/gpt-oss-120b` | 100% | 18/18 | 0 | 2.1s | $0.00054 |
| `groq:openai/gpt-oss-20b` | 100% | 18/18 | 1 | 5.1s | $0.00035 |

Full report, per-category breakdown and failure taxonomy:
[`eval/results/benchmark.md`](eval/results/benchmark.md).

**Read this honestly:**

- **Both models scored 100%, which means the suite is too easy to separate
  them.** 18 tasks over two datasets is a smoke test that the pipeline works
  end-to-end, not a ranking. Do not read it as a model comparison.
- **Latency is wall-clock on a free tier, so it is mostly queueing.** The timer
  wraps the whole call including rate-limit backoff. It is not model speed.
- **The self-correction loop fired ~0 times here**, because these tasks are
  within both models' comfort zone. Its behaviour is covered by unit tests in
  `tests/test_agent.py` rather than by this table. An earlier run showed 6
  repairs — all of which turned out to be our own validator falsely rejecting
  `df.rename()`, which is what prompted the blocklist narrowing described above.
  That is the more useful finding: the benchmark caught a bug in the product.
- `--verify` runs every task's reference implementation through the same scoring
  path and must be 18/18. That is what makes a low score attributable to the
  model and not to a broken check. CI runs it on every push.

```bash
python -m eval.run_benchmark --verify                    # ground truth, no API calls
python -m eval.run_benchmark --models groq:openai/gpt-oss-120b
```

---

## Demo mode — the app works with no API key

A deployed link that dies when a key rotates or a free quota trips is a dead
link, and that is exactly when someone is most likely to open it. So missing
credentials is not an error state here:

- With a key → questions are answered live, on any CSV you upload.
- Without a key → the app loads the sample datasets and **replays recorded model
  responses**, offered as one-click questions.

**Only the network call is replayed.** The recorded response still goes through
code extraction, the AST validator, and real execution in the sandbox worker —
so every number and chart a visitor sees was computed from the CSV at the moment
they asked. The UI labels replayed answers so nobody mistakes one for live
generation.

The fixture is generated by `--record`, which writes **only tasks the benchmark
scored correct**. The recorded text is therefore genuine model output that is
verified-good by construction, not hand-written text dressed up as a model
response. `tests/test_replay.py` executes all 18 recorded answers in the real
sandbox on every CI run.

```bash
python -m eval.run_benchmark --models groq:openai/gpt-oss-120b --record
```

---

## Setup

Requires **Python 3.13+**.

```bash
git clone <your-repo-url> && cd my-data-analyst
python -m venv .venv && .venv\Scripts\activate     # Windows
# python3 -m venv .venv && source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

That already works — it starts in demo mode. To analyse your own data, add a
key:

```bash
cp .env.example .env
# GROQ_API_KEY=...   free, no card: https://console.groq.com/keys
```

<details>
<summary><b>AWS Bedrock instead of Groq</b></summary>

```bash
pip install boto3
```

```dotenv
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
BEDROCK_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

Credentials come from the standard AWS chain (env vars, `~/.aws/credentials`, or
an instance role). The model must be enabled for your account in the Bedrock
console for that region. If both providers are configured, the sidebar switches
between them at runtime.
</details>

<details>
<summary><b>Docker</b></summary>

```bash
docker build -t data-analyst .
docker run -p 8501:8501 --env-file .env data-analyst
```

Runs as a non-root user. Omit `--env-file` to see demo mode. This is also the
only configuration where the sandbox memory limit is genuinely enforced.
</details>

<details>
<summary><b>Streamlit Cloud</b></summary>

Point it at `app.py`, then **set the Python version to 3.13** in advanced
settings — the default is older and `pyproject.toml` requires 3.13. Add
`GROQ_API_KEY` under *App settings → Secrets*. Without it the app still deploys
and runs in demo mode.
</details>

### Configuration

Every tunable is an environment variable with a working default; see
[`.env.example`](.env.example) for the annotated list. Groq retires model ids
periodically — if a call returns "model does not exist", pick a current one from
[the Groq model list](https://console.groq.com/docs/models) and set `GROQ_MODEL`.

---

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest                                # 249 tests
python -m eval.run_benchmark --verify
python -m demo_data.generate          # regenerate the seeded CSVs
```

The test suite is organised around failure modes rather than around modules.
Every sandbox escape in `tests/test_validator.py` was verified to bypass an
earlier version of this code, and is kept permanently so containment cannot
silently regress.

CI runs lint, the full suite, and benchmark ground-truth verification on Python
3.13.

## Datasets

Two seeded, committed CSVs, generated by `demo_data/generate.py` with fixed
random seeds so a published benchmark number is reproducible:

| Dataset | Shape | Contains |
|---|---|---|
| `retail_orders.csv` | 3,000 × 12 | orders, regions, categories, sales, profit, returns |
| `customer_churn.csv` | 2,500 × 12 | tenure, contract, charges, support calls, churn |

Both include missing values and mixed dtypes on purpose — clean data would not
exercise the data-quality tasks.
