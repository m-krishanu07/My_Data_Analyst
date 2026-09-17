# Benchmark results

`benchmark.md` and `benchmark.csv` land here. They are **not** committed with
placeholder numbers — an unrun benchmark should look unrun.

To produce them:

```bash
# 1. Prove the harness before trusting it. No API calls, must be 18/18.
python -m eval.run_benchmark --verify

# 2. Score one or more models end to end.
python -m eval.run_benchmark --models groq:llama-3.1-8b-instant \
                                      groq:llama-3.3-70b-versatile
```

Step 1 exists because a benchmark that cannot fail is not a benchmark. It runs
each task's reference implementation through the same sandbox and the same
scoring path a model's answer takes. If a reference fails, the check is wrong
and any model score printed alongside it is meaningless.
