# Benchmarks — measured, not estimated

One script, the numbers reviewers ask about. Run it:

```bash
.venv/Scripts/python.exe benchmarks/bench.py
```

It reproduces the headline flip as a side effect — exit code 0 means the
decision flipped on your machine too.

## Reference machine (RTX 4090 · torch 2.11.0+cu128)

| What | Measured |
|---|---|
| USD predicate resolve (composed stage) | **19 µs** |
| USD author entire stage + resolve | 553 µs |
| Echo-guard check | 1.1 µs |
| Model load (warm cache, bf16 → cuda) | 4.6 s |
| Resident VRAM | 8.51 GB |
| Prefill, forced choice (107 tok) | 229 ms |
| Peak VRAM | 8.63 GB |
| Full A-prime pass, end to end | **0.47 s** |

**The headline is a ratio, not an absolute:** resolving the composed-stage
predicate is **~12,000× cheaper** than one model forward. Authoring the entire
USD world-model from scratch *and* resolving it is still **~400× cheaper**.
The gate is free relative to inference — composition adds recognition, not
latency.

Numbers are environment-dependent. [`results.json`](results.json) carries the
full record including the hardware block; rerun the script to get yours.
