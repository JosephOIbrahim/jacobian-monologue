# Benchmarks — measured, not estimated

One script, the numbers reviewers ask about. Run it:

```bash
.venv/Scripts/python.exe benchmarks/bench.py
```

It reproduces the headline flip as a side effect — exit code 0 means the
decision flipped on your machine too.

## Reference machine (RTX 4090 · torch 2.11.0+cu128)

<!-- BENCH:TABLE:START -->
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
<!-- BENCH:TABLE:END -->

<!-- BENCH:RATIO:START -->
**The headline is a ratio, not an absolute:** resolving the composed-stage
predicate is **~12,000× cheaper** than one model forward. Authoring the entire
USD world-model from scratch *and* resolving it is still **~400× cheaper**.
The gate is free relative to inference — composition adds recognition, not
latency.
<!-- BENCH:RATIO:END -->

Numbers are environment-dependent. [`results.json`](results.json) carries the
full record including the hardware block; rerun the script to get yours.

## Automation

`run.ps1` is the whole loop in one command:

```powershell
benchmarks\run.ps1            # measure -> regression-check -> sync docs -> commit+push on material change
benchmarks\run.ps1 -NoPush    # same, stop before push
benchmarks\run.ps1 -Force     # commit a refresh even if nothing moved
```

What it enforces:

- **Docs can't drift.** The table above, the ratio paragraph, and the cost
  line in the root README are regenerated from `results.json` between marker
  comments — [`autodoc.py`](autodoc.py) is the only writer.
- **Regressions are loud.** Fresh numbers compare against the committed
  baseline at HEAD: resolve >2×, prefill >1.5×, peak VRAM >1.1×, end-to-end
  >2×, or a failed flip trips exit 2 and a `REGRESSION` commit. A regression
  is a result, not noise.
- **Jitter is silent.** Displayed numbers hold still inside per-metric noise
  bands; if nothing moves beyond tolerance and nothing trips, the run leaves
  the tree clean — no noise commits.

CI runs the CPU subset (`bench.py --cpu-only`: USD gate + echo guard) plus the
unit tests on every push — deliberately no badge, per the v0.1.0 restraint
lock. For weekly hands-free local runs: [`register_weekly.ps1`](register_weekly.ps1)
(run it yourself once — registering a scheduled task stays a human call).
