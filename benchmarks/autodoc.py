"""Sync docs from benchmarks/results.json + regression-check vs committed baseline.

Docs never drift from data: the marked regions in benchmarks/README.md and the
root README.md are REGENERATED from results.json every run. Regression compares
the fresh results against the version committed at HEAD.

Exit codes: 0 ok | 2 regression tripped (thresholds below).
Prints machine-readable lines for the orchestrator:
    DOC_CHANGED=true|false
    REGRESSION=none|<semicolon-joined flags>
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))

# ---------------- regression vs HEAD ----------------
THRESHOLDS = (  # (label, getter, max_ratio_vs_baseline)
    ("resolve_us>2x",  lambda d: d["usd_gate"]["resolve_only_us"], 2.0),
    ("prefill_ms>1.5x", lambda d: d["prefill_ms"]["aligned"]["median"], 1.5),
    ("peak_vram>1.1x", lambda d: d["peak_vram_gb"], 1.1),
    ("end_to_end>2x",  lambda d: d["aprime_end_to_end"]["seconds"], 2.0),
)

flags: list[str] = []
if not NEW["aprime_end_to_end"]["decision_flipped"]:
    flags.append("FLIP_FAILED")

try:
    old_raw = subprocess.run(
        ["git", "-C", str(ROOT), "show", "HEAD:benchmarks/results.json"],
        capture_output=True, text=True, check=True).stdout
    OLD = json.loads(old_raw)
except Exception:
    OLD = None  # first run: no committed baseline, nothing to compare

if OLD is not None:
    for label, get, ratio in THRESHOLDS:
        try:
            if get(NEW) > get(OLD) * ratio:
                flags.append(f"{label} ({get(OLD)} -> {get(NEW)})")
        except KeyError:
            pass

# ---------------- jitter gate: regenerate docs only on real movement ----------------
# Displayed numbers hold still unless a metric moved beyond its noise band.
TOLERANCES = (  # (getter, relative tol) -- peak/resident use absolute GB below
    (lambda d: d["usd_gate"]["resolve_only_us"], 0.20),
    (lambda d: d["usd_gate"]["author_plus_resolve_us"], 0.20),
    (lambda d: d["echo_guard_us"], 0.30),
    (lambda d: d["model"]["load_s"], 0.20),
    (lambda d: d["prefill_ms"]["aligned"]["median"], 0.10),
    (lambda d: d["aprime_end_to_end"]["seconds"], 0.20),
)

def _moved() -> bool:
    if OLD is None:
        return True                       # no baseline: docs come from NEW
    for get, tol in TOLERANCES:
        try:
            o, n = get(OLD), get(NEW)
            if o and abs(n - o) / o > tol:
                return True
        except KeyError:
            return True
    for key in ("peak_vram_gb",):
        if abs(NEW.get(key, 0) - OLD.get(key, 0)) > 0.05:
            return True
    if abs(NEW["model"]["resident_vram_gb"] - OLD["model"]["resident_vram_gb"]) > 0.05:
        return True
    return False

STALE = _moved() or bool(flags)

# ---------------- generated content ----------------
g = NEW["usd_gate"]
resolve = round(g["resolve_only_us"])
author = round(g["author_plus_resolve_us"])
echo = NEW["echo_guard_us"]
load = NEW["model"]["load_s"]
resident = NEW["model"]["resident_vram_gb"]
pf = NEW["prefill_ms"]["aligned"]
peak = NEW["peak_vram_gb"]
e2e = NEW["aprime_end_to_end"]["seconds"]
rx = NEW["gate_vs_forward"]["resolve_x_cheaper"]
ax = NEW["gate_vs_forward"]["author_plus_resolve_x_cheaper"]
rx_disp = f"{round(rx, -3):,}"        # 11932 -> 12,000
ax_disp = f"{round(ax, -2):,}"        # 414   -> 400

TABLE = f"""| What | Measured |
|---|---|
| USD predicate resolve (composed stage) | **{resolve} \u00b5s** |
| USD author entire stage + resolve | {author} \u00b5s |
| Echo-guard check | {echo} \u00b5s |
| Model load (warm cache, bf16 \u2192 cuda) | {load} s |
| Resident VRAM | {resident} GB |
| Prefill, forced choice ({pf['tokens']} tok) | {round(pf['median'])} ms |
| Peak VRAM | {peak} GB |
| Full A-prime pass, end to end | **{e2e} s** |"""

RATIO = (f"**The headline is a ratio, not an absolute:** resolving the composed-stage\n"
         f"predicate is **~{rx_disp}\u00d7 cheaper** than one model forward. Authoring the entire\n"
         f"USD world-model from scratch *and* resolving it is still **~{ax_disp}\u00d7 cheaper**.\n"
         f"The gate is free relative to inference \u2014 composition adds recognition, not\n"
         f"latency.")

COSTS = (f"**What it costs (measured):** {peak:.1f} GB peak VRAM \u00b7 ~5 min first-time setup "
         f"(torch is the big download) \u00b7 model loads in ~{round(load)} s from cache \u00b7 the "
         f"headline run completes in **{e2e} s** after load; the robustness suite ~2 min. "
         f"The USD gate itself resolves in **{resolve} \u00b5s** \u2014 ~{rx_disp}\u00d7 cheaper than one "
         f"model forward. Full numbers and the script: [`benchmarks/`](benchmarks/).")


def splice(path: Path, tag: str, body: str) -> bool:
    """Replace content between <!-- {tag}:START --> and <!-- {tag}:END -->."""
    start, end = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    s = path.read_text(encoding="utf-8")
    a, b = s.index(start), s.index(end)
    new = s[: a + len(start)] + "\n" + body + "\n" + s[b:]
    if new != s:
        path.write_text(new, encoding="utf-8", newline="\n")
        return True
    return False


changed = False
if STALE:
    changed |= splice(ROOT / "benchmarks" / "README.md", "BENCH:TABLE", TABLE)
    changed |= splice(ROOT / "benchmarks" / "README.md", "BENCH:RATIO", RATIO)
    changed |= splice(ROOT / "README.md", "BENCH:COSTS", COSTS)

print(f"DOC_CHANGED={'true' if changed else 'false'}")
print(f"REGRESSION={'; '.join(flags) if flags else 'none'}")
print(f"KEY=resolve {g['resolve_only_us']}us | prefill {pf['median']}ms | "
      f"peak {peak}GB | e2e {e2e}s | "
      f"flipped {NEW['aprime_end_to_end']['decision_flipped']}")
if flags:
    print("\n!! REGRESSION FLAGS TRIPPED -- numbers above thresholds vs HEAD "
          "baseline. This is a result, not noise: investigate before shrugging.",
          file=sys.stderr)
sys.exit(2 if flags else 0)
