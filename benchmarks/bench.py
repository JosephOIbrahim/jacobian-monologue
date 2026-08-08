"""Instrument benchmarks -- measured costs, not estimates.

What this measures and why:

  USD GATE      author_stage() + resolve_wake() latency. The question a
                reviewer asks: does composing the world-model in OpenUSD add
                meaningful latency to the loop? (Spoiler shape: the gate is
                microseconds; one model forward is ~a second. Gating is free.)
  ECHO GUARD    echo_clean() throughput on the real aligned prompt.
  MODEL         cold load time (bf16 -> cuda) and resident VRAM.
  PREFILL       forced-choice forward latency, aligned + counterfactual
                prompts, median of 10 after 2 warmups, plus peak VRAM.
  END-TO-END    one full A-prime pass: author -> resolve -> deliver ->
                two forwards -> argmax compare.

Numbers are ENVIRONMENT-DEPENDENT. The committed benchmarks/results.json is
the reference machine's record (hardware block included); reproduce yours with:

    .venv\\Scripts\\python.exe benchmarks\\bench.py
"""
from __future__ import annotations

import json
import os
import platform
import statistics as st
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)                                   # run_aprime inserts "src" relative
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "m7_usd_wake"))

OUT = ROOT / "benchmarks" / "results.json"
now = time.perf_counter


def loop_us(fn, n: int) -> float:
    """Median-of-5 batches, microseconds per call."""
    batches = []
    for _ in range(5):
        t0 = now()
        for _ in range(n):
            fn()
        batches.append((now() - t0) / n * 1e6)
    return st.median(batches)


def main() -> int:
    R: dict = {"benchmark": "instrument-costs", "run_date": date.today().isoformat()}

    # ---------------- USD gate (no GPU needed) ----------------
    from stage import ALIGNED, COUNTERFACTUAL, author_stage, resolve_wake
    from pxr import Usd

    R["usd_version"] = list(Usd.GetVersion())
    stage_fixed = author_stage(ALIGNED)
    R["usd_gate"] = {
        "author_plus_resolve_us": round(loop_us(
            lambda: resolve_wake(author_stage(ALIGNED)), 300), 1),
        "resolve_only_us": round(loop_us(
            lambda: resolve_wake(stage_fixed), 3000), 1),
    }
    print(f"usd gate: author+resolve {R['usd_gate']['author_plus_resolve_us']} us"
          f" | resolve-only {R['usd_gate']['resolve_only_us']} us")

    # ---------------- model load ----------------
    import torch
    import transformers
    from run_aprime import FORBIDDEN, GIST, MODEL_ID, build_prompt

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

    torch.cuda.reset_peak_memory_stats()
    t0 = now()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
    torch.cuda.synchronize()
    load_s = now() - t0
    resident_gb = torch.cuda.memory_allocated() / 1e9
    R["model"] = {"model_id": MODEL_ID, "load_s": round(load_s, 1),
                  "resident_vram_gb": round(resident_gb, 2)}
    print(f"model: load {load_s:.1f}s | resident {resident_gb:.2f} GB")

    # ---------------- echo guard ----------------
    from probe.exclusions import echo_clean
    mem = f"Relevant prior experience: {GIST}\n"
    mem_ids = tok.encode(mem)
    targets = []
    for w in FORBIDDEN:
        ids = tok.encode(" " + w, add_special_tokens=False)
        targets.append((" " + w, ids[0] if len(ids) == 1 else -1))
    R["echo_guard_us"] = round(loop_us(
        lambda: echo_clean(mem, mem_ids, targets), 2000), 1)
    print(f"echo guard: {R['echo_guard_us']} us per check")

    # ---------------- prefill latency + peak VRAM ----------------
    prompts = {"aligned": build_prompt(tok, True),
               "counterfactual": build_prompt(tok, False)}
    encs = {k: tok(v, return_tensors="pt").to("cuda") for k, v in prompts.items()}

    @torch.no_grad()
    def forward(enc):
        out = hf(**enc).logits[0, -1]
        torch.cuda.synchronize()
        return out

    for enc in encs.values():                       # warmup
        forward(enc); forward(enc)

    torch.cuda.reset_peak_memory_stats()
    R["prefill_ms"] = {}
    for name, enc in encs.items():
        times = []
        for _ in range(10):
            t0 = now(); forward(enc); times.append((now() - t0) * 1000)
        R["prefill_ms"][name] = {
            "median": round(st.median(times), 1),
            "tokens": int(enc["input_ids"].shape[1]),
        }
        print(f"prefill {name}: {R['prefill_ms'][name]['median']} ms "
              f"({R['prefill_ms'][name]['tokens']} tok)")
    R["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(f"peak vram during prefill: {R['peak_vram_gb']} GB")

    # ---------------- end-to-end A-prime pass ----------------
    @torch.no_grad()
    def choice(prompt):
        enc = tok(prompt, return_tensors="pt").to("cuda")
        p = torch.softmax(hf(**enc).logits[0, -1].float(), -1)
        out = {}
        for L in ["A", "B", "C"]:
            ids = (tok.encode(L, add_special_tokens=False)
                   + tok.encode(" " + L, add_special_tokens=False))
            out[L] = max(p[i].item() for i in ids)
        t = sum(out.values())
        return {k: v / t for k, v in out.items()}

    t0 = now()
    picks = {}
    for name, wm in (("aligned", ALIGNED), ("counterfactual", COUNTERFACTUAL)):
        woke = resolve_wake(author_stage(wm))
        d = choice(build_prompt(tok, woke))
        picks[name] = max(d, key=d.get)
    torch.cuda.synchronize()
    e2e = now() - t0
    flipped = picks == {"aligned": "B", "counterfactual": "A"}
    R["aprime_end_to_end"] = {"seconds": round(e2e, 2), "picks": picks,
                              "decision_flipped": flipped}
    print(f"a-prime end-to-end: {e2e:.2f}s | picks {picks} | flipped={flipped}")

    # ---------------- hardware block ----------------
    R["hardware"] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
    }
    fwd_us = R["prefill_ms"]["aligned"]["median"] * 1000
    R["gate_vs_forward"] = {
        "resolve_x_cheaper": round(fwd_us / R["usd_gate"]["resolve_only_us"]),
        "author_plus_resolve_x_cheaper": round(
            fwd_us / R["usd_gate"]["author_plus_resolve_us"]),
    }
    R["note"] = ("environment-dependent reference numbers; the headline is the "
                 f"measured ratio: predicate resolution is "
                 f"{R['gate_vs_forward']['resolve_x_cheaper']}x cheaper than one "
                 "model forward on this machine -- the USD gate is free "
                 "relative to inference")

    OUT.write_text(json.dumps(R, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0 if flipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
