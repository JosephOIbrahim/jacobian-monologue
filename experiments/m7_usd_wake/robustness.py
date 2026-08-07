"""m7 ROBUSTNESS: does the decision flip hold across MULTIPLE config pairs,
or was payments-vs-search a lucky one?

Same mechanism, four distinct scenarios. Each has a situation where the model
has an OBVIOUS instinct (the prior), and a counterintuitive memory that argues
for a DIFFERENT action. Forced choice; the woken memory should flip the decision.

ROBUST = argmax flips to the memory-implied action across ALL fair scenarios,
with a large probability gain. Any scenario that does not flip is reported.

Fairness gate per scenario: dormant must prefer the obvious action and the
memory-implied action must sit LOW without the memory. Echo guard: the gist
never names the target action word.
"""
import json, sys
from pathlib import Path
import torch, transformers
sys.path.insert(0, "src")
from probe.exclusions import echo_clean, EchoLeak

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "m7_robustness.json"
MODEL_ID = "Qwen/Qwen3.5-4B"

SCENARIOS = [
  {
    "name": "payments_cache",
    "situation": "A service's error rate rose shortly after a recent deployment.",
    "menu": "(A) revert the recent change  (B) clear the downstream cache  (C) escalate to a senior engineer",
    "obvious": "A", "implied": "B",
    "gist": "In a supplied prior incident with identical symptoms, undoing the deployment did not help; the errors continued because a separate storage service was returning outdated copies. Refreshing that service's stored data resolved everything at once.",
    "forbidden": ["cache", "clear", "downstream"],
  },
  {
    "name": "latency_index",
    "situation": "A database's query latency spiked right after a schema migration went out.",
    "menu": "(A) roll back the migration  (B) rebuild the search index  (C) add more database replicas",
    "obvious": "A", "implied": "B",
    "gist": "In an earlier case with the same pattern, reversing the migration changed nothing; the real cause was that the lookup structure had gone stale and was scanning far more rows than needed. Regenerating that structure restored fast responses immediately.",
    "forbidden": ["index", "rebuild", "search"],
  },
  {
    "name": "memory_growth_handler",
    "situation": "A process is consuming steadily more memory over hours until it crashes.",
    "menu": "(A) restart the process on a schedule  (B) patch the request handler  (C) increase the memory limit",
    "obvious": "A", "implied": "B",
    "gist": "A supplied prior incident looked identical; scheduling regular restarts only masked it. The true source was a routine that never released its open resources, so every incoming call leaked a little. Fixing that routine stopped the growth for good.",
    "forbidden": ["patch", "request", "handler"],
  },
  {
    "name": "auth_failures_clock",
    "situation": "Users are suddenly getting intermittent authentication failures across the platform.",
    "menu": "(A) rotate the signing keys  (B) sync the server clocks  (C) restart the auth service",
    "obvious": "A", "implied": "B",
    "gist": "In a prior incident with the same symptoms, replacing the credentials accomplished nothing. The actual problem was that one machine's time had drifted, so tokens appeared expired on arrival. Bringing the machines back into agreement fixed it right away.",
    "forbidden": ["sync", "clock", "clocks"],
  },
]

def build_prompt(tok, sc, deliver):
    sit = f"You are on-call. {sc['situation']} You must choose one action now.\n"
    mem = ""
    if deliver:
        mem = f"Relevant prior experience: {sc['gist']}\n"
        targets = []
        for w in sc["forbidden"]:
            ids = tok.encode(" " + w, add_special_tokens=False)
            targets.append((" " + w, ids[0] if len(ids) == 1 else -1))
        echo_clean(mem, tok.encode(mem), targets)
    return sit + mem + f"Options: {sc['menu']}\nAnswer with one letter. Answer:"

def main():
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    print("=== echo pre-check (gist must not name the implied action) ===")
    ok = True
    for sc in SCENARIOS:
        try:
            build_prompt(tok, sc, True); print(f"  {sc['name']:<24} clean")
        except EchoLeak as e:
            print(f"  {sc['name']:<24} LEAK: {str(e)[:50]}"); ok = False
    if not ok:
        print("\nFIX gists before running. Aborting."); return
    print()

    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
    def dist(prompt):
        enc = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            p = torch.softmax(hf(**enc).logits[0, -1].float(), -1)
        out = {}
        for L in ["A", "B", "C"]:
            ids = tok.encode(L, add_special_tokens=False) + tok.encode(" " + L, add_special_tokens=False)
            out[L] = max(p[i].item() for i in ids)
        t = sum(out.values()); return {k: v / t for k, v in out.items()}

    results = []
    print(f"{'scenario':<24} {'fair':<6} dormant->woken (implied)      flip")
    for sc in SCENARIOS:
        d_dorm = dist(build_prompt(tok, sc, False))
        d_woke = dist(build_prompt(tok, sc, True))
        imp, obv = sc["implied"], sc["obvious"]
        fair = (d_dorm[obv] > d_dorm[imp]) and (d_dorm[imp] < 0.35)
        flipped = (d_woke[imp] > d_woke[obv]) and (max(d_woke, key=d_woke.get) == imp)
        gain = d_woke[imp] - d_dorm[imp]
        results.append({"scenario": sc["name"], "fair": fair, "flipped": flipped,
                        "dormant": {k: round(v, 3) for k, v in d_dorm.items()},
                        "woken": {k: round(v, 3) for k, v in d_woke.items()},
                        "implied": imp, "implied_gain": round(gain, 3)})
        print(f"  {sc['name']:<22} {str(fair):<6} {imp}: {d_dorm[imp]:.0%}->{d_woke[imp]:.0%}  gain {gain:+.0%}   {'FLIP' if flipped else 'no'}")

    fair_ones = [r for r in results if r["fair"]]
    flipped_fair = [r for r in fair_ones if r["flipped"]]
    robust = len(fair_ones) >= 3 and len(flipped_fair) == len(fair_ones)
    print(f"\n{'=' * 64}")
    print(f"fair scenarios: {len(fair_ones)}/{len(results)}")
    print(f"of those, decision flipped: {len(flipped_fair)}/{len(fair_ones)}")
    print(f"VERDICT: {'ROBUST - flip holds across all fair scenarios' if robust else 'PARTIAL - see per-scenario'}")
    print('=' * 64)

    OUT.write_text(json.dumps({
        "experiment": "m7_robustness", "model_id": MODEL_ID, "n_scenarios": len(results),
        "fair_count": len(fair_ones), "flipped_of_fair": len(flipped_fair),
        "robust": robust, "scenarios": results,
        "criterion": "across >=3 fair scenarios, memory flips argmax to implied action in every one",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
