"""Mile 2 - exclusions. Gate: pytest green + covert fraction on Mile 1 prompts.

Cross-check: Mile 1b measured peak covert 0.21 at L21 with inline logic. This
recomputes it through probe.exclusions. Disagreement means the harness is
wrong, not the instrument.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import torch
import transformers

import jlens
from probe.exclusions import covert_fraction, covert_hit_layerwise, covert_hit_strict, echo_clean, rank_of
from probe.exclusions import EchoLeak
from probe.pins import BAND, LENS_FILENAME, LENS_REPO, LENS_REVISION, MODEL_ID, SUBSTRATE_VERSION
from probe.progress import bar

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "m2_exclusions.json"

spec = importlib.util.spec_from_file_location(
    "m1run", ROOT / "experiments" / "m1_instrument" / "run.py"
)
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)


def main() -> int:
    assert BAND is not None, "BAND is None -- Mile 1 did not close"
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to("cuda").eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILENAME, revision=LENS_REVISION
    )
    norm, head = hf.model.norm, hf.lm_head

    prompts, echo_rejects = [], []
    for p, a in m1.CANDIDATES:
        ids = tok.encode(a, add_special_tokens=False)
        if len(ids) != 1:
            continue
        try:
            echo_clean(p, tok.encode(p), [(a, ids[0])])
        except EchoLeak as e:
            echo_rejects.append((a, str(e)[:70]))
            continue
        prompts.append((p, a, ids[0]))

    print(f"{len(prompts)} prompts pass echo_clean; {len(echo_rejects)} rejected")
    for a, why in echo_rejects:
        print(f"  REJECT {a!r}: {why}")

    layers = None
    layerwise, strict = {}, {}
    for prompt, answer, tid in bar(prompts, label="mile-2 exclusions"):
        ll, _, _ = lens.apply(model, prompt, positions=[-1])
        if layers is None:
            layers = sorted(ll.keys())
            layerwise = {l: [] for l in layers}
            strict = {l: [] for l in layers}
        enc = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = hf(**enc, output_hidden_states=True)
        hs = out.hidden_states
        final_rank = rank_of(out.logits[0, -1], tid)
        off = len(hs) - len(layers)
        for l in layers:
            lens_r = rank_of(ll[l][0], tid)
            logit_r = rank_of(head(norm(hs[l + off][0, -1].unsqueeze(0)))[0], tid)
            layerwise[l].append(covert_hit_layerwise(lens_r, logit_r))
            strict[l].append(covert_hit_strict(lens_r, final_rank))

    lw = {l: covert_fraction(layerwise[l]) for l in layers}
    st = {l: covert_fraction(strict[l]) for l in layers}
    peak_l = max(lw, key=lw.get)
    band_lw = sum(lw[l] for l in range(BAND[0], BAND[1] + 1)) / (BAND[1] - BAND[0] + 1)
    band_st = sum(st[l] for l in range(BAND[0], BAND[1] + 1)) / (BAND[1] - BAND[0] + 1)

    print(f"\n--- covert fraction, BAND {BAND} ---")
    print("  layer  layerwise  strict")
    for l in range(max(0, BAND[0] - 3), min(layers[-1], BAND[1] + 4) + 1):
        mark = " <-" if BAND[0] <= l <= BAND[1] else ""
        print(f"  L{l:>3}    {lw[l]:6.2f}    {st[l]:5.2f}{mark}")

    print(f"\npeak layerwise {lw[peak_l]:.2f} at L{peak_l}   (Mile 1b: 0.21 at L21)")
    print(f"BAND mean layerwise {band_lw:.2f}   strict {band_st:.2f}")
    match = abs(lw[21] - 0.21) < 0.02
    print(f"cross-check vs Mile 1b at L21: {'MATCH' if match else 'MISMATCH'} ({lw[21]:.2f})")
    print(f"\nK4 needs >= 0.70 at Mile 4. BAND mean is {band_lw:.2f}.")

    OUT.write_text(json.dumps({
        "mile": 2, "gate_passed": match, "band": list(BAND),
        "covert_layerwise": {str(l): lw[l] for l in layers},
        "covert_strict": {str(l): st[l] for l in layers},
        "band_mean_layerwise": band_lw, "band_mean_strict": band_st,
        "peak_layerwise": lw[peak_l], "peak_layer": peak_l,
        "m1b_crosscheck_l21": lw[21], "crosscheck_match": match,
        "n_prompts": len(prompts), "echo_rejects": echo_rejects,
        "substrate_version": SUBSTRATE_VERSION,
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout.strip(),
    }, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
