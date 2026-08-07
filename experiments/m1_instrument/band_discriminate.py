"""Mile 1b - discriminate workspace band from mouth band.

Mile 1 recall alone cannot do this: known-answer prompts are built so the model
DOES say the answer, so late-layer recall is guaranteed and meaningless.

Per layer we compute two things at the final prompt position:
  J-lens rank of the target   (what the workspace is disposed to say)
  logit-lens rank of target   (what the output is already committed to)

covert = J-lens top-5 AND logit-lens rank > 50. That is the BLUEPRINT's
mouth-exclusion definition applied layerwise. BAND is where covert peaks.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import transformers

import jlens
from probe.pins import LENS_FILENAME, LENS_REPO, LENS_REVISION, MODEL_ID
from probe.progress import bar

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "m1_instrument.json"
prev = json.loads(RES.read_text(encoding="utf-8"))

import importlib.util
spec = importlib.util.spec_from_file_location("m1run", Path(__file__).parent / "run.py")
m1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m1)

hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILENAME, revision=LENS_REVISION)

norm = hf.model.norm
head = hf.lm_head

prompts = []
for p, a in m1.CANDIDATES:
    ids = tok.encode(a, add_special_tokens=False)
    if len(ids) == 1 and a.strip().lower() not in p.lower():
        prompts.append((p, a, ids[0]))

layers = None
jl_hit = {}; ll_hit = {}; covert = {}

for prompt, answer, tid in bar(prompts, label="mile-1b discriminate"):
    lens_logits, _, _ = lens.apply(model, prompt, positions=[-1])
    if layers is None:
        layers = sorted(lens_logits.keys())
        jl_hit = {l: 0 for l in layers}; ll_hit = {l: 0 for l in layers}; covert = {l: 0 for l in layers}
    enc = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        hs = hf(**enc, output_hidden_states=True).hidden_states
    off = len(hs) - len(layers)
    for l in layers:
        j_rank = (lens_logits[l][0] > lens_logits[l][0][tid]).sum().item()
        h = hs[l + off][0, -1]
        lg = head(norm(h.unsqueeze(0)))[0]
        l_rank = (lg > lg[tid]).sum().item()
        if j_rank < 5: jl_hit[l] += 1
        if l_rank < 5: ll_hit[l] += 1
        if j_rank < 5 and l_rank > 50: covert[l] += 1

n = len(prompts)
print(f"\n--- layer | J-lens | logit-lens | COVERT (J hit, mouth silent) --- n={n}")
print("  layer   jlens  logit  covert")
for l in layers:
    j, lo, c = jl_hit[l]/n, ll_hit[l]/n, covert[l]/n
    print(f"  L{l:>3}   {j:5.2f}  {lo:5.2f}  {c:5.2f} |{'#'*int(c*40):<40}|")

best = max(covert, key=lambda l: covert[l])
thresh = 0.5 * covert[best]
band_layers = [l for l in layers if covert[l] >= thresh and covert[l] > 0]
band = (min(band_layers), max(band_layers)) if band_layers else None
print(f"\npeak covert {covert[best]/n:.2f} at L{best}")
print(f"band at >=50% of peak covert: {band}")
print(f"naive Mile-1 band (recall only, mouth-contaminated): {tuple(prev['proposed_band'])}")

prev["covert_by_layer"] = {str(l): covert[l]/n for l in layers}
prev["logit_lens_by_layer"] = {str(l): ll_hit[l]/n for l in layers}
prev["proposed_band_naive"] = prev["proposed_band"]
prev["proposed_band_covert"] = band
prev["peak_covert_layer"] = best
RES.write_text(json.dumps(prev, indent=2), encoding="utf-8")
print(f"\nupdated {RES}")
