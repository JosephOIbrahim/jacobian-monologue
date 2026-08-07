"""Mile 4 - the sweep. Axis 1 then Axis 2. Raw R only; no verdict, no plot.

Metric R(f,c) = mean over BAND of log( (m_new + eps) / (m_old + eps) ), where
m_new/m_old are lens softmax mass on the new/old target tokens at the final
prompt position. Positive R = workspace leans to the newly-bound concept.

Axis 1: memory absent vs present@pos1  -- manipulation check.
Axis 2: target swept through its reachable ranks at fixed content -- the claim.

echo_clean runs on EVERY prompt before it is scored. A leak voids that run.
Covert fraction is tracked per the frozen K4 definition (layerwise).
"""
import json
import math
import subprocess
from pathlib import Path

import torch
import transformers

import jlens
from probe.context_builder import Embedder, build_conditions
from probe.exclusions import EchoLeak, covert_fraction, covert_hit_layerwise, echo_clean, rank_of
from probe.factset import validate
from probe.pins import BAND, LENS_FILENAME, LENS_REPO, LENS_REVISION, MODEL_ID, SUBSTRATE_VERSION
from probe.progress import bar

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "m4_sweep.json"
EPS = 1e-12
DTS = tuple(round(x, 1) for x in
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 68, 76, 85, 95,
             108, 122, 140, 160, 185, 215, 250, 290, 340, 400, 470, 550])


class Lens:
    def __init__(self):
        self.hf = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16
        ).to("cuda").eval()
        self.tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = jlens.from_hf(self.hf, self.tok)
        self.lens = jlens.JacobianLens.from_pretrained(
            LENS_REPO, filename=LENS_FILENAME, revision=LENS_REVISION
        )
        self.norm, self.head = self.hf.model.norm, self.hf.lm_head
        self.band = list(range(BAND[0], BAND[1] + 1))

    @torch.no_grad()
    def score(self, prompt, id_new, id_old):
        ll, _, _ = self.lens.apply(self.model, prompt, positions=[-1])
        enc = self.tok(prompt, return_tensors="pt").to("cuda")
        out = self.hf(**enc, output_hidden_states=True)
        hs = out.hidden_states
        layers = sorted(ll.keys())
        off = len(hs) - len(layers)

        logs, covert = [], []
        for l in self.band:
            p = torch.softmax(ll[l][0].float(), dim=-1)
            logs.append(math.log((p[id_new].item() + EPS) / (p[id_old].item() + EPS)))
            lens_r = rank_of(ll[l][0], id_new)
            logit_r = rank_of(self.head(self.norm(hs[l + off][0, -1].unsqueeze(0)))[0], id_new)
            covert.append(covert_hit_layerwise(lens_r, logit_r))
        return sum(logs) / len(logs), covert


def main() -> int:
    assert BAND is not None
    L = Lens()
    tok = L.tok
    embed = Embedder(L.hf, tok)

    pairs, _ = validate(tok, verbose=False)
    pairs = [p for p in pairs if p.key.startswith("country:")]
    print(f"{len(pairs)} country pairs; BAND {BAND}; scoring\n")

    axis1, axis2 = {}, {}
    covert_all, echo_voids = [], []
    kept = 0

    for pair in bar(pairs, label="mile-4 sweep"):
        conds, _ = build_conditions(tok, embed, pair, dts=DTS)
        if len(conds) < 3:
            continue

        targets = [(pair.t_new, pair.id_new), (pair.t_old, pair.id_old)]
        absent_prompt = f"Question: {pair.probe}\nAnswer:"
        try:
            echo_clean(absent_prompt, tok.encode(absent_prompt), targets)
            for c in conds.values():
                echo_clean(c.prompt, tok.encode(c.prompt), targets)
        except EchoLeak as e:
            echo_voids.append((pair.key, str(e)[:70]))
            continue

        r_absent, cov_a = L.score(absent_prompt, pair.id_new, pair.id_old)
        pos1 = min(conds)
        r_present, cov_p = L.score(conds[pos1].prompt, pair.id_new, pair.id_old)
        axis1[pair.key] = {"absent": r_absent, "present": r_present,
                           "delta": r_present - r_absent}
        covert_all += cov_a + cov_p

        row = {}
        for pos, c in sorted(conds.items()):
            r, cov = L.score(c.prompt, pair.id_new, pair.id_old)
            row[pos] = {"R": r, "dt": c.weight}
            covert_all += cov
        axis2[pair.key] = row
        kept += 1

    covf = covert_fraction(covert_all)
    print(f"\n{kept} pairs scored; {len(echo_voids)} echo-voided")
    for k, w in echo_voids[:5]:
        print(f"  VOID {k}: {w}")
    print(f"\ncovert fraction over BAND (K4 needs >=0.70): {covf:.3f}")
    if axis1:
        d1 = sorted(v["delta"] for v in axis1.values())
        print(f"Axis 1 median delta R: {d1[len(d1)//2]:.3f} nat  (check: >1.0)")

    OUT.write_text(json.dumps({
        "mile": 4, "band": list(BAND), "eps": EPS,
        "n_pairs_scored": kept, "echo_voids": echo_voids,
        "covert_fraction": covf, "covert_n": len(covert_all),
        "axis1": axis1, "axis2": axis2,
        "model_id": MODEL_ID, "lens_repo": LENS_REPO, "lens_revision": LENS_REVISION,
        "lens_filename": LENS_FILENAME, "SUBSTRATE_VERSION": SUBSTRATE_VERSION,
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout.strip(),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print("Mile 4 records raw R + covert only. Verdict is Mile 6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
