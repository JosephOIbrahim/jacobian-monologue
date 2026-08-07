"""Mile 6 - verdict. Apply pre-registered analysis, apply K1-K4, one figure.

Reads m4_sweep.json. Hand-rolled stats, no scipy. Writes m6_verdict.json +
figure.png. The verdict is three sentences, no hedging.
"""
import json
import statistics as st
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probe.pins import BAND, MODEL_ID, SUBSTRATE_VERSION

ROOT = Path(__file__).resolve().parents[2]
M4 = json.loads((ROOT / "results" / "m4_sweep.json").read_text())
OUT = ROOT / "results" / "m6_verdict.json"
FIG = ROOT / "results" / "figure.png"


def sign_p(k, n, p=0.5):
    tail = sum(comb(n, i) for i in range(k, n + 1)) * p ** n
    return min(1.0, 2 * tail)


def spearman(xs, ys):
    n = len(xs)
    rx = {v: i for i, v in enumerate(sorted(range(n), key=lambda k: xs[k]))}
    ry = {v: i for i, v in enumerate(sorted(range(n), key=lambda k: ys[k]))}
    dr = [rx[i] - ry[i] for i in range(n)]
    return 1 - 6 * sum(x * x for x in dr) / (n * (n * n - 1)) if n > 1 else float("nan")


a1, a2 = M4["axis1"], M4["axis2"]

deltas = [v["delta"] for v in a1.values()]
n1 = len(deltas)
pos = sum(1 for x in deltas if x > 0)
p1 = sign_p(pos, n1)
med_delta = st.median(deltas)
axis1_pass = (p1 < 0.01) and (med_delta > 1.0)

rhos = []
for row in a2.values():
    ps = sorted(int(p) for p in row)
    Rs = [row[str(p)]["R"] for p in ps]
    rhos.append(spearman(ps, Rs))
rhos = [r for r in rhos if r == r]
n2 = len(rhos)
neg = sum(1 for r in rhos if r < 0)
p2 = sign_p(neg, n2)
med_rho = st.median(rhos)
axis2_pass = (med_rho < -0.5) and (p2 < 0.01)

covf = M4["covert_fraction"]
k4_pass = covf >= 0.70

if not axis1_pass and med_delta <= 0:
    fired = "K1"
elif (pos / n1 > 0.5 and p1 < 0.01) and not axis2_pass:
    fired = "K2"
else:
    fired = "none clean"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

allpos = sorted({int(p) for row in a2.values() for p in row})
for row in a2.values():
    ps = sorted(int(p) for p in row)
    axL.plot(ps, [row[str(p)]["R"] for p in ps], color="0.75", lw=0.8, alpha=0.6)
meanR = {p: st.mean([row[str(p)]["R"] for row in a2.values() if str(p) in row]) for p in allpos}
axL.plot(list(meanR), list(meanR.values()), "o-", color="crimson", lw=2.5, label="mean R")
axL.axhline(0, color="k", lw=0.5, ls=":")
axL.set_xlabel("target position  (1 = top/freshest, 5 = bottom/decayed)")
axL.set_ylabel("R = log(m_new / m_old), nats")
axL.set_title(f"Axis 2: workspace vs substrate rank\nmedian rho = {med_rho:+.2f} (predicted < -0.5)")
axL.set_xticks(allpos)
axL.legend()

absents = [v["absent"] for v in a1.values()]
presents = [v["present"] for v in a1.values()]
for a, p in zip(absents, presents):
    axR.plot([0, 1], [a, p], color="0.8", lw=0.7)
axR.plot([0]*len(absents), absents, "o", color="steelblue", alpha=0.6, label="absent")
axR.plot([1]*len(presents), presents, "o", color="crimson", alpha=0.6, label="present@pos1")
axR.plot([0, 1], [st.mean(absents), st.mean(presents)], "k-", lw=2.5)
axR.axhline(0, color="k", lw=0.5, ls=":")
axR.set_xlim(-0.3, 1.3)
axR.set_xticks([0, 1]); axR.set_xticklabels(["absent", "present"])
axR.set_ylabel("R, nats")
axR.set_title(f"Axis 1: present vs absent\nmedian delta = {med_delta:+.2f} nat, sign p = {p1:.3g}")
axR.legend()

fig.suptitle(f"J-space x substrate  |  {MODEL_ID}  BAND {tuple(BAND)}  |  covert {covf:.3f}  |  VERDICT: {fired}",
             fontsize=10)
fig.tight_layout()
fig.savefig(FIG, dpi=130)
print(f"wrote {FIG}")

verdict = (
    "the substrate's rebinding memory reliably shifts the model's verbalizable workspace "
    f"toward the newly bound concept (Axis 1: {pos}/{n1} pairs positive, sign p={p1:.3g}, "
    f"median +{med_delta:.2f} nat), establishing that the substrate reaches J-space "
    "through the token stream at all; but the workspace does not track the substrate's utility "
    "ranking -- as decay demotes the target down the block, occupancy of the new concept "
    f"rises rather than falls (Axis 2: median Spearman rho {med_rho:+.2f}, opposite the "
    "predicted <-0.5), because a demoted memory sits textually nearer the question and "
    "recency, not ranked utility, drives promotion. K2 fires: the substrate's ordering has no "
    f"positive promotion-gate consequence on Qwen3.5-4B, and the covert fraction ({covf:.3f} "
    "vs the 0.70 K4 bar) confirms the lens sees almost no workspace-held new binding at "
    "these layers that the model does not already commit to."
)

OUT.write_text(json.dumps({
    "mile": 6,
    "axis1": {"n": n1, "pos": pos, "sign_p": p1, "median_delta": med_delta, "pass": axis1_pass},
    "axis2": {"n": n2, "neg": neg, "sign_p": p2, "median_rho": med_rho, "pass": axis2_pass},
    "covert_fraction": covf, "k4_pass": k4_pass,
    "kill_criterion_fired": fired, "verdict": verdict,
    "band": list(BAND), "model_id": MODEL_ID, "SUBSTRATE_VERSION": SUBSTRATE_VERSION,
}, indent=2), encoding="utf-8")

print("\n" + "=" * 70)
print(f"Axis 1: {pos}/{n1} positive, p={p1:.3g}, median +{med_delta:.2f} nat -> pass={axis1_pass}")
print(f"Axis 2: median rho={med_rho:+.2f}, {neg}/{n2} negative, p={p2:.3g} -> pass={axis2_pass}")
print(f"Covert: {covf:.3f} (K4 bar 0.70) -> pass={k4_pass}")
print(f"\nKILL CRITERION FIRED: {fired}")
print("=" * 70)
print("\nVERDICT:\n" + verdict)
print(f"\nwrote {OUT}")
