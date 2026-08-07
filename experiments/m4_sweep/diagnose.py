"""Mile 4 diagnosis: why did covert collapse, and is Axis 2 monotone?
Reads m4_sweep.json only. No model. Does not alter results."""
import json
import math
from pathlib import Path

d = json.loads((Path(__file__).resolve().parents[2] / "results" / "m4_sweep.json").read_text())

print("=== AXIS 1: absent vs present@pos1 ===")
a1 = d["axis1"]
deltas = sorted(v["delta"] for v in a1.values())
absents = [v["absent"] for v in a1.values()]
presents = [v["present"] for v in a1.values()]
n = len(deltas)
print(f"  n={n}")
print(f"  R absent : median {sorted(absents)[n//2]:+.3f}  range [{min(absents):+.2f},{max(absents):+.2f}]")
print(f"  R present: median {sorted(presents)[n//2]:+.3f}  range [{min(presents):+.2f},{max(presents):+.2f}]")
print(f"  delta    : median {deltas[n//2]:+.3f}  range [{min(deltas):+.2f},{max(deltas):+.2f}]")
pos_delta = sum(1 for x in deltas if x > 0)
print(f"  pairs with delta>0 (memory pushed toward new concept): {pos_delta}/{n}")
# sign test p-value, two-sided, hand-rolled
def sign_p(k, n, p=0.5):
    from math import comb
    tail = sum(comb(n, i) for i in range(k, n+1)) * p**n
    return min(1.0, 2*tail)
print(f"  sign test (delta>0) p = {sign_p(pos_delta, n):.4g}")

print("\n=== AXIS 2: does R fall as target position worsens? ===")
a2 = d["axis2"]
# Spearman rho per pair between position and R, hand-rolled
def spearman(xs, ys):
    n = len(xs)
    rx = {v:i for i,v in enumerate(sorted(range(n), key=lambda k: xs[k]))}
    ry = {v:i for i,v in enumerate(sorted(range(n), key=lambda k: ys[k]))}
    dr = [rx[i]-ry[i] for i in range(n)]
    return 1 - 6*sum(x*x for x in dr)/(n*(n*n-1)) if n>1 else float("nan")
rhos = []
for key, row in a2.items():
    positions = sorted(int(p) for p in row)
    Rs = [row[str(p)]["R"] for p in positions]
    rho = spearman(positions, Rs)
    rhos.append(rho)
rhos_s = sorted(r for r in rhos if r==r)
m = len(rhos_s)
print(f"  n={m}   median rho(position, R) = {rhos_s[m//2]:+.3f}  (claim: <-0.5)")
neg = sum(1 for r in rhos_s if r < 0)
print(f"  pairs with rho<0 (R falls as position worsens): {neg}/{m}")
print(f"  sign test (rho<0) p = {sign_p(neg, m):.4g}")

print("\n  sample per-pair position->R:")
for key in list(a2)[:5]:
    row = a2[key]
    seq = {int(p): round(row[str(p)]["R"],2) for p in sorted(row, key=int)}
    print(f"    {key:<26} {seq}")

print("\n=== COVERT collapse ===")
print(f"  Mile 2 (known-answer): 0.21   Mile 4 (rebinding): {d['covert_fraction']:.3f}")
print(f"  covert samples: {d['covert_n']}")
print("  interpretation: on rebinding prompts the NEW token rarely enters lens")
print("  top-5 while the layer's logit-lens stays silent. Either the workspace")
print("  is not promoting the new binding at these layers, or the lens cannot")
print("  see it. Axis 1 delta sign + Axis 2 rho sign are what adjudicate K1/K2.")
