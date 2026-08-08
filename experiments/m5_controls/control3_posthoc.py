"""Mile 5, Control 3 -- position-only null. RUN POST HOC, and labelled as such.

PROVENANCE. The BLUEPRINT pre-registered three Mile 5 controls. They were not
executed before the verdict: K2's firing condition was already unambiguous in
the Mile 4 data (median rho +1.00 against a predicted < -0.5) and the K2
protocol -- ship the negative and stop -- was followed from Mile 4 to Mile 6.
This script executes Control 3 after the fact. It cannot retroactively become
the pre-registered control; it is a post-hoc robustness check on the verdict's
interpretation, and the results file records that status explicitly.

WHAT IT TESTS. The Mile 6 verdict attributes the Axis 2 inversion to recency:
a demoted TARGET sits textually nearer the question and gains influence. The
alternative it must rule out: the metric R has position structure with NO
target present at all -- i.e. the block geometry itself, not the target's
content, drives R. Here every block is five neutral distractors (no rebinding
deposit anywhere). One designated neutral line is swept through positions 1-5
while the other four hold a fixed order. R is scored exactly as in Mile 4,
against each pair's (T_new, T_old) token ids.

CRITERION, STATED BEFORE THIS RUN. FLAT (verdict interpretation survives) =
|median Spearman rho(position, R)| < 0.5 AND neither sign test (rho<0, rho>0)
reaches p < 0.01. Scale is reported alongside: median per-pair R-range here
vs the median per-pair R-range of Mile 4 Axis 2. If the null shows Axis-2-scale
structure, the recency interpretation does NOT survive and the record says so.
"""
import importlib.util
import json
import sys
from datetime import date
from math import comb
from pathlib import Path
from statistics import median

sys.path.insert(0, "src")

from probe.exclusions import EchoLeak, echo_clean
from probe.factset import distractors_for, validate
from probe.pins import BAND, MODEL_ID
from probe.progress import bar

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "m5_control3_posthoc.json"
M4 = json.loads((ROOT / "results" / "m4_sweep.json").read_text(encoding="utf-8"))

# Reuse Mile 4's Lens verbatim -- same metric, same band, same code path.
_spec = importlib.util.spec_from_file_location(
    "m4run", ROOT / "experiments" / "m4_sweep" / "run.py"
)
_m4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m4)

PROMPT_TEMPLATE = "Notes:\n{block}\n\nQuestion: {probe}\nAnswer:"
POSITIONS = (1, 2, 3, 4, 5)


def sign_p(k: int, n: int, p: float = 0.5) -> float:
    tail = sum(comb(n, i) for i in range(k, n + 1)) * p ** n
    return min(1.0, 2 * tail)


def spearman(xs, ys):
    n = len(xs)
    rx = {v: i for i, v in enumerate(sorted(range(n), key=lambda k: xs[k]))}
    ry = {v: i for i, v in enumerate(sorted(range(n), key=lambda k: ys[k]))}
    dr = [rx[i] - ry[i] for i in range(n)]
    return 1 - 6 * sum(x * x for x in dr) / (n * (n * n - 1)) if n > 1 else float("nan")


def main() -> int:
    assert BAND is not None
    L = _m4.Lens()
    tok = L.tok

    pairs, _ = validate(tok, verbose=False)
    pairs = [p for p in pairs if p.key.startswith("country:")]
    print(f"{len(pairs)} country pairs; BAND {BAND}; position-only null (no target)\n")

    per_pair, rhos, ranges, voids = {}, [], [], []
    for pair in bar(pairs, label="m5 control-3 null"):
        dists = distractors_for(tok, pair, n=5)   # five neutral lines, echo-clean
        base, swept = dists[:4], dists[4]
        targets = [(pair.t_new, pair.id_new), (pair.t_old, pair.id_old)]

        Rs = {}
        try:
            for pos in POSITIONS:
                lines = base[: pos - 1] + [swept] + base[pos - 1 :]
                prompt = PROMPT_TEMPLATE.format(block="\n".join(lines), probe=pair.probe)
                echo_clean(prompt, tok.encode(prompt), targets)
                r, _cov = L.score(prompt, pair.id_new, pair.id_old)
                Rs[pos] = r
        except EchoLeak as e:
            voids.append((pair.key, str(e)[:70]))
            continue

        rho = spearman(list(POSITIONS), [Rs[p] for p in POSITIONS])
        rhos.append(rho)
        ranges.append(max(Rs.values()) - min(Rs.values()))
        per_pair[pair.key] = {"R": {str(p): round(Rs[p], 4) for p in POSITIONS},
                              "rho": round(rho, 4)}

    n = len(rhos)
    neg = sum(1 for r in rhos if r < 0)
    pos_ = sum(1 for r in rhos if r > 0)
    p_neg, p_pos = sign_p(neg, n), sign_p(pos_, n)
    med_rho = median(rhos)
    med_range = median(ranges)

    m4_ranges = []
    for row in M4["axis2"].values():
        vals = [row[k]["R"] for k in row]
        m4_ranges.append(max(vals) - min(vals))
    m4_med_range = median(m4_ranges)

    flat = abs(med_rho) < 0.5 and p_neg >= 0.01 and p_pos >= 0.01
    verdict = (
        "FLAT: with no target present, R shows no reliable position structure "
        f"(median rho {med_rho:+.2f}; sign tests ns; median per-pair range "
        f"{med_range:.3f} nat vs Mile 4 Axis 2 {m4_med_range:.3f} nat). The block "
        "geometry alone does not drive the metric; the Axis 2 inversion required "
        "the target's content, consistent with the recency interpretation."
        if flat else
        "STRUCTURED: the position-only null shows reliable structure "
        f"(median rho {med_rho:+.2f}, sign p rho<0 {p_neg:.3g} / rho>0 {p_pos:.3g}; "
        f"median per-pair range {med_range:.3f} nat vs Mile 4 {m4_med_range:.3f}). "
        "The metric carries position structure without any target, and the Mile 6 "
        "recency interpretation does NOT survive this control as stated."
    )

    print(f"\nmedian rho {med_rho:+.3f} | rho<0 {neg}/{n} p={p_neg:.3g} | "
          f"rho>0 {pos_}/{n} p={p_pos:.3g}")
    print(f"median per-pair R-range: {med_range:.3f} nat (Mile 4 Axis 2: {m4_med_range:.3f})")
    print(f"echo-voided: {len(voids)}")
    print(f"\nVERDICT: {verdict}")

    OUT.write_text(json.dumps({
        "experiment": "m5_control3_position_only_null",
        "post_hoc": True,
        "run_date": date.today().isoformat(),
        "provenance": "pre-registered in BLUEPRINT Mile 5; executed after the "
                      "Mile 6 verdict as a robustness check, not as the "
                      "pre-registered control",
        "criterion": "FLAT iff |median rho|<0.5 and both sign tests ns at 0.01",
        "n_pairs": n, "median_rho": med_rho,
        "sign_p_rho_neg": p_neg, "sign_p_rho_pos": p_pos,
        "median_R_range": med_range, "m4_axis2_median_R_range": m4_med_range,
        "flat": flat, "verdict": verdict, "per_pair": per_pair,
        "echo_voids": voids, "band": list(BAND), "model_id": MODEL_ID,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
