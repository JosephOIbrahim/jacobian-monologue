"""Mile 3 pilot: does elapsed decay move the target THROUGH positions, or drop
it OUT of the block? This decides the Axis 2 redesign. No lens, 10 pairs.

Mechanism:
  - age the target by `dt` seconds of decay; keep distractors fresh.
  - the substrate ranks the aged target lower as `dt` grows; standing follows a
    descending decay curve while distractors hold at the fresh ceiling.
  - read the target's realised position.
Zero wall-clock waiting -- ageing is applied through the ranker, not by
sleeping. All substrate calls go through the ranker interface (probe/substrate).
"""
import json
from pathlib import Path

import transformers

from probe.factset import distractors_for, validate
from probe.substrate import SubstrateRanker

ROOT = Path(__file__).resolve().parents[2]
DTS = [0.0, 30.0, 60.0, 120.0, 300.0]  # seconds of decay: descending standing

tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
pairs, _ = validate(tok, verbose=False)
pilot = pairs[:10]
ranker = SubstrateRanker.default()
print(f"pilot: {len(pilot)} pairs")
print(f"dt sweep (s of decay): {DTS}\n")


def emb(i, n):  # target most probe-relevant, distractors graded. GPU-free pilot.
    return [1.0 - i / n, i / n]


def realised(pair, dt):
    dists = distractors_for(tok, pair)
    payloads = [pair.deposit] + dists
    embs = [emb(i, len(payloads)) for i in range(len(payloads))]
    order, standing = ranker.rank_block(
        payloads, embs, embs[0], aged_index=0, age_seconds=dt
    )
    pos = order.index(pair.deposit) + 1 if pair.deposit in order else None
    return pos, round(standing.get(pair.deposit, float("nan")), 4)


rows = []
for pair in pilot:
    by_dt = {}
    for dt in DTS:
        pos, u = realised(pair, dt)
        by_dt[dt] = {"pos": pos, "in_top5": bool(pos and pos <= 5), "standing": u}
    rows.append({"key": pair.key, "by_dt": by_dt})
    print(f"  {pair.key:<28} pos {[by_dt[d]['pos'] for d in DTS]}  standing {[by_dt[d]['standing'] for d in DTS]}")

moved = sum(1 for r in rows if len({r["by_dt"][d]["pos"] for d in DTS}) > 1)
dropped = sum(1 for r in rows if any(not r["by_dt"][d]["in_top5"] for d in DTS))
slid = sum(1 for r in rows
           if all(r["by_dt"][d]["in_top5"] for d in DTS)
           and len({r["by_dt"][d]["pos"] for d in DTS}) > 1)

print(f"\n{'='*52}")
print(f"target position moved at all     : {moved}/10")
print(f"target ever left top-5 (dropped) : {dropped}/10")
print(f"target SLID within block         : {slid}/10")
print('='*52)
if slid >= 6:
    verdict = "MEDIATES_POSITION"
elif dropped >= 6:
    verdict = "ACTS_DIRECTLY"
else:
    verdict = "WEAK"
print(f"VERDICT: {verdict}")

OUT = ROOT / "results" / "m3_pilot.json"
OUT.write_text(json.dumps({
    "dts": DTS, "n": len(pilot),
    "moved": moved, "dropped": dropped, "slid": slid, "verdict": verdict,
    "rows": rows,
}, indent=2), encoding="utf-8")
print(f"wrote {OUT}")
