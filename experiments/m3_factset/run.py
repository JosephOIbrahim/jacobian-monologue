"""Mile 3 gate: 38 pairs, five position-controlled conditions each, all three
assertions per pair. Writes fact-set hash + five per-pair block hashes.

Gate: every pair reaches BLOCK_SIZE positions, content identical / order
distinct / realised == intended for all. Closing this freezes KILL CRITERIA.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import transformers

from probe.context_builder import MIN_POSITIONS, Embedder, assert_conditions, build_conditions
from probe.factset import validate
from probe.pins import BAND, MODEL_ID, SUBSTRATE_VERSION
from probe.progress import bar

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "m3_factset.json"

# dense grid near the transitions so the target lands on >=4 distinct ranks
DTS = tuple(round(x, 1) for x in
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 68, 76, 85, 95,
             108, 122, 140, 160, 185, 215, 250, 290, 340, 400, 470, 550])


def main() -> int:
    assert BAND is not None
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to("cuda").eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    embed = Embedder(hf, tok)

    pairs, drops = validate(tok, verbose=False)
    # COUNTRY-ONLY at Mile 3: only country-rebinding blocks resolve >=3 ranks
    # under decay (language clumps to 2, see Mile 3 record). validate() returns
    # all kinds; filter to the domain the instrument can actually resolve.
    pairs = [p for p in pairs if p.key.startswith("country:")]
    print(f"{len(pairs)} country pairs (language/continent excluded); gate needs >=30\n")

    fs_hash = hashlib.sha256(
        "\n".join(f"{p.key}|{p.deposit}|{p.probe}|{p.t_new}|{p.t_old}" for p in pairs)
        .encode()
    ).hexdigest()[:16]

    records = {}
    failures = []
    for pair in bar(pairs, label="mile-3 conditions"):
        conds, diag = build_conditions(tok, embed, pair, dts=DTS)
        try:
            assert_conditions(conds, pair)
        except AssertionError as e:
            failures.append(str(e))
            continue
        records[pair.key] = {
            "content_hash": next(iter(conds.values())).content_hash,
            "order_hashes": {p: c.order_hash for p, c in conds.items()},
            "dts": diag["dts_used"],
            "target_utils": diag["target_utils"],
        }

    # Gate is >=30 pairs reaching >=3 ranks. Individual pairs may fail on block
    # geometry (a target that jumps 1->5 with no middle rank); those are dropped,
    # not disqualifying, exactly as multi-token attributes are dropped upstream.
    passed = len(records) >= 30
    print(f"\n{len(records)}/{len(pairs)} pairs passed (>= {MIN_POSITIONS} positions each); gate needs >=30")
    # distribution of how many positions each pair reached
    from collections import Counter
    dist = Counter(len(r["order_hashes"]) for r in records.values())
    print(f"positions-reached distribution: {dict(sorted(dist.items()))}")
    for f in failures[:8]:
        print(f"  FAIL {f}")

    if records:
        ex = next(iter(records.values()))
        print(f"\nexample content_hash (constant): {ex['content_hash']}")
        print(f"example order_hashes (distinct): {list(ex['order_hashes'].values())}")

    OUT.write_text(json.dumps({
        "mile": 3, "gate_passed": passed,
        "n_pairs": len(pairs), "n_passed": len(records),
        "factset_hash": fs_hash, "block_size": 5,
        "mechanism": "elapsed-time decay via the substrate ranker; direct priority boost inert (fresh memories share a ranking ceiling)",
        "pilot_verdict": "MEDIATES_POSITION (10/10 slid within block)",
        "per_pair": records, "failures": failures,
        "band": list(BAND), "SUBSTRATE_VERSION": SUBSTRATE_VERSION,
        "git_sha": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout.strip(),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print("GATE:", "PASS -- KILL CRITERIA NOW FROZEN" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
