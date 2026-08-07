"""Mile 3 - substrate wiring and position-controlled context blocks.

The memory substrate is the independent variable. Its decay-over-time ranking
decides where the target memory sits in the retrieved block. The target is
deposited as "already aged" and the distractors fresh, so the substrate ranks
the target lower purely through elapsed-time decay. Sweeping the target's age
slides it down the ranking.

WHY DECAY, NOT A DIRECT PRIORITY BOOST. The original design tried to raise the
target's priority directly. That lever proved inert on this substrate: freshly
written memories sit at a ranking ceiling, so a direct boost cannot move them
and ranking reduces to relevance. Decay over time is the only lever that moves
a memory's standing -- and it is the substrate's core thesis. The Mile 3 pilot
(m3_pilot.json) confirmed 10/10 targets SLIDE within the block under decay
rather than dropping out -- so position, not mere presence, is manipulated.

The substrate's concrete call surface is held behind the SubstrateRanker
interface below (this repo does not vendor or document the substrate's
internals -- it is proprietary and tested here only through its public
behaviour). Swap in any ranker exposing the same three operations to reproduce.

Content is held constant across all conditions. Only order moves. See
BLUEPRINT "Position control for Axis 2" for the three assertions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from probe.factset import BLOCK_SIZE, Pair, distractors_for
from probe.pins import BAND
from probe.substrate import SubstrateRanker

PROMPT_TEMPLATE = "Notes:\n{block}\n\nQuestion: {probe}\nAnswer:"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Condition:
    position: int          # 1-based, intended AND realised (asserted equal)
    lines: tuple[str, ...]
    prompt: str
    weight: float
    content_hash: str
    order_hash: str


class Embedder:
    """Mean-pooled hidden states from the layer below BAND.

    One model in memory, zero extra dependencies, and retrieval semantics stay
    aligned with the model doing the reasoning.
    """

    def __init__(self, hf, tok, layer: int | None = None):
        self.hf, self.tok = hf, tok
        assert BAND is not None, "BAND is None -- Mile 1 did not close"
        self.layer = BAND[0] - 1 if layer is None else layer

    @torch.no_grad()
    def __call__(self, text: str) -> list[float]:
        enc = self.tok(text, return_tensors="pt").to("cuda")
        hs = self.hf(**enc, output_hidden_states=True).hidden_states
        vec = hs[self.layer][0].mean(dim=0).float()
        vec = vec / (vec.norm() + 1e-12)
        return vec.cpu().tolist()


def _retrieve(ranker: SubstrateRanker, payloads, embs, probe_emb, age: float) -> tuple[list[str], dict]:
    """Rank one block through the substrate. Target (index 0) is aged by `age`;
    distractors stay fresh. Returns (order, standing-per-item).

    Single lever: only the target's standing moves, via decay over `age`.
    Ranking = relevance gated by the one standing we are sweeping. No distractor
    manipulation, which would add a second axis and reintroduce the confound the
    two-axis redesign removed. All substrate calls go through the ranker
    interface; see probe/substrate.py.
    """
    order, standing = ranker.rank_block(
        payloads, embs, probe_emb, aged_index=0, age_seconds=age
    )
    return order, standing


def build_conditions(
    tok,
    embed: Embedder,
    pair: Pair,
    dts=(0.0, 30.0, 60.0, 120.0, 300.0),
    ranker: SubstrateRanker | None = None,
) -> tuple[dict[int, Condition], dict]:
    """Age the target across `dts`; keep the first age that lands it at each
    realised position 1..BLOCK_SIZE. Returns (conditions, diag).

    `dts` (seconds of decay) maps to a descending standing curve. If a coarse
    sweep misses a position, callers may pass a finer grid. `ranker` defaults to
    the configured substrate ranker (see probe/substrate.py).
    """
    distractors = distractors_for(tok, pair)
    payloads = [pair.deposit] + distractors
    embs = [embed(p) for p in payloads]
    probe_emb = embed(pair.probe)
    n = len(payloads)

    if ranker is None:
        ranker = SubstrateRanker.default()

    found: dict[int, tuple[float, list[str], dict]] = {}
    for dt in dts:
        order, standing = _retrieve(ranker, payloads, embs, probe_emb, dt)
        if len(order) != n:
            continue
        pos = order.index(pair.deposit) + 1
        found.setdefault(pos, (dt, order, standing))

    conditions: dict[int, Condition] = {}
    for pos, (dt, order, standing) in sorted(found.items()):
        block = "\n".join(order)
        conditions[pos] = Condition(
            position=pos,
            lines=tuple(order),
            prompt=PROMPT_TEMPLATE.format(block=block, probe=pair.probe),
            weight=dt,  # the decay age (s) that produced this position
            content_hash=_sha("\u0000".join(sorted(order))),
            order_hash=_sha(block),
        )

    diag = {
        "positions_reached": sorted(found),
        "positions_missing": [p for p in range(1, n + 1) if p not in found],
        "dts_used": {p: c.weight for p, c in conditions.items()},
        "target_standing": {p: round(found[p][2].get(pair.deposit, float("nan")), 4)
                            for p in found},
    }
    return conditions, diag


MIN_POSITIONS = 3  # see note below; the block geometry caps reachable ranks


def assert_conditions(conditions: dict[int, Condition], pair: Pair) -> None:
    """The three BLUEPRINT assertions. Loud, not annotated.

    POSITION COUNT: relaxed BLOCK_SIZE(5) -> 3 at Mile 3, on measured geometry,
    not convenience. With real Qwen embeddings the four distractors span a
    ~0.03 probe-similarity band (0.62-0.65) while the target sits at 0.72, well
    above the clump. As the target decays its score crosses the entire
    distractor band almost at once, so it lands on ranks {1,4,5} and skips 2,3
    -- those ranks are not physically realisable for this block, and no dt grid
    recovers them (verified: dense 29-point sweep still yields {1,4,5}).

    This is itself a result: the substrate's decay ranking has COARSE RESOLUTION when
    candidates are near-equidistant. Three monotone (utility, position) points
    still support a Spearman rank correlation and a sign test -- weaker than
    five, but honest. Forcing more ranks would require engineering the
    distractors' relevance spread, which is instrument-shaping to hit a target
    number, exactly the move the kill criteria exist to prevent.
    """
    assert conditions, f"{pair.key}: no conditions built"
    assert len(conditions) >= MIN_POSITIONS, (
        f"{pair.key}: {len(conditions)}/{BLOCK_SIZE} positions reachable "
        f"({sorted(conditions)}), need >={MIN_POSITIONS} for a rank correlation."
    )

    # 1. content identical across conditions
    ch = {c.content_hash for c in conditions.values()}
    assert len(ch) == 1, f"{pair.key}: content drifted across conditions -- {ch}"

    # 2. order distinct across conditions
    oh = [c.order_hash for c in conditions.values()]
    assert len(set(oh)) == len(oh), (
        f"{pair.key}: duplicate order hashes -- conditions are the same prompt twice"
    )

    # 3. realised position == intended
    for pos, c in conditions.items():
        got = c.lines.index(pair.deposit) + 1
        assert got == pos == c.position, f"{pair.key}: intended {pos}, realised {got}"
