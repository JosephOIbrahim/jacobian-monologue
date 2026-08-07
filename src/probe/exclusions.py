"""Mile 2 - echo and mouth exclusions.

Two independent guards. Both must hold for a run to count.

ECHO: the target concept must not be present in the context in any form.
Token-ID disjointness alone is NOT sufficient. " six" and "six" are distinct
IDs, so a context containing "sixty" passes an ID check while leaking the
concept at the surface. Mile 1's prompt filter caught exactly that case. We
check IDs and surface form, and either one firing voids the run.

MOUTH: a lens hit counts only if the output has not already committed. There
are two defensible readings of "committed" and they disagree sharply -- see
covert_hit_layerwise vs covert_hit_strict below.
"""

from __future__ import annotations

from typing import Iterable, Sequence

TOPK = 5
MOUTH_FLOOR = 50


class EchoLeak(AssertionError):
    """Target concept found in context. The run is void, not annotated."""


def echo_clean(
    context_text: str,
    context_ids: Sequence[int],
    targets: Sequence[tuple[str, int]],
) -> None:
    """Raise EchoLeak if any target leaks into the context. Returns None."""
    ids = set(context_ids)
    haystack = context_text.lower()
    for text, tid in targets:
        if tid in ids:
            raise EchoLeak(f"token id {tid} ({text!r}) present in context ids")
        needle = text.strip().lower()
        if needle and needle in haystack:
            raise EchoLeak(
                f"surface form {needle!r} present in context text "
                f"(token id {tid} was absent -- ID check alone would have passed)"
            )


def rank_of(logits, token_id: int) -> int:
    """0-based rank of token_id in a 1-D logit vector. rank 0 == argmax."""
    return int((logits > logits[token_id]).sum().item())


def covert_hit_layerwise(
    lens_rank: int,
    logit_lens_rank: int,
    *,
    topk: int = TOPK,
    mouth_floor: int = MOUTH_FLOOR,
) -> bool:
    """K4 GATE. Is the workspace holding the concept before THIS layer's
    residual commits to it?

    logit_lens_rank is the rank under unembed(final_norm(h_l)) -- the model's
    own readout of the same layer the J-lens is reading.
    """
    return lens_rank < topk and logit_lens_rank > mouth_floor


def covert_hit_strict(
    lens_rank: int,
    model_final_rank: int,
    *,
    topk: int = TOPK,
    mouth_floor: int = MOUTH_FLOOR,
) -> bool:
    """DIAGNOSTIC ONLY. Is the concept absent from the model's FINAL output?

    This is the BLUEPRINT's original wording. It answers a different and
    narrower question: does the workspace carry content the model never says?
    That is the paper's alignment-audit framing, and it is the wrong gate here.
    In Mile 4 a successful rebinding means the model DOES say the target, so
    this returns False on exactly the runs the experiment is designed to
    measure. Kept because a near-zero value is informative, not because it
    gates anything.
    """
    return lens_rank < topk and model_final_rank > mouth_floor


def covert_fraction(hits: Iterable[bool]) -> float:
    hits = list(hits)
    return sum(hits) / len(hits) if hits else 0.0
