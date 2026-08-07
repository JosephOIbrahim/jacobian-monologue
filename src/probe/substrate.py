"""Substrate ranking interface.

This experiment tests a proprietary memory substrate (patents pending) purely
through its observable ranking behaviour. The substrate itself is NOT vendored
or described here -- only the three-operation contract the probe depends on:

    1. write a memory (payload + embedding),
    2. age a chosen memory by N seconds of decay,
    3. rank the memories against a probe embedding, returning order + standing.

Any object implementing rank_block() with these semantics reproduces the
experiment. The default() ranker binds to the substrate when it is installed
and importable; without it, supply your own ranker to build_conditions().

The substrate's key observable properties, established in Mile 3 and reported
in the paper, are behavioural facts a reproducer needs -- not implementation:

  * Freshly written memories share a ranking ceiling: a memory cannot be
    pushed ABOVE that ceiling by a direct priority signal. The only lever that
    moves a memory's standing is decay over elapsed time.
  * Decay has a minimum timescale; sub-second ages are not meaningful.
  * When candidate relevances are tightly clustered, decay resolves ~3
    distinct ranks, not 5.

These are the properties that shaped the design. How the substrate implements
them is out of scope for this repository.
"""

from __future__ import annotations

from typing import Protocol


class Ranker(Protocol):
    def rank_block(
        self,
        payloads: list[str],
        embeddings: list[list[float]],
        probe_embedding: list[float],
        *,
        aged_index: int,
        age_seconds: float,
    ) -> tuple[list[str], dict]:
        """Return (payloads ordered best-first, {payload: standing})."""


class SubstrateRanker:
    """Binds to the proprietary substrate when available.

    Kept deliberately thin: the public repo carries the experiment and its
    results, not the substrate. `default()` raises a clear error if the
    substrate is not installed, directing the reproducer to supply their own
    Ranker implementation instead.
    """

    @staticmethod
    def default() -> "Ranker":
        try:
            from probe._substrate_impl import build_default_ranker
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "No substrate ranker available. The proprietary substrate is not "
                "included in this repository. Install it and provide a private "
                "probe/_substrate_impl.py exposing build_default_ranker(), or pass "
                "your own ranker=... to build_conditions(). See probe/substrate.py "
                "for the three-operation contract."
            ) from exc
        return build_default_ranker()
