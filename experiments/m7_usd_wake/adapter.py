"""m7 - delivery adapter. Woken memory -> prompt tokens, echo-guarded.

The honesty layer. A woken memory carries a `gist` (a paraphrase of the
situation) and `concepts` (the tokens we MEASURE in J-space). The adapter
delivers ONLY the gist. Concepts are never spoken -- they are what we watch
for. If a concept word appears in the delivered prompt, echo_clean voids it.

This is what makes the experiment falsifiable rather than circular: the model
must INFER the concepts from the situation, not read them.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass

sys.path.insert(0, "src")
from probe.exclusions import echo_clean, EchoLeak


@dataclass(frozen=True)
class Memory:
    path: str
    gist: str                 # DELIVERED (paraphrase, must not name concepts)
    concepts: tuple[str, ...] # MEASURED (target tokens, never delivered)
    origin: str = "synthetic"


# The synthetic episodic memory from the proposal. Gist describes the SITUATION;
# concepts are what should light up if the model recognizes it.
CONNECTION_POOL_INCIDENT = Memory(
    path="/Memories/ConnectionPoolIncident",
    gist=(
        "In a supplied prior scenario, a change to how the service pooled its "
        "sessions caused the service to exhaust its available capacity. "
        "Reverting that change restored the service, after the number of live "
        "sessions and the database's health were checked."
    ),
    # measured, NOT delivered. NOTE: "rollback"/"connection"/"verify" must NOT
    # appear in the gist above -- the adapter asserts this.
    concepts=("rollback", "connection", "verify"),
)

PROMPT_TEMPLATE = (
    "You are assessing a live situation.\n{memory}\n"
    "Current situation: a production service is showing elevated errors "
    "following a recent change.\n"
    "Assess the current situation and choose the next action.\nAssessment:"
)


def build_prompt(tok, memory: Memory | None) -> str:
    """Compose the prompt. If memory is None (dormant), no memory block is added.

    Asserts echo-cleanliness against the memory's OWN concepts: the gist may not
    contain any concept word or token. A violation raises EchoLeak -- the run is
    void, never silently delivered.
    """
    if memory is None:
        mem_block = ""      # dormant: the model gets the situation only
    else:
        mem_block = f"Relevant prior experience: {memory.gist}\n"
        # GUARD: the delivered gist must not echo any measured concept
        targets = []
        for c in memory.concepts:
            ids = tok.encode(" " + c, add_special_tokens=False)
            targets.append((" " + c, ids[0] if len(ids) == 1 else -1))
        echo_clean(mem_block, tok.encode(mem_block), targets)

    return PROMPT_TEMPLATE.format(memory=mem_block)


def concept_token_ids(tok, memory: Memory) -> dict:
    """The token ids we measure in J-space. Single-token concepts only."""
    out = {}
    for c in memory.concepts:
        ids = tok.encode(" " + c, add_special_tokens=False)
        out[c] = ids[0] if len(ids) == 1 else None
    return out


if __name__ == "__main__":
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    m = CONNECTION_POOL_INCIDENT

    print("concept token ids (measured, single-token check):")
    for c, tid in concept_token_ids(tok, m).items():
        print(f"  {c:<12} -> {tid}  {'OK' if tid is not None else 'MULTI-TOKEN -- unmeasurable'}")
    print()

    print("=== ALIGNED prompt (memory woken, gist delivered) ===")
    try:
        p = build_prompt(tok, m)
        print(p)
        print("  --> echo guard PASSED: gist delivered, no concept word leaked")
    except EchoLeak as e:
        print(f"  --> ECHO LEAK (gist names a concept!): {e}")
    print()

    print("=== COUNTERFACTUAL prompt (memory dormant, no delivery) ===")
    p0 = build_prompt(tok, None)
    print(p0)
