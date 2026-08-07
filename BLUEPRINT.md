# BLUEPRINT — J-space x the substrate probe

**Two weekends. Six miles. One plot.**

---

## The claim under test

> the substrate's utility scalar predicts workspace occupancy of a memory's bound concept, at fixed context content.

That is the whole thing. Everything below exists to make that sentence falsifiable.

---

## Why two axes, not two arms

The original split — *dose-response* and *ordering-at-fixed-content* — was wrong. A utility sweep that changes **which** memories appear in context is confounded with "different text in the prompt," and any positive result is uninterpretable.

the substrate's only runtime channel is the token stream. So utility can move exactly one thing without changing content: **position within the retrieved block.**

That collapses the design:

| | Manipulation | Holds constant | Role |
|---|---|---|---|
| **Axis 1** | memory absent -> present | — | **manipulation check** |
| **Axis 2** | position 1 -> 5 within block | full block content, byte-identical | **the finding** |

Axis 1 is not a result. It establishes the instrument reads anything at all. Axis 2 is the claim.

---

## Metric

For fact pair `f`, condition `c`, at the final prompt position (pre-generation):

```
1. Prefill. Capture residual stream h_l at every layer l.
2. lens_logits[l] = lens.apply(model, prompt, positions=[-1])
3. P_l = softmax(lens_logits[l])              # full vocabulary
4. m_new(l) = sum_{t in T_new} P_l[t]
   m_old(l) = sum_{t in T_old} P_l[t]
5. R(f,c) = mean_{l in BAND} log( (m_new(l) + eps) / (m_old(l) + eps) )
   eps = 1e-12
```

`R` is a log mass ratio in nats. Positive means the workspace leans toward the newly-bound concept.

**`BAND` is measured, not assumed.** Mile 1 determines it empirically as the contiguous layer range where the known-answer test achieves top-5 lens recall of the held-out answer token. The paper's structural finding — coherent content only in an intermediate layer band — is a prediction to confirm on this model, not a parameter to hardcode.

---

## Exclusions

Both are `assert`. A run that trips either is void, not annotated.

### Echo exclusion
No token in `T_new` or `T_old` may appear anywhere in the tokenised context. Check token IDs, not strings — BPE fragments leak. If a target appears as a continuation fragment inside another word, the run is void.

### Mouth exclusion
A lens hit counts only if the model is not already about to say it:

```
covert_hit(t, l) := rank_lens(t, l) < 5  AND  rank_logit_lens(t, l) > 50

where rank_logit_lens(t, l) = rank of t under unembed(final_norm(h_l))
      i.e. the model's OWN readout of the same layer the J-lens is reading.

AMENDED AT MILE 2 (pre-freeze). The original wording used
rank_model(t, final_layer) -- the model's final output. That is degenerate
here: measured across all 31 layers on 28 known-answer prompts it returns
0.00 EVERYWHERE, because those prompts are built so the model does say the
answer. In Mile 4 a successful rebinding likewise means the model says the
target, so the original definition would return False on exactly the runs the
experiment exists to measure. Both are implemented in probe.exclusions; the
strict form is retained as a diagnostic and gates nothing.

THE K4 THRESHOLD IS NOT AMENDED. Only the definition changed.
```

Reading the output two layers early is not a workspace measurement. Track the covert fraction per run; it is a gate (see K4).

---

## Fact set

`n >= 30` pairs. Each pair:

```python
{
  "deposit":  "Home base moved from Paris to Beijing last month.",
  "probe":    "What currency should I budget in?",
  "T_new":    ["yuan"],      # must tokenise to exactly one token
  "T_old":    ["euro"],      # must tokenise to exactly one token
}
```

**Build-time assertions, in `factset.py`:**

- Every target tokenises to exactly one token, in both bare and leading-space form. jlens readouts are single-token only — a multi-token concept is invisible, and you will misread that as a null.
- `T_new` and `T_old` are disjoint from `tokenize(deposit)`.
- `T_new` and `T_old` are disjoint from `tokenize(probe)`.

Fail at build time. Never at run time.

**Domain choice is not free.** Anchor on entity rebinding — the multi-fact editing structure is the one paper claim that replicated cleanly under external review. Rhyme planning and mental arithmetic failed to replicate; multi-hop probe-swap was weak. Do not build fact pairs that depend on those structures.

---

## Substrate wiring

The substrate is accessed only through the ranker interface (see
`probe/substrate.py`). Conceptually, each condition:

```python
# pseudo-code against the ranker contract; the substrate itself is proprietary
order, standing = ranker.rank_block(
    payloads,            # target memory + distractors
    embeddings,
    probe_embedding,
    aged_index=0,        # the target
    age_seconds=age,     # decay applied to the target's standing
)
```

**Embedder:** mean-pooled hidden states from the same Qwen model, taken from the layer below `BAND`. One model in memory, zero extra dependencies, and retrieval semantics stay aligned with the model doing the reasoning.

**Position control for Axis 2:** the target memory's rank within the returned block is the independent variable. Reach each position by ageing the target (decay over elapsed time) against the four fresh distractors -- the substrate's only lever that moves a memory's standing, since fresh memories share a ranking ceiling that no direct priority signal can exceed. `context_builder.py` must assert three things per pair:

1. `content_hash = sha256(sorted(lines))` is **IDENTICAL** across all conditions -- not one character of any memory changes.
2. `order_hash = sha256("\n".join(lines))` is **DISTINCT** across all -- proving the order actually moved and the conditions are not silently duplicates.
3. The realised index of the target equals the intended position.

AMENDED AT MILE 3 (pre-freeze). The original wording said the *concatenated*
block text must be byte-identical across conditions. That is self-contradictory:
permuting order necessarily changes the concatenation. The intent was that
CONTENT is held constant while ORDER varies, which is assertions 1 and 2 taken
together. Assertion 2 is the one the original wording would have lost, and it is
the one that catches the worst failure mode -- five "conditions" that are
actually the same prompt five times.

---

## Miles

Each mile is one Claude Code session. Hit the gate, write the JSON, stop.

### Mile 1 — Instrument
*No the substrate. No Mile 2 code.*

Install jlens as a dependency (Apache-2.0, github.com/anthropics/jacobian-lens). Load Qwen3.5-4B bf16 + `neuronpedia/jacobian-lens` revision `qwen-n1000`. Run the repo's known-answer example. Sweep all layers, determine `BAND`.

**Gate:** known-answer token in lens top-5 across a contiguous band of >=6 layers. `results/m1_instrument.json` records `BAND`, per-layer recall, wall-clock per prefill, and native-Windows vs WSL2.

### Mile 2 — Exclusions
*Still no the substrate.*

Build `exclusions.py` and its pytest suite. Echo exclusion on synthetic positive and negative cases. Mouth exclusion with the covert-hit definition above, validated against a hand-labelled set of 10 prompts where you know the answer is or is not about to be said.

**Gate:** `pytest tests/test_exclusions.py` green. Covert fraction reported on the Mile 1 known-answer prompts.

### Mile 3 — Fact set + the substrate wiring
Build `factset.py` with all build-time assertions. Generate >=30 pairs. Build `context_builder.py` with the byte-identity assertion.

**Gate:** 30 pairs pass every assertion. Five conditions per pair produce byte-identical block text with the target at the intended position, verified by hash. `results/m3_factset.json` carries the fact-set hash.

> **§KILL CRITERIA freezes when this gate closes.** From here forward it is read-only.

### Mile 4 — The sweep
Axis 1 then Axis 2. `n x 2` runs, then `n x 5` runs. Prefill only; generate just enough tokens for mouth exclusion.

**Gate:** `results/m4_sweep.json`, all runs echo-clean, covert fraction recorded.

### Mile 5 — Controls
Three, all required:

1. **No-memory baseline** — probe alone, no retrieved block.
2. **Shuffled-memory baseline** — block members replaced with unrelated deposits of equal length and count.
3. **Position-only null** — five distractor memories, no target present, swept through the same five positions. `R` should be flat and near zero. *If this control shows structure, position itself is driving the metric and Axis 2 is uninterpretable.*

**Gate:** `results/m5_controls.json`. Control 3 flat within noise.

### Mile 6 — Verdict
Apply the kill criteria. One matplotlib figure: `R` vs position, error bars across fact pairs, Axis 1 delta as a horizontal reference band. Verdict paragraph — three sentences, no hedging.

**Gate:** `results/m6_verdict.json` + `results/figure.png` + verdict paragraph in the JSON.

---

## Pre-registered analysis

Decided before Mile 4 runs. Not adjustable afterward.

- **Axis 1:** paired sign test across fact pairs. Pass = `p < 0.01` **and** median `delta R > 1.0` nat.
- **Axis 2:** Spearman rho between position and `R`, per fact pair. Pass = median `rho < -0.5` **and** sign test on `rho < 0` at `p < 0.01`.
- Sign tests hand-rolled from `math` and `statistics`. No scipy.

---

## KILL CRITERIA

**FROZEN at Mile 3 close (git tag mile-3, commit recorded in results/m3_factset.json). Do not edit, soften, or add exceptions.**

Mile 3 also established two hard facts about the instrument, recorded before the freeze:
- **Axis 2 mechanism is elapsed decay, not direct priority boost** (fresh memories share a ranking ceiling; the boost lever is inert).
- **Blocks resolve 3 ranks, not 5**, for country-rebinding pairs; language pairs resolve only 2 and were dropped. 31 country pairs cleared the >=30 gate at 3 ranks each. K4's 0.70 covert threshold and the Spearman/sign-test structure are UNCHANGED; only the number of positions per pair fell, from 5 to 3.

> ### K1 — Axis 1 fails
> Presence vs absence produces no reliable `delta R`.
> **The instrument is dead on this box. Stop. Do not run Mile 4.**
> Return to Mile 1 and re-derive `BAND`, or change model.

> ### K2 — Axis 1 passes, Axis 2 fails
> Ordering at fixed content does not move workspace occupancy.
> **the substrate's utility ranking has no promotion-gate consequence. Ship the negative and stop.**
> This is a real result. It says the substrate's ranking is retrieval convenience, not cognition — and it is worth more than a soft positive.

> ### K3 — Echo exclusion fires
> Any target token found in tokenised context.
> **Every result from that run is void.** Not annotated. Void.

> ### K4 — Covert fraction below 70%
> More than 30% of hits fail mouth exclusion.
> **You are reading the mouth, not the workspace.** Narrow `BAND` toward earlier layers and re-run Mile 2 before touching Mile 4.

---

## Known limits, stated upfront

- **Single-token readouts only.** Multi-token concepts are invisible to this instrument. A null is a null *for single-token concepts*, nothing broader.
- **The lens produces false positives.** External review found many. The covert-hit gate mitigates but does not eliminate this.
- **jlens is unmaintained reference code.** Adapt, never depend.
- **Automatic circuits bypass the workspace entirely.** Anthropic says so directly. A negative here cannot rule out that the effect exists somewhere the lens cannot see.
- **n=30 on one model.** This is an existence probe, not a paper.
- **Measured covert baseline (Mile 2): 0.15 mean across BAND (20,22), peak 0.21
  at L21, on 28 known-answer prompts.** K4 requires >= 0.70. That gap is real
  and it is recorded here rather than resolved by moving the threshold. Mile 3's
  prompts differ in kind -- retrieved context, contested binding, no single
  obvious completion -- so 0.15 is not a prediction of Mile 4's value. It is
  also not encouraging. If K4 fires, it fires.

---

## Out of scope

Custom lens fitting. Persistent-storage integration. Multi-turn persistence. Dictionary / SAE cross-check. Downstream training work. Other components of the broader (proprietary) architecture.

A dictionary-learning cross-check on a different model is a **separate blueprint** and it changes the model choice at Mile 1.
