# SESSIONS - copy-paste, one per Claude Code session

Open Claude Code in `<repo-root>`. Read `CLAUDE.md` and `BLUEPRINT.md` first in every session.

**Run the gate before every mile. No exceptions.**

```
.\.venv\Scripts\python.exe scripts\verify.py
```

Seven checks, exit 0 required. It catches CPU-only torch, a broken hidden-state
path, and the substrate pin drift - the three failures that would otherwise surface as
bad science three miles later instead of as a red line here.

One mile per session. Stop at the gate.

---

## Session 0 - Scaffold   [DONE - commit e6e6fd3]

Already executed. **Do not re-run - it would rebuild the venv.**

What landed:

- venv on CPython 3.12.10 (`uv venv --python 3.12`)
- `torch 2.11.0+cu128` - CUDA True on the 4090. *Default PyPI served a CPU-only
  wheel first; reinstalled from the cu128 index, which caps torch at 2.11.0.*
- `transformers 5.14.1`, matplotlib, pytest
- `jlens 0.1.0` installed as a dependency (Apache-2.0, github.com/anthropics/jacobian-lens)
- `substrate (proprietary build)` editable from `<path-to-substrate>` at SHA **substrate-mile1-frozen**
- `internal-monologue` installed editable - `import probe` needs no PYTHONPATH
- `src/probe/progress.py` - stdlib progress bar, TTY and non-TTY paths
- `src/probe/pins.py` - frozen SHA + model/lens IDs, single source of truth
- `scripts/verify.py` - the gate
- `.gitattributes` pinning LF (Mile 3 asserts byte-identical text; platform
  line-ending rewrites would produce a phantom failure)

Still to create, at the mile that needs them: `experiments/m1..m6/`, `tests/`.

---

## Session 1 - Instrument

```
Read CLAUDE.md and BLUEPRINT.md. Mile 1 only.

FIRST: run scripts/verify.py. If it does not exit 0, stop and report.

Goal: prove the lens reads on this box and measure BAND.

1. Load Qwen3.5-4B bf16 + neuronpedia/jacobian-lens revision qwen-n1000.
   IDs are in src/probe/pins.py - use them, do not retype them.
2. Run the known-answer example from the jlens README.
3. Sweep every layer. For ~20 known-answer prompts, record top-5 lens recall
   of the held-out answer token per layer. Use probe.progress.bar for the sweep.
4. BAND = longest contiguous run of layers with recall above chance by a clear
   margin. Report the margin. Do not pick a threshold silently.
5. Write BAND back into src/probe/pins.py.

Write results/m1_instrument.json: BAND, per-layer recall, wall-clock per prefill,
model + lens revisions, SUBSTRATE_VERSION from pins, git SHA, native-Windows or WSL2.

Gate: known-answer token in lens top-5 across >=6 contiguous layers.

Note: torch is 2.11.0+cu128 (the cu128 index caps there) against transformers
5.14.1. If you hit an API incompatibility between them, say so plainly - do not
paper over it, it changes the instrument.

Do not import substrate  # proprietary; see probe/substrate.py. Do not write exclusions.py. Stop at the gate.
```

---

## Session 2 - Exclusions

```
Read CLAUDE.md and BLUEPRINT.md. Mile 2 only.
FIRST: run scripts/verify.py. If it does not exit 0, stop and report.

Build src/probe/exclusions.py:
- echo_clean(context_ids, target_ids) -> None. Asserts disjoint. Token IDs, not strings.
- covert_hit(lens_ranks, model_ranks, token, layer) -> bool, per the BLUEPRINT definition.
- covert_fraction(run) -> float.

Build tests/test_exclusions.py:
- echo: synthetic positives, negatives, and one BPE-fragment leak case that MUST fail.
- mouth: 10 hand-labelled prompts where you know the answer is / is not imminent.

Then report covert_fraction on the Mile 1 known-answer prompts.

Gate: pytest green + covert fraction reported.

Do not import substrate  # proprietary; see probe/substrate.py. Stop at the gate.
```

---

## Session 3 - Fact set + wiring

```
Read CLAUDE.md and BLUEPRINT.md. Mile 3 only.
FIRST: run scripts/verify.py. If it does not exit 0, stop and report.

1. src/probe/factset.py - 30+ entity-rebinding pairs, every build-time assertion
   from the BLUEPRINT enforced at import. Single-token, bare and leading-space.
   Echo-disjoint from both deposit and probe.

2. src/probe/context_builder.py - the substrate ephemeral, mean-pooled embedder from the
   layer below BAND, five conditions per pair.
   Two hard assertions:
     a. realised rank of target == intended position
     b. concatenated block text byte-identical across all five conditions (hash it)

Write results/m3_factset.json with the fact-set hash and five per-pair block hashes.

Gate: 30 pairs pass everything; five conditions per pair hash-identical.

When this gate closes, BLUEPRINT.md KILL CRITERIA is frozen. Say so out loud
in your final message. Stop.
```

---

## Session 4 - The sweep

```
Read CLAUDE.md and BLUEPRINT.md. Mile 4 only.
FIRST: run scripts/verify.py. If it does not exit 0, stop and report.

Axis 1: n x 2 runs (absent, present@pos1).
Axis 2: n x 5 runs (positions 1-5, content fixed).

Wrap both sweeps in probe.progress.bar. Non-TTY path prints every 5s, which is
what you want in a Claude Code log.

Prefill only. Generate the minimum tokens needed for mouth exclusion.
Run echo_clean on every single run before recording it.

Write results/m4_sweep.json: raw R per (pair, condition), covert fraction,
full resolved config, seed, SUBSTRATE_VERSION.

Gate: all runs echo-clean, JSON written.

Do NOT compute the verdict. Do NOT apply kill criteria. Do NOT plot. Stop.
```

---

## Session 5 - Controls

```
Read CLAUDE.md and BLUEPRINT.md. Mile 5 only.
FIRST: run scripts/verify.py. If it does not exit 0, stop and report.

Three controls, all required:
1. No-memory baseline.
2. Shuffled-memory baseline (unrelated deposits, equal length and count).
3. Position-only null (five distractors, no target, same five positions).

Write results/m5_controls.json.

Gate: control 3 flat within noise. If control 3 shows structure, say so plainly
and stop - Axis 2 is uninterpretable and Mile 6 does not run.

Stop at the gate.
```

---

## Session 6 - Verdict

```
Read CLAUDE.md and BLUEPRINT.md. Mile 6 only.
FIRST: run scripts/verify.py. If it does not exit 0, stop and report.

1. Apply the pre-registered analysis exactly as written. Hand-rolled sign tests,
   no scipy.
2. Apply K1-K4 in order. State which fired, if any.
3. One matplotlib figure: R vs position, error bars across pairs, Axis 1 delta
   as a horizontal reference band. results/figure.png.
4. Verdict paragraph, three sentences, no hedging, into results/m6_verdict.json.

If a kill criterion fires, write the negative with the same care as a positive.
Do not soften it. Do not propose a follow-up experiment in the verdict - that is
a separate conversation.

Stop.
```

---

## Between sessions

After each gate, before starting the next:

```
git add -A ; git commit -m "mile N: <gate result>"
```

The results JSONs are the record. There is no other record.
