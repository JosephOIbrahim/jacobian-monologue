# CLAUDE.md â€” InternalMonologue

J-space x the substrate probe. Experiment repo, not a product. Optimise for a defensible verdict, not for coverage.

## Resolved environment facts

- Repo root: `<repo-root>`
- the substrate source: `<path-to-substrate>` (editable install target)
- System Python is 3.14.2. **Do not build the venv on it.** Pin 3.12 â€” `uv venv --python 3.12`. Torch/CUDA wheel availability on 3.14 is the single most likely cause of a lost first session, and that failure would land on Mile 1 disguised as an instrument problem.
- Windows + RTX 4090 (24 GB), CUDA, PowerShell.
- Desktop Commander file tools now reach `<repo-root>` (granted at Mile 1 close, scoped to this directory only, not to `D:\`). DC's shell always could. The Filesystem MCP server also has access but hung mid-edit once during Mile 0 â€” prefer DC.

## Hard rules

1. **The Jacobian lens is an external dependency.** Install it (Apache-2.0, github.com/anthropics/jacobian-lens); do not vendor or patch it. Wrap it in `src/probe/`.
2. **The substrate runs ephemeral.** Use an in-memory, throwaway substrate instance only — no persistent tier, no external storage engine anywhere in Miles 1-6. If you find yourself standing up persistence, you have left the experiment.
3. **No new runtime dependency without asking.** The allowed set is pinned in `pyproject.toml`. Adding `sentence-transformers`, `scipy`, `pandas`, or anything else requires an explicit go-ahead from Joe.
4. **`BLUEPRINT.md` Â§KILL CRITERIA freezes when Mile 3 closes.** Do not edit, soften, reword, or add escape hatches. If a criterion looks wrong mid-run, stop and say so â€” do not adjust it.
5. **One mile per session.** Hit the gate, write the results JSON, stop. Do not continue into the next mile even if it looks easy.
6. **Every run writes `results/mN_<name>.json`** with the full resolved config embedded â€” model revision, lens revision, layer band, seed, fact-set hash, git SHA. A result with no embedded config is void.
7. **Assertions fail loud.** Echo exclusion and single-token validation are `assert`, not warnings. A run that trips one is discarded, not annotated.
8. **No process ceremony.** No SCOUT / KICKOFF / HANDOFF / SURGERY documents. This repo produces code, results JSONs, and one plot.

## Style

- Terse. No docstring essays. Type hints on public functions only.
- `pytest` for the exclusion harnesses. Nothing beyond that.
- No CLI framework. `python -m experiments.mN_name.run` is the interface.
- Plots: matplotlib, one figure, zero styling work.

## Scope boundary

**In scope:** fitting nothing, reading a pre-fitted lens, building fact sets, wiring the substrate as a context builder, the two-axis sweep, controls, verdict.

**Out of scope, do not start:** fitting a custom lens, any persistent-storage integration, multi-turn persistence, SAE / dictionary cross-checks, downstream training work, or anything belonging to the broader (proprietary) architecture beyond this experiment.

## The gate

Run before every mile, no exceptions:

```
.\.venv\Scripts\python.exe scripts\verify.py
```

Seven checks, exit 0 required. It exists because an import check is a weak gate -
it passes on a CPU-only torch build, on a torch/transformers pairing that cannot
capture hidden states, and on a drifted the substrate SHA. All three would surface as
bad science several miles later instead of as a red line here.

`src/probe/pins.py` is the single source of truth for SUBSTRATE_VERSION, MODEL_ID,
LENS_REPO, LENS_REVISION, and BAND. Import them; never retype them. Every
results JSON embeds them.

If `verify.py` reports PIN DRIFT, stop. Archive `results/` and restart at Mile 1.
The substrate under test is the instrument - a moved pin is a moved instrument.
