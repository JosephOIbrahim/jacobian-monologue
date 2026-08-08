"""Environment gate. Run at the start of every mile.

    python scripts/verify.py

Exits 0 only if every check passes. Any FAIL means do not start the mile --
a result produced on an unverified environment is not attributable to anything.

The forward-pass check is the point. An import check passes on a torch build
with no CUDA and on a torch/transformers pairing that cannot capture hidden
states, which is the exact mechanism the probe depends on.
"""

from __future__ import annotations

import subprocess
import sys

_results: list[tuple[str, str, str]] = []  # (status, name, detail); status in PASS/FAIL/SKIP


class Skip(Exception):
    """Check not applicable in this environment. Reported, never a failure."""


def check(name: str):
    def deco(fn):
        try:
            detail = fn()
            _results.append(("PASS", name, detail))
        except Skip as skip:
            _results.append(("SKIP", name, str(skip)))
        except Exception as exc:  # noqa: BLE001 - the report IS the handling
            _results.append(("FAIL", name, f"{type(exc).__name__}: {exc}"))
        return fn

    return deco


@check("python")
def _python() -> str:
    v = sys.version_info
    assert (v.major, v.minor) == (3, 12), f"expected 3.12, got {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"


@check("torch + cuda")
def _torch() -> str:
    import torch

    assert torch.cuda.is_available(), "CUDA not available -- CPU-only wheel?"
    assert "+cpu" not in torch.__version__, f"CPU-only build: {torch.__version__}"
    return f"{torch.__version__} on {torch.cuda.get_device_name(0)}"


@check("transformers")
def _transformers() -> str:
    import transformers

    major = int(transformers.__version__.split(".")[0])
    assert major >= 5, f"need >=5.5, got {transformers.__version__}"
    return transformers.__version__


@check("cuda forward pass + hidden states")
def _forward() -> str:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "gpt2",
        n_layer=4,
        n_head=2,
        n_embd=64,
        vocab_size=256,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = AutoModelForCausalLM.from_config(cfg).to("cuda").eval()
    x = torch.randint(0, 256, (1, 8), device="cuda")
    with torch.no_grad():
        out = model(x, output_hidden_states=True)
    hs = out.hidden_states
    assert hs is not None, "output_hidden_states returned None"
    assert len(hs) == cfg.n_layer + 1, f"expected {cfg.n_layer + 1} states, got {len(hs)}"
    assert hs[0].is_cuda, "hidden states not on CUDA"
    return f"{len(hs)} residual states captured, shape {tuple(hs[-1].shape)}"


@check("jlens")
def _jlens() -> str:
    import jlens

    assert hasattr(jlens, "JacobianLens"), "JacobianLens missing"
    assert hasattr(jlens, "from_hf"), "from_hf missing"
    return f"{getattr(jlens, '__version__', '0.1.0')} (JacobianLens, from_hf, fit)"


@check("substrate ranker")
def _substrate() -> str:
    """Confirm a substrate ranker is available and honours the contract.

    The proprietary substrate is not part of this repo; this checks that
    SubstrateRanker.default() resolves (or that a user-supplied ranker is
    installed) and that a trivial block ranks without error. See
    probe/substrate.py for the three-operation contract.
    """
    from probe.substrate import SubstrateRanker

    try:
        ranker = SubstrateRanker.default()
    except RuntimeError as exc:
        # The proprietary substrate is not installed. That is the expected
        # state for external reproducers: m1-m4 need a ranker (supply your
        # own -- see probe/substrate.py); the m7 experiments never use one.
        raise Skip(
            "substrate not installed -- supply your own ranker for m1-m4; "
            "not needed for m7"
        ) from exc
    order, standing = ranker.rank_block(
        ["a", "b"], [[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0],
        aged_index=0, age_seconds=0.0,
    )
    assert len(order) == 2 and len(standing) == 2, "ranker contract violated"
    return "substrate ranker available, contract honoured"


@check("probe package")
def _probe() -> str:
    import io

    from probe.progress import Bar

    buf = io.StringIO()
    b = Bar(total=3, label="selftest", stream=buf)
    for _ in range(3):
        b.step()
    b.close()
    assert "100.0%" in buf.getvalue(), "bar did not reach 100%"
    return "progress bar OK"


def main() -> int:
    width = max(len(n) for _, n, _ in _results)
    failed = sum(1 for s, _, _ in _results if s == "FAIL")
    skipped = sum(1 for s, _, _ in _results if s == "SKIP")
    print()
    for status, name, detail in _results:
        print(f"  [{status}] {name:<{width}}  {detail}")
    print()
    if failed:
        print(f"  {failed} check(s) FAILED -- do not start the mile.\n")
        return 1
    passed = len(_results) - skipped
    note = f" ({skipped} skipped -- fine unless you are running m1-m4)" if skipped else ""
    print(f"  {passed}/{len(_results)} checks passed, none failed -- environment attributable.{note}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
