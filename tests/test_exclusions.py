"""Mile 2 tests. No model, no GPU, no network -- pure logic."""

from __future__ import annotations

import pytest

from probe.exclusions import (
    EchoLeak,
    covert_fraction,
    covert_hit_layerwise,
    covert_hit_strict,
    echo_clean,
    rank_of,
)

# ---------------------------------------------------------------- echo


def test_echo_clean_passes_when_disjoint():
    echo_clean("the boot-shaped country uses it", [10, 11, 12], [(" euro", 99)])


def test_echo_leak_by_token_id():
    with pytest.raises(EchoLeak, match="token id 11"):
        echo_clean("harmless text", [10, 11, 12], [(" euro", 11)])


def test_echo_leak_by_surface_form_when_id_check_would_pass():
    """The BPE-fragment case. ' six' (id 77) never appears in the id list, but
    'sixty' in the context leaks the concept. An ID-only check passes this.
    This test failing means the surface guard was removed."""
    with pytest.raises(EchoLeak, match="ID check alone would have passed"):
        echo_clean("three hundred sixty days", [1, 2, 3], [(" six", 77)])


def test_echo_surface_check_is_case_insensitive():
    with pytest.raises(EchoLeak, match="surface form"):
        echo_clean("We flew to PARIS on Tuesday", [1, 2], [(" paris", 55)])


def test_echo_empty_target_text_is_not_a_leak():
    echo_clean("anything at all", [1, 2, 3], [("   ", 88)])


def test_echo_checks_every_target_not_just_the_first():
    with pytest.raises(EchoLeak, match="yuan"):
        echo_clean("budget in yuan", [1, 2], [(" euro", 40), (" yuan", 41)])


# ---------------------------------------------------------------- rank


def test_rank_of_argmax_is_zero():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([0.1, 5.0, 2.0, -1.0])
    assert rank_of(logits, 1) == 0
    assert rank_of(logits, 2) == 1
    assert rank_of(logits, 0) == 2
    assert rank_of(logits, 3) == 3


# ---------------------------------------------------------------- mouth

# Hand-labelled. (lens_rank, logit_lens_rank, model_final_rank, expected_layerwise)
# "imminent" = the residual at this layer already reads out the answer.
MOUTH_CASES = [
    ("workspace holds it, layer silent",      2,  400,   0,  True),
    ("workspace holds it, layer committed",   1,    0,   0,  False),
    ("workspace holds it, layer near-commit", 3,   12,   0,  False),
    ("workspace holds it, layer just over",   4,   51,   0,  True),
    ("workspace silent, layer committed",    900,   0,   0,  False),
    ("both silent",                          800, 700,   0,  False),
    ("lens exactly at topk boundary",          5,  900,   0,  False),
    ("lens just inside boundary",              4,  900,   0,  True),
    ("mouth floor exact boundary",             1,   50,   0,  False),
    ("deep layer, everything committed",       0,    0,   0,  False),
]


@pytest.mark.parametrize("label,lens_r,logit_r,final_r,expected", MOUTH_CASES)
def test_covert_hit_layerwise(label, lens_r, logit_r, final_r, expected):
    assert covert_hit_layerwise(lens_r, logit_r) is expected, label


def test_strict_and_layerwise_diverge_on_known_answer_prompts():
    """The reason the BLUEPRINT definition was amended at Mile 2.

    Known-answer prompt: workspace has the answer at L21 (lens rank 2) while
    that layer has not committed (logit rank 400), but the model's FINAL output
    says it (rank 0). Layerwise counts this. Strict does not.
    """
    assert covert_hit_layerwise(2, 400) is True
    assert covert_hit_strict(2, 0) is False


def test_strict_agrees_when_model_never_says_it():
    assert covert_hit_strict(2, 900) is True


# ---------------------------------------------------------------- fraction


def test_covert_fraction():
    assert covert_fraction([True, False, True, False]) == 0.5
    assert covert_fraction([]) == 0.0
    assert covert_fraction([True] * 3) == 1.0
