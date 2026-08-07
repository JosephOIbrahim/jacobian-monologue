"""Frozen version pins for the experiment.

Single source of truth. Imported by scripts/verify.py and by every experiment
run, which embeds these into results/*.json.

Changing SUBSTRATE_VERSION mid-experiment invalidates every result recorded
before the change. That is not a style rule -- the substrate under test is the
instrument. If you deliberately move the pin, archive results/ first and
restart at Mile 1.
"""

from __future__ import annotations

# Substrate build identifier at experiment start. The substrate is proprietary
# and not included in this repo; this pin records which build produced the
# results, and reproductions should pin their own ranker build here.
SUBSTRATE_VERSION = "substrate-mile1-frozen"

# Model + lens the probe is fitted to. Set at Mile 1, do not drift.
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_DTYPE = "bfloat16"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"

# Determined empirically at Mile 1. None until then -- verify.py does not
# enforce this one, but experiment runs must refuse to proceed while it is None.
BAND: tuple[int, int] | None = (20, 22)
LENS_FILENAME = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
# Chosen over the sibling early-stopped fit (417 prompts, identity_distance
# 0.514843). Same 406 MB shape. The qwen-n1000 revision exists for this file and
# shallow layers converge slowest, so the fuller fit buys margin for free.

# BAND RATIFIED AT MILE 1 = (20, 22).
# Chosen on covert readout, not raw recall. Raw top-5 recall proposed (22, 30),
# but L23-30 is the mouth: J-lens and logit-lens agree to within noise there
# (covert 0.00 across all of L23-30). L20-22 is the only span where the lens
# carries the answer while the output has not committed -- J 0.32/0.46/0.54
# against logit 0.14/0.14/0.32.
#
# RISK CARRIED FORWARD: peak covert fraction is 0.21 at L21 on easy known-answer
# prompts. K4 requires >= 0.70 at Mile 4. This is an early warning that K4 may
# fire. Do NOT widen BAND to chase it -- widening imports mouth contamination,
# which is the failure K3/K4 exist to catch.
