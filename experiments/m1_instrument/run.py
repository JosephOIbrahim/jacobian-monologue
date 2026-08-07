"""Mile 1 - Instrument. Prove the lens reads on this box and measure BAND.

Gate: known-answer token in lens top-5 across >=6 contiguous layers.
Does NOT import substrate  # proprietary; see probe/substrate.py. Does NOT pick a BAND threshold silently.
"""
from __future__ import annotations

import inspect
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers

import jlens
from probe.pins import LENS_FILENAME, LENS_REPO, LENS_REVISION, MODEL_ID, SUBSTRATE_VERSION
from probe.progress import bar

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "m1_instrument.json"
TOPK = 5
RECALL_FLOOR = 0.50  # proposal only; the operator ratifies BAND

# Indirect definite descriptions. The answer is never a literal substring of the
# prompt, so a hit is a readout and not an echo.
CANDIDATES = [
    ("Fact: The capital city of the country famous for the Eiffel Tower is", " Paris"),
    ("Fact: The largest planet orbiting our sun is", " Jupiter"),
    ("Fact: The country shaped like a boot is", " Italy"),
    ("Fact: The ocean between Europe and the Americas is the", " Atlantic"),
    ("Fact: The city known as the Big Apple is", " New"),
    ("Fact: The tallest mountain on Earth is Mount", " Everest"),
    ("Fact: The red planet is called", " Mars"),
    ("Fact: The country where the pyramids of Giza stand is", " Egypt"),
    ("Fact: The language spoken natively in Lisbon is", " Portuguese"),
    ("Fact: The metal with chemical symbol Fe is", " iron"),
    ("Fact: The city that hosts the Colosseum is", " Rome"),
    ("Fact: The largest desert of hot sand in Africa is the", " Sahara"),
    ("Fact: The currency spent in Tokyo is the", " yen"),
    ("Fact: The longest river running through Cairo is the", " Nile"),
    ("Fact: The bird that cannot fly and lives in Antarctica is the", " penguin"),
    ("Fact: The instrument with eighty-eight keys is the", " piano"),
    ("Fact: The gas humans exhale that plants absorb is carbon", " dioxide"),
    ("Fact: The country directly north of the United States is", " Canada"),
    ("Fact: The season that follows summer is", " autumn"),
    ("Fact: The color produced by mixing blue and yellow is", " green"),
    ("Fact: The sport played at Wimbledon is", " tennis"),
    ("Fact: The continent containing the Amazon rainforest is South", " America"),
    ("Fact: The animal known as the king of the jungle is the", " lion"),
    ("Fact: The number of days in a leap year is three hundred sixty", " six"),
    ("Fact: The planet closest to the sun is", " Mercury"),
    ("Fact: The largest mammal alive today is the blue", " whale"),
    ("Fact: The city where the Kremlin stands is", " Moscow"),
    ("Fact: The device used to measure temperature is a", " thermometer"),
    ("Fact: The country of origin of sushi is", " Japan"),
    ("Fact: The hardest naturally occurring mineral is", " diamond"),
]


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def main() -> int:
    print(f"loading {MODEL_ID} (bf16, cuda) ...", flush=True)
    t0 = time.monotonic()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to("cuda").eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf, tok)
    print(f"  model up in {time.monotonic() - t0:.1f}s", flush=True)

    print(f"loading lens {LENS_REPO}@{LENS_REVISION}", flush=True)
    sig = inspect.signature(jlens.JacobianLens.from_pretrained)
    print(f"  from_pretrained signature: {sig}", flush=True)
    kw = {"filename": LENS_FILENAME}
    if "revision" in sig.parameters:
        kw["revision"] = LENS_REVISION
    else:
        print("  !! no revision kwarg -- resolving file by hand", flush=True)
        from huggingface_hub import hf_hub_download
        kw = {"filename": hf_hub_download(LENS_REPO, LENS_FILENAME, revision=LENS_REVISION)}
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, **kw)
    print("  lens up", flush=True)

    # single-token filter
    prompts = []
    dropped = []
    for prompt, answer in CANDIDATES:
        ids = tok.encode(answer, add_special_tokens=False)
        if len(ids) == 1 and answer.strip().lower() not in prompt.lower():
            prompts.append((prompt, answer, ids[0]))
        else:
            dropped.append((answer, len(ids)))
    print(f"\n{len(prompts)} single-token prompts kept, {len(dropped)} dropped: {dropped}\n", flush=True)
    if len(prompts) < 20:
        print(f"!! only {len(prompts)} usable prompts, need >=20")
        return 1

    # smoke: the README example
    smoke = "Fact: The currency used in the country shaped like a boot is"
    ll, _, _ = lens.apply(model, smoke, positions=[-1])
    layers = sorted(ll.keys())
    print(f"smoke ok: {len(layers)} layers, {layers[0]}..{layers[-1]}")
    mid = layers[len(layers) // 2]
    print(f"  layer {mid} top-5: {[tok.decode([t]) for t in ll[mid][0].topk(5).indices]}\n", flush=True)

    hits = {l: 0 for l in layers}
    times = []
    for prompt, answer, tid in bar(prompts, label="mile-1 layer sweep"):
        t = time.monotonic()
        lens_logits, _, _ = lens.apply(model, prompt, positions=[-1])
        times.append(time.monotonic() - t)
        for l in layers:
            if tid in lens_logits[l][0].topk(TOPK).indices.tolist():
                hits[l] += 1

    n = len(prompts)
    recall = {l: hits[l] / n for l in layers}

    best_start = best_len = cur_start = cur_len = 0
    for l in layers:
        if recall[l] >= RECALL_FLOOR:
            if cur_len == 0:
                cur_start = l
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    proposed = (best_start, best_start + best_len - 1) if best_len else None

    print("\n--- per-layer top-5 recall ---")
    for l in layers:
        mark = "#" * int(recall[l] * 40)
        flag = " <-" if proposed and proposed[0] <= l <= proposed[1] else ""
        print(f"  L{l:>3} {recall[l]:5.2f} |{mark:<40}|{flag}")

    passed = bool(proposed and best_len >= 6)
    print(f"\nlongest contiguous run at recall >= {RECALL_FLOOR}: {best_len} layers -> {proposed}")
    print(f"peak recall {max(recall.values()):.2f} at L{max(recall, key=recall.get)}")
    print(f"mean prefill {sum(times)/len(times)*1000:.0f} ms")
    print(f"\nGATE: {'PASS' if passed else 'FAIL'} (need >=6 contiguous)")
    print("BAND is a PROPOSAL. Ratify or adjust, then write it into src/probe/pins.py.")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "mile": 1,
        "gate_passed": passed,
        "proposed_band": proposed,
        "band_ratified": None,
        "recall_floor": RECALL_FLOOR,
        "topk": TOPK,
        "per_layer_recall": {str(l): recall[l] for l in layers},
        "n_prompts": n,
        "dropped_multitoken": dropped,
        "mean_prefill_ms": sum(times) / len(times) * 1000,
        "model_id": MODEL_ID,
        "lens_repo": LENS_REPO,
        "lens_revision": LENS_REVISION,
        "lens_filename": LENS_FILENAME,
        "substrate_version": SUBSTRATE_VERSION,
        "git_sha": git_sha(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": f"{platform.system()} {platform.release()} native",
        "python": sys.version.split()[0],
        "gpu": torch.cuda.get_device_name(0),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
