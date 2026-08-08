# Install - the no-panic version

Written for someone who thinks in **layers, timelines, and color** more than in terminals. If a command scares you, that's fine - copy it, paste it, watch what happens. You can't break anything that a fresh clone doesn't fix.

---

## 🎨 The mental model first

Think of this like opening someone else's **Houdini scene** or a layered **PSD** you didn't build.

- The **model** (Qwen) is the render engine. Big, does the heavy work. You download it once.
- The **lens** is your loupe - it lets you *see into* a layer that's normally hidden.
- **the substrate** is the thing on trial. It's a separate project, like a linked asset you reference in.
- The **venv** is a clean workspace so nothing bleeds into your other projects. Like working in a fresh scene file instead of your messy one.

You're not writing code. You're **setting up the scene, then hitting render.**

---

## 🧰 What you need on the machine

Three things. Check them like checking you have the right plugins before opening a file:

| You need | Why | Check it |
|---|---|---|
| **Python 3.12** | the language everything runs in | `python --version` |
| **A CUDA GPU** (NVIDIA) | the model needs real horsepower - a 4090 is plenty | you have one if you do VFX |
| **`uv`** | a fast installer, cleaner than the old `pip` | `uv --version` |
| **A substrate checkout** | the asset being tested - point at it, or supply your own ranker | it's the whole point |

Missing Python or uv? Install Python from python.org, then `pip install uv`. That's it.

---

## 🎬 Setup - six moves, in order

Open a terminal **in the project folder**. Paste these one at a time. Read the note under each - it tells you what you just did.

**1 - Make the clean workspace**
```bash
uv venv --python 3.12
```
*Creates an isolated sandbox. Nothing here touches your system Python or your other work. Like a new empty scene.*

**2 - Get the render engine (the GPU version - this matters)**
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```
*This is a ~2.5 GB download, so grab a coffee. The `--index-url` bit is not optional - leave it off and you get a version that runs on CPU, which is like rendering on your laptop's onboard graphics. Slow enough to ruin your day.*

**3 - Get the loupe (the Jacobian lens)**
```bash
uv pip install "jlens @ git+https://github.com/anthropics/jacobian-lens"
```
*This is Anthropic's lens - the tool that lets you see into the model's hidden layer. It's a separate open-source project (Apache-2.0), so you install it straight from its home, not from this repo. This repo references it; it doesn't ship it.*

**4 - Install the experiment itself**
```bash
uv pip install -e .
```
*Wires up the probe code. Fast. The `-e` means "editable" - changes to the code take effect live, like a referenced file instead of an imported copy.*

**5 - Point at the substrate (the asset on trial)**
```bash
uv pip install -e /path/to/substrate
```
*Swap `/path/to/substrate` for wherever the memory substrate lives on your machine. This is the proprietary piece being measured - it isn't included in this repo. If you don't have it, supply your own ranker (see `src/probe/substrate.py` for the interface) and pass it in.*

**6 - Check the scene loaded clean**
```bash
python scripts/verify.py
```
*Runs 7 checks. You want a wall of green `[PASS]`. This is your "did all my textures load" moment - if something's red, it tells you exactly what, and you fix that one thing before going further.*

---

## ✅ You're good when

`verify.py` prints a board of green `[PASS]` lines and **no red `[FAIL]`**. That's the render preview coming up clean.

One line may say `[SKIP] substrate ranker` - that's normal if you don't have the proprietary substrate. It's like a missing optional plugin: the scene still opens. You only need a ranker (yours or the substrate) for the m1–m4 experiments; **the m7 headline runs without it.**

---

## 🆘 When it goes sideways

Nothing here is fatal. Match your symptom:

| What you see | What it means | The move |
|---|---|---|
| `torch ... +cpu` in the checks | you got the laptop-graphics version | redo step 2, keep the `--index-url` |
| `CUDA not available` | GPU isn't being seen | check your NVIDIA drivers are current |
| `[SKIP] substrate ranker` | no ranker installed | expected without the proprietary substrate; not needed for m7. For m1–m4, do step 5 or supply your own ranker (see `src/probe/substrate.py`) |
| substrate check shows red `[FAIL]` | a ranker IS installed but broke the contract | fix the path in step 5, or fix your ranker's `rank_block()` |
| `python: command not found` | try `python3` instead | some machines name it that |

**The universal reset:** delete the `.venv` folder and start again from step 1. Costs you one coffee's worth of download. You genuinely cannot corrupt anything else.

---

## 🎥 What running it actually looks like

Once it's set up, each stage is one line, like `experiments/m4_sweep/run.py`. You'll see a **progress bar** crawl across - that's the model reasoning through each memory pair, one at a time. When it finishes, it drops a `.json` in `results/` and, at the end, **one plot**.

That plot is the whole point. Everything above is just getting the scene to open.
