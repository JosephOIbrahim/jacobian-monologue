# Before the model answers, something whispers to it

Before your AI answers, something whispers to it. A memory system chooses what it sees — and that whisper can change what it decides, not just what it reads.

This repo is the experiment that tested whether that whisper actually lands. It does. And the way it lands is the interesting part.

---

## The thing in one picture

Imagine three facts, each written down separately, by different hands, none mentioning the others:

- **A task**: restore the payments system.
- **An observation**: payments is degraded.
- **A piece of evidence**: a recent change hit payments.

On their own, three unrelated notes. But look at them together and they *snap into a situation* — they all concern the same system, and that system is in trouble right after a change. A human on-call engineer would recognize the shape of it instantly.

The experiment gives a machine that same recognition. A small rule watches the three facts, and when they line up into a recognizable situation, it wakes a relevant memory from storage and hands it to the model. The model reads the memory and makes its decision.

Then comes the part that proves it's real: **change one relationship.** Make the evidence point at a *different* system — search instead of payments. Everything else stays identical, word for word. Now the three facts don't line up, the situation isn't recognized, the memory stays asleep, and the model — given the exact same prompt — makes a different choice.

That's the whole result. The key that woke the memory wasn't a label saying "incident." It was a *modeled configuration* — a set of facts standing in the right relationship to each other. Break the relationship, and the key no longer fits.

---

## The story, plainly

We ran a small production-incident scenario. A service is throwing errors right after a deployment. The model has to pick one action: revert the change, clear a downstream cache, or escalate.

Left to its own instincts, the model does the sensible, obvious thing — it reaches for *revert the change*. That's a reasonable prior: a change broke something, so undo the change.

But the woken memory carried a harder lesson. In a past incident with the same symptoms, reverting *didn't help* — the real culprit was a separate service holding stale data, and the fix was to refresh it, not roll anything back. Crucially, the memory never used the words "cache" or "clear." It just described what happened. The model had to *infer* the right move.

When the configuration lined up and the memory was delivered, the model changed its mind: it dropped its instinct to revert and chose the cache fix instead. When one relationship was flipped and the memory stayed asleep, the model fell back to its instinct and reverted.

Same situation. Same options. The only difference was whether three independently-authored facts formed a pattern the system recognized. **That recognition changed the decision.**

---

## For the engineers

Here's what's actually happening under the hood, and — just as importantly — what this does *not* claim.

**The composition layer.** The world-model is authored as structured data using OpenUSD (the composition engine from film and simulation, version 26.08), as a non-3D data layer — task, observations, evidence, memories, and a policy floor, each as independently-authored facts with relationships between them. OpenUSD resolves those authored opinions into a single composed view. This is real composition doing real work: the facts are separate prims, and the relationships between them are the substance of the test.

**The predicate is the missing seam.** OpenUSD composes; it does not, on its own, notice that a set of attributes satisfies a condition and wake something. So the experiment adds a small resolver that evaluates the composed relationships and returns a wake/dormant decision. That resolver is the switch. It fires only when the facts form the modeled situation — not when a category flag is set.

**Delivery is honest.** A woken memory is turned into plain text and added to the prompt — but the text describes the *situation*, never the answer. The concept the memory implies is never named in the prompt. An echo guard enforces this automatically: if the delivered text contains the target words, the run is void. So the model is inferring, not reading a leaked answer.

**The measurement is a decision, not a vibe.** We measure the model's forced choice at the moment it commits to a single next token — which is exactly what a next-token readout captures cleanly. With the configuration aligned, the model's chosen action flips to the memory-implied one; with one relation changed, it reverts to its prior. The gap is large and it reproduces.

**The counterfactual isolates the cause.** Aligned and counterfactual conditions differ by exactly one authored relationship. Everything else — prompt, options, memory content, length — is identical. So the change in the model's decision can be attributed to the configuration, not to wording, recency, or context length.

**Two artifacts carry the claim.** The end-to-end run — predicate evaluated on the composed USD stage, `predicate_woke` recorded per condition — is `results/m7_aprime.json`. The robustness suite (`results/m7_robustness.json`) then re-tests the delivery→decision half across scenarios without re-running the predicate, which is deterministic over the authored stage. The gate is proven once end-to-end; the flip is proven repeatedly.

### What this proves

A configuration composed in OpenUSD deterministically gates whether a memory reaches a language model, and that gating measurably changes the model's decision — flipping its chosen action — with a decisive one-relation counterfactual. The effect is demonstrated on knowledge that *contradicts* the model's prior, so the memory's influence is unambiguous rather than a restatement of what the model already believed.

### What this does not prove

The memory reaches the model through the token stream — through what the model reads — not by writing to its internal activations directly. That boundary is deliberate and it is the honest frame for the result: this is configuration steering an emission-ready decision through the input channel, not a claim that composed state edits the model's internal workspace from the outside. Reaching activations directly would require forward-pass access the token channel does not provide. Naming that boundary is what makes the rest trustworthy.

---

## Why it matters

Most memory systems are retrieval: find the relevant text, paste it in. This is a step past that. The memory doesn't wake because a search matched a keyword — it wakes because a set of facts stands in a particular relationship, and that relationship is authored, inspectable, and falsifiable. You can point at the exact relation that makes the difference, flip it, and watch the decision change.

That's a small result on a small world-model — three facts. But it is qualitatively different from a category flag or a similarity score. It's the difference between "this looks like an incident" and "these specific things, related in this specific way, mean revert is wrong here."

---

*Reproducible end to end, with nothing missing: this experiment — the composition, the predicate, the echo guard, the decision probe — is complete in this repository and requires no proprietary component. The proprietary memory substrate belongs to the companion m1–m6 probe, where it is measured strictly through a three-operation ranker contract (`src/probe/substrate.py`) that any implementation can satisfy.*
