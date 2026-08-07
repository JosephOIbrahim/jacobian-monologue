# Social drafts

Two versions. Attach the schema image (aligned-vs-counterfactual) to either.

---

## LinkedIn

**Before an AI answers, something whispers to it — and I ran the experiment to find out whether that whisper actually changes its mind.**

It does. Here's the setup, in plain terms:

Three facts, each written separately, none mentioning the others — a task ("restore payments"), an observation ("payments is degraded"), a piece of evidence ("a change hit payments"). On their own, three unrelated notes. Together, they snap into a recognizable situation.

A small rule watches those facts. When they line up, it wakes a relevant memory from storage and hands it to the model.

The scenario: a service is erroring after a deploy. Pick one action — revert, clear a cache, or escalate. Left alone, the model's instinct is to revert (a change broke it, so undo the change). But the woken memory carried a harder lesson: last time, reverting didn't help — the real fix was a stale downstream cache. The memory never said "cache." The model had to infer it.

When the facts lined up and the memory woke, the model changed its mind and chose the cache fix.

Then the part that proves it's real: I flipped one relationship — made the evidence point at a different system. Same prompt, word for word. The facts no longer lined up, the memory stayed asleep, and the model fell back to its instinct and reverted.

One relationship changed the decision.

The honest boundary: the memory reaches the model through what it reads, not by editing its internals. That's the point, not a caveat — it's configuration steering a decision through the input channel, with a clean counterfactual isolating the cause.

Built on OpenUSD (yes, the film/simulation composition engine) as a non-3D data layer, with an echo guard so the model infers rather than reads the answer. Repo + full writeup in comments.

What would you author into a situation your AI should recognize?

---

## Twitter / X

**Before an AI answers, something whispers to it. I tested whether that whisper changes its mind.**

It does — and flipping one fact flips the decision.

🧵

---

Three facts, written separately:
· task: restore payments
· observation: payments is degraded
· evidence: a change hit payments

Alone, unrelated notes. Together, they snap into a situation a rule can recognize — and wake a memory.

---

Scenario: service erroring after a deploy. Pick: revert / clear cache / escalate.

The model's instinct: revert.

The woken memory's lesson: last time revert failed — it was a stale cache. (The memory never says "cache." The model infers it.)

Memory delivered → model picks the cache fix.

---

Then the proof: I flip ONE relationship — evidence now points at a different system. Same prompt, word for word.

Facts no longer line up → memory stays asleep → model reverts to instinct.

One relation. Different decision.

---

Honest boundary: the memory works through what the model reads, not by editing its internals. That's the point — configuration steering a decision through the input channel, with a clean counterfactual.

Built on OpenUSD as a data layer. Repo down below.
