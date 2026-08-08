# Social posts — final, as posted (2026-08-08)

Attach the schema image to both — it carries the 9% → 65% result for anyone
who never taps the link. (PNG export: render `schema-social.svg` — or its
dark twin `schema-social-dark.svg` — at ~1360px or wider.)

---

## LinkedIn (VFX audience)

I used USD to change an AI's mind.

Same OpenUSD we use for scene assembly. Except the scene is a situation, not geometry.

Three facts, authored as separate prims:

• a task — restore the payments system
• an observation — payments is degraded
• evidence — a change just hit payments

When the relationships line up, a rule wakes a relevant memory and hands it to the AI.

The AI changes its decision. The right call went from 9% to 65%.

Then I flipped one relationship — pointed the evidence at a different system. Same prompt, word for word. The memory stays asleep. The AI goes back to its old answer.

One relation. Different decision.

If you work in USD, you already have the instincts for this. Cognitive state composes the way scenes do.

Repo and full writeup:
https://github.com/JosephOIbrahim/jacobian-monologue

What would you author into a situation your tools should recognize?

---

## Twitter / X (AI labs audience)

I gated an AI's memory with a scene graph. Flipping one relation flips its decision.

Three facts authored separately in OpenUSD: a task, an observation, evidence. When the relations align, a rule wakes a dormant memory. The model reads it.

P(right action): 9% → 65%. Decision flips.

Flip one relation. Same prompt, word for word. Memory sleeps. Model reverts to its prior.

Held across scenarios behind a pre-registered fairness gate.

The boundary, stated plainly: this works through what the model reads — the token stream — not activation writes.

The gate costs 19µs. One forward costs ~230ms. ~12,000× cheaper.

Also in the repo: a pre-registered null. Does memory ranking drive workspace occupancy? No — recency won. The kill criterion fired. Reported in full.

https://github.com/JosephOIbrahim/jacobian-monologue
