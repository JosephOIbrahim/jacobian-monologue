# The elevator pitch

*Read whichever length fits the moment. All three say the same true thing.*

---

## 🎤 One breath

> I built a memory system for AI, then used Anthropic's brain-reading tool to check if it actually changes what the AI *thinks* - not just what it reads. It does reach in. But the part I expected to be the lever turned out not to be. I have the receipts either way.

---

## ☕ One coffee

An AI model holds a few concepts "in mind" while it reasons - a workspace, separate from the text it outputs. Anthropic shipped a tool this summer that reads that workspace directly.

I've been building **the substrate**, a memory system that decides which memories an AI sees and how they fade over time. Obvious question: **does the substrate's choices change the workspace?**

So I ran it. Solo. Locked every judgment call *before* seeing the data, so I couldn't fool myself.

**Result, in two halves:**
- Feeding the model a memory measurably shifts its inner workspace toward the right concept. The substrate reaches inside - through nothing but the text it contributes. That's the boundary I care about, and it held.
- But the *specific* thing I predicted would drive it - the substrate's ranking - didn't. Something simpler won: how close the memory sat to the question.

The second half is a **negative result**, and it's the valuable one. It's honest, it's pre-registered, and it tells me exactly which experiment to run next. Most people show you the demo that worked. I can show you the one that didn't, and prove why.

---

## 🍷 One glass, for someone technical

Anthropic's July 2026 workspace paper introduced the Jacobian lens - a cheap, weight-grounded readout of the concepts a model is *disposed to verbalize*, its "J-space." I paired it with my memory substrate to test one claim: **does the substrate's utility ranking predict workspace occupancy of a bound concept, at fixed context content?**

Design: 31 entity-rebinding pairs ("the office moved from Lyon to Osaka" → "which country?"), targets never appearing in-context so a lens hit is a readout, not an echo. Two axes - presence (manipulation check) and order-at-fixed-content (the claim). Kill criteria and statistics frozen before the sweep.

**Axis 1 passed:** rebinding memory shifts the workspace toward the new concept, 24/31 pairs, sign-test p=0.003. Concrete evidence a substrate influences J-space *purely through the token stream* - no forward-pass access, exactly as the architecture predicts.

**Axis 2 inverted:** predicted Spearman ρ < −0.5, measured **+1.0**. The confound was clean and instructive - decaying a memory down the substrate's ranking placed it textually nearer the probe, and recency drove promotion instead of utility. **K2 fired.**

Two bonus facts about the substrate itself, surfaced by the instrument: a direct priority boost cannot raise a fresh memory (fresh memories share a ranking ceiling), and its decay ranking resolves ~3 distinct positions, not 5, when candidates are similar. Both are now in the record, both contradict the naive framing, and I'd rather learn that from my own probe than from a reviewer.

n=31, one model, one layer band, single-token targets. It doesn't prove ranking is inert in general - it proves *this instrument couldn't see a positive effect and recency was strong enough to invert the sign.* The follow-up that kills the confound is specified and waiting.
