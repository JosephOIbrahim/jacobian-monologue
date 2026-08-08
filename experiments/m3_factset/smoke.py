"""Mile 3 smoke: one pair, full weight sweep, show what the substrate actually does."""
import transformers
import torch

from probe.context_builder import Embedder, assert_conditions, build_conditions
from probe.factset import validate

hf = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B", dtype=torch.bfloat16
).to("cuda").eval()
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
embed = Embedder(hf, tok)
print(f"embedder layer = {embed.layer}\n")

pairs, _ = validate(tok, verbose=False)
p = pairs[0]
print(f"pair: {p.key}\n")

conds, diag = build_conditions(tok, embed, p)
print("positions reached :", diag["positions_reached"] and sorted(set(diag["positions_reached"])))
print("positions missing :", diag["positions_missing"])
print("decay ages used   :", diag["dts_used"])
print()

for pos in sorted(conds):
    c = conds[pos]
    print(f"--- position {pos}  (w={c.weight})  content={c.content_hash} order={c.order_hash}")
    for i, ln in enumerate(c.lines, 1):
        tag = "  <== TARGET" if ln == p.deposit else ""
        print(f"    {i}. {ln[:64]}{tag}")
    print()

try:
    assert_conditions(conds, p)
    print("ASSERTIONS PASS")
except AssertionError as e:
    print("ASSERTION FAIL:", e)

print()
print("--- full prompt, position 1 ---")
print(conds[min(conds)].prompt)
