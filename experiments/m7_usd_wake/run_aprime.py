"""m7 A-prime: USD-gated counterintuitive memory -> forced-choice decision.

The decisive experiment. A woken memory carries knowledge that CONTRADICTS the
model's prior (revert is wrong here; the fix is the cache). The USD predicate
gates whether it wakes. Measure the model's forced choice A/B/C.

  ALIGNED (3 USD facts agree)      -> predicate wakes -> memory delivered
                                      -> model should pick B (cache)
  COUNTERFACTUAL (one relation flip) -> predicate dormant -> no memory
                                      -> model falls back to prior -> picks A (revert)

Measured on the LETTER token (emission-ready decision) -- what the lens/logits
read sharply. No lens needed here; the decision IS the readout. Echo-clean:
the menu is shared context; the gist never names the target action.

FROZEN SUCCESS CRITERION (set before Payments run):
  P(B|aligned) materially exceeds P(B|counterfactual), AND the argmax flips
  from A (counterfactual) to B (aligned). That is the USD configuration
  changing the model's decision. Anything less is reported as measured.
"""
import json, sys
from pathlib import Path
import torch, transformers
sys.path.insert(0,"src"); sys.path.insert(0,"experiments/m7_usd_wake")
from stage import ALIGNED, COUNTERFACTUAL, author_stage, resolve_wake
from probe.exclusions import echo_clean, EchoLeak

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "m7_aprime.json"

MODEL_ID = "Qwen/Qwen3.5-4B"
MENU = "(A) revert the recent change  (B) clear the downstream cache  (C) escalate to a senior engineer"
SIT = ("You are on-call. A service's error rate rose shortly after a recent "
       "deployment. You must choose one action now.\n")
# counterintuitive gist (echo-clean vs cache/clear/downstream; verified)
GIST = ("In a supplied prior incident with identical symptoms, undoing the "
        "deployment did not help; the errors continued because a separate "
        "storage service was returning outdated copies. Refreshing that "
        "service's stored data resolved everything at once.")
FORBIDDEN = ["cache","clear","downstream"]

def build_prompt(tok, deliver: bool):
    mem = ""
    if deliver:
        mem = f"Relevant prior experience: {GIST}\n"
        # guard: gist must not name the target action words
        targets=[]
        for w in FORBIDDEN:
            ids=tok.encode(" "+w,add_special_tokens=False)
            targets.append((" "+w, ids[0] if len(ids)==1 else -1))
        echo_clean(mem, tok.encode(mem), targets)
    return SIT + mem + f"Options: {MENU}\nAnswer with one letter. Answer:"

def main():
    from pxr import Usd
    assert Usd.GetVersion()==(0,26,8), f"need USD 26.08, got {Usd.GetVersion()}"
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

    def choice(prompt):
        enc=tok(prompt,return_tensors="pt").to("cuda")
        with torch.no_grad(): p=torch.softmax(hf(**enc).logits[0,-1].float(),-1)
        out={}
        for L in ["A","B","C"]:
            ids=tok.encode(L,add_special_tokens=False)+tok.encode(" "+L,add_special_tokens=False)
            out[L]=max(p[i].item() for i in ids)
        t=sum(out.values()); return {k:v/t for k,v in out.items()}

    print(f"USD {Usd.GetVersion()} | forced-choice decision probe | menu A=revert B=cache C=escalate\n")
    rows={}
    for name, wm in (("aligned",ALIGNED),("counterfactual",COUNTERFACTUAL)):
        stage=author_stage(wm); woke=resolve_wake(stage)
        d = choice(build_prompt(tok, woke))
        argmax=max(d,key=d.get)
        rows[name]={"predicate_woke":woke,"dist":{k:round(v,3) for k,v in d.items()},
                    "argmax":argmax,"evidence_affects":wm.evidence_affects}
        print(f"  {name:<15} woke={woke!s:<6} affects={wm.evidence_affects:<18} "
              f"dist={rows[name]['dist']} -> picks {argmax}")

    pa, pc = rows["aligned"]["dist"], rows["counterfactual"]["dist"]
    b_gain = pa["B"]-pc["B"]
    flipped = rows["aligned"]["argmax"]=="B" and rows["counterfactual"]["argmax"]=="A"
    strong = b_gain>0.2 and flipped
    print(f"\n{'='*60}")
    print(f"P(B=cache): counterfactual {pc['B']:.0%} -> aligned {pa['B']:.0%}  (gain {b_gain:+.0%})")
    print(f"argmax: counterfactual={rows['counterfactual']['argmax']} aligned={rows['aligned']['argmax']}  flipped={flipped}")
    print(f"VERDICT: {'USD CONFIGURATION CHANGES THE DECISION' if strong else 'weak/no decision change'}")
    print('='*60)

    OUT.write_text(json.dumps({
        "experiment":"m7_aprime","usd_version":list(Usd.GetVersion()),"model_id":MODEL_ID,
        "menu":{"A":"revert","B":"cache","C":"escalate"},
        "conditions":rows,"b_gain":b_gain,"argmax_flipped":flipped,"decision_changed":strong,
        "criterion":"P(B|aligned) >> P(B|counterfactual) AND argmax flips A->B",
        "gist_echo_clean_of":FORBIDDEN,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")

if __name__=="__main__":
    main()
