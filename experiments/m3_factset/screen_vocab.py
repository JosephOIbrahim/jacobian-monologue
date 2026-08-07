"""Screen candidate attribute vocabulary against the tokenizer.

Runs BEFORE the fact set is built. Single-tokenness is a property of the
tokenizer alone -- filtering on it cannot bias the experimental outcome, which
is why screening here is legitimate and relaxing to first-token matching is not.
"""
import transformers

tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")


def ok(w: str) -> bool:
    w = w.strip()
    return (
        len(tok.encode(w, add_special_tokens=False)) == 1
        and len(tok.encode(" " + w, add_special_tokens=False)) == 1
    )


VOCAB = {
    "continent": "Europe Asia Africa America Australia Antarctica",
    "country": (
        "France Germany Japan China Italy Spain Egypt India Brazil Canada "
        "Mexico Russia Poland Greece Turkey Norway Sweden Portugal Austria "
        "Belgium Ireland Kenya Nigeria Chile Peru Vietnam Thailand Korea "
        "Israel Iran Iraq Cuba Chad Mali Ghana Finland Denmark Hungary"
    ),
    "capital": (
        "Paris Berlin Rome Madrid Cairo Tokyo Beijing Moscow Oslo Lima "
        "Athens Dublin Vienna Warsaw Ankara Seoul Delhi Lisbon Prague Bern "
        "Doha Kiev Riga Baku Quito Havana Manila Bogota Nairobi Sydney"
    ),
    "language": (
        "French German Spanish Italian Japanese Arabic Russian Greek Turkish "
        "English Chinese Hebrew Latin Hindi Danish Polish Korean Dutch"
    ),
    "currency": "euro yen won dollar pound rupee peso franc real krone",
    "colour": "red blue green black white yellow orange purple brown grey gold",
    "sport": "tennis golf soccer hockey rugby cricket boxing skiing swimming",
    "metal": "gold silver iron copper steel tin lead zinc nickel",
}

for kind, words in VOCAB.items():
    ws = words.split()
    good = [w for w in ws if ok(w)]
    bad = [w for w in ws if not ok(w)]
    print(f"{kind:<10} {len(good):>2}/{len(ws):<2}")
    print(f"{'':<10} OK: {' '.join(good)}")
    if bad:
        print(f"{'':<10} XX: {' '.join(bad)}")
    print()
