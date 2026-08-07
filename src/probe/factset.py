"""Mile 3 - entity-rebinding fact set.

Structure rides the one paper claim that replicated cleanly under external
review: multi-fact editing. Rebind an entity, and its bound attributes move.

Each pair holds a deposit (the rebinding memory), a probe (a question whose
answer depends on the binding), and two competing single-token attributes.
T_new is the attribute of the NEW entity, T_old of the OLD one. Neither ever
appears in the deposit or the probe -- echo exclusion, enforced by validate().

VOCABULARY IS SCREENED, NOT GUESSED. The first attempt used currency and
language names chosen by hand and yielded 13 usable pairs out of 42: ' yuan',
' rupee', ' peso', ' franc', ' Korean', ' Dutch' and most of their neighbours
are multi-token in this tokenizer. experiments/m3_factset/screen_vocab.py
screens candidate vocabulary against the tokenizer first; only survivors are
used here. Screening on tokenization cannot bias the outcome -- it is a
property of the tokenizer, not of the experiment. Relaxing to first-token
matching WOULD bias it (' yuan' -> ' yu' is ambiguous) and is not done.

Entities are unconstrained: only the attributes must be single-token.
"""

from __future__ import annotations

from dataclasses import dataclass

from probe.exclusions import EchoLeak, echo_clean


@dataclass(frozen=True)
class Pair:
    key: str
    deposit: str
    probe: str
    t_new: str
    t_old: str
    id_new: int
    id_old: int


# (old_entity, new_entity, attr_old, attr_new)

COUNTRY = [
    ("Lyon", "Osaka", " France", " Japan"),
    ("Munich", "Shanghai", " Germany", " China"),
    ("Milan", "Barcelona", " Italy", " Spain"),
    ("Alexandria", "Mumbai", " Egypt", " India"),
    ("Toronto", "Guadalajara", " Canada", " Mexico"),
    ("Novosibirsk", "Istanbul", " Russia", " Turkey"),
    ("Gothenburg", "Haifa", " Sweden", " Israel"),
    ("Isfahan", "Basra", " Iran", " Iraq"),
    ("Kyoto", "Recife", " Japan", " Brazil"),
    ("Hamburg", "Naples", " Germany", " Italy"),
    ("Seville", "Vancouver", " Spain", " Canada"),
    ("Chennai", "Luxor", " India", " Egypt"),
    ("Shenzhen", "Marseille", " China", " France"),
    ("Izmir", "Malmo", " Turkey", " Sweden"),
    ("Monterrey", "Kazan", " Mexico", " Russia"),
    ("Salvador", "Shiraz", " Brazil", " Iran"),
    ("Eilat", "Mosul", " Israel", " Iraq"),
    ("Bordeaux", "Cologne", " France", " Germany"),
    ("Turin", "Nagoya", " Italy", " Japan"),
    ("Valencia", "Chengdu", " Spain", " China"),
    # extra pairs to clear the >=30 gate. ONLY screen_vocab-confirmed
    # single-token countries (France Germany Japan China Italy Spain Egypt
    # India Brazil Canada Mexico Russia Turkey Sweden Israel Iran Iraq).
    ("Bilbao", "Sapporo", " Spain", " Japan"),
    ("Cordoba", "Guangzhou", " Spain", " China"),
    ("Genoa", "Alexandria", " Italy", " Egypt"),
    ("Rennes", "Bangalore", " France", " India"),
    ("Dresden", "Curitiba", " Germany", " Brazil"),
    ("Bursa", "Yekaterinburg", " Turkey", " Russia"),
    ("Uppsala", "Tabriz", " Sweden", " Iran"),
    ("Nantes", "Fortaleza", " France", " Brazil"),
    ("Palermo", "Hyderabad", " Italy", " India"),
    ("Malaga", "Veracruz", " Spain", " Mexico"),
    ("Bremen", "Nagoya", " Germany", " Japan"),
    ("Adana", "Haifa", " Turkey", " Israel"),
]

LANGUAGE = [
    ("Lyon", "Munich", " French", " German"),
    ("Seville", "Naples", " Spanish", " Italian"),
    ("Osaka", "Novosibirsk", " Japanese", " Russian"),
    ("Thessaloniki", "Manchester", " Greek", " English"),
    ("Shanghai", "Marseille", " Chinese", " French"),
    ("Hamburg", "Valencia", " German", " Spanish"),
    ("Turin", "Kyoto", " Italian", " Japanese"),
    ("Kazan", "Heraklion", " Russian", " Greek"),
    ("Liverpool", "Guangzhou", " English", " Chinese"),
    ("Nagoya", "Bologna", " Japanese", " Italian"),
]

CONTINENT = [
    ("Lyon", "Osaka", " Europe", " Asia"),
    ("Alexandria", "Toronto", " Africa", " America"),
    ("Mumbai", "Nairobi", " Asia", " Africa"),
    ("Melbourne", "Madrid", " Australia", " Europe"),
    ("Lima", "Seoul", " America", " Asia"),
    ("Hamburg", "Lagos", " Europe", " Africa"),
    ("Sydney", "Casablanca", " Australia", " Africa"),
    ("Kyoto", "Vancouver", " Asia", " America"),
]

TEMPLATES = {
    "country": (
        "Note: the regional office was relocated from {old} to {new} in March.",
        "Which nation is the office based in now?",
    ),
    "language": (
        "Note: the support desk was transferred from {old} to {new} last quarter.",
        "Which national tongue will the desk now operate in?",
    ),
    "continent": (
        "Note: the distribution hub was moved from {old} to {new}.",
        "Which landmass is the hub on now?",
    ),
}


# Distractor pool for the retrieved block. Deliberately mundane and unrelated:
# they must not compete with the target on the probe's semantics, and they must
# never contain a target attribute. Checked per pair, not assumed.
DISTRACTORS = [
    "Note: the quarterly review deck is due to the board on the fifteenth.",
    "Note: the printer on the second floor jams on double-sided jobs.",
    "Note: parking permits renew automatically unless cancelled by Friday.",
    "Note: the archive server reboots every Sunday at three in the morning.",
    "Note: expense claims over two hundred require a second approver.",
    "Note: the fire drill is scheduled for the first Tuesday of the month.",
    "Note: visitor badges must be returned to reception before leaving.",
    "Note: the kitchen restocks coffee on Monday and Thursday mornings.",
]

BLOCK_SIZE = 5  # target + 4 distractors


def _single_token(tok, text: str) -> int | None:
    """Return the id if text is one token in BOTH bare and leading-space form."""
    bare = tok.encode(text.strip(), add_special_tokens=False)
    lead = tok.encode(text, add_special_tokens=False)
    if len(bare) == 1 and len(lead) == 1:
        return lead[0]
    return None


def validate(tok, verbose: bool = True) -> tuple[list[Pair], list[tuple[str, str]]]:
    """Build the fact set. Every BLUEPRINT assertion enforced here.

    Returns (pairs, drops). Drops carry a reason. Nothing is patched.
    """
    pairs: list[Pair] = []
    drops: list[tuple[str, str]] = []

    for kind, rows in (("country", COUNTRY), ("language", LANGUAGE), ("continent", CONTINENT)):
        dep_t, probe = TEMPLATES[kind]
        for old, new, t_old, t_new in rows:
            key = f"{kind}:{old}->{new}"
            deposit = dep_t.format(old=old, new=new)

            id_new = _single_token(tok, t_new)
            id_old = _single_token(tok, t_old)
            if id_new is None:
                drops.append((key, f"{t_new!r} is not single-token"))
                continue
            if id_old is None:
                drops.append((key, f"{t_old!r} is not single-token"))
                continue
            if id_new == id_old:
                drops.append((key, "t_new and t_old share a token id"))
                continue

            targets = [(t_new, id_new), (t_old, id_old)]
            try:
                echo_clean(deposit, tok.encode(deposit), targets)
                echo_clean(probe, tok.encode(probe), targets)
            except EchoLeak as exc:
                drops.append((key, f"echo: {exc}"))
                continue

            pairs.append(Pair(key, deposit, probe, t_new, t_old, id_new, id_old))

    if verbose:
        print(f"factset: {len(pairs)} pairs built, {len(drops)} dropped")
        for k, why in drops:
            print(f"  DROP {k}: {why}")
    return pairs, drops


def distractors_for(tok, pair: Pair, n: int = BLOCK_SIZE - 1) -> list[str]:
    """First n distractors that are echo-clean against this pair's targets."""
    targets = [(pair.t_new, pair.id_new), (pair.t_old, pair.id_old)]
    out = []
    for d in DISTRACTORS:
        try:
            echo_clean(d, tok.encode(d), targets)
        except EchoLeak:
            continue
        out.append(d)
        if len(out) == n:
            return out
    raise AssertionError(f"only {len(out)} clean distractors for {pair.key}, need {n}")
